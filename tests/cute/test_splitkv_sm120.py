"""Split-KV + paged varlen correctness for the FA4 SM120 kernels.

Checks FA4 against an f32 reference and, when a vllm FA2 (sm_120 build) is
available, against FA2 for mixed decode/prefill batches, GQA 24/4,
head_dim=256, page size 16, causal (bottom-right aligned). Run:

    CUDA_LAUNCH_BLOCKING=1 python tests/cute/test_splitkv_sm120.py
"""

import os
import sys
import types

import torch

# `flash_attn/__init__.py` exports the FA2 extension first, which needs a built
# flash_attn_2_cuda. Stub the parent package so `flash_attn.cute` imports in a
# pure-CuTe checkout (no FA2 build).
try:
    import flash_attn  # noqa: F401
except Exception:
    _pkg = types.ModuleType("flash_attn")
    _pkg.__path__ = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "flash_attn"))
    ]
    sys.modules["flash_attn"] = _pkg

from flash_attn.cute import flash_attn_varlen_func as fa4

# FA2 cross-check is optional (needs a vllm build exposing sm_120 FA2).
try:
    from vllm.vllm_flash_attn import flash_attn_interface as fa2
except Exception:
    fa2 = None

NHQ, NHK, D, PAGE = 24, 4, 256, 16
SCALE = D ** -0.5


def ref_attn(q, kr, vr, q_lens, k_lens):
    """f32 reference, GQA broadcast, bottom-right causal alignment."""
    outs, qi, ki = [], 0, 0
    for ql, kl in zip(q_lens, k_lens):
        qq = q[qi : qi + ql].float().view(ql, NHK, NHQ // NHK, D)
        kk = kr[ki : ki + kl].float().view(kl, NHK, D)
        vv = vr[ki : ki + kl].float().view(kl, NHK, D)
        qi += ql
        ki += kl
        s = torch.einsum("qhgd,khd->qhgk", qq, kk) * SCALE
        i = torch.arange(ql, device=q.device)
        j = torch.arange(kl, device=q.device)
        mask = j[None, :] <= (kl - ql) + i[:, None]  # (ql, kl)
        s = s.masked_fill(~mask[:, None, None, :], float("-inf"))
        p = s.softmax(-1)
        outs.append(torch.einsum("qhgk,hkd->qhgd", p, vv.permute(1, 0, 2)))
    o = torch.cat(outs)  # (total_q, NHK, G, D)
    return o.view(-1, NHQ, D)


def make_paged(q_lens, k_lens):
    torch.manual_seed(0)
    tq, tk = sum(q_lens), sum(k_lens)
    q = torch.randn(tq, NHQ, D, dtype=torch.bfloat16, device="cuda")
    kr = torch.randn(tk, NHK, D, dtype=torch.bfloat16, device="cuda")
    vr = torch.randn_like(kr)
    k_pages, v_pages = tk // PAGE, tk // PAGE
    kp = torch.zeros(k_pages * PAGE, NHK, D, dtype=torch.bfloat16, device="cuda")
    vp = torch.zeros_like(kp)
    kp[:tk] = kr
    vp[:tk] = vr
    k = kp.view(k_pages, PAGE, NHK, D)
    v = vp.view(v_pages, PAGE, NHK, D)
    cu_q = torch.tensor(
        [0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device="cuda"
    )
    su_k = torch.tensor(k_lens, dtype=torch.int32, device="cuda")
    width = max(k_lens) // PAGE
    rows, p = [], 0
    for kl in k_lens:
        n = kl // PAGE
        row = torch.arange(p, p + n, dtype=torch.int32)
        if n < width:
            row = torch.cat([row, torch.full((width - n,), n - 1, dtype=torch.int32)])
        rows.append(row)
        p += n
    pt = torch.stack(rows).to(torch.int32).cuda()
    return q, k, v, cu_q, su_k, pt, kr, vr


def run_case(q_lens, k_lens, num_splits):
    q, k, v, cu_q, su_k, pt, kr, vr = make_paged(q_lens, k_lens)
    mq, mk = max(q_lens), max(k_lens)
    out4 = fa4(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        seqused_k=su_k,
        max_seqlen_q=mq,
        max_seqlen_k=mk,
        page_table=pt,
        softmax_scale=SCALE,
        causal=True,
        num_splits=num_splits,
    )
    o4 = out4[0] if isinstance(out4, tuple) else out4
    e2 = float("nan")
    if fa2 is not None:
        o2 = fa2.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=mq,
            cu_seqlens_q=cu_q,
            max_seqlen_k=mk,
            block_table=pt,
            seqused_k=su_k,
            causal=True,
            softmax_scale=SCALE,
            fa_version=2,
        )
        o2 = o2[0] if isinstance(o2, tuple) else o2
        e2 = (o2.float() - ref_attn(q, kr, vr, q_lens, k_lens)).abs().max().item()
    r = ref_attn(q, kr, vr, q_lens, k_lens)
    e4 = (o4.float() - r).abs().max().item()
    ok = e4 < 0.01 and (fa2 is None or e2 < 0.01)
    fa2_str = f"err_fa2={e2:.5f}" if fa2 is not None else "fa2=n/a"
    print(
        f"qs={q_lens} ks={k_lens} splits={num_splits}: "
        f"err_ref={e4:.5f} {fa2_str} {'OK' if ok else 'MISMATCH'}",
        flush=True,
    )
    return ok


CASES = [
    ([1], [4096], -1),
    ([1], [4096], 8),
    ([1], [4096], 32),
    ([1, 1, 1, 1], [512, 1024, 2048, 4096], -1),
    ([1] * 8, [304, 784, 1024, 2000, 4096, 512, 112, 2352], 16),
    ([1, 128], [256, 4096], -1),
    ([64, 128], [2048, 4096], -1),
    ([1], [64], 4),
]


def main():
    allok = True
    for q_lens, k_lens, ns in CASES:
        allok &= run_case(q_lens, k_lens, ns)
    print("ALL OK" if allok else "SOME FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
