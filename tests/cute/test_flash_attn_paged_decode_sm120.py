"""SM120 D=256 paged decode: the dispatched KV tile must stay 64x64.

The paged cp.async KV load corrupts whole heads non-deterministically at
narrow KV tiles: 64x32 (the previous paged decode tile) produced per-head
errors of 0.08-0.72 against 3e-4 clean on an idle GPU -- 6/30 and 7/30 bad
runs at B=1 KLEN=2048 G=4 causal, 23/30 and 21/30 at B=4 KLEN=4096, 30/30
non-causal, 27/30 at B=8 mixed klen. The same shapes through the dense
kernel are clean at 64x32 (0/10), so the defect is in the paged load, not
the tile math. 64x64 is 0/420 over that matrix. These cases pin the
dispatch (see known issue 2 in the PR description).

Run: pytest tests/cute/test_flash_attn_paged_decode_sm120.py
     python tests/cute/test_flash_attn_paged_decode_sm120.py  (no pytest needed)
"""

import pytest
import torch

from flash_attn.cute import flash_attn_varlen_func as fa4

NHK, D, PAGE = 4, 256, 16
SCALE = D ** -0.5
RUNS = 10  # fresh data per run: the 64x32 corruption is intermittent

# (q_lens, k_lens, nheads_per_kvhead, causal)
CASES = [
    # 30/30 bad runs at the old 64x32 paged decode tile.
    ([1] * 4, [4096] * 4, 4, False),
    # The shape from the original contention-time report.
    ([1], [2048], 4, True),
    ([1] * 2, [4096] * 2, 2, True),
]


def _paged_decode_ref(k, v, q, k_lens, q_lens, nheads_per_kvhead, causal):
    """fp32 reference, one varlen-pool batch row at a time."""
    out, q_off, k_off = [], 0, 0
    for ql, kl in zip(q_lens, k_lens):
        kk = k[k_off:k_off + kl].repeat_interleave(nheads_per_kvhead, dim=1).float()
        vv = v[k_off:k_off + kl].repeat_interleave(nheads_per_kvhead, dim=1).float()
        qq = q[q_off:q_off + ql].float()
        s = torch.einsum("qhd,khd->qhk", qq, kk) * SCALE
        if causal:
            # decode alignment: query j of the row sits at key position kl - ql + j
            qi = torch.arange(ql, device="cuda") + (kl - ql)
            ki = torch.arange(kl, device="cuda")
            s = s.masked_fill(ki[None, :] > qi[:, None], float("-inf"))
        out.append(torch.einsum("qhk,khd->qhd", s.softmax(-1), vv))
        q_off += ql
        k_off += kl
    return torch.cat(out)


@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] != 12,
    reason="SM120 D=256 paged decode dispatch",
)
@pytest.mark.parametrize("q_lens,k_lens,nheads_per_kvhead,causal", CASES)
def test_flash_attn_paged_decode_sm120_tile(q_lens, k_lens, nheads_per_kvhead, causal):
    nheads = NHK * nheads_per_kvhead
    max_klen = max(k_lens)
    total_q = sum(q_lens)
    for run in range(RUNS):
        torch.manual_seed(7919 + run)
        q = torch.randn(total_q, nheads, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(sum(k_lens), NHK, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn_like(k)

        # Batch b, logical page p -> physical page (varlen offset of b)/PAGE + p.
        rows, offset = [], 0
        for kl in k_lens:
            n = kl // PAGE
            rows.append(torch.cat([
                torch.arange(offset // PAGE, offset // PAGE + n, dtype=torch.int32, device="cuda"),
                torch.zeros(max_klen // PAGE - n, dtype=torch.int32, device="cuda"),
            ]))
            offset += kl
        page_table = torch.stack(rows)
        cu_seqlens_q = torch.tensor(
            [0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device="cuda"
        )
        seqused_k = torch.tensor(k_lens, dtype=torch.int32, device="cuda")

        out = fa4(
            q,
            k.view(-1, PAGE, NHK, D),
            v.view(-1, PAGE, NHK, D),
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max_klen,
            page_table=page_table,
            softmax_scale=SCALE,
            causal=causal,
        )
        out = (out[0] if isinstance(out, tuple) else out).float()
        ref = _paged_decode_ref(k, v, q, k_lens, q_lens, nheads_per_kvhead, causal)
        # Clean runs are <= 4e-4; the narrow-tile corruption is >= 0.08.
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=0)

if __name__ == "__main__":
    if torch.cuda.get_device_capability()[0] != 12:
        print("skipped: needs SM120")
        raise SystemExit(0)
    for case in CASES:
        test_flash_attn_paged_decode_sm120_tile(*case)
        print(f"ok {case}")
