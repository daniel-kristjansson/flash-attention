"""Regression test for the wasted-tile OOB seqlen read (commit f16403e).

Calling the varlen paged kernel repeatedly with *changing* shapes previously
crashed with cudaErrorIllegalAddress: wasted grid tiles (batch_idx ==
num_batch) read mSeqUsedK / cu_seqlens[batch_idx+1] out of bounds, and stale
allocator memory from the previous launch became seqlen_k. Fresh processes
happened to hold zeros there; reused processes did not.

Run: CUDA_LAUNCH_BLOCKING=1 .venv/bin/python tests/test_repeated_call_sm120.py
"""

import sys
import torch

sys.path.insert(0, "/home/danielk/code")
from fa4_sm120 import flash_attn_varlen_func as fa4  # noqa: E402

NHQ, NHK, D, PAGE = 24, 4, 256, 16
SCALE = D ** -0.5


def call(q_lens, k_lens, num_splits):
    tq, tk = sum(q_lens), sum(k_lens)
    q = torch.randn(tq, NHQ, D, dtype=torch.bfloat16, device="cuda")
    k_pages = tk // PAGE
    k = torch.randn(k_pages, PAGE, NHK, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
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
    out = fa4(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        seqused_k=su_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        page_table=pt,
        softmax_scale=SCALE,
        causal=True,
        num_splits=num_splits,
    )
    o = out[0] if isinstance(out, tuple) else out
    assert not torch.isnan(o).any(), "NaN in output"
    return o


# (q_lens, k_lens, num_splits) pairs; the original crash was mlen:
# callA [1,128] k[256,4096] s2 -> callB [1,128] k[2048,4096] s2.
MODES = {
    "mlen": (
        ([1, 128], [256, 4096], 2),
        ([1, 128], [2048, 4096], 2),
    ),
    "qlen": (
        ([1, 1], [256, 4096], 2),
        ([64, 64], [2048, 4096], 2),
    ),
    "splits": (
        ([1], [4096], 4),
        ([1], [4096], 32),
    ),
    "klen": (
        ([1, 128], [256, 512], 2),
        ([1, 128], [2048, 4096], 2),
    ),
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "mlen"
    a, b = MODES[mode]
    call(*a)
    torch.cuda.synchronize()
    print("callA ok", flush=True)
    call(*b)
    torch.cuda.synchronize()
    print(f"callB({mode}) ok", flush=True)


if __name__ == "__main__":
    main()
