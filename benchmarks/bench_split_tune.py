"""Sweep split-KV counts for decode at D=256, GQA 24/4.

Run: python benchmarks/bench_split_tune.py
"""

import sys
import time

import torch

from flash_attn.cute import flash_attn_varlen_func as fa4

try:
    from vllm.vllm_flash_attn import flash_attn_interface as fa2
except Exception:
    fa2 = None

NHQ, NHK, D, PAGE = 24, 4, 256, 16
SCALE = D ** -0.5


def timeit(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def make(batch, sl):
    torch.manual_seed(0)
    q = torch.randn(batch, NHQ, D, dtype=torch.bfloat16, device="cuda").view(-1, NHQ, D)
    nk = batch * (sl // PAGE)
    k = torch.randn(nk, PAGE, NHK, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    cu = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda")
    su = torch.full((batch,), sl, dtype=torch.int32, device="cuda")
    pt = torch.arange(nk, dtype=torch.int32, device="cuda").reshape(batch, -1)
    return q, k, v, cu, su, pt


def main():
    for batch, sl in [(1, 2048), (1, 4096), (1, 8192), (8, 4096), (16, 2048)]:
        q, k, v, cu, su, pt = make(batch, sl)
        if fa2 is not None:
            f2 = lambda: fa2.flash_attn_varlen_func(
                q=q, k=k, v=v, max_seqlen_q=1, cu_seqlens_q=cu, max_seqlen_k=sl,
                block_table=pt, seqused_k=su, causal=True, softmax_scale=SCALE, fa_version=2,
            )
            t2 = timeit(f2)
        else:
            t2 = float("nan")
        row = f"b{batch:<3} k{sl:<5} FA2 {t2:>7.1f} |" if fa2 is not None else f"b{batch:<3} k{sl:<5} FA2     n/a |"
        for ns in [1, 4, 8, 16, 32, 47, 64]:
            f4 = lambda: fa4(
                q, k, v, cu_seqlens_q=cu, seqused_k=su, max_seqlen_q=1, max_seqlen_k=sl,
                page_table=pt, softmax_scale=SCALE, causal=True, num_splits=ns,
            )
            t4 = timeit(f4)
            ratio = f"({t2/t4:4.2f})" if fa2 is not None else ""
            row += f" s{ns:<3} {t4:>6.1f}{ratio}|"
        print(row, flush=True)


if __name__ == "__main__":
    main()
