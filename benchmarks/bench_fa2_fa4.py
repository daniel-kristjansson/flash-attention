"""FA4 SM120 vs FA2 (sm_120 build): decode and prefill at D=256, GQA 24/4.

Decode uses num_splits=-1 (auto heuristic). Run:

    python benchmarks/bench_fa2_fa4.py
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


def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def make_decode(batch, seqlen_k):
    torch.manual_seed(0)
    q = torch.randn(batch, NHQ, D, dtype=torch.bfloat16, device="cuda").view(-1, NHQ, D)
    nk = batch * (seqlen_k // PAGE)
    k = torch.randn(nk, PAGE, NHK, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    cu_q = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda")
    su_k = torch.full((batch,), seqlen_k, dtype=torch.int32, device="cuda")
    pt = torch.arange(nk, dtype=torch.int32, device="cuda").reshape(batch, -1)
    return q, k, v, cu_q, su_k, pt


def make_prefill(total, nseq=1):
    per = total // nseq
    torch.manual_seed(0)
    q = torch.randn(total, NHQ, D, dtype=torch.bfloat16, device="cuda")
    nk = total // PAGE
    k = torch.randn(nk, PAGE, NHK, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, nseq * per], dtype=torch.int32, device="cuda")
    su_k = torch.full((nseq,), per, dtype=torch.int32, device="cuda")
    pt = torch.arange(nk, dtype=torch.int32, device="cuda").reshape(nseq, -1)
    return q, k, v, cu_q, su_k, pt


def main():
    if fa2 is None:
        print("WARN: vllm FA2 not available; FA2 columns will be n/a.")
    print(f"{'case':<28} {'FA2 (us)':>10} {'FA4 (us)':>10} {'speedup':>8}")
    for batch, sl in [
        (1, 2048),
        (1, 4096),
        (8, 2048),
        (16, 2048),
        (32, 4096),
        (64, 4096),
        (64, 8192),
    ]:
        q, k, v, cu, su, pt = make_decode(batch, sl)
        if fa2 is not None:
            f2 = lambda: fa2.flash_attn_varlen_func(
                q=q, k=k, v=v, max_seqlen_q=1, cu_seqlens_q=cu, max_seqlen_k=sl,
                block_table=pt, seqused_k=su, causal=True, softmax_scale=SCALE, fa_version=2,
            )
            t2 = timeit(f2)
        else:
            t2 = float("nan")
        f4 = lambda: fa4(
            q, k, v, cu_seqlens_q=cu, seqused_k=su, max_seqlen_q=1, max_seqlen_k=sl,
            page_table=pt, softmax_scale=SCALE, causal=True, num_splits=-1,
        )
        t4 = timeit(f4)
        sp = f"{t2/t4:.2f}x" if fa2 is not None else "n/a"
        print(f"decode b{batch:<3} k{sl:<6} {t2:>10.1f} {t4:>10.1f} {sp:>8}")
    print()
    for total in [1024, 2048, 4096, 8192]:
        q, k, v, cu, su, pt = make_prefill(total)
        f4 = lambda: fa4(
            q, k, v, cu_seqlens_q=cu, seqused_k=su, max_seqlen_q=total, max_seqlen_k=total,
            page_table=pt, softmax_scale=SCALE, causal=True,
        )
        t4 = timeit(f4, iters=20, warmup=5)
        fl4 = 2 * total * total * NHQ * D / 1e9 / t4 * 1e6
        if fa2 is not None:
            f2 = lambda: fa2.flash_attn_varlen_func(
                q=q, k=k, v=v, max_seqlen_q=total, cu_seqlens_q=cu, max_seqlen_k=total,
                block_table=pt, seqused_k=su, causal=True, softmax_scale=SCALE, fa_version=2,
            )
            t2 = timeit(f2, iters=20, warmup=5)
            fl2 = 2 * total * total * NHQ * D / 1e9 / t2 * 1e6
            print(f"prefill {total:<6} {t2:>10.1f} {t4:>10.1f} {t2/t4:>7.2f}x   ({fl2:>6.0f} / {fl4:>6.0f} TFLOPS)")
        else:
            print(f"prefill {total:<6} {'n/a':>10} {t4:>10.1f} {'n/a':>8}   ({fl4:>6.0f} TFLOPS)")


if __name__ == "__main__":
    main()
