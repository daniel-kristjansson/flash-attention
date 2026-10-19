# SM120 (Blackwell GeForce / DGX Spark) forward: split-KV, paged KV, TMA 8-warp, block-sparse, dropout, D=256

This branch orders the open SM120 PRs onto `upstream/main` and ports the
validated local SM120 deltas from the `fa4_sm120` bundle. Target: SM 12.0
(RTX PRO 6000 / DGX Spark), kernels are CuTeDSL.

## Merged upstream PRs (in order, each `--no-ff`)

| # | PR | What it adds |
|---|----|--------------|
| 1 | #2336 | SM120 split-KV / FlashDecoding with FP32 partial outputs |
| 2 | #2348 | SM120 kernel-level paged KV cache |
| 3 | #2599 | SM120 TMA forward, 8 MMA warps (supersedes #2349) |
| 4 | #2389 | SM80/SM120 block-sparse forward |
| 5 | #2657 | FA4/CuTe dropout forward + backward |

**#2439 (dropout) was closed without merging; operator approved #2657 (its open
successor) as the substitution.** #2349 was skipped (superseded by #2599).

## Local SM120 deltas (from the `fa4_sm120` bundle)

- **D=256** (own commit, trivially revertable): 64x64 base tile + 1-KV-stage
  dispatch + `FA4_SM120_FORCE_STAGES`, 64x16 ping-pong decode (9f7d3fa, later
  removed — see decode tile performance), paged decode on 64x64 (the 64x32
  and 64x16 KV tiles are corrupted — see known issue 2). Required for
  `head_dim=256` to dispatch on SM120's
  99 KB SMEM (the previous `head_dim>64 -> 128x64` needs 128 KB for D=256).
- **PDL** on the split-KV fwd->combine boundary (default on, `FA4_SM120_PDL`)
  + the `use_pdl=False` fix on the TMA kernel.
- **Fix window_size Int32 conversion** for SM80/SM120 host signatures.
- **Fix wasted-tile OOB seqlen reads** in the varlen grid.
- **Force pack_gqa off** on the SM120 (SM80-code-path) kernel + document G=6
  inexpressibility.
- **Tests/benchmarks**: `tests/cute/test_splitkv_sm120.py`,
  `tests/cute/test_repeated_call_sm120.py`, `benchmarks/bench_fa2_fa4.py`,
  `benchmarks/bench_split_tune.py` (adapted to the repo API; FA2 cross-check is
  optional).
- **TMA kernel alignment** (bug fixes): `make_rmem_tensor` (DSL 4.8), `__call__`
  positional slots, `AuxData` handling — so the #2599 TMA kernel compiles and
  runs against the merged tree.

**Not ported (out of scope):** the bundle's HF-snapshot base divergence
(`mask.py`, `softmax.py`, `utils.py`, `paged_kv.py`, ... non-SM120 files) and
the bundle's own dropout tile reduction (superseded by #2657's philox dropout).

## Validation (RTX PRO 6000, SM 12.0, bf16, vs f32 matmul reference)

- `compileall` clean on `flash_attn/cute` + tests/benchmarks.
- D=128 dense causal + non-causal (TMA path): OK (max_err ~2e-3 / 8e-3).
- D=256 dense causal + non-causal (64x64, 1 stage): OK.
- D=256 dense decode (64x64, the post-sweep default; 64x16 ping-pong was
  removed — see decode tile performance): OK (max_err ~3.4e-4, B=1 and B=32
  KLEN=4096 causal; B=1 30/30 clean stress, vs 26/30 corrupted at the old
  64x16 default).
- D=128 / D=192 paged decode (split-KV): OK (max_err ~3.5e-4).
- D=256 paged prefill (QL=128, causal): OK (max_err ~4e-4).
- D=256 paged decode (64x64): OK — idle-GPU stress, 420 runs over 7
  configs, 0 bad (see known issue 2 for the table and for the 64x32 tile
  this replaces); max_err <= 3.6e-3 vs fp32 SDPA (bf16 quantization).
  `compute-sanitizer --tool memcheck` clean on the guarded paged loads.
- D=256 paged decode with split-KV (num_splits 2 and 4, KLEN 2048/4096/8192):
  OK (max_err <= 3.6e-4).
- Multi-sequence varlen paged decode (B=2 mixed klen, B=3 mixed qlen, B=8
  mixed klen, G=1/2): OK (max_err ~1.4e-3). See known issue 3 — the earlier
  "B=4 corruption" was the repro's page table, not the kernel.

## Production readiness (SM120)

Kernel-level (FA4 API on an RTX PRO 6000, SM 12.0, bf16 in / fp32 out,
compared against fp32 SDPA; tolerance ~1e-2 per-head absolute, clean runs
land at 3e-4..3.6e-3). The paged D=256 decode rows were (re)measured with the
stress harness on an idle GPU; the rest are this branch's validation record.

**Validated**

- Dense D=128 causal + non-causal (TMA path): max_err ~2e-3 / 8e-3.
- Dense D=256 forward: prefill and decode (64x64, 1 KV stage), causal and
  non-causal, GQA; decode stress 30/30 clean at B=1 and B=32 KLEN=4096,
  against 26/30 corrupted at the removed 64x16 tile.
- Paged D=256 prefill (QL=128, causal): max_err ~4e-4.
- Paged D=256 decode, 64x64, non-split: 7-config stress matrix, 420 runs on
  an idle GPU, **0 bad** — B=1/4/8, KLEN 128..8192, mixed klen, mixed qlen,
  G=1/2/4, causal and non-causal, page_size 16, per-batch page tables.
  Re-run through the shipped default dispatch: 0/210.
- Paged D=256 decode with split-KV (`num_splits` 2 and 4; KLEN 2048/4096/
  8192): 0/48, max_err <= 3.6e-4.
- Paged D=128 / D=192 decode (split-KV): max_err ~3.5e-4.
- `compute-sanitizer --tool memcheck` clean on the guarded paged loads.
- Regression tests: `tests/cute/test_flash_attn_paged_decode_sm120.py`,
  `tests/cute/test_repeated_call_sm120.py`, `tests/cute/test_splitkv_sm120.py`.

**Not validated**

- FP8 KV cache / fp8 inputs (`float8_e4m3fn`), and descale paths, on any SM120
  shape.
- `page_size != 16`: that takes the `paged_kv_non_tma` loader, which the
  stress matrix did not exercise.
- B > 8 or KLEN > 8192 for paged D=256 decode; `num_splits` other than
  explicit 2/4 (the `num_splits < 1` heuristic is not stress-tested).
- Non-default `tile_mn`, local/windowed attention, softcap, `score_mod`,
  `mask_mod`, block sparsity, dropout, and the backward pass on D=256 paged
  paths (the SM120 deltas here are forward-only additions).
- CUDA graph capture, `torch.compile`, and PDL under stress (PDL is on by
  default; the stress matrix above ran with it).
- Anything on SM80/SM90/SM100/SM110: this branch's changes are gated to
  `arch // 10 == 12` and those paths were not re-run here.
- End-to-end vLLM: the numbers above are kernel-level FA4 API calls, not
  vLLM integration tests.

## Decode tile performance (SM120 D=256)

B-sweep, dense decode, D=256, qlen=1, causal, G=4 (NHQ=16), bf16, RTX PRO
6000 (188 SMs, CC 12.0); median of 20 cuda-event iters, 3 warmup, via
`flash_attn_func`. B=12/13 probe the old gate's boundary. CTA count = B*G
(pack_gqa: one CTA per (batch, kv head)).

KLEN=4096 (us; % vs 64x32):

| B | CTAs | 64x16pp (old default, B<=11) | 64x32 | 64x64 (new default) |
|---:|-----:|-----------------------------:|------:|--------------------:
| 1  | 4    | 299.6 (+13.1%)               | 264.8 | 256.6 (-3.1%) |
| 2  | 8    | 291.8 (+11.6%)               | 261.4 | 253.8 (-2.9%) |
| 4  | 16   | 288.1 (+10.2%)               | 261.5 | 252.8 (-3.3%) |
| 8  | 32   | 311.8 (+14.0%)               | 273.5 | 270.8 (-1.0%) |
| 12 | 48   | (gate off: 533.9)            | 549.4 | 531.7 (-3.2%) |
| 13 | 52   | (gate off: 536.0)            | 548.0 | 532.7 (-2.8%) |
| 16 | 64   | (gate off: 529.6)            | 551.2 | 532.1 (-3.5%) |
| 32 | 128  | (gate off: 836.3)            | 868.4 | 834.6 (-3.9%) |

KLEN=8192 (us; % vs 64x32):

| B  | CTAs | 64x16pp | 64x32  | 64x64 (new default) |
|---:|-----:|--------:|-------:|--------------------:
| 4  | 16   | 556.1 (+14.4%)  | 486.0  | 465.9 (-4.1%) |
| 16 | 64   | (gate off: 1020.7) | 1061.6 | 1033.5 (-2.6%) |
| 32 | 128  | (gate off: 1608.5) | 1673.2 | 1604.6 (-4.1%) |

(An earlier B=1 KLEN sweep gave the same picture: at KLEN=4096, 292/265/256
us for 64x16pp/64x32/64x64.)

Correctness: 64x32 and 64x64 clean at every (B, KLEN) (max_err <= 3.8e-4 vs
fp32 SDPA at B in {2, 8, 32}). **64x16pp is intermittently wrong**: 26/30
runs corrupted at B=1 KLEN=4096 with per-head output errors up to 0.11
(same signature as the paged per-head race, known issue 2); also wrong at
B=2 and B=8. The GPU was shared with another workload, which may widen race
windows, but the tile loses on timing regardless.

Decision: **the ping-pong decode path is removed** (this branch). Mechanism:
64x16 (48 KB, 1 KV stage) only pays off if two CTAs co-reside per SM so one
CTA's MMA hides the other's cp.async — i.e. the SMs must be CTA-starved, which
requires more than 188 CTAs (B > 46 for G=4). The old gate
(`total_q * num_head < 188`) compared *q-head slots* (Gx the CTA count under
pack_gqa) to the SM count, so for G=4 it cut off at B=12 (48 CTAs) — the
16x16 tile never ran in the regime where it could win, it only halved the KV
tile (4x more loop iterations at KLEN=4096, forced 1 KV stage, no
load/MMA overlap) and it is also the corrupted path. 64x64 (96 KB, 1 KV
stage, the only D=256 width that fits the 99 KB SMEM) was the fastest or
tied-fastest tile at every measured B and KLEN, so dense D=256 decode now
always dispatches to 64x64. Paged decode now uses the same 64x64 tile: the
paged 64x32 tile is corrupted (known issue 2) and 64x64 is also 10-13%
faster on paged decode (B=1 KLEN=2048: 165.6 vs 183.2 us; B=1 KLEN=4096:
279.5 vs 312.0 us; B=4 KLEN=4096: 275.7 vs 315.5 us; B=8 mixed klen: 516.3
vs 584.2 us; median of 20).

## Known issues

1. **D=256 paged decode at 64x16: smem OOB — FIXED (this branch).**
   Root cause: the SM80-family paged KV load's gmem copy thread tile spans
   `num_threads // gmem_threads_per_row` rows per m unit — 32 rows with
   SM120's 256 threads — while the 64x16 ping-pong decode tile is 16 KV rows
   wide. Rows 16–31 of the tile issued unpredicated `cp.async`s past the end
   of the sK/sV smem allocation (illegal address, error 700). A source-size
   predicate is not sufficient: a size-0 `cp.async` still writes its 16B
   zero-fill to the out-of-bounds smem destination. Fixed in
   `paged_load_K`/`paged_load_V` with a per-row branch around the copy, gated
   by `const_expr` so every configuration whose thread tile fits the KV tile
   (tile_n >= 32 on SM120, all D=128/D=192, all SM80/90/100) compiles the
   original path unchanged. The pre-fix stopgap (64x16 restricted to non-paged)
   is superseded by the dispatch below.

2. **Paged decode at narrow KV tiles: non-deterministic per-head corruption
   — WORKED AROUND, dispatch now avoids it (root cause OPEN).**
   The paged cp.async KV load corrupts whole heads at narrow KV tiles. It
   first showed at tile_n=16 (~2/5 runs, err 0.2–0.33 vs ~3e-4 clean heads),
   then at tile_n=32 under GPU contention (4/20 runs, B=1 KLEN=2048). An
   idle-GPU stress matrix (RTX PRO 6000, D=256, NHK=4, page_size 16, bf16,
   fresh random data every run, fp32 SDPA reference, 188-SM device otherwise
   idle) settles it: **64x32 is broken at idle too, far worse than the
   contention-time rate suggested**. Runs bad (>1e-2 per-head err; clean runs
   are <= 3.6e-3) out of 30, two independent seed sets where repeated:

   | config (G=4 unless stated) | 64x32 | 64x64 |
   |---|---:|---:|
   | B=1 KLEN=2048 causal | 6/30, 7/30 | 0/30, 0/30 |
   | B=1 KLEN=4096 causal | 9/30 | 0/30, 0/30 |
   | B=4 KLEN=4096 causal | 23/30, 21/30 | 0/30, 0/30 |
   | B=8 mixed klen 128..8192 causal | 27/30 | 0/30, 0/30 |
   | B=1 KLEN=2048 causal, G=1 | 0/30 | 0/30, 0/30 |
   | B=4 KLEN=4096 non-causal | 30/30, 30/30 | 0/30, 0/30 |
   | B=3 mixed qlen [1,2,1] non-causal | 30/30 | 0/30, 0/30 |
   | **total** | **183/300** (125/210 + 58/90) | **0/420** |

   Bad runs lose one to several whole heads (err 0.076–0.72, e.g. B=8 mixed
   klen: heads 1, 7, 8 at 0.098/0.384/0.721 while the rest are ~2e-3); the
   affected heads differ run to run.

   Localization: the same B=4 KLEN=4096 shape through the **dense** kernel at
   tile 64x32 is 0/10 clean while paged is 10/10 bad, so the defect is in the
   paged load, not the tile math. It is not the number of KV stages
   (`FA4_SM120_FORCE_STAGES=1` at 64x32 is still 10/10 bad), not a GQA
   head-mapping swap (the `i % NHK` mapping explains none of the bad heads),
   and not present in split-KV (`num_splits` 2/4 paged decode is clean at
   both tiles, 0/48 each) — the trigger is the non-split paged decode loop
   with a narrow KV tile. PTX inspection earlier showed ptxas hoisting the
   second m unit of the unrolled load past the caller's commit/wait/barrier.

   **Outcome:** paged D=256 decode dispatches to 64x64, which is clean over
   the whole matrix (0/420, and 0/210 again with the shipped default) and is
   10-13% faster than 64x32, so the safe tile costs nothing. Pinned by
   `tests/cute/test_flash_attn_paged_decode_sm120.py` (fails 3/3 when the
   tile is forced back to 64x32, passes 3/3 on the default).

   **Residual risk:** the root cause is unfixed — the narrow-tile paged load
   path is simply not used any more for D=256. Anything that reintroduces a
   KV tile below 64 rows on a paged path (a future tile, page-size-specific
   loader, or `FA4_SM120_TILE_N_256=32/16`) re-exposes it. The matrix covers
   page_size 16 only, G=1/2/4, B<=8, KLEN<=8192; other page sizes take the
   `paged_kv_non_tma` loader, which this sweep did not exercise.

3. **B>1 multi-sequence paged decode: "corruption" — RESOLVED, not a kernel
   bug.** The original repro (`/tmp/paged_b4.py`) built its page table as
   `arange(total_pages).expand(B, -1)` — every batch row identical, so for
   batch b > 0 logical page p mapped to physical page p (sequence 0's
   physical pages) instead of b*256+p. The kernel read exactly what the table
   said: cross-check confirms batch b>0 outputs equal attention of q_b over
   **sequence 0's KV** (err ~3e-4) and differ from own-KV attention (~0.12),
   i.e. correct attention for the (invalid) table. With a correct per-batch
   table the same shape is clean: B=4 KLEN=4096 -> per-seq
   `[0.0002, 0.0003, 0.0003, 0.0003]`; B=2 klen=[2048,512], B=8 mixed klen
   (128..8192) G=2, B=3 mixed qlen [1,2,1] all clean (max_err ~1.4e-3).
   The batch index was already handled correctly: `PagedKVManager.create`
   slices `mPageTable[bidb, None]` with the per-CTA `batch_size` from the
   tile scheduler. vLLM batched decode on SM120 is unaffected.
