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
  removed — see decode tile performance), paged decode on the 64x32 tile
  (see known issues). Required for `head_dim=256` to dispatch on SM120's
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
- D=256 paged decode (64x32): OK; the previously-crashing 64x16 configs
  (512 G=1 causal, 2048 G=2 causal, 4096 G=2 non-causal) run clean 15/15
  via the 64x32 dispatch (max_err ~3e-4..7e-4); `compute-sanitizer --tool
  memcheck` clean on the guarded loads.
- Multi-sequence varlen paged decode (B=2 mixed klen, B=3 mixed qlen, B=8
  mixed klen, G=1/2): OK (max_err ~1.4e-3). See known issue 3 — the earlier
  "B=4 corruption" was the repro's page table, not the kernel.

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
always dispatches to 64x64. Paged decode is unchanged (64x32).

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

2. **Paged 64x16: non-deterministic per-head corruption (OPEN, follow-up).**
   With the OOB fixed, paged KV at tile_n=16 corrupts one (or adjacent
   pairs of) heads in ~30% of runs: e.g. KLEN=512 G=1 causal flaked 2/5
   (bad head err 0.2–0.33 vs ~3e-4 clean heads); also seen at KLEN=2048/4096
   G=2. Every flaky config has KLEN % 16 == 0 (seqlen predicates are no-ops,
   so the trigger is the new out-of-tile row-skip branch); dense 64x16 with
   the same tiles is clean 9/9, localizing the race to the paged load path.
   PTX inspection shows ptxas hoisting the second m unit of the unrolled load
   past the caller's commit/wait/barrier (benign at tile_n=16, noted).
   **Update:** the dense 64x16 path was removed from dispatch entirely (see
   decode tile performance) — it too misfired, 26/30 runs corrupted at
   B=1 KLEN=4096 — and dense D=256 decode now uses 64x64. Paged decode
   still dispatches to 64x32, but under a loaded GPU the same per-head
   corruption was also observed at paged 64x32 (4/20 runs, B=1 KLEN=2048;
   identical before and after this branch's dispatch change; paged 64x64 was
   0/20 clean in the same session), so the race's scope may be broader than
   tile_n=16 or contention may widen the window — needs idle-GPU
   re-validation; paged 64x64 is the candidate fallback.

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
