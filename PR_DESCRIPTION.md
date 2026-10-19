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
  dispatch + `FA4_SM120_FORCE_STAGES`, 64x16 ping-pong decode (9f7d3fa),
  paged decode on the 64x32 tile (see known issues). Required for
  `head_dim=256` to dispatch on SM120's 99 KB SMEM (the previous
  `head_dim>64 -> 128x64` needs 128 KB for D=256).
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
- D=256 dense decode (64x16 ping-pong): OK (max_err ~3e-4, 4096 nc / 2048 G4 c).
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

Micro-benchmark, B=1, G=4 (NHQ=16), qlen=1 causal, bf16, KLEN in
{2048, 4096, 8192}; median of 20 iters with cuda events, 3 warmup;
dense via `flash_attn_func`, paged via varlen + page table.
64x16 = the natural dense-decode dispatch (ping-pong); paged 64x16 is
not measured (dispatch disabled pending the race in known issue 2).

| KLEN | dense 64x16pp | dense 64x32 | dense 64x64 | paged 64x32 | paged 64x64 |
|------:|--------------:|------------:|------------:|------------:|------------:|
| 2048 | 170 us | 158 us | 152 us | 188 us | 169 us |
| 4096 | 292 us | 265 us | 256 us | 321 us | 285 us |
| 8192 | 537 us | 485 us | 465 us | 587 us | 517 us |

Observations: with only 4 CTAs in flight (B=1, G=4), the larger KV tile
is fastest and the 64x16 ping-pong tile is ~8-10% *slower* than 64x32
(its design benefit — a second CTA per SM hiding the first's cp.async —
needs CTA co-residency, which does not occur at this CTA count on a
188-SM part). Paged carries a ~10-15% fixed page-table overhead over
dense at the same tile. The dispatch therefore keeps 64x16 ping-pong
for dense decode (bundle 9f7d3fa design) and 64x32 for paged decode
(clean + close to 64x64), but a G>=4 B=1 workload would be fastest at
64x64; revisiting the decode tile choice with higher-parallelism
measurements is a candidate follow-up.

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
   past the caller's commit/wait/barrier (benign at tile_n=16, noted). Until
   root-caused, **paged D=256 decode dispatches to 64x32** (validated clean
   15/15 above); dense D=256 decode keeps the 64x16 ping-pong tile.

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
