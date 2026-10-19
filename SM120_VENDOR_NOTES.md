# SM120 vendored FA4 branch (`vendor-sm120`)

Vendored, gate-validated SM120 FA4 forward tree. Base is vLLM's flash-attention
fork (`vfa/main` = `b273a026`); the four unmerged Dao-AILab SM120 feature PRs
were ported on top of it, followed by the local SM120 fixes from the validated
`sm120` branch.

Production vLLM (`branch dtk`) loads this tree instead of the flat
`/home/danielk/code/fa4_sm120` bundle.

## Base

| | |
|---|---|
| Base commit | `b273a026` — "[Bugfix] Fix SM90 hdim 512 forward race on the shared P tile" (`vllm-project/flash-attention` main, synced from Dao-AILab `ce088ab9` + the fork's ABI / FA2-dropout / FA4 fake-tensor / SM90 FP8-KV work) |
| Branch | `vendor-sm120` (pushes fast-forward only; `sm120` untouched) |

The base did **not** contain the SM120 feature work: `interface.py` asserted
`page_table is None` and `not is_split_kv` for `arch // 10 == 12`, `flash_fwd.py`
had no paged-KV path at all, and `flash_fwd_sm120_tma.py` did not exist. Those
features live only in unmerged Dao-AILab PRs which the validated `sm120` branch
had merged locally, so they are the first four commits here.

## Per-commit table

| # | Commit | Source | What it does | Gate |
|---|---|---|---|---|
| 1 | `c3d53aed` | sm120 merge `d029cac0` (PR #2336) | SM120 SplitKV (FlashDecoding) with FP32 partials: kernel takes `is_split_kv`, O/LSE gain the split dim, `split_idx` flows through BlockInfo/scheduler, empty splits skip the store (interface pre-fills O=0 / LSE=-inf) | dense causal/non-causal 0/3; SplitKV ns=4/8 causal + ns=4 non-causal 0/3 (worst 8.7e-03) |
| 2 | `aca531b0` | sm120 merge `e8c02891` (PR #2348) | SM120 kernel-level paged KV: inline page-table lookup + row-wise cp.async gather (`paged_load_K/V`), PagedKVManager in the SM80-style kernel, `num_stages=2` where SMEM allows | paged decode D=128/D=64 0/12 (worst 4.3e-04); dense + SplitKV still 0 |
| 3 | `78219ffe` | sm120 merge `b6136b39` (PR #2599) | SM120 TMA forward (new `flash_fwd_sm120_tma.py`, 1003 lines): cp.async.bulk Q/K/V, 1 DMA warp + 8 MMA warps, TMA O store; SM120 `num_threads` 128→256; `num_stages` from `can_implement` + `FA4_SM120_FORCE_STAGES`; plus the DSL-4.8/`AuxData` alignment the file needs on this tree | dense (TMA route confirmed via `can_implement` hook) 0, worst 8.7e-03; paged D=128/D=64 0/8 |
| 4 | `517526f6` | sm120 merge `a8adc478` (PR #2389) | SM80/SM120 block-sparse forward: `run_block_sparse_mainloop_sm80` + `mma_one_n_block_bs`, const_expr switch in the kernel, TMA route excludes block-sparse | block-sparse block-causal B=1/B=2 S=256 D=128 0, err == dense 8.4e-03 |
| 5 | `a52650da` | local `525df91f` | wasted-tile OOB seqlen clamp in `SeqlenInfoQK` (varlen grid padding) | smoke 0 bad (D=256 paged gate not yet runnable) |
| 6 | `8a6698e0` | local `5f1c1a2b` | paged cp.async SMEM OOB guard when the KV tile is narrower than the gmem thread-tile rows | smoke 0 bad |
| 7 | `7899eca5` | local `748f506c`+`9e31883e`+`3b2ba236` end state | SM120 D=256 64x64 tile + `FA4_SM120_TILE_N_256` override (top-level 64x16 ping-pong deliberately not ported) | D=256 paged gate 0/21 (3 runs/case) |
| 8 | `918798d7` | local `de0a0b60` | SplitKV PDL fwd→combine (`griddepcontrol_launch_dependents`), `FA4_SM120_PDL`; TMA kernel `use_pdl=False` | D=256 paged gate 0/21 |
| 9 | `f07cdb3f` | local `da125569` + `4ebdc338` | `window_size` → `Int32` for the SM80/SM120 host signature, defined above the compile-cache guard (the `4ebdc338` scoping fix was reproduced and folded in) | D=256 paged gate 0/21 |
| 10 | `863dbef0` | local `0f1929c2` | force `pack_gqa = False` on the SM120 kernel (G=6 pack_gqa is structurally inexpressible there) | D=256 paged gate 0/21; dense + SplitKV 0/5 |
| 11 | `a1d8541a` | local `1749bc3c` | `tests/cute/test_splitkv_sm120.py`, `tests/cute/test_repeated_call_sm120.py`, `benchmarks/bench_fa2_fa4.py`, `benchmarks/bench_split_tune.py` | `test_splitkv_sm120.py` → ALL OK (8/8, 3.1e-04..2.1e-03; FA2 cross-check agrees); `test_repeated_call_sm120.py` OK; gate 0/21 |
| 12 | `6ae8bb8c` | local `e3182a57` + `3c257059` | code already in place from commit 7; adds `tests/cute/test_flash_attn_paged_decode_sm120.py` | 3/3 cases ok, 10 fresh-data runs each |

## Final full matrix (D=256 paged decode, 30 runs/config)

`CUDA_VISIBLE_DEVICES=0`, RTX PRO 6000 (SM 12.0), bf16 vs f32 reference,
`atol = 1e-2`, fresh random data per run.

| config | bad | worst good |
|---|---|---|
| B=1 klen=2048 G=4 causal | 0/30 | 5.80e-04 |
| B=4 klen=4096 G=4 causal | 0/30 | 4.03e-04 |
| B=8 mixed klen (128..8192) G=4 causal | 0/30 | 2.98e-03 |
| B=4 klen=4096 G=4 non-causal | 0/30 | 4.03e-04 |
| B=2 klen=4096 G=2 causal | 0/30 | 3.54e-04 |
| B=1 klen=2048 G=1 causal | 0/30 | 5.56e-04 |
| B=1 klen=4096 G=4 causal | 0/30 | 3.46e-04 |
| **TOTAL** | **0/210** | |

Paged decode timings (median of 20 cuda-event iterations):

| config | median | min |
|---|---|---|
| B=1 KLEN=2048 G=4 causal | 155.4 us | 148.0 us |
| B=1 KLEN=4096 G=4 causal | 270.2 us | 263.8 us |
| B=4 KLEN=4096 G=4 causal | 271.6 us | 264.0 us |
| B=8 mixed klen G=4 causal | 504.2 us | 496.8 us |
| B=4 KLEN=4096 G=4 non-causal | 266.3 us | 259.0 us |
| B=2 KLEN=4096 G=2 causal | 269.5 us | 257.0 us |

## Wholesale-replaced / new files

No file was taken wholesale from `sm120` — every file is the vendor base's
version with the ported hunks hand-merged. One file is new:

- `flash_attn/cute/flash_fwd_sm120_tma.py` — new, taken verbatim from the PR
  #2599 merge (`b6136b39`) and then adapted for this tree (DSL 4.8
  `make_rmem_tensor`, `AuxData` aux slot, positional slots the SM120
  `compile_args` fill). This is the one place where vLLM-fork content could not
  be inherited, because the base had no such file.

`flash_fwd.py` and `interface.py` (the two files with vLLM-fork content) were
hand-merged, never replaced.

## vLLM-fork work: preserved vs dropped

Preserved (verified present in the final tree):

- FA2 ABI / `flash_api_stable` and the FA2 dropout removal — untouched
  (`csrc/`, `flash_attn/flash_attn_interface.py`).
- FA4 compile / fake-tensor warmup (`caaa4eb5`): the `_CompileOnlyTensorSpec`
  and fake-tensor compile paths in `_flash_attn_fwd` are the base's, untouched;
  the SplitKV partial-buffer allocation keeps that structure and only changes
  `torch.empty` → `torch.zeros` / `torch.full(-inf)` so skipped splits fold
  correctly.
- SM90 FP8-KV dequant (`617264c1`): `flash_fwd_sm90.py` untouched; the SM80
  `_check_type`/`o_dtype` plumbing that it uses is preserved (the SM80/SM120
  epilogue keeps `self.o_dtype` from `mO`).
- `pack_gqa_layout` in `flash_fwd.py`: preserved (still applied for the SM80
  path); SM120 forces it off in its own class.
- `AuxData` / philox interface: preserved (`AuxData` is still the aux slot for
  every kernel; the ported TMA kernel was adapted to take `AuxData` rather than
  the PR's older raw tensor list).
- `mCuTotalMBlocks` / `mCuTotalSplitsMBlocks` scheduler plumbing: preserved and
  still passed to the SM80/SM120 kernel; the TMA kernel accepts the slots and
  ignores them (it is only routed for dense, non-varlen requests).
- Rope dim: untouched (`flash_attn/cute` has no rope code path changes here).
- SM90 hdim512 / SplitKV fixes from the base: untouched (`flash_fwd_sm90.py`,
  `flash_fwd_combine.py` — the only combine change is `use_pdl=True`).

Dropped / not taken (all intentional, none of them vLLM behavior):

- PR #2336's older mainloop/epilogue restructuring (the base's newer version was
  kept) and its `lpt` / `is_persistent` / `element_size` scheduler arguments.
- The local 64x16 ping-pong decode tile and its `FA4_SM120_FORCE_STAGES`
  per-D knob from `748f506c`/`9e31883e`/`3b2ba236`: superseded by the end state
  (`e3182a57` deleted the ping-pong path again). `FA4_SM120_FORCE_STAGES` is
  still honored as a debug override of the stage count.
- `PR_DESCRIPTION.md` from `e3182a57`/`3c257059` (not part of this tree).

Behavior changes worth knowing:

- SM120 uses 256 threads (8 MMA warps) and, for dense requests, the TMA kernel.
- SM120 SplitKV partial buffers are zero/`-inf` initialized (needed for empty
  splits); SM90/SM100 pay the same memset on their split path.
- SM80 (`arch // 10 == 8`) is clamped to `num_splits = 1` — the SM80-family
  kernel keeps a single split, matching the base's hard-disabled state.
- `split_idx` for the SM80-family kernel now comes from the work tile instead of
  a constant; with `num_splits == 1` it is always 0.

## Skipped items and why

- `ff4ea3b2` (SM120 TMA DSL 4.8 / mask.py / interface alignment): **not
  skipped** — the base did not contain it and the TMA kernel would not compile
  or run without it. Folded into commit 3 (`78219ffe`).
- `4ebdc338` (`window_size_left_c` scoping in the fwd cache-hit path): **not
  skipped** — the port reproduced the bug. Folded into commit 9 (`f07cdb3f`).
- `e3182a57` / `3c257059` code: satisfied by commit 7 (`7899eca5`), which went
  straight to the 64x64 end state; only their regression test is added (commit
  12).
- The 64x16 ping-pong tile: deliberately not resurrected.

## Rollback

- Point production back at the bundle: set `VLLM_FA4_SM120_DIR` to the old path,
  or revert the loader commit in `/home/danielk/code/vllm` (`branch dtk`).
- Revert the vendor tree: `git -C /home/danielk/code/flash-attention-sm120
  reset --hard b273a026` (keep `sm120` and `/home/danielk/code/fa4_sm120` as the
  known-good references).
- Delete the pushed branch if it should not exist:
  `git push origin --delete vendor-sm120`.
