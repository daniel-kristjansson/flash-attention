"""Minimal FA4 decode repro for ncu: b1 k4096, D=256, GQA 24/4, paged, causal.

Usage: ncu ... .venv/bin/python benchmarks/profile_decode_b1.py [num_splits] [tile_n] [force_stages]

(sudo strips env vars, so tile/stage config is taken from argv and set here
before the bundle import reads it.)
"""

import os
import sys
import torch

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
if len(sys.argv) > 2:
    os.environ["FA4_SM120_TILE_N_256"] = sys.argv[2]
if len(sys.argv) > 3:
    os.environ["FA4_SM120_FORCE_STAGES"] = sys.argv[3]

sys.path.insert(0, "/home/danielk/code")
from fa4_sm120 import flash_attn_varlen_func as fa4  # noqa: E402

NHQ, NHK, D, PAGE = 24, 4, 256, 16
SL = 4096

torch.manual_seed(0)
q = torch.randn(1, NHQ, D, dtype=torch.bfloat16, device="cuda")
nk = SL // PAGE
k = torch.randn(nk, PAGE, NHK, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn_like(k)
cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
su = torch.tensor([SL], dtype=torch.int32, device="cuda")
pt = torch.arange(nk, dtype=torch.int32, device="cuda").reshape(1, -1)

for _ in range(3):
    fa4(
        q, k, v, cu_seqlens_q=cu, seqused_k=su, max_seqlen_q=1, max_seqlen_k=SL,
        page_table=pt, softmax_scale=D ** -0.5, causal=True, num_splits=NS,
    )
torch.cuda.synchronize()
print("done splits=", NS)
