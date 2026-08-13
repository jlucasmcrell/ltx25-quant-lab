"""Build the F16 GGUF of an LTX-2.5 22B transformer - the master every quant
is cut from.

Convention copied from the LTX-2.3 GGUF this lab already ships and that
ComfyUI-GGUF accepts:
  - general.architecture = "ltxv", and nothing else in the KV block
  - the "model.diffusion_model." prefix stripped from every tensor name
  - GGUF stores dims reversed; GGUFWriter.add_tensor does that itself

Why this model is worth the disk, in one number: every weight matrix in
LTX-2.5 has a last dimension divisible by 256, so K-quants are legal on all of
them. The only exceptions are 306 gate-logit bias vectors of length 32, 2 MB in
total, which stay F16 like every other bias. MiniMax-H3 could not do this - its
2688-wide tensors made K-quants impossible - so the LTX-2.5 quant ladder can go
places the H3 one could not.

Streams tensor by tensor: 42 GB in, 42 GB out, a few hundred MB resident.
"""
import json
import os
import struct
import sys
import time

import numpy as np
import gguf
from gguf import GGMLQuantizationType as QT
from safetensors import safe_open

SRC = sys.argv[1] if len(sys.argv) > 1 else (
    "F:/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/"
    "models/diffusion_models/LTX New/ltx-2.5-22b-distilled-transformer-bf16.safetensors")
OUT = sys.argv[2] if len(sys.argv) > 2 else "G:/ltx-lab/build/ltx25/LTX25-distilled-DiT-F16.gguf"
PREFIX = "model.diffusion_model."
F16_MAX = 65504.0

os.makedirs(os.path.dirname(OUT), exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open(SRC, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
SRC_META = hdr.pop("__metadata__", None) or {}
names = sorted(hdr)
print(f"[f16] {len(names)} tensors, {sum(v['data_offsets'][1] - v['data_offsets'][0] for v in hdr.values()) / 1e9:.2f} GB in")

w = gguf.GGUFWriter(OUT + ".building", "ltxv")

# THE TWO THINGS A NAIVE CONVERSION LOSES, and why self-converted LTX-2.5
# GGUFs "do not load":
#
# 1. the safetensors __metadata__ config. ComfyUI decides which LTX class to
#    build and how wide the audio embeddings connector is from that JSON. Drop
#    it and comfy falls back to defaults - the connector comes out 3840 wide
#    against this checkpoint's 2048 and the state dict refuses to load.
#    ComfyUI-GGUF's get_gguf_metadata() collects every simple STRING/INT/F32
#    KV and hands it to comfy as `metadata`, so carrying the fields across is
#    enough.
# 2. leading singleton dimensions. GGUF's ne[] has no way to say [1, 4096], so
#    keyframes_abs_pos_embedding round-trips as [4096]. ComfyUI-GGUF restores
#    it from a `comfy.gguf.orig_shape.<name>` array if one is present.
for _k, _v in SRC_META.items():
    if isinstance(_v, str):
        w.add_string(_k, _v)
print(f"[f16] carried {sum(1 for v in SRC_META.values() if isinstance(v, str))}"
      f" metadata field(s): {sorted(SRC_META)}")
t0 = time.time()
overflow = []
odd_shapes = []
done = 0
with safe_open(SRC, framework="pt") as st:
    for k in names:
        t = st.get_tensor(k)
        arr = t.float().numpy() if t.dtype.__str__() == "torch.bfloat16" else t.numpy()
        if arr.dtype == np.float32:
            mx = float(np.abs(arr).max()) if arr.size else 0.0
            if mx > F16_MAX:
                overflow.append((k, mx))
                out = arr                      # keep it in F32 rather than clip
            else:
                out = arr.astype(np.float16)
        else:
            out = arr.astype(np.float16)
        name = k[len(PREFIX):] if k.startswith(PREFIX) else k
        if 1 in tuple(arr.shape):
            w.add_array(f"comfy.gguf.orig_shape.{name}",
                        [int(d) for d in arr.shape])
            odd_shapes.append((name, tuple(arr.shape)))
        w.add_tensor(name, out)
        done += 1
        if done % 500 == 0:
            print(f"   {done}/{len(names)}  {time.time() - t0:.0f}s", flush=True)

print(f"[f16] orig_shape recorded for {len(odd_shapes)}: {odd_shapes}")
print("[f16] writing...", flush=True)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file(progress=False)
w.close()
os.replace(OUT + ".building", OUT)
print(f"[f16] {OUT}  {os.path.getsize(OUT) / 1e9:.2f} GB  in {time.time() - t0:.0f}s")
if overflow:
    print(f"[f16] {len(overflow)} tensor(s) exceeded the F16 range and were kept F32:")
    for k, mx in overflow[:10]:
        print(f"        {k}  max|w| = {mx:.1f}")
else:
    print("[f16] no tensor exceeded the F16 range")
