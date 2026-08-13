"""Per-family mixed-precision GGUFs for LTX-2.5 - "AV-protected" quants.

llama-quantize gives you one type for the whole model (its k-quant mixture
heuristics key off llama tensor names like `attn_v` and `ffn_down`, which an
`ltxv` file does not have, so every tensor gets the base type). That is a blunt
instrument for a model like this one, where the weights are not equally worth
protecting:

    feed-forward           30.6%   the bulk, and the most quant-tolerant part
    video attention        30.4%
    audio branch           ~15%    audio_ff, audio_attn1/2
    cross-modal attention  ~7%     audio_to_video / video_to_audio
    connectors + adaln     ~6%     small, and they steer everything
    norms, biases, gates   <1%     tiny, and catastrophic to round

LTX-2.5's selling point is that it generates picture and sound in one pass. The
audio branch and the two cross-modal attentions are the part that does that, and
together they are only about a fifth of the weights - so protecting them costs
almost nothing in file size and defends exactly the thing the model is bought
for. Meanwhile the feed-forward blocks, a third of the file, take the beating.

This writes GGUFs where each tensor's type is chosen by name. Only the types the
pure-python gguf quantiser can produce are used (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
F16), so no llama.cpp round trip and no requantisation loss.
"""
import json
import os
import re
import struct
import sys
import time

import numpy as np
import gguf
from gguf import GGMLQuantizationType as QT
from safetensors import safe_open

PREFIX = "model.diffusion_model."

# name -> (matcher, type). First match wins, so order matters.
def _rules(bulk, attn, audio, cross):
    return [
        # 1-D everything (biases, norms, scales, gate logits): never quantised.
        (lambda k, s: len(s) < 2, QT.F32),
        # the steering layers: small, and they set every block's modulation
        (lambda k, s: "adaln" in k or "embeddings_connector" in k
         or "time_embed" in k or "timestep_embedder" in k
         or "caption_projection" in k or "proj_in" in k or "proj_out" in k,
         QT.Q8_0),
        # the joint-AV machinery
        (lambda k, s: "audio_to_video_attn" in k or "video_to_audio_attn" in k,
         cross),
        (lambda k, s: k.startswith("audio_") or ".audio_" in k, audio),
        # ordinary video attention
        (lambda k, s: ".attn" in k, attn),
        # feed-forward and whatever is left
        (lambda k, s: True, bulk),
    ]


RECIPES = {
    # ~4.9 bpw. The 16 GB headline: everything cheap except what makes it LTX.
    "AVQ4": dict(bulk=QT.Q4_0, attn=QT.Q4_1, audio=QT.Q5_1, cross=QT.Q8_0),
    # ~5.6 bpw. For 24 GB: audio and cross-modal left almost untouched.
    "AVQ5": dict(bulk=QT.Q5_0, attn=QT.Q5_1, audio=QT.Q8_0, cross=QT.Q8_0),
    # ~4.5 bpw floor. Same protection, bulk pushed as far as pure python goes.
    "AVQ4S": dict(bulk=QT.Q4_0, attn=QT.Q4_0, audio=QT.Q5_0, cross=QT.Q5_1),
}


def build(src, out, recipe, label):
    rules = _rules(**RECIPES[recipe])
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    names = sorted(hdr)
    w = gguf.GGUFWriter(out + ".building", "ltxv")
    t0 = time.time()
    tally = {}
    with safe_open(src, framework="pt") as st:
        for i, k in enumerate(names, 1):
            shape = hdr[k]["shape"]
            qt = next(t for m, t in rules if m(k, shape))
            t = st.get_tensor(k)
            arr = t.float().numpy() if str(t.dtype) == "torch.bfloat16" else t.numpy()
            arr = arr.astype(np.float32)
            if qt == QT.F32:
                data = arr
            elif qt == QT.F16:
                data = arr.astype(np.float16)
            else:
                data = gguf.quants.quantize(arr, qt)
            w.add_tensor(k[len(PREFIX):] if k.startswith(PREFIX) else k, data,
                         raw_dtype=qt, raw_shape=arr.shape)
            tally[qt.name] = tally.get(qt.name, 0) + arr.nbytes // 2
            if i % 500 == 0:
                print(f"   {i}/{len(names)} {time.time() - t0:.0f}s", flush=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    os.replace(out + ".building", out)
    gb = os.path.getsize(out) / 1e9
    print(f"[{label}] {out}  {gb:.2f} GB  ({time.time() - t0:.0f}s)")
    print(f"[{label}] fp16-equivalent bytes by type: "
          + ", ".join(f"{k} {v / 1e9:.1f}G" for k, v in sorted(tally.items())))
    return gb


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    src, out, recipe = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    build(src, out, recipe, recipe)
