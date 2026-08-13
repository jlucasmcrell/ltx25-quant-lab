"""Quantise LTX-2.5 into ComfyUI's OWN native formats - no custom nodes needed.

ComfyUI 0.32 carries a quantisation system of its own (comfy/quant_ops.py plus
the comfy_kitchen kernels): each quantised Linear stores `weight`, a
`weight_scale`, and a tiny `comfy_quant` blob of JSON naming the format. Stock
ComfyUI loads it. No ComfyUI-GGUF, no dequant-on-the-fly through a custom node,
and the kernels run in the quantised domain rather than unpacking to bf16.

Lightricks' own `comfy-int8-convrot` release is a working example of the format,
so the layer SET is mirrored from it rather than guessed: the 1440 Linears they
chose to quantise get quantised, and everything they left in bf16 - adaLN, the
timestep embedders, every norm and bias, the scale_shift tables - stays bf16
here too. Those are the layers that steer every block; they are 6% of the file
and rounding them is how a quantised DiT dies.

What that leaves us room to change is the FORMAT:

    int8_tensorwise   8 bits   ~21.5 GB   what Lightricks already ship
    float8_e4m3fn     8 bits   ~21.5 GB   wider GPU support than int8 kernels
    asym_w4a8_int8    4 bits   ~11.5 GB   <- the 16 GB card lands here
    convrot_w4a4      4 bits   ~11.5 GB   different 4-bit math, same budget
    nvfp4             4 bits   ~11.5 GB   Blackwell only (5070 Ti / 5080 16 GB)

The 4-bit formats are the point. A 22B joint audio+video DiT does not otherwise
fit on a 16 GB card without spilling to system RAM, and spilling costs more than
the quantisation does.
"""
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, "G:/ComfyUI_LTX25/ComfyUI")
from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor  # noqa: E402

M = ("F:/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/"
     "models/diffusion_models/LTX New/")
CANON = M + "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"

FORMATS = {
    "w4a8":    ("asym_w4a8_int8", dict(group_size=16, convrot_groupsize=256)),
    "w4a4":    ("convrot_w4a4", dict(convrot_groupsize=256, quant_group_size=64)),
    "fp8":     ("float8_e4m3fn", dict(scale="recalculate")),
    "nvfp4":   ("nvfp4", dict(scale="recalculate")),
    "int8":    ("int8_tensorwise", dict(per_channel=True, convrot=True,
                                        convrot_groupsize=256)),
}


def quant_conf(fmt, kwargs):
    """The comfy_quant JSON, matching what ops.py writes for each format."""
    c = {"format": fmt}
    if fmt == "int8_tensorwise" and kwargs.get("convrot"):
        c["convrot"] = True
        c["convrot_groupsize"] = kwargs.get("convrot_groupsize", 256)
    elif fmt == "convrot_w4a4":
        c["convrot_groupsize"] = kwargs.get("convrot_groupsize", 256)
    elif fmt == "asym_w4a8_int8":
        c["group_size"] = kwargs.get("group_size", 16)
        c["convrot_groupsize"] = kwargs.get("convrot_groupsize", 256)
    return c


def canon_layers(path):
    """The exact set of Linears Lightricks quantised, and the file metadata."""
    with open(path, "rb") as f:
        import struct
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    md = hdr.pop("__metadata__", {})
    layers = {k[:-len(".comfy_quant")] for k in hdr if k.endswith(".comfy_quant")}
    return layers, md


def build(src, out, key, canon=CANON, device="cuda", limit=None):
    fmt, kwargs = FORMATS[key]
    layout = QUANT_ALGOS[fmt]["comfy_tensor_layout"]
    layers, md = canon_layers(canon)
    conf = quant_conf(fmt, kwargs)
    blob = torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)
    print(f"[{key}] {fmt} via {layout}  {len(layers)} layers  conf={conf}")

    sd = {}
    t0 = time.time()
    n_q = n_copy = 0
    with safe_open(src, framework="pt") as f:
        keys = list(f.keys())
        for i, k in enumerate(keys, 1):
            base = k[:-len(".weight")] if k.endswith(".weight") else None
            if base is not None and base in layers:
                w = f.get_tensor(k).to(device=device, dtype=torch.bfloat16)
                qt = QuantizedTensor.from_float(w, layout, **kwargs)
                for kk, vv in qt.state_dict(k).items():
                    sd[kk] = vv.cpu()
                sd[f"{base}.comfy_quant"] = blob.clone()
                n_q += 1
                del w, qt
                if n_q % 200 == 0:
                    torch.cuda.empty_cache()
                    print(f"   {i}/{len(keys)}  quantised {n_q}  "
                          f"{time.time() - t0:.0f}s", flush=True)
            else:
                sd[k] = f.get_tensor(k)
                n_copy += 1
            if limit and n_q >= limit:
                print(f"[{key}] limit {limit} reached, stopping early")
                break
    print(f"[{key}] {n_q} quantised, {n_copy} copied, {time.time() - t0:.0f}s")
    md_out = {k: v for k, v in md.items()}
    md_out["quantized_by"] = "riftcast/ltx25-quant-lab"
    md_out["quant_format"] = fmt
    save_file(sd, out + ".building", metadata=md_out)
    os.replace(out + ".building", out)
    print(f"[{key}] {out}  {os.path.getsize(out) / 1e9:.2f} GB")


if __name__ == "__main__":
    key = sys.argv[1]
    src = sys.argv[2] if len(sys.argv) > 2 else M + "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    out = sys.argv[3] if len(sys.argv) > 3 else f"G:/ltx-lab/build/ltx25/LTX25-distilled-DiT-comfy-{key}.safetensors"
    canon = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else CANON
    lim = int(sys.argv[5]) if len(sys.argv) > 5 else None
    os.makedirs(os.path.dirname(out), exist_ok=True)
    build(src, out, key, canon=canon, limit=lim)
