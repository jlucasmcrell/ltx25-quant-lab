"""Load-check a ComfyUI-native quantised file without a GPU or a render.

Drives `comfy.ops._load_quantized_module` - the real function ComfyUI calls per
layer - on one layer of every format the file declares, then runs a forward pass.
The Linear has to come from `mixed_precision_ops()`, because that is the ops
family whose `__init__` sets `factory_kwargs`, which the loader reads at line
1114. Building a `manual_cast.Linear` instead fails with a bare AttributeError
that says nothing about the actual problem.

The point is to catch a bad file in seconds rather than after a render, and in
particular to prove that a file declaring TWO formats across its layers loads -
`ops.py` reads `comfy_quant` per prefix, so it should, but should is not tested.
"""
import collections
import json
import struct
import sys

import torch

sys.path.insert(0, "G:/ComfyUI_LTX25/ComfyUI")
import comfy.ops as ops  # noqa: E402
from safetensors import safe_open  # noqa: E402


PACKED_4BIT = {"asym_w4a8_int8", "convrot_w4a4", "nvfp4"}


def check(path, forward=True):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    md = hdr.pop("__metadata__", {})
    blobs = [k for k in hdr if k.endswith(".comfy_quant")]
    print(f"\n{path.rsplit('/', 1)[-1]}")
    print(f"  tensors {len(hdr)} | quantised layers {len(blobs)} | "
          f"declared {md.get('quant_format')!r}")

    ok = True
    with safe_open(path, framework="pt") as f:
        fmts = collections.Counter()
        first = {}
        for k in blobs:
            fmt = json.loads(bytes(f.get_tensor(k).tolist()))["format"]
            fmts[fmt] += 1
            first.setdefault(fmt, k)
        print(f"  formats {dict(fmts)}")

        for fmt, k in first.items():
            base = k[: -len(".comfy_quant")]
            sd = {kk[len(base) + 1:]: f.get_tensor(kk)
                  for kk in hdr if kk.startswith(base + ".")}
            sd["comfy_quant"] = f.get_tensor(k)
            Ops = ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
            w = sd["weight"]
            # 4-bit formats pack two values per byte, so the stored weight is
            # half as wide as the layer. Passing the on-disk width makes the
            # loader compute a scale shape half the size of the real one and
            # reject a perfectly good file.
            out_f, in_f = (w.shape[0], w.shape[1]) if w.dim() == 2 else (1, w.numel())
            if fmt in PACKED_4BIT:
                in_f *= 2
            lin = Ops.Linear(in_f, out_f, bias=False, device="cpu")
            miss, unexp = [], []
            try:
                ops._load_quantized_module(
                    lin, lambda *a, **kw: None,
                    {f"x.{kk}": vv for kk, vv in sd.items()},
                    "x.", {}, False, miss, unexp, [])
            except Exception as e:
                print(f"    {fmt:18s} LOAD FAIL {type(e).__name__}: {str(e)[:110]}")
                ok = False
                continue
            note = f"quant_format={lin.quant_format} layout={lin.layout_type}"
            if forward:
                try:
                    x = torch.randn(2, lin.in_features, dtype=torch.bfloat16)
                    y = lin(x)
                    note += f" forward{tuple(y.shape)}"
                    if not torch.isfinite(y.float()).all():
                        note += " NON-FINITE"
                        ok = False
                except Exception as e:
                    note += f" forward FAIL {type(e).__name__}: {str(e)[:70]}"
            print(f"    {fmt:18s} load OK  {note}")
    return ok


if __name__ == "__main__":
    D = "G:/ltx-lab/build/ltx25/"
    files = sys.argv[1:] or [
        D + "LTX25-distilled-DiT-comfy-mix4x8-13.8GB.safetensors",
        D + "LTX25-distilled-DiT-comfy-mix4x8-17GB.safetensors",
        D + "LTX25-distilled-DiT-comfy-int8.safetensors",
        D + "LTX25-distilled-DiT-comfy-w4a8.safetensors",
    ]
    allok = all(check(p) for p in files)
    print(f"\n{'ALL OK' if allok else 'FAILURES ABOVE'}")
