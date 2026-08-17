"""MiniMax-H3 in ComfyUI's own quantisation formats - no custom node needed.

The LTX-2.5 lab, pointed at a different model. Two things carry over unchanged
and one is new.

Carries over: the layer SET is mirrored from a canonical file rather than
guessed. Comfy-Org publish `minimax_h3_ref2va_pruned_int8_convrot.safetensors`,
and whichever Linears they chose to quantise are the ones quantised here. For
LTX-2.5 that was Lightricks' int8-convrot; same idea, same reason - the steering
layers are where a quantised DiT dies and this is not the place to have an
opinion.

Carries over: the streaming writer, because `save_file` wants the whole state
dict resident and this model is 40 GB of bf16.

New, and the reason this is worth doing at all: H3's canonical set includes
**adaln**, which the LTX set did not. On the PRUNED lineage adaln is small
(the pruning strips adaln_proj); on the full one it is 38% of the file. Build
from pruned bf16 or the arithmetic does not work - full-lineage 4-bit lands at
19.9 GB, which is what the existing GGUF Q4_0 already achieves, while pruned
4-bit lands near 12.5 GB.

    python h3_native_quant.py w4a8 --src minimax_h3_ref2va_pruned_bf16.safetensors         --canon minimax_h3_ref2va_pruned_int8_convrot.safetensors --out my-w4a8.safetensors
    python h3_native_quant.py nvfp4 --src <pruned fl2va bf16> --canon <ref2va canon> --out ...

--comfy points at a ComfyUI checkout (0.30+) so comfy.quant_ops imports; it
defaults to the COMFYUI_DIR environment variable, then ../../ComfyUI.
"""
import argparse
import json
import os
import struct
import sys
import time

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _import_comfy(comfy_dir):
    """comfy.quant_ops lives inside a ComfyUI checkout; point at one."""
    cands = [comfy_dir, os.environ.get("COMFYUI_DIR"),
             os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "..", "ComfyUI")]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "comfy")):
            sys.path.insert(0, os.path.abspath(c))
            break
    try:
        from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor  # noqa: E402
    except ImportError as e:
        sys.exit("cannot import comfy.quant_ops - pass --comfy <path to a "
                 "ComfyUI 0.30+ checkout> or set COMFYUI_DIR (%s)" % e)
    return QUANT_ALGOS, QuantizedTensor


QUANT_ALGOS = QuantizedTensor = None


# --- streaming safetensors writer (inlined from ltx25_mixed_native so this file
# runs on its own; that module imports comfy at import time) ---------------
ST_DTYPE = {
    torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32",
    torch.float64: "F64", torch.uint8: "U8", torch.int8: "I8",
    torch.int16: "I16", torch.int32: "I32", torch.int64: "I64",
    torch.bool: "BOOL", torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e5m2: "F8_E5M2",
    # mxfp8 block scales are e8m0; safetensors has no name for it and ComfyUI
    # reads them back with .view(torch.float8_e8m0fnu), so store the raw bytes
    torch.float8_e8m0fnu: "U8",
}


def _raw(t):
    """Contiguous little-endian bytes of any tensor, without a dtype-specific path."""
    t = t.contiguous().cpu()
    return t.view(-1).view(torch.uint8).numpy().tobytes()


class StreamingSafetensors:
    """Write a safetensors file one tensor at a time.

    `save_file` wants the whole state dict resident, which for a 22B model means
    tens of GB on top of whatever the source read is holding - the exact machine
    this project exists to avoid needing. The format does not require it: it is
    an 8-byte header length, a JSON header of {name: {dtype, shape,
    data_offsets}}, then the tensor bytes back to back in offset order. So bodies
    stream to a sidecar as they are produced, and the header is written once the
    offsets are known.
    """

    def __init__(self, out):
        self.out = out
        self.tmp = out + ".data"
        self.fh = open(self.tmp, "wb")
        self.entries, self.off, self._synced = {}, 0, 0

    def add(self, name, t):
        b = _raw(t)
        self.fh.write(b)
        self.entries[name] = {"dtype": ST_DTYPE[t.dtype], "shape": list(t.shape),
                              "data_offsets": [self.off, self.off + len(b)]}
        self.off += len(b)
        # Push written bytes out of the modified-page cache. Without this the
        # sidecar's dirty pages count as unavailable memory and look exactly like
        # a leak, which is what tripped the RAM guard on the first attempt.
        if self.off - self._synced > (512 << 20):
            self.fh.flush()
            os.fsync(self.fh.fileno())
            self._synced = self.off

    def close(self, metadata):
        self.fh.close()
        hdr = dict(self.entries)
        if metadata:
            hdr["__metadata__"] = {k: str(v) for k, v in metadata.items()}
        blob = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
        blob += b" " * ((8 - len(blob) % 8) % 8)          # data must start 8-aligned
        with open(self.out + ".building", "wb") as o:
            o.write(struct.pack("<Q", len(blob)))
            o.write(blob)
            with open(self.tmp, "rb") as d:
                while True:
                    chunk = d.read(64 << 20)
                    if not chunk:
                        break
                    o.write(chunk)
        os.remove(self.tmp)
        os.replace(self.out + ".building", self.out)

D = os.environ.get("H3_QUANT_DIR", "./h3native/diffusion_models/")

# VARIANTS are the presets used for the published builds; --src / --canon /
# --out override them, so nothing here needs editing to quantise your own file.
#
# fl2va deliberately reuses the REF2VA canon. Verified against the headers:
# both pruned bf16 files carry byte-identical key sets, shapes and dtypes (532
# keys), and all 200 canon layer names resolve in fl2va - so the arch is the
# same and fetching the 21 GB fl2va canon buys nothing. Note the trap: the
# UNPRUNED fl2va canon quantises 250 layers because it includes
# blocks.N.adaln_proj.linear. Against a pruned source that would be wrong.
VARIANTS = {
    "ref2va": dict(
        src=D + "minimax_h3_ref2va_pruned_bf16.safetensors",
        canon=D + "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        outdir="./h3native/",
        source_tag="minimax_h3_ref2va_pruned_bf16"),
    "fl2va": dict(
        src=D + "minimax_h3_fl2va_pruned_bf16.safetensors",
        canon=D + "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        outdir="./h3native_fl2va/",
        source_tag="minimax_h3_fl2va_pruned_bf16"),
}
SRC = CANON = OUTDIR = SOURCE_TAG = None    # bound from --variant in __main__

FORMATS = {
    "int8":    ("int8_tensorwise", dict(per_channel=True, convrot=True,
                                        convrot_groupsize=256)),
    "w4a8":    ("asym_w4a8_int8", dict(group_size=16, convrot_groupsize=256)),
    "w4a4":    ("convrot_w4a4", dict(convrot_groupsize=256, quant_group_size=64)),
    "nvfp4":   ("nvfp4", dict(scale="recalculate")),
    "mxfp8":   ("mxfp8", dict(scale="recalculate")),
    "fp8":     ("float8_e4m3fn", dict(scale="recalculate")),
    "fp8e5m2": ("float8_e5m2", dict(scale="recalculate")),
}


def quant_conf(fmt, kw):
    c = {"format": fmt}
    if fmt == "int8_tensorwise" and kw.get("convrot"):
        c["convrot"] = True
        c["convrot_groupsize"] = kw.get("convrot_groupsize", 256)
    elif fmt == "convrot_w4a4":
        c["convrot_groupsize"] = kw.get("convrot_groupsize", 256)
    elif fmt == "asym_w4a8_int8":
        c["group_size"] = kw.get("group_size", 16)
        c["convrot_groupsize"] = kw.get("convrot_groupsize", 256)
    return c


def canon_layers(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    md = hdr.pop("__metadata__", {})
    return {k[:-len(".comfy_quant")] for k in hdr
            if k.endswith(".comfy_quant")}, md


def build(key, out):
    global SRC, CANON, SOURCE_TAG
    import gc
    fmt, kw = FORMATS[key]
    layout = QUANT_ALGOS[fmt]["comfy_tensor_layout"]
    layers, md = canon_layers(CANON)
    blob = torch.tensor(list(json.dumps(quant_conf(fmt, kw)).encode("utf-8")),
                        dtype=torch.uint8)
    print(f"[{key}] {fmt} via {layout} | {len(layers)} canonical layers", flush=True)

    w_ = StreamingSafetensors(out)
    t0, n, skipped = time.time(), 0, []
    with safe_open(SRC, framework="pt") as f:
        keys = list(f.keys())
        for i, k in enumerate(keys, 1):
            base = k[:-7] if k.endswith(".weight") else None
            if base is not None and base in layers:
                w = f.get_tensor(k).to(dtype=torch.bfloat16)
                try:
                    qt = QuantizedTensor.from_float(w, layout, **kw)
                except Exception as e:
                    # a layer the format cannot take stays bf16 rather than
                    # killing an hour-long build
                    skipped.append((base, f"{type(e).__name__}"))
                    w_.add(k, w)
                    del w
                    continue
                for kk, vv in qt.state_dict(k).items():
                    w_.add(kk, vv.cpu())
                w_.add(f"{base}.comfy_quant", blob)
                n += 1
                del w, qt
            else:
                t = f.get_tensor(k)
                w_.add(k, t)
                del t
            if i % 400 == 0:
                gc.collect()
                print(f"   {i}/{len(keys)}  quantised {n}  {time.time()-t0:.0f}s",
                      flush=True)
    md = dict(md)
    md["quantized_by"] = "riftcast/ltx25-quant-lab"
    md["quant_format"] = fmt
    md["quant_source"] = SOURCE_TAG
    w_.close(md)
    print(f"[{key}] {n} layers -> {out}  {os.path.getsize(out)/1e9:.2f} GB  "
          f"{time.time()-t0:.0f}s", flush=True)
    if skipped:
        print(f"[{key}] {len(skipped)} layer(s) left bf16: {skipped[:4]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("format", choices=sorted(FORMATS))
    ap.add_argument("--variant", default="ref2va", choices=sorted(VARIANTS),
                    help="preset (paths under $H3_QUANT_DIR); overridden by --src/--canon")
    ap.add_argument("--src", default=None,
                    help="your bf16 safetensors (use the PRUNED lineage)")
    ap.add_argument("--canon", default=None,
                    help="Comfy-Org's minimax_h3_ref2va_pruned_int8_convrot.safetensors - "
                         "its quantised layer set is mirrored")
    ap.add_argument("--out", default=None)
    ap.add_argument("--comfy", default=None,
                    help="path to a ComfyUI 0.30+ checkout (for comfy.quant_ops)")
    a = ap.parse_args()
    QUANT_ALGOS, QuantizedTensor = _import_comfy(a.comfy)
    globals()["QUANT_ALGOS"], globals()["QuantizedTensor"] = QUANT_ALGOS, QuantizedTensor
    v = VARIANTS[a.variant]
    globals()["SRC"] = a.src or v["src"]
    globals()["CANON"] = a.canon or v["canon"]
    globals()["OUTDIR"] = v["outdir"]
    globals()["SOURCE_TAG"] = (os.path.splitext(os.path.basename(a.src))[0]
                               if a.src else v["source_tag"])
    OUTDIR = v["outdir"]
    for p in (SRC, CANON):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")
    os.makedirs(OUTDIR, exist_ok=True)
    out = a.out or (f"{OUTDIR}MiniMax-H3-{a.variant}-pruned-comfy-"
                    f"{a.format}.safetensors")
    if os.path.exists(out):
        sys.exit(f"refusing to overwrite an existing build: {out}")
    print(f"variant={a.variant}  src={os.path.basename(SRC)}  "
          f"canon={os.path.basename(CANON)}", flush=True)
    build(a.format, out)
