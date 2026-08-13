"""INT4/INT8 mixed-precision LTX-2.5, chosen by measured error rather than by rule.

ComfyUI stores the quantisation format in a `comfy_quant` blob **per layer**
(ops.py:1136 pops `{prefix}comfy_quant` and sets `module.quant_format` from it),
so a file may carry a different format on every Linear and stock ComfyUI will
load it. Nothing else about the format has to change.

That makes the interesting question "which layers deserve the extra 4 bits", and
it is answerable rather than guessable. For every one of the 1440 quantised
Linears we measure the actual reconstruction error at both w4a8 and int8, then
solve the obvious knapsack: promoting layer i from 4-bit to 8-bit costs
p_i * (1.002 - 0.564) bytes and buys a reduction in squared Frobenius error of
||W_i||^2 * (e4_i^2 - e8_i^2). Sort by benefit per byte, promote until the budget
is spent.

Weighting by ||W||^2 rather than by relative error matters: relative error alone
says every layer is equally worth promoting, which throws bytes at small layers
whose absolute contribution to the residual is nil.

Run:
    python ltx25_mixed_native.py measure          # one pass, caches to JSON
    python ltx25_mixed_native.py plan 13.8
    python ltx25_mixed_native.py build 13.8 out.safetensors
"""
import json
import os
import struct
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, "G:/ComfyUI_LTX25/ComfyUI")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor  # noqa: E402
from ltx25_native_quant import CANON, FORMATS, M, canon_layers, quant_conf  # noqa: E402

SRC = M + "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
CACHE = "G:/ltx-lab/build/ltx25/mixed_layer_errors.json"
LO, HI = "w4a8", "int8"          # the two formats being mixed
MIN_FREE_GIB = 6.0               # abort the write pass rather than wedge the box
MAX_RSS_GIB = 12.0               # ...but only if WE are the ones holding it


def _q(w, key):
    fmt, kw = FORMATS[key]
    return QuantizedTensor.from_float(w, QUANT_ALGOS[fmt]["comfy_tensor_layout"], **kw)


def _bytes(sd):
    return sum(v.numel() * v.element_size() for v in sd.values())


def measure():
    layers, _ = canon_layers(CANON)
    out, t0 = {}, time.time()
    with safe_open(SRC, framework="pt") as f:
        keys = [k for k in f.keys() if k.endswith(".weight") and k[:-7] in layers]
        for i, k in enumerate(keys, 1):
            w = f.get_tensor(k).to(dtype=torch.bfloat16)
            wf = w.float()
            fro = wf.norm().item()
            rec = {"params": w.numel(), "fro": fro}
            for key in (LO, HI):
                try:
                    qt = _q(w, key)
                    d = qt.dequantize().float()
                    rec[f"err_{key}"] = ((wf - d).norm() / (fro + 1e-12)).item()
                    rec[f"bpp_{key}"] = _bytes(qt.state_dict(k)) / w.numel()
                    del qt, d
                except Exception as e:                       # layer can't take it
                    rec[f"err_{key}"] = None
                    rec[f"fail_{key}"] = f"{type(e).__name__}: {e}"[:160]
            out[k[:-7]] = rec
            del w, wf
            if i % 100 == 0:
                print(f"  {i}/{len(keys)}  {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(out, open(CACHE, "w"), indent=0)
    print(f"measured {len(out)} layers in {time.time() - t0:.0f}s -> {CACHE}")


def _untouched_bytes():
    with open(SRC, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    layers, _ = canon_layers(CANON)
    tot = 0
    for k, v in hdr.items():
        if k.endswith(".weight") and k[:-7] in layers:
            continue
        n_ = 1
        for d in v["shape"]:
            n_ *= d
        tot += n_ * 2                                        # bf16
    return tot


def plan(target_gb):
    d = json.load(open(CACHE))
    base = _untouched_bytes()
    # everything starts at the cheap format
    cur = base + sum(r["params"] * r[f"bpp_{LO}"] for r in d.values())
    budget = target_gb * 1e9 - cur
    cand = []
    for name, r in d.items():
        if r.get(f"err_{HI}") is None or r.get(f"err_{LO}") is None:
            continue
        cost = r["params"] * (r[f"bpp_{HI}"] - r[f"bpp_{LO}"])
        gain = (r["fro"] ** 2) * (r[f"err_{LO}"] ** 2 - r[f"err_{HI}"] ** 2)
        if cost > 0 and gain > 0:
            cand.append((gain / cost, cost, gain, name))
    cand.sort(reverse=True)
    promoted, spent, got = set(), 0.0, 0.0
    total_gain = sum(c[2] for c in cand)
    for _, cost, gain, name in cand:
        if spent + cost > budget:
            continue
        promoted.add(name)
        spent += cost
        got += gain
    print(f"  floor {cur/1e9:.2f} GB (all {LO}) | ceiling "
          f"{(base + sum(r['params']*r[f'bpp_{HI}'] for r in d.values()))/1e9:.2f} GB (all {HI})")
    print(f"  target {target_gb} GB -> promote {len(promoted)}/{len(d)} layers to {HI}, "
          f"{spent/1e9:.2f} GB spent, final {(cur+spent)/1e9:.2f} GB")
    print(f"  squared-error removed: {100*got/total_gain:.1f}% of what full {HI} would remove")
    return promoted


    # safetensors dtype names, keyed by torch dtype
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


def build(target_gb, out):
    import gc

    import psutil
    promoted = plan(target_gb)
    layers, md = canon_layers(CANON)
    blobs = {}
    for key in (LO, HI):
        fmt, kw = FORMATS[key]
        conf = quant_conf(fmt, kw)
        blobs[key] = torch.tensor(list(json.dumps(conf).encode("utf-8")),
                                  dtype=torch.uint8)
    w_ = StreamingSafetensors(out)
    t0, n, lowest = time.time(), {LO: 0, HI: 0}, 1e9
    with safe_open(SRC, framework="pt") as f:
        keys = list(f.keys())
        for i, k in enumerate(keys, 1):
            base = k[:-7] if k.endswith(".weight") else None
            if base is not None and base in layers:
                key = HI if base in promoted else LO
                w = f.get_tensor(k).to(dtype=torch.bfloat16)
                qt = _q(w, key)
                for kk, vv in qt.state_dict(k).items():
                    w_.add(kk, vv.cpu())
                w_.add(f"{base}.comfy_quant", blobs[key])
                n[key] += 1
                del w, qt
            else:
                t = f.get_tensor(k)
                w_.add(k, t)
                del t
            if i % 400 == 0:
                gc.collect()
                free = psutil.virtual_memory().available / 2**30
                rss = psutil.Process().memory_info().rss / 2**30
                lowest = min(lowest, free)
                print(f"   {i}/{len(keys)}  {HI}={n[HI]} {LO}={n[LO]}  "
                      f"{time.time()-t0:.0f}s  free {free:.1f} GiB  rss {rss:.1f} GiB",
                      flush=True)
                if free < MIN_FREE_GIB and rss > MAX_RSS_GIB:
                    raise MemoryError(
                        f"free RAM {free:.1f} GiB below floor {MIN_FREE_GIB} and this "
                        f"process holds {rss:.1f} GiB - aborting rather than wedging the box")
    md = dict(md)
    md["quantized_by"] = "riftcast/ltx25-quant-lab"
    # a budget above the ceiling promotes everything, which is a pure build, not a mix
    md["quant_format"] = (FORMATS[HI][0] if n[LO] == 0 else
                          FORMATS[LO][0] if n[HI] == 0 else f"mixed:{LO}+{HI}")
    md["quant_mixed_hi_layers"] = str(n[HI])
    w_.close(md)
    print(f"[mix] {n[HI]} at {HI}, {n[LO]} at {LO} -> {out}  "
          f"{os.path.getsize(out)/1e9:.2f} GB  {time.time()-t0:.0f}s  "
          f"low-water RAM {lowest:.1f} GiB")


def pure(key, out):
    """One format across every quantised layer, through the streaming writer."""
    import gc

    layers, md = canon_layers(CANON)
    fmt, kw = FORMATS[key]
    blob = torch.tensor(list(json.dumps(quant_conf(fmt, kw)).encode("utf-8")),
                        dtype=torch.uint8)
    w_ = StreamingSafetensors(out)
    t0, n = time.time(), 0
    print(f"[{key}] {fmt} across {len(layers)} layers")
    with safe_open(SRC, framework="pt") as f:
        keys = list(f.keys())
        for i, k in enumerate(keys, 1):
            base = k[:-7] if k.endswith(".weight") else None
            if base is not None and base in layers:
                w = f.get_tensor(k).to(dtype=torch.bfloat16)
                qt = _q(w, key)
                for kk, vv in qt.state_dict(k).items():
                    w_.add(kk, vv.cpu())
                w_.add(f"{base}.comfy_quant", blob)
                n += 1
                del w, qt
            else:
                t = f.get_tensor(k)
                w_.add(k, t)
                del t
            if i % 800 == 0:
                gc.collect()
                print(f"   {i}/{len(keys)}  quantised {n}  {time.time()-t0:.0f}s", flush=True)
    md = dict(md)
    md["quantized_by"] = "riftcast/ltx25-quant-lab"
    md["quant_format"] = fmt
    w_.close(md)
    print(f"[{key}] {n} layers -> {out}  {os.path.getsize(out)/1e9:.2f} GB  "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "measure":
        measure()
    elif cmd == "plan":
        plan(float(sys.argv[2]))
    elif cmd == "build":
        build(float(sys.argv[2]), sys.argv[3])
    elif cmd == "pure":
        pure(sys.argv[2], sys.argv[3])
