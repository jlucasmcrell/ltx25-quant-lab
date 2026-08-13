# LTX-2.5 quant lab

Tooling that produced [**joeygambino/LTX-2.5-Quantized-16GB**](https://huggingface.co/joeygambino/LTX-2.5-Quantized-16GB) —
eleven quantisations of Lightricks' LTX-2.5 22B audio+video transformer, cut so
the model runs on a card that does not have 42 GB.

The weights live on Hugging Face. **This repo is the how**, because the two
things that made it work are not obvious and cost a day to find.

---

## The two findings

### 1. GGUF silently loses the config, and the audio branch breaks four hours later

ComfyUI does not infer LTX-2.5's transformer config from the tensor shapes. It
reads it out of the safetensors `__metadata__` block. GGUF has no such block, so
a straight conversion produces a file that loads, samples, and then decodes
audio through a connector sized **3840** instead of this checkpoint's **2048**.

Nothing errors. You get a video with wrong sound.

The fix is in [`tools/ltx25_gguf_f16.py`](tools/ltx25_gguf_f16.py): carry every
string field of `__metadata__` across as a GGUF KV. `llama-quantize` passes
unknown KVs straight through, so this survives the quantisation step without
patching llama.cpp.

The same file handles the second half of the problem — **GGUF drops leading
singleton dimensions**. `keyframes_abs_pos_embedding` is `[1, 4096]` and comes
back `[4096]`. ComfyUI-GGUF will restore the original shape if you tell it, so
the writer emits `comfy.gguf.orig_shape.<tensor>` as an INT32 array for every
tensor whose shape would change.

### 2. K-quants are legal here; IQ-quants are not

4041 of 4349 tensors have a last dimension divisible by 256. Of the 308 that do
not, 306 are biases that stay F16 anyway — 304 gate-logit biases of length 32,
plus `proj_out.bias` and `audio_proj_out.bias` at length 128 — and **two are real
weight matrices**, `patchify_proj.weight` and `audio_patchify_proj.weight`, whose
128-wide axis cannot take a K-quant. They fall back to F16, which is the
right answer for input projections regardless.

`llama-quantize` **refuses IQ types** for this class of file:

```
failed to quantize: Invalid quantization type for image model (Not supported)
```

So no IQ2/IQ3/IQ4 ladder exists for LTX-2.5. If you find one published, it was
made some other way.

Also worth knowing: the architecture string has to be
`general.architecture = "ltxv"` exactly. ComfyUI-GGUF checks it against
`IMG_ARCH_LIST` in `loader.py` — thirteen architectures, of which `ltxv` is the
one this model must claim — and anything outside that list is rejected before a
single tensor is read.

---

## Why these files are smaller than the other LTX-2.5 GGUFs

Against **realrebelai** it is a flat ~0.9 GB at every level (0.90–0.93 GB, Q2_K
through Q8_0), and the reason is boring and checkable: the 2605 tensors that are
never quantised — norms, biases, scale-shift tables, the two patchify
projections — are written **F16 here and F32 elsewhere**. Same weights, half the
bytes, no quality argument involved.

Against **vantagewithai / Abiray** the same ~0.9 GB holds at Q6_K and Q8_0 but
the gap widens to 2.3–4.2 GB below Q5, which that argument does not explain —
their Q2_K and Q3_K_M are only 0.8 GB apart, so much of their low-bit ladder is
not actually quantised.

| level | here | realrebelai | vantagewithai / Abiray |
|---|---|---|---|
| Q2_K | **7.91** | 8.83 | 12.13 |
| Q3_K_M | **10.60** | 11.53 | 12.92 |
| Q4_K_S | **12.93** | 13.85 | 15.33 |
| Q4_K_M | **14.17** | 15.09 | 15.69 |
| Q6_K | **17.75** | 18.66 | 18.62 |
| Q8_0 | **22.73** | 23.63 | 23.60 |

---

## The comfy-native branch

ComfyUI 0.32 ships its own quantisation system — a `comfy_quant` uint8 JSON blob
per layer, read by the comfy-kitchen kernels. No custom node required.

[`tools/ltx25_native_quant.py`](tools/ltx25_native_quant.py) writes
`float8_e4m3fn`, `nvfp4`, `asym_w4a8_int8` and `convrot_w4a4` builds. The
important part is not the arithmetic, it is the **layer selection**: the set of
1440 Linears to quantise is *mirrored from Lightricks' own `int8-convrot`
release* rather than guessed. adaLN, the timestep embedders, every norm and bias
and the scale-shift tables stay bf16. Those steering layers are about 6% of the
file, and rounding them is how a quantised DiT dies.

---

## Tools

| file | what it does |
|---|---|
| `tools/ltx25_gguf_f16.py` | bf16 safetensors → F16 GGUF master, with the metadata and orig_shape fixes above. Feed the result to `llama-quantize`. |
| `tools/ltx25_native_quant.py` | bf16 safetensors → ComfyUI-0.32-native quantised safetensors. `QuantizedTensor.from_float(w, layout, **kw)`. |
| `tools/ltx25_mixed_gguf.py` | per-tensor-class mixed ladders (keep attention high, drop the FFN) for sizes the standard levels do not hit. |
| `tools/ltx25_loadcheck.py` | loads a built file on CPU through ComfyUI-GGUF's `GGMLOps` and reports shape/KV mismatches — catches the two findings above in seconds instead of after a render. |
| `tools/ltx25_verify.py` | header/KV inspection of a finished GGUF. |
| `tools/quant_arms.py` | render harness: queues the same scene, seed and size through every arm so the only variable is the weights. |

`ltx25_loadcheck.py` is the one to run first on anything you build.

### Run these with ComfyUI's Python, not your system one

`ltx25_native_quant.py` and `ltx25_mixed_native.py` need `comfy_kitchen`, which
ships inside ComfyUI's embedded interpreter. Under a system Python you get an
`ERROR:root:Failed to import comfy_kitchen` line, a **stub** `QuantizedTensor`
class with no `from_float`, and then an `AttributeError` several steps later that
does not name the real cause.

```bash
./python_embeded/python.exe tools/ltx25_native_quant.py w4a8
```

The GGUF tools have no such dependency and run anywhere.

---

## How to reproduce a level

```bash
python tools/ltx25_gguf_f16.py                 # writes LTX25-distilled-DiT-F16.gguf
llama-quantize LTX25-distilled-DiT-F16.gguf LTX25-distilled-DiT-Q4_K_S.gguf Q4_K_S
python tools/ltx25_loadcheck.py LTX25-distilled-DiT-Q4_K_S.gguf
```

Stock `llama-quantize`. No patch. The metadata survives because unknown KVs are
passed through.

---

## What is not here

- **The VAEs are not quantised and should not be.** 1.5 GB and 0.4 GB — the
  saving is inside the noise of a 16 GB budget, and decode is where artefacts are
  most visible.
- **No imatrix ladder.** Not attempted; the calibration path for a joint
  audio+video DiT is not the text-model one and has not been worked out here.
- **`w4a4` and `nvfp4` were built and tested on Blackwell.** The kernel paths
  declare SM 7.5+ / 8.0+, but neither has been run here on an Ada or Ampere 16 GB
  card, which is most of this repo's audience. NVFP4 is Blackwell-only by
  construction. The GGUF ladder has no such question over it.

## License

Tooling: MIT. The weights it produces inherit the
[LTX-2.x Community License](https://huggingface.co/Lightricks/LTX-2.5/blob/main/LICENSE)
from Lightricks/LTX-2.5; the license text travels inside every `.safetensors`
file's metadata.
