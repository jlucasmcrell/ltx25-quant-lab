"""Load-only check: does the quant produce the RIGHT model, on CPU, no sampling.

The failure this exists to catch is the one the community is already hitting -
"if you convert LTX-2.5 to GGUF yourself, it will not load". A GGUF carries no
config metadata, so ComfyUI has to infer the architecture from tensor shapes,
and an inference that lands on LTX-2.3 gives you a model that either refuses to
load or loads and renders garbage. Cheaper to find out here than 20 minutes into
a render, and this needs no VRAM, so it runs while the card is busy.

Pass = the detected model class and its key dimensions match what the bf16
original produces.
"""
import sys

sys.path.insert(0, "G:/ComfyUI_LTX25/ComfyUI")
ARGS = sys.argv[1:]                           # comfy's arg parser eats sys.argv
sys.argv = [sys.argv[0]]                      # on import, so keep our own copy

import torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402


def describe(model_patcher, tag):
    m = model_patcher.model
    conf = m.model_config
    diff = m.diffusion_model
    print(f"[{tag}] model_config = {type(conf).__name__}")
    print(f"[{tag}] diffusion_model = {type(diff).__name__}")
    n_blocks = len(getattr(diff, "transformer_blocks", []) or [])
    print(f"[{tag}] transformer_blocks = {n_blocks}")
    for attr in ("inner_dim", "num_attention_heads", "attention_head_dim",
                 "caption_channels", "in_channels", "out_channels"):
        v = getattr(diff, attr, None)
        if v is not None:
            print(f"[{tag}]   {attr} = {v}")
    p = sum(x.numel() for x in diff.parameters())
    print(f"[{tag}] parameters = {p / 1e9:.2f} B")
    return {"config": type(conf).__name__, "blocks": n_blocks,
            "params": round(p / 1e9, 2)}


def load_gguf(path):
    # the pack is a package with relative imports, so import it as one
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ComfyUI_GGUF",
        "G:/ComfyUI_LTX25/ComfyUI/custom_nodes/ComfyUI-GGUF/__init__.py",
        submodule_search_locations=["G:/ComfyUI_LTX25/ComfyUI/custom_nodes/ComfyUI-GGUF"])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI_GGUF"] = pkg
    spec.loader.exec_module(pkg)
    loader = importlib.import_module("ComfyUI_GGUF.loader")
    ops_mod = importlib.import_module("ComfyUI_GGUF.ops")
    sd, extra = loader.gguf_sd_loader(path)    # newer packs return (sd, extra)
    print(f"    gguf tensors: {len(sd)}  extra: {list(extra)}")
    # the packed bytes only make sense with the pack's own ops - without them
    # comfy tries to copy Q3_K blocks straight into bf16 parameters
    return comfy.sd.load_diffusion_model_state_dict(
        sd, model_options={"custom_operations": ops_mod.GGMLOps()},
        metadata=extra.get("metadata", {}))


def load_safetensors(path):
    sd, md = comfy.utils.load_torch_file(path, return_metadata=True)
    print(f"    safetensors tensors: {len(sd)}  metadata keys: {list(md or {})[:4]}")
    return comfy.sd.load_diffusion_model_state_dict(sd, model_options={}, metadata=md)


if __name__ == "__main__":
    out = {}
    for path in ARGS or [
            "G:/ltx-lab/build/ltx25/LTX25-distilled-DiT-Q3_K_M.gguf"]:
        tag = path.rsplit("/", 1)[-1]
        print(f"=== {tag}")
        try:
            mp = load_gguf(path) if path.endswith(".gguf") else load_safetensors(path)
            if mp is None:
                print(f"[{tag}] LOAD RETURNED None - comfy could not detect a model")
                out[tag] = None
                continue
            out[tag] = describe(mp, tag)
            del mp
        except Exception as e:
            import traceback
            print(f"[{tag}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            out[tag] = "FAILED"
    print("\n--- summary ---")
    for k, v in out.items():
        print(f"  {k}: {v}")
