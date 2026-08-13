"""The quant comparison, rendered as RIFT content rather than as throwaways.

Joe's rule: what we render has to be useful. So the arms are a real SIGHTINGS
scene from the factory, vertical, at a size that can be posted - and the
comparison still holds, because the ONLY thing that differs between arms is
which weights are loaded. Same script, same seed, same sigmas, same size.

Order is "what nobody else has, first":
  1. comfy-w4a4    11.2 GB  the smallest LTX-2.5 transformer published in any
                            format, and the only convrot_w4a4 build anywhere
  2. comfy-nvfp4   12.5 GB  ours against Lightricks' 18.7 GB nvfp4
  3. Q3_K_S         9.7 GB  no equivalent at this tier elsewhere
  4. Q2_K           7.9 GB  the 12 GB-card rung
  5. Q3_K_M        10.6 GB  the 16 GB recommendation
  6. comfy-w4a8    12.5 GB
  7. Q4_K_S        12.9 GB
"""
import json
import os
import sys
import urllib.error
import urllib.request

S = "http://127.0.0.1:8190"
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = json.load(open(os.path.join(HERE, "ltx_api_graph.json")))
SCRIPT_JSON = (r"F:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable"
               r"\ComfyUI\custom_nodes\comfyui-inspire-pack\prompts\example"
               r"\RIFT PROMPTS\_GEN\SIGHTINGS_two_adult_neighbours_stand_in_JOYECHO.json")

# 544x960 = 0.52 MP, the transpose of the official 0.5 MP row in LTX-2.5's
# own resolution table. The previous run used 704x1216 = 0.86 MP, four times
# the official pass-1 size, and the four-step pass 2 could not resolve it -
# gravel and chain-link came back as a noise carpet in every arm.
W, H, LENGTH, SEED = 544, 960, 385, 771120
SHOTS = 3            # LTX generates all of them in one pass

ARMS = [
    ("comfy-w4a4",  r"ltx25quant\LTX25-distilled-DiT-comfy-w4a4.safetensors", "native"),
    ("comfy-nvfp4", r"ltx25quant\LTX25-distilled-DiT-comfy-nvfp4.safetensors", "native"),
    ("Q3_K_S",      r"ltx25quant\LTX25-distilled-DiT-Q3_K_S.gguf", "gguf"),
    ("Q2_K",        r"ltx25quant\LTX25-distilled-DiT-Q2_K.gguf", "gguf"),
    ("Q3_K_M",      r"ltx25quant\LTX25-distilled-DiT-Q3_K_M.gguf", "gguf"),
    ("comfy-w4a8",  r"ltx25quant\LTX25-distilled-DiT-comfy-w4a8.safetensors", "native"),
    ("Q4_K_S",      r"ltx25quant\LTX25-distilled-DiT-Q4_K_S.gguf", "gguf"),
    ("Q4_K_M",      r"ltx25quant\LTX25-distilled-DiT-Q4_K_M.gguf", "gguf"),
    ("Q5_K_M",      r"ltx25quant\LTX25-distilled-DiT-Q5_K_M.gguf", "gguf"),
    ("Q6_K",        r"ltx25quant\LTX25-distilled-DiT-Q6_K.gguf", "gguf"),
    ("Q8_0",        r"ltx25quant\LTX25-distilled-DiT-Q8_0.gguf", "gguf"),
    ("comfy-fp8",   r"ltx25quant\LTX25-distilled-DiT-comfy-fp8_e4m3fn.safetensors", "native"),
]


def script_text():
    d = json.load(open(SCRIPT_JSON, encoding="utf-8"))
    shots = d.get("prompts") or d.get("shots")
    return "\n---\n".join(shots[:SHOTS])


def graph_for(tag, name, kind, script):
    g = json.loads(json.dumps(BASE))
    if kind == "gguf":
        g["12"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": name}}
    else:
        g["12"] = {"class_type": "UNETLoader",
                   "inputs": {"unet_name": name, "weight_dtype": "default"}}
    for k, v in g.items():
        ct = v["class_type"]
        i = v["inputs"]
        if ct == "RiftEngineScript":
            i["script"] = script
        elif ct == "PrimitiveBoolean" and i.get("value") in (True, False):
            pass
        elif ct == "H3StudioControls":
            i["width"], i["height"] = W, H
        elif ct == "PrimitiveInt" and isinstance(i.get("value"), int) and i["value"] > 100:
            i["value"] = LENGTH
        elif ct == "RandomNoise":
            i["noise_seed"] = SEED
        elif ct == "SaveVideo":
            i["filename_prefix"] = f"video/SIGHTINGS_05MP/{tag}"
    # ENGINE -> LTX-2.5
    for k, v in g.items():
        if v["class_type"] == "PrimitiveBoolean" and k == "1":
            v["inputs"]["value"] = True
    return g


if __name__ == "__main__":
    script = script_text()
    print(f"script: {len(script)} chars, {len(script.split(chr(10)+'---'+chr(10)))} shots")
    only = sys.argv[1:] or None
    for tag, name, kind in ARMS:
        if only and tag not in only:
            continue
        body = json.dumps({"prompt": graph_for(tag, name, kind, script),
                           "client_id": "quantarms"}).encode()
        req = urllib.request.Request(S + "/prompt", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            pid = json.load(urllib.request.urlopen(req))["prompt_id"]
            print(f"  queued {tag:<12} -> {pid[:8]}")
        except urllib.error.HTTPError as e:
            d = json.loads(e.read().decode())
            print(f"  REJECTED {tag}: {d.get('error', {}).get('message', '')}")
            for n, errs in (d.get("node_errors") or {}).items():
                for x in errs.get("errors", []):
                    print("     ", n, x.get("type"), str(x.get("message"))[:110])
