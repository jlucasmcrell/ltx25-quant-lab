"""Render the same seed and prompt through every LTX-2.5 quant we built.

A quant that loads is not a quant that works. The only honest test is the one
the user will run: same graph, same seed, same sigmas, swap the weights, look at
the result. The graph is lifted from an actual completed job in ComfyUI's own
history, so nothing about it is reconstructed.

GGUF files go through ComfyUI-GGUF's UnetLoaderGGUF; the comfy-native
quantisations go through the stock UNETLoader, which is the whole point of them.
"""
import json
import os
import sys
import urllib.error
import urllib.request

S = "http://127.0.0.1:8190"
BASE = json.load(open(os.path.join(os.path.dirname(__file__), "ltx_api_graph.json")))

# small and fast: the point is fidelity of the weights, not a showcase
W, H, LEN = 512, 288, 121


def graph_for(name, kind, tag, seed=12345):
    g = json.loads(json.dumps(BASE))
    if kind == "gguf":
        g["12"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": name}}
    else:
        g["12"] = {"class_type": "UNETLoader",
                   "inputs": {"unet_name": name, "weight_dtype": "default"}}
    # pin size and length: MASTER CONTROLS and the length primitive feed these
    for k, v in g.items():
        ct = v["class_type"]
        if ct == "H3StudioControls":
            v["inputs"]["width"] = W
            v["inputs"]["height"] = H
        elif ct == "PrimitiveInt" and isinstance(v["inputs"].get("value"), int) \
                and v["inputs"]["value"] in (193, 385):
            v["inputs"]["value"] = LEN
        elif ct == "RandomNoise":
            v["inputs"]["noise_seed"] = seed
        elif ct == "SaveVideo":
            v["inputs"]["filename_prefix"] = f"video/QUANTCHECK/{tag}"
    return g


def submit(g, tag):
    body = json.dumps({"prompt": g, "client_id": "quantcheck"}).encode()
    req = urllib.request.Request(S + "/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
        print(f"  queued {tag} -> {pid[:8]}")
        return pid
    except urllib.error.HTTPError as e:
        d = json.loads(e.read().decode())
        print(f"  REJECTED {tag}: {d.get('error', {}).get('message', '')}")
        for n, errs in (d.get("node_errors") or {}).items():
            for x in errs.get("errors", []):
                print("     ", n, x.get("type"), str(x.get("message"))[:120])
        return None


if __name__ == "__main__":
    todo = []
    for a in sys.argv[1:]:
        kind = "gguf" if a.lower().endswith(".gguf") else "native"
        tag = os.path.splitext(os.path.basename(a))[0].replace("LTX25-distilled-DiT-", "")
        todo.append((a, kind, tag))
    for name, kind, tag in todo:
        submit(graph_for(name, kind, tag), tag)
