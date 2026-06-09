#!/usr/bin/env python3
"""ComfyUI API - Anything V5, LoRA 없이 샴고양이"""
import json, urllib.request, urllib.parse, urllib.error, time
from pathlib import Path

COMFY_URL = "http://127.0.0.1:8188"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

def queue_prompt(workflow):
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req)
        return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print("API 에러:", e.read().decode()); raise

def wait_for_result(prompt_id, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        with urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}") as r:
            h = json.loads(r.read())
        if prompt_id in h: return h[prompt_id]
        time.sleep(3)
    raise TimeoutError("타임아웃")

def fetch_image(filename, subfolder, folder_type):
    params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
    with urllib.request.urlopen(f"{COMFY_URL}/view?{params}") as r:
        return r.read()

def generate(seed, prefix="siamese_anv5"):
    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "anything-v5.safetensors"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text":
            "masterpiece, best quality, "
            "(no humans:2.0), (solo:2.0), (animal focus:2.0), "
            "(siamese cat:1.9), (1cat:1.9), (seal point:1.7), "
            "(dark brown face:1.7), (dark brown ears:1.7), (cream white body:1.6), "
            "(blue eyes:2.0), pointed ears, whiskers, sitting, "
            "ghibli style, soft watercolor, warm pastel light"
        }},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text":
            "(human:2.5), (person:2.5), (girl:2.5), (boy:2.5), "
            "(2cats:2.5), (multiple cats:2.5), "
            "(yellow eyes:2.5), (green eyes:2.5), "
            "(black cat:2.0), (merged faces:2.5), (fused faces:2.5), "
            "realistic, 3d, blurry, low quality, watermark, deformed"
        }},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0],
            "latent_image": ["5", 0], "seed": seed, "steps": 50,
            "cfg": 9.5, "sampler_name": "dpm_2", "scheduler": "karras", "denoise": 1.0
        }},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}}
    }
    print(f"[*] 생성 중... seed={seed}")
    result = queue_prompt(workflow)
    prompt_id = result["prompt_id"]
    history = wait_for_result(prompt_id)
    for node_output in history.get("outputs", {}).values():
        for img in node_output.get("images", []):
            data = fetch_image(img["filename"], img["subfolder"], img["type"])
            out_path = OUT_DIR / f"{prefix}_s{seed}.png"
            out_path.write_bytes(data)
            print(f"[+] 저장됨: {out_path}")

if __name__ == "__main__":
    for seed in [42, 456, 789]:
        generate(seed)
    print("[+] 완료!")
