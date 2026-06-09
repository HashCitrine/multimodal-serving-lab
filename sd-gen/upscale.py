#!/usr/bin/env python3
"""
Pillow Lanczos 4x 업스케일 + 언샤프 마스킹 샤프닝
- 얼굴 디테일 강화 효과
"""
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path

OUT_DIR = Path(__file__).parent / "outputs"

# 업스케일할 파일들
targets = [
    "ghibli_face_s10.png",
    "ghibli_face_s20.png",
    "ghibli_face_s30.png",
]

for fname in targets:
    src = OUT_DIR / fname
    if not src.exists():
        print(f"[!] 없음: {src}")
        continue

    img = Image.open(src)
    w, h = img.size
    print(f"[*] {fname}: {w}x{h} → {w*4}x{h*4}")

    # 4x 업스케일 (Lanczos — 최고품질 필터)
    upscaled = img.resize((w * 4, h * 4), Image.LANCZOS)

    # 언샤프 마스킹: 엣지/디테일 선명하게
    sharpened = upscaled.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=2))

    # 채도 살짝 올리기 (지브리 색감 강조)
    enhancer = ImageEnhance.Color(sharpened)
    final = enhancer.enhance(1.1)

    out_path = OUT_DIR / fname.replace(".png", "_4x.png")
    final.save(out_path)
    print(f"[+] 저장됨: {out_path}")

print("[+] 완료!")
