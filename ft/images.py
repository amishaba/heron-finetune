"""Image handling shared by load, prepare and the providers.

Images are downscaled once at load (longest side <= task.input.max_side) and saved as JPEG under the
run's data/images/, named by content hash. Baselines, training and the tuned model therefore all see
the same pixels, and identical images share a path (so the split stage's duplicate check catches them).
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
from pathlib import Path

from PIL import Image

JPEG_QUALITY = 90
PATCH_PX = 32  # OpenAI's patch-based vision models and Qwen3.5 (16px patches merged 2x2) both use 32px


def save_downscaled(img: Image.Image, out_dir: Path, max_side: int) -> tuple[str, dict[str, int]]:
    """Save as RGB JPEG with longest side <= max_side. Returns (file name, size info)."""
    original = img.size
    img = img.convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    data = buf.getvalue()
    name = hashlib.sha1(data).hexdigest()[:16] + ".jpg"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    if not path.exists():
        path.write_bytes(data)
    return name, {"width": img.width, "height": img.height, "orig_width": original[0], "orig_height": original[1]}


def data_url(path: str | Path, max_side: int | None = None) -> str:
    """Base64 JPEG data URL. With max_side, any image (PNG, huge photos) is re-encoded smaller first."""
    if max_side is None:
        return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()
    with Image.open(path) as im:
        img = im.convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def patches(path: str | Path) -> int:
    """32x32 patches covering the image: the unit both OpenAI and Qwen3.5 bill image tokens in (estimate)."""
    with Image.open(path) as im:
        w, h = im.size
    return math.ceil(w / PATCH_PX) * math.ceil(h / PATCH_PX)
