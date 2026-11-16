"""Regenerates the tiny_receipts fixture: 30 synthetic receipt images plus labels.jsonl in CORD's
ground_truth shape (gt_parse.menu / sub_total / total), so the same target_map as examples/cord works.

    .venv/bin/python examples/fixtures/tiny_receipts/make.py
"""

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ITEMS = ["KOPI SUSU", "ES TEH", "NASI GORENG", "ROTI BAKAR", "MIE AYAM", "AIR MINERAL", "TAHU ISI", "SATE"]


def money(n: int) -> str:
    return f"{n:,}".replace(",", ".")  # Indonesian style, like CORD


def main() -> None:
    rng = random.Random(7)
    (HERE / "images").mkdir(exist_ok=True)
    rows = []
    for i in range(30):
        menu = []
        for name in rng.sample(ITEMS, rng.randint(1, 4)):
            cnt, unit = rng.randint(1, 3), rng.choice([8, 12, 15, 18, 25]) * 1000
            menu.append({"nm": name, "cnt": str(cnt), "price": money(cnt * unit)})
        sub = sum(int(m["price"].replace(".", "")) for m in menu)
        tax = sub // 10 if i % 3 else 0
        img = Image.new("RGB", (320, 120 + 28 * len(menu) + 90), "white")
        d = ImageDraw.Draw(img)
        d.text((20, 20), f"WARUNG {i:02d}", fill="black")
        y = 60
        for m in menu:
            d.text((20, y), f"{m['cnt']} {m['nm']}", fill="black")
            d.text((220, y), m["price"], fill="black")
            y += 28
        d.text((20, y + 10), "SUBTOTAL", fill="black")
        d.text((220, y + 10), money(sub), fill="black")
        if tax:
            d.text((20, y + 34), "PB1 10%", fill="black")
            d.text((220, y + 34), money(tax), fill="black")
        d.text((20, y + 58), "TOTAL", fill="black")
        d.text((220, y + 58), money(sub + tax), fill="black")
        name = f"r{i:03d}.png"
        img.save(HERE / "images" / name)
        sub_total = {"subtotal_price": money(sub), **({"tax_price": money(tax)} if tax else {})}
        gt = {
            "gt_parse": {
                "menu": menu if len(menu) > 1 else menu[0],
                "sub_total": sub_total,
                "total": {"total_price": money(sub + tax)},
            }
        }
        split = "train" if i < 20 else ("val" if i < 25 else "test")
        rows.append({"image": name, "split": split, "ground_truth": json.dumps(gt)})
    (HERE / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
