# ───── grab_openimages_resilient.py ─────
"""
Download           : 1 000 usable pictures from dalle-mini/open-images
Split              : 500 → images_train/ , 500 → images_val/
JSON file listing  : train_files.json , val_files.json
"""
import json, random, io, urllib.request, urllib.error, time
from pathlib import Path

from datasets import load_dataset
from PIL import Image

DATASET  = "dalle-mini/open-images"
TARGET   = 1_000
TRAIN_N  = 500
SEED     = 42
TIMEOUT  = 8            # seconds per HTTP request

# ---------- output folders ---------------------------------------------------
ROOT = Path("openimages_subset")
TR   = ROOT / "images_train"
VA   = ROOT / "images_val"
TR.mkdir(parents=True, exist_ok=True)
VA.mkdir(parents=True, exist_ok=True)

random.seed(SEED)

def fetch_pil(sample):
    """Return a PIL.Image or raise IOError/HTTPError."""
    if "image" in sample:                        # already decoded
        return sample["image"].convert("RGB")

    url = sample.get("image_url") or sample.get("url")
    if not url:
        raise IOError("no URL")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        buf = io.BytesIO(resp.read())
    return Image.open(buf).convert("RGB")

def main():
    ds = load_dataset(DATASET, split="train", streaming=True)
    usable = []                          # (image_pil, file_name)

    seen = 0
    for samp in ds:
        if len(usable) >= TARGET:
            break

        seen += 1
        try:
            img = fetch_pil(samp)
        except (urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                IOError) as e:
            # dead link or corrupted file → skip
            if seen % 5000 == 0:
                print(f"skipped {seen} rows so far…")
            continue

        fname = samp.get("image_id") or samp.get("id") or f"{seen:08d}.jpg"
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            fname += ".jpg"

        usable.append((img, fname))

        if len(usable) % 100 == 0:
            print(f"[progress] {len(usable)}/{TARGET} images OK")

    print(f"[✓] gathered {len(usable)} usable images out of {seen} rows")

    # shuffle deterministically, then split
    random.shuffle(usable)
    train = usable[:TRAIN_N]
    val   = usable[TRAIN_N:]

    def dump(split, folder, json_path):
        names = []
        for img, fname in split:
            img.save(folder / fname, quality=90)
            names.append(fname)
        json_path.write_text(json.dumps(names, indent=2))
        print(f"[✓] saved {len(names)} → {folder}")

    dump(train, TR, ROOT / "train_files.json")
    dump(val,   VA, ROOT / "val_files.json")

if __name__ == "__main__":
    main()
