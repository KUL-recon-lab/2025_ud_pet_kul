import json
import os
from pathlib import Path

from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import mm
from PIL import Image

if os.name == "nt":
    mdir = Path(
        "//uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
    )
else:
    mdir = Path(
        "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
    )

cfg_path = Path("../data_stats/config_biograph_fdg.json")
mode = "train"  # "train" or "val"

################################################################################
################################################################################
################################################################################

with open(cfg_path, "r") as f:
    data = json.load(f)

if mode == "train":
    s_dirs = [mdir / x for x in data["training_s_dirs"]]
else:
    s_dirs = [mdir / x for x in data["validation_s_dirs"]]

# collect all .mi.png and .mip.png images under each training subject dir
img_entries = []
for s_dir in s_dirs:
    s_dir = Path(s_dir)
    if not s_dir.exists():
        print(f"Warning: training subject dir not found: {s_dir}")
        continue
    # search for both patterns to be tolerant to filename variants
    for p in sorted((s_dir / "ref").rglob("*.mip.png")):
        img_entries.append((p, s_dir.name))

if not img_entries:
    print("No training images (*.mi.png / *.mip.png) found. Exiting.")
else:
    out_pdf = cfg_path.with_suffix(f".{mode}.pdf")
    c = rl_canvas.Canvas(str(out_pdf), pagesize=A4)
    pw, ph = A4
    margin = 10 * mm
    max_w = pw - 2 * margin
    max_h = ph - 2 * margin

    for i, (img_path, sname) in enumerate(img_entries):
        print(f"Processing image {i + 1}/{len(img_entries)}: {img_path}")
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Could not open image {img_path}: {e}")
            continue

        iw, ih = pil.size
        scale = min(max_w / iw, max_h / ih, 1.0)
        draw_w = iw * scale
        draw_h = ih * scale
        x = (pw - draw_w) / 2
        y = (ph - draw_h) / 2

        img_reader = ImageReader(pil)
        # draw header (subject/acq name) near top
        header_y = ph - (margin / 2)
        c.setFont("Helvetica", 10)
        c.drawCentredString(pw / 2, header_y, str(sname))

        # draw the image centered on the page
        c.drawImage(
            img_reader, x, y, draw_w, draw_h, preserveAspectRatio=True, mask="auto"
        )
        c.showPage()

    c.save()
    print(f"Wrote {out_pdf} with {len(img_entries)} pages.")
