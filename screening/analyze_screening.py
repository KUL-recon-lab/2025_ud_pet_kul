import pandas as pd
import numpy as np
import os
from pathlib import Path
from PIL import Image
import re

if os.name == "nt":
    mdir = Path(
        "//uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
    )
else:
    mdir = Path(
        "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
    )


df_all_data = pd.read_csv(Path("../data_stats/") / "data_stats.csv")
df_screened = pd.read_csv("mips_screening_cc.csv")
# trim all whitespace values in df_screened.cat_name
df_screened["cat_name"] = df_screened["cat_name"].str.strip()

# count how many cases we have per "cat_name", print a nice summary
# print(df_screened["cat_name"].value_counts())

# loop over all the cat_names

for group, df_group in df_screened.groupby("cat_name"):
    print(f"Group: {group}, n={len(df_group)}\n")
    # collect PNGs for this group
    pngs: list[Path] = []
    for row in df_group.itertuples():
        mips_file = mdir / row.acq / "ref" / "resampled_1.65.nii.mip.png"
        if mips_file.exists():
            pngs.append(mips_file)
        else:
            # try alternate naming patterns (some files might be .mip.png or .mip.PNG)
            alt = mips_file.with_suffix(".mip.PNG")
            if alt.exists():
                pngs.append(alt)

    if len(pngs) == 0:
        print(f"  No PNGs found for group '{group}', skipping PDF creation.")
        continue

    out_dir = Path("screening_pdfs")
    out_dir.mkdir(exist_ok=True)
    out_pdf = out_dir / f"{group}.pdf"

    # open images and convert to RGB
    pil_images = []
    for p in pngs:
        try:
            im = Image.open(p)
            if im.mode != "RGB":
                im = im.convert("RGB")
            pil_images.append(im)
        except Exception as e:
            print(f"  Warning: could not open {p}: {e}")

    if len(pil_images) == 0:
        print(f"  No valid images for group '{group}' after loading, skipping.")
        continue

    try:
        first, rest = pil_images[0], pil_images[1:]
        first.save(out_pdf, "PDF", resolution=150, save_all=True, append_images=rest)
        print(f"  Wrote PDF: {out_pdf} ({len(pil_images)} pages)")
    except Exception as e:
        print(f"  Error writing PDF for group '{group}': {e}")
