import pandas as pd
import os
from pathlib import Path

from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import mm

write_pdf = False

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
    # collect PNGs for this group and keep acquisition label
    pngs: list[tuple[Path, str]] = []
    for row in df_group.itertuples():
        mips_file = mdir / row.acq / "ref" / "resampled_1.65.nii.mip.png"
        acq_label = str(row.acq)
        pngs.append((mips_file, acq_label))

    ############################################################################
    ############################################################################
    ############################################################################

    if write_pdf:
        if len(pngs) == 0:
            print(f"  No PNGs found for group '{group}', skipping PDF creation.")
            continue

        out_dir = Path("screening_pdfs")
        out_dir.mkdir(exist_ok=True)
        out_pdf = out_dir / f"{group}.pdf"

        try:
            c = rl_canvas.Canvas(str(out_pdf), pagesize=A4)
            pw, ph = A4
            margin = 12 * mm
            header_h = 12 * mm

            for idx, (p, acq_label) in enumerate(pngs):
                try:
                    img = ImageReader(str(p))
                    iw, ih = img.getSize()
                except Exception as e:
                    print(f"  Warning: could not read image {p}: {e}")
                    continue

                avail_w = pw - 2 * margin
                avail_h = ph - 2 * margin - header_h
                scale = min(avail_w / iw, avail_h / ih)
                draw_w = iw * scale
                draw_h = ih * scale

                header_x = pw / 2
                header_y = ph - margin - (header_h / 2)
                img_x = (pw - draw_w) / 2
                img_y = margin

                key = f"{group}_p{idx}"
                c.bookmarkPage(key)
                c.addOutlineEntry(str(acq_label), key, level=0, closed=False)

                c.setFont("Helvetica-Bold", 12)
                c.drawCentredString(header_x, header_y, str(acq_label))

                try:
                    c.drawImage(
                        img,
                        img_x,
                        img_y,
                        width=draw_w,
                        height=draw_h,
                        preserveAspectRatio=True,
                    )
                except Exception as e:
                    print(f"  Warning: could not draw image {p}: {e}")

                c.showPage()

            c.save()
            print(f"  Wrote PDF: {out_pdf} ({len(pngs)} pages)")
        except Exception as e:
            print(f"  Error writing PDF for group '{group}': {e}")
