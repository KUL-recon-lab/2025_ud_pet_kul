import pandas as pd
import os
import math
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

################################################################################
################################################################################
################################################################################

# from df_all_data, get all rows where tracer is not FDG nor Fluorodeoxyglucose and recon is ref
df_non_fdg = df_all_data[
    (df_all_data["tracer"] != "FDG") & (df_all_data["tracer"] != "Fluorodeoxyglucose")
]
df_non_fdg = df_non_fdg[df_all_data["recon"] == "ref"]

non_fdg_acqs = df_non_fdg["acq"].values
# write the last 4 cases to val_non_fdg.txt, the rest to train_non_fdg.txt
with open("val_non_fdg.txt", "w") as f:
    for acq in non_fdg_acqs[-4:]:
        f.write(f"{acq}\n")
with open("train_non_fdg.txt", "w") as f:
    for acq in non_fdg_acqs[:-4]:
        f.write(f"{acq}\n")

################################################################################
################################################################################
################################################################################


df_screened = pd.read_csv("mips_screening_cc.csv")
# trim all whitespace values in df_screened.cat_name
df_screened["cat_name"] = df_screened["cat_name"].str.strip()

# count how many cases we have per "cat_name", print a nice summary
# print(df_screened["cat_name"].value_counts())

# loop over all the cat_names
groups_90_10 = {
    "low_contrast_lesion",
    "many_lesion",
    "solid_lesion",
    "unusual_phys_uptake",
}

for group, df_group in df_screened.groupby("cat_name"):
    train_list: list[str] = []
    val_list: list[str] = []
    print(f"Group: {group}, n={len(df_group)}\n")
    # collect PNGs for this group and keep acquisition label
    pngs: list[tuple[Path, str]] = []
    acq_labels = []
    tracers = []

    for row in df_group.itertuples():
        mips_file = mdir / row.acq / "ref" / "resampled_1.65.nii.mip.png"
        acq_label = str(row.acq)
        pngs.append((mips_file, acq_label))
        acq_labels.append(acq_label)
        # get tracer from df_all_data where acq matches row.acq and recon == "ref"
        tracer = df_all_data.loc[
            (df_all_data["acq"] == row.acq) & (df_all_data["recon"] == "ref"), "tracer"
        ].values
        if len(tracer) > 0:
            tracers.append(tracer[0])
        else:
            tracers.append("unknown")

    # for each unique tracer in tracers, print how many times it occurs
    unique, counts = (
        pd.Series(tracers).value_counts().index,
        pd.Series(tracers).value_counts().values,
    )
    print("  Tracer counts:")
    for u, c in zip(unique, counts):
        print(f"    {u}: {c}")

    ############################################################################
    ############################################################################
    ############################################################################

    n = len(acq_labels)
    if group in groups_90_10 and n > 0:
        split = int(math.floor(0.9 * n))
        train_list.extend(acq_labels[:split])
        val_list.extend(acq_labels[split:])
    elif group == "none":
        # first 50 to train, next 10 to val
        train_list.extend(acq_labels[:50])
        val_list.extend(acq_labels[50:60])
    else:
        raise ValueError(f"Group '{group}' not in predefined split groups")

    # write train and val lists to text files
    out_dir = Path(".")
    train_file = out_dir / f"train_{group}.txt"
    val_file = out_dir / f"val_{group}.txt"
    with open(train_file, "w") as f:
        for item in train_list:
            f.write(f"{item}\n")
    with open(val_file, "w") as f:
        for item in val_list:
            f.write(f"{item}\n")

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
