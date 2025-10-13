import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
import getpass
import pandas as pd
from PIL import Image
from datetime import datetime
from time import strptime
import os

random.seed(42)

# get the user name that is running the script
user = getpass.getuser()
print(user)

outfile = Path(".") / f"mips_screening_{user}.txt"
# create file if it does not exist
if not outfile.exists():
    outfile.touch()
# write a line with the current datetime to file
with open(outfile, "a") as f:
    f.write(f"# Screening started: {datetime.now().isoformat()}\n")

# read data_stats.csv
df = pd.read_csv("data_stats.csv")

if os.name == "nt":
    mdir = Path(
        "//uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
    )
else:
    mdir = Path(
        "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
    )

# get all subject dirs using iterdir
s_dirs = [mdir / x for x in np.unique(df.acq.values)]
# shuffle s_dirs
random.shuffle(s_dirs)

categories = [
    "none",
    "many_lesion",
    "solid_lesion",
    "low_contrast_lesion",
    "unusual_phys_uptake",
]
cat_str = "\n".join([f"{i}: {c}" for i, c in enumerate(categories)])

# write categories to file
with open(outfile, "a") as f:
    f.write(f"# Categories: {cat_str.replace("\n",",")}\n")

for i, s_dir in enumerate(s_dirs):
    check_file = s_dir / f"mips_screened_{user}"
    mip_file = s_dir / "ref" / "resampled_1.65.nii.mip.png"

    if mip_file.exists() and not check_file.exists():
        print(f"{i:03d}: {s_dir.name}")

        img = Image.open(mip_file)

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(img)
        fig.show()

        # have the user input a category using a number from 0, 1, 3, ...
        cat = -1

        while cat < 0 or cat >= len(categories):
            input_str = input(cat_str + "\nq -> exit\n")
            if input_str.lower() in ["q"]:
                cat = -2
                break
            try:
                cat = int(input_str)
            except:
                cat = -1

        # plt.close(fig)

        if cat == -2:
            print("Exiting...")
            break
        else:
            with open(outfile, "a") as f:
                f.write(f"{s_dir.name}, {cat}, {categories[cat]}\n")
            # write a hidden file to indicate that this subject has been processed
            check_file.touch()
