import nibabel as nib
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
import getpass
import pydicom
from datetime import datetime
from time import strptime
from datetime import timedelta

random.seed(42)

# get the user name that is running the script
user = getpass.getuser()
print(user)

outfile = Path(".") / f"screening_{user}.txt"
# create file if it does not exist
if not outfile.exists():
    outfile.touch()
# write a line with the current datetime to file
with open(outfile, "a") as f:
    f.write(f"# Screening started: {datetime.now().isoformat()}\n")

mdir = (
    "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
)

# get all subject dirs using iterdir
s_dirs = sorted([d for d in Path(mdir).iterdir() if d.is_dir()])
# shuffle s_dirs
random.shuffle(s_dirs)

categories = [
    "none",
    "many_lesion",
    "solid_lesion",
    "low_contrast_lesion",
    "unusual_phys_uptake",
    "obese",
]
cat_str = "\n".join([f"{i}: {c}" for i, c in enumerate(categories)])

# write categories to file
with open(outfile, "a") as f:
    f.write(f"# Categories: {cat_str.replace("\n",",")}\n")

for i, s_dir in enumerate(s_dirs):
    check_file = s_dir / f".screened_{user}"

    if not check_file.exists():
        print(f"{i:03d}: {s_dir.name}")
        ref_file = s_dir / "ref" / "resampled_1.65.nii.gz"
        dcm = pydicom.dcmread(s_dir / "ref" / "_sample.dcm")
        # get patient weight and height from dicom if exists
        if "PatientWeight" in dcm:
            weight = float(dcm.PatientWeight)
        else:
            weight = -1.0
        if "PatientSize" in dcm:
            height = float(dcm.PatientSize)
        else:
            height = -1.0
        if weight > 0 and height > 0:
            bmi = weight / height**2
        else:
            bmi = -1.0

        # get the radiopharmaceutical from dicom if exists
        if "RadiopharmaceuticalInformationSequence" in dcm:
            rads = dcm.RadiopharmaceuticalInformationSequence[0]
            # get the injection dose from dicom if exists
            if "RadionuclideTotalDose" in rads:
                dose = rads.RadionuclideTotalDose / 1e6
            else:
                dose = -1
            if "Radionuclide" in rads:
                radionuclide = rads.Radionuclide
            else:
                radionuclide = "unknown"
            if "Radiopharmaceutical" in rads:
                radiopharmaceutical = rads.Radiopharmaceutical
            else:
                radiopharmaceutical = "unknown"
            if "RadiopharmaceuticalStartTime" in rads:
                injection_time: str = (
                    rads.RadiopharmaceuticalStartTime
                )  # '131313.000000'
        else:
            dose = -1
            radionuclide = "unknown"
            radiopharmaceutical = "unknown"
            injection_time = "000000.000000"

        if "ManufacturerModelName" in dcm:
            scanner = dcm.ManufacturerModelName
        else:
            scanner = "unknown"

        ref_time: str = dcm.SeriesTime  # '141842.000000'

        t1 = strptime(ref_time.split(".")[0], "%H%M%S")
        t2 = strptime(injection_time.split(".")[0], "%H%M%S")

        # time difference ref_time - injection_time in seconds
        dt_min = (
            timedelta(hours=t1.tm_hour, minutes=t1.tm_min, seconds=t1.tm_sec)
            - timedelta(hours=t2.tm_hour, minutes=t2.tm_min, seconds=t2.tm_sec)
        ).seconds / 60

        vol = nib.load(ref_file).get_fdata()

        mip1 = np.max(vol, axis=0).T
        mip2 = np.flip(np.max(vol, axis=1).T, 1)

        fig, ax = plt.subplots(1, 4, figsize=(9, 8), layout="constrained")
        ax[3].imshow(mip1, cmap="Greys", origin="lower", vmin=0, vmax=2)
        ax[2].imshow(mip2, cmap="Greys", origin="lower", vmin=0, vmax=2)
        ax[1].imshow(np.exp(mip1) - 1, cmap="Greys", origin="lower", vmin=0, vmax=10)
        ax[0].imshow(np.exp(mip2) - 1, cmap="Greys", origin="lower", vmin=0, vmax=10)

        fig.suptitle(
            f"{i}, {radiopharmaceutical}, {scanner}, {dose}MBq, {dt_min:.1f}min p.i., BMI {bmi:.1f}",
            fontsize="medium",
        )

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

        plt.close(fig)

        if cat == -2:
            print("Exiting...")
            break
        else:
            with open(outfile, "a") as f:
                f.write(f"{s_dir.name}, {cat}, {categories[cat]}\n")
            # write a hidden file to indicate that this subject has been processed
            # check_file.touch()
