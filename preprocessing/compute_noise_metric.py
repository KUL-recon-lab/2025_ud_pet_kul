import nibabel as nib
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pydicom
from time import strptime
from datetime import timedelta
from data import noise_metric

random.seed(42)

# parse optional chunk argument to run in data-parallel chunks
parser = argparse.ArgumentParser(description="Compute noise metrics over subject dirs")
parser.add_argument(
    "ichunk", type=int, help="which chunk index to process", choices=range(6)
)

args = parser.parse_args()
ichunk = args.ichunk
nchunks = 6

# %%

# validate ichunk
if ichunk is not None:
    if ichunk < 0 or ichunk >= nchunks:
        raise SystemExit(f"--ichunk must be in [0,{nchunks-1}] but got {ichunk}")

# get the user name that is running the script
mdir = (
    "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
)

outfile = Path(".") / f"data_stats_{ichunk}.csv"
if not outfile.exists():
    outfile.touch()

# get all subject dirs using iterdir
s_dirs = sorted([d for d in Path(mdir).iterdir() if d.is_dir()])
# shuffle s_dirs
random.shuffle(s_dirs)

# if running in chunked mode, select only the subset for this chunk
old_n = len(s_dirs)
s_dirs = s_dirs[ichunk::nchunks]
print(
    f"Running chunk {ichunk}/{nchunks}: processing {len(s_dirs)} of {old_n} subject dirs"
)

for i, s_dir in enumerate(s_dirs):
    print(f"{i:03d}/{len(s_dirs)}: {s_dir.name}")
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
            injection_time: str = rads.RadiopharmaceuticalStartTime  # '131313.000000'
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

    for acq in ["100", "50", "20", "10", "4", "ref"]:
        in_file = s_dir / acq / "resampled_1.65.nii.gz"
        noise_metric_file = s_dir / acq / "noise_metric.txt"

        if in_file.exists() and not noise_metric_file.exists():
            vol = nib.load(in_file).get_fdata()
            noise_met = noise_metric(vol)
            # write to noise_metric_file
            with open(noise_metric_file, "w") as f:
                f.write(f"{noise_met:.8f}\n")
            print(acq, f"{noise_met:.4f}")

            # append a line to the outfile containing s_dir.name, weight, height, bmi, radiopharmaceutical, scanner, dose, dt_min, acq, noise_met
            with open(outfile, "a") as f:
                f.write(
                    f"{s_dir.name},{weight:.1f},{height:.2f},{bmi:.1f},{radiopharmaceutical},{scanner},{dose:.2f},{dt_min:.2f},{acq},{noise_met:.8f}\n"
                )

            mip1 = np.max(vol, axis=0).T
            mip2 = np.flip(np.max(vol, axis=1).T, 1)

            fig, ax = plt.subplots(1, 4, figsize=(9, 8), layout="constrained")
            ax[3].imshow(mip1, cmap="Greys", origin="lower", vmin=0, vmax=2)
            ax[2].imshow(mip2, cmap="Greys", origin="lower", vmin=0, vmax=2)
            ax[1].imshow(
                np.exp(mip1) - 1, cmap="Greys", origin="lower", vmin=0, vmax=10
            )
            ax[0].imshow(
                np.exp(mip2) - 1, cmap="Greys", origin="lower", vmin=0, vmax=10
            )

            fig.suptitle(
                f"{i}, {radiopharmaceutical}, {scanner}, {dose}MBq, {dt_min:.1f}min p.i., BMI {bmi:.1f}",
                fontsize="medium",
            )

            fig.savefig(in_file.with_suffix(".mip.png"), dpi=150)
            plt.close(fig)
