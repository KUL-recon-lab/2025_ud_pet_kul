import torchio as tio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import nibabel as nib

from scipy.ndimage import gaussian_filter
from pathlib import Path
from data import SUVLogCompress, patch_inference
import matplotlib as mpl

mpl.rcParams.update(
    {
        "savefig.transparent": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

target_voxsize_mm = 1.65

################################################################################
################################################################################
# (3) setup the resampling and intensity transform we need (data preprocessing)
################################################################################
################################################################################

resample_transform = tio.Resample(target=target_voxsize_mm)
intensity_norm_transform = SUVLogCompress()
inverse_intensity_norm_transform = intensity_norm_transform.inverse()
canonical_transform = tio.transforms.ToCanonical()

################################################################################
################################################################################
# (4) iterate over all test data sets and do the prediction
################################################################################
################################################################################

isub = 0
drf = 4

cfg_path = Path("inference_model_config.json")
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

model_dict = {}
for entry in cfg.get("models", []):
    model_dict[(entry["manufacturer"], int(entry["drf"]))] = (
        Path(cfg["model_mdir"]) / entry["model_dir"]
    )

model_dir = model_dict[("uEXPLORER", drf)]

with open("config_uexp_fdg.json", "r") as f:
    config = json.load(f)

val_dir = Path(config["mdir"]) / config["validation_s_dirs"][isub]

input_path = list((val_dir / str(drf)).glob("???_*.nii.gz"))[0]
ref_path = list((input_path.parent.parent / "ref").glob("???_*.nii.gz"))[0]

with open(input_path.parent.parent / "suv_factor.txt", "r") as f:
    suv_fac = float(f.read().strip())

subject = canonical_transform(
    tio.Subject(
        {
            "input": tio.ScalarImage(input_path),
            "ref": tio.ScalarImage(ref_path),
            "suv_fac": suv_fac,
        }
    )
)

subject_resampled = resample_transform(subject)
subject_resampled_normalized = intensity_norm_transform(subject_resampled)

patch_size = 96
patch_overlap = 48
batch_size = 20

mip_out_dir = Path("tmp_mips")
if not mip_out_dir.exists():
    mip_out_dir.mkdir(parents=True, exist_ok=True)

sm_fwhm_mm = 7.0
asp = subject["input"].spacing[2] / subject["input"].spacing[0]

inp = suv_fac * subject["input"].data.numpy().squeeze()
inp_sm = gaussian_filter(
    inp, sigma=sm_fwhm_mm / (2.25 * np.array(subject["input"].spacing))
)

ref = suv_fac * subject["ref"].data.numpy().squeeze()

mip1 = np.max(inp, axis=0).T
mip2 = np.flip(np.max(inp, axis=1).T, 1)
mip1b = np.max(inp_sm, axis=0).T
mip2b = np.flip(np.max(inp_sm, axis=1).T, 1)

mipa = np.max(ref, axis=0).T
mipb = np.flip(np.max(ref, axis=1).T, 1)

do_pred = True

for i in range(1, 31):
    ofile = (
        mip_out_dir / f"sub{isub:02}_drf{drf:03}_{model_dir.stem[11:-16]}_{i:03}.png"
    )
    if do_pred and not ofile.exists():
        ckpt = f"model_{i:04}_scripted.pt"
        model_path = model_dir / ckpt
        print(model_path)
        output_tensor = patch_inference(
            subject_resampled_normalized,
            scripted_model_path=model_path,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            batch_size=batch_size,
            verbose=True,
        ).cpu()
        subject_resampled_normalized["output"] = tio.ScalarImage(
            tensor=output_tensor, affine=subject_resampled_normalized["input"].affine
        )

        # undo SUV compression
        out_subject_resampled = inverse_intensity_norm_transform(
            subject_resampled_normalized
        )

        # undo resampling
        subject["output"] = tio.Resample(subject["input"])(
            out_subject_resampled["output"]
        )
        pred = suv_fac * subject["output"].data.numpy().squeeze()

        mip3 = np.max(pred, axis=0).T
        mip4 = np.flip(np.max(pred, axis=1).T, 1)

        ims1 = dict(vmin=0, vmax=10, origin="lower", aspect=asp, cmap="Greys")
        fig, ax = plt.subplots(1, 8, figsize=(26, 12), layout="constrained")
        ax[0].imshow(mipa, **ims1)
        ax[1].imshow(mipb, **ims1)
        ax[2].imshow(mip1, **ims1)
        ax[3].imshow(mip2, **ims1)
        ax[4].imshow(mip1b, **ims1)
        ax[5].imshow(mip2b, **ims1)
        if do_pred:
            ax[6].imshow(mip3, **ims1)
            ax[7].imshow(mip4, **ims1)
        ax[0].set_title("full dose", fontsize="medium")
        ax[2].set_title(f"DRF {drf}", fontsize="medium")
        ax[4].set_title(f"DRF {drf} {sm_fwhm_mm}mm smoothed", fontsize="medium")
        ax[6].set_title(f"{model_dir.stem[11:-16]}", fontsize="medium")
        ax[7].set_title(f"{ckpt}")
        fig.show()
        fig.savefig(
            ofile,
            dpi=150,
        )
        plt.close(fig)
