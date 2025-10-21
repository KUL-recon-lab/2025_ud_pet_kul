import torch
import torchio as tio
import pandas as pd
import pymirc.viewer as pv
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter
from pathlib import Path
from data import SUVLogCompress, patch_inference

import matplotlib as mpl

mpl.rcParams.update(
    {
        "savefig.transparent": False,  # don’t write alpha
        "savefig.facecolor": "white",  # solid background for saved figures
        "figure.facecolor": "white",  # UI/background while drawing
        "axes.facecolor": "white",
    }
)

#############################
# TO BE FINALIZED
model_mdir = Path("denoising_models")
model_dict = {}
model_dict[("uEXPLORER", 4)] = (
    # model_mdir / "uexp_fdg" / "4" / "run_config_uexp_fdg_4_RobustL1_16_20251018_182651"
    model_mdir
    / "uexp_fdg"
    / "4"
    / "run_config_uexp_fdg_4_RobustL1_32_20251020_213648"
)
model_dict[("uEXPLORER", 10)] = (
    model_mdir
    / "uexp_fdg"
    / "10"
    / "run_config_uexp_fdg_10_RobustL1_32_20251020_213652"
    # / "run_config_uexp_fdg_10_RobustL1_16_20251018_102329"
)
model_dict[("uEXPLORER", 20)] = (
    model_mdir
    / "uexp_fdg"
    / "20"
    / "run_config_uexp_fdg_20_RobustL1_32_20251020_213648"
    # / "run_config_uexp_fdg_20_RobustL1_16_20251018_102328"
)
model_dict[("uEXPLORER", 50)] = (
    model_mdir
    / "uexp_fdg"
    / "50"
    / "run_config_uexp_fdg_50_RobustL1_32_20251020_213649"
    # / "run_config_uexp_fdg_50_RobustL1_16_20251018_182638"
)
model_dict[("uEXPLORER", 100)] = (
    model_mdir
    / "uexp_fdg"
    / "100"
    / "run_config_uexp_fdg_100_RobustL1_32_20251020_220254"
    # / "run_config_uexp_fdg_100_RobustL1_16_20251018_182635"
)


model_dict[("Biograph128_Vision Quadra Edge", 4)] = (
    model_mdir
    / "biograph_fdg"
    / "4"
    / "run_config_biograph_fdg_4_RobustL1_16_20251018_134655"
)
model_dict[("Biograph128_Vision Quadra Edge", 10)] = (
    model_mdir
    / "biograph_fdg"
    / "10"
    / "run_config_biograph_fdg_10_RobustL1_16_20251018_134657"
)
model_dict[("Biograph128_Vision Quadra Edge", 20)] = (
    model_mdir
    / "biograph_fdg"
    / "20"
    / "run_config_biograph_fdg_20_RobustL1_16_20251018_134642"
)
model_dict[("Biograph128_Vision Quadra Edge", 50)] = (
    model_mdir
    / "biograph_fdg"
    / "50"
    / "run_config_biograph_fdg_50_RobustL1_16_20251018_224024"
)
model_dict[("Biograph128_Vision Quadra Edge", 100)] = (
    model_mdir
    / "biograph_fdg"
    / "100"
    / "run_config_biograph_fdg_100_RobustL1_16_20251018_224023"
)

# loop over model_dict and load scripted models
for key, model_dir in model_dict.items():
    # load training metrics from model_dir / "train_metrics.csv"
    train_mets = np.loadtxt(model_dir / "train_metrics.csv", delimiter=",")
    val_nrmse = train_mets[:, -2]
    best_epoch = np.argmin(val_nrmse) + 1
    scripted_model_path = model_dir / f"model_{best_epoch:04}_scripted.pt"
    model_dict[key] = scripted_model_path
    print(key, scripted_model_path, val_nrmse[best_epoch - 1])

#############################

target_voxsize_mm: float = 1.65

test_input_dir = Path(
    "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/TestData"
)

input_df = pd.read_csv(test_input_dir / "PET_info_noNORMAL.csv")

# calculate the injected dose decay corrected to the scan start time
input_df["A"] = (
    0.5
    ** (
        input_df["TimeFromInjectionToAcquisition_s"]
        / input_df["RadionuclideHalfLife_s"]
    )
) * input_df["RadionuclideTotalDose_Bq"]

input_df["suv_fac"] = 1000 * input_df["PatientWeight_kg"] / input_df["A"]

show = True
################################################################################


resample_transform = tio.Resample(target=target_voxsize_mm)
intensity_norm_transform = SUVLogCompress()
inverse_intensity_norm_transform = intensity_norm_transform.inverse()
canonical_transform = tio.transforms.ToCanonical()

# iterate over all rows in input_df
for i, row in input_df.iterrows():
    out_dir = (
        Path(
            "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/TestData/pred"
        )
        / f"{row['SubjectID']}"
    )
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    print(row)
    subject = canonical_transform(
        tio.Subject(
            {
                "input": tio.ScalarImage(test_input_dir / row["NiftiFileName"]),
                "suv_fac": row["suv_fac"],
            }
        )
    )

    subject_resampled = resample_transform(subject)
    subject_resampled_normalized = intensity_norm_transform(subject_resampled)

    output_tensor = patch_inference(
        subject_resampled_normalized,
        scripted_model_path=model_dict[
            (row["ManufacturerModelName"], row["DoseReductionFactor"])
        ],
        patch_size=96,
        patch_overlap=48,
        batch_size=20,
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
    subject["output"] = tio.Resample(subject["input"])(out_subject_resampled["output"])

    ###
    if show:
        sm_fwhm_mm = 7.0
        asp = subject["input"].spacing[2] / subject["input"].spacing[0]

        inp = row["suv_fac"] * subject["input"].data.numpy().squeeze()
        inp_sm = gaussian_filter(
            inp, sigma=sm_fwhm_mm / (2.25 * np.array(subject["input"].spacing))
        )
        pred = row["suv_fac"] * subject["output"].data.numpy().squeeze()

        inp_norm = subject_resampled_normalized["input"].data.numpy().squeeze()
        pred_norm = subject_resampled_normalized["output"].data.numpy().squeeze()

        mip1 = np.max(inp, axis=0).T
        mip2 = np.flip(np.max(inp, axis=1).T, 1)
        mip1b = np.max(inp_sm, axis=0).T
        mip2b = np.flip(np.max(inp_sm, axis=1).T, 1)
        mip3 = np.max(pred, axis=0).T
        mip4 = np.flip(np.max(pred, axis=1).T, 1)

        ims1 = dict(vmin=0, vmax=7, origin="lower", aspect=asp, cmap="Greys")
        fig, ax = plt.subplots(1, 6, figsize=(22, 12), layout="constrained")
        ax[0].imshow(mip1, **ims1)
        ax[1].imshow(mip2, **ims1)
        ax[2].imshow(mip1b, **ims1)
        ax[3].imshow(mip2b, **ims1)
        ax[4].imshow(mip3, **ims1)
        ax[5].imshow(mip4, **ims1)
        ax[1].set_title(f"{row['SubjectID']} DRF {row['DoseReductionFactor']:03}")
        ax[0].set_title("Input")
        ax[2].set_title(f"Input {sm_fwhm_mm}mm smoothed")
        ax[4].set_title("AI denoised")
        fig.savefig(
            out_dir / f"{row['SubjectID']}_{row['DoseReductionFactor']:03}.mip1.png",
            dpi=100,
        )
        fig.show()
        plt.close(fig)

        mip5 = np.max(inp_norm, axis=0).T
        mip6 = np.flip(np.max(inp_norm, axis=1).T, 1)
        mip7 = np.max(pred_norm, axis=0).T
        mip8 = np.flip(np.max(pred_norm, axis=1).T, 1)

        ims2 = dict(vmin=0, vmax=3.5, origin="lower", cmap="Greys")
        fig2, ax2 = plt.subplots(1, 4, figsize=(16, 12), layout="constrained")
        ax2[0].imshow(mip5, **ims2)
        ax2[1].imshow(mip6, **ims2)
        ax2[2].imshow(mip7, **ims2)
        ax2[3].imshow(mip8, **ims2)

        ax2[1].set_title(f"{row['SubjectID']} DRF {row['DoseReductionFactor']:03}")
        ax2[0].set_title("Input (lognorm)")
        ax2[2].set_title("AI denoised (lognorm)")

        fig2.savefig(
            out_dir / f"zz{row['SubjectID']}_{row['DoseReductionFactor']:03}.mip2.png",
            dpi=100,
        )
        fig2.show()
        plt.close(fig2)

        # vi1 = pv.ThreeAxisViewer(inp, imshow_kwargs=dict(vmin=0, vmax=7))
        # vi2 = pv.ThreeAxisViewer(pred, imshow_kwargs=dict(vmin=0, vmax=7))
        # vi2 = pv.ThreeAxisViewer(
        #    pred - inp, imshow_kwargs=dict(vmin=-2, vmax=2, cmap="seismic")
        # )

        # tmp = input("press enter to continue")
