import shutil
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

# replace hard-coded vars with argparse (use parse_known_args so importing in notebooks/IDE won't fail)
parser = argparse.ArgumentParser(description="Run test predictions")
parser.add_argument(
    "--sid",
    type=int,
    default=-1,
    help="Subject ID number (matches Anonymous-<sid>)",
    choices=list(range(1, 51)),
)
parser.add_argument(
    "--df",
    type=int,
    default=-1,
    help="DoseReductionFactor",
    choices=[4, 10, 20, 50, 100],
)
parser.add_argument(
    "--nomips",
    action="store_true",
    help="Whether to skip MIP writing",
)
parser.add_argument(
    "--patch_size",
    type=int,
    default=96,
    help="Patch size for inference",
)
parser.add_argument(
    "--patch_overlap",
    type=int,
    default=48,
    help="Patch overlap for inference",
)
parser.add_argument(
    "--batch_size", type=int, default=20, help="Batch size for inference"
)
parser.add_argument(
    "--odir", type=str, default=None, help="sub directory for output predictions"
)
parser.add_argument(
    "--input_dir",
    type=str,
    default="/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/TestData",
    help="Input directory containing test NIfTI files",
)

args = parser.parse_args()
sid: int = args.sid
df: int = args.df
show: bool = not args.nomips
patch_size: int = args.patch_size
patch_overlap: int = args.patch_overlap
batch_size: int = args.batch_size
odir: str | None = args.odir
test_input_dir: Path = Path(args.input_dir)

if odir is None:
    out_dir = test_input_dir / "predictions_0"
    i = 1
    while out_dir.exists():
        out_dir = test_input_dir / f"predictions_{i}"
        i += 1

out_dir.mkdir(parents=True, exist_ok=True)
print(f"Writing predictions to {out_dir}")

model_cfg_path = Path(".") / "inference_model_config.json"
challenge_csv_path = Path(".") / "PET_info_noNORMAL.csv"

target_voxsize_mm: float = 1.65

# copy model_cfg_path to out_dir for reference
shutil.copy(model_cfg_path, out_dir / "inference_model_config.json")
shutil.copy(challenge_csv_path, out_dir / "test_data_info.csv")

################################################################################
################################################################################
# (1) read the test data set csv and create the SUV factor (factor that converts from Bq/ml to SUV)
################################################################################
################################################################################

input_df = pd.read_csv(challenge_csv_path)

# filter to only SubjectID and DoseReductionFactor if specified
if sid != -1:
    input_df = input_df[input_df[f"SubjectID"] == f"Anonymous-{sid:02}"]
if df != -1:
    input_df = input_df[input_df["DoseReductionFactor"] == df]

# calculate the injected dose decay corrected to the scan start time
input_df["A"] = (
    0.5
    ** (
        input_df["TimeFromInjectionToAcquisition_s"]
        / input_df["RadionuclideHalfLife_s"]
    )
) * input_df["RadionuclideTotalDose_Bq"]

input_df["suv_fac"] = 1000 * input_df["PatientWeight_kg"] / input_df["A"]

################################################################################
################################################################################
# (2) create a model dictionary mapping (manufacturer, drf) -> model_path
################################################################################
################################################################################

# load models from inference_model_config.json
if not model_cfg_path.exists():
    raise FileNotFoundError(f"Missing config: {model_cfg_path}")
with model_cfg_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

base_dir = cfg.get("model_mdir")
if base_dir:
    base_dir = Path(base_dir)
    if not base_dir.is_absolute():
        base_dir = (Path(__file__).parent / base_dir).resolve()
else:
    base_dir = Path(__file__).parent

model_dict = {}
for entry in cfg.get("models", []):
    manuf = entry["manufacturer"]
    drf = int(entry["drf"])
    model_dir = Path(entry["model_dir"])
    if not model_dir.is_absolute():
        model_dir = (base_dir / model_dir).resolve()

    checkpoint = entry.get("checkpoint")
    if checkpoint:
        cp = Path(checkpoint)
        if not cp.is_absolute():
            cp = (model_dir / cp).resolve()
        model_path = cp
    else:
        val_loss_file = model_dir / "val_loss.csv"
        if val_loss_file.exists():
            val_loss_data = np.loadtxt(val_loss_file, delimiter=",")
            epochs = val_loss_data[:, 0].astype(int)
            val_loss = val_loss_data[:, 1]
            best_row = int(np.nanargmin(val_loss)) + 1
            best_epoch = int(epochs[np.nanargmin(val_loss)])

            if best_row != best_epoch:
                raise ValueError(
                    f"Epoch mismatch in {val_loss_file}: row {best_row} vs epoch {best_epoch}"
                )
            model_path = (model_dir / f"model_{best_epoch:04}_scripted.pt").resolve()
        else:
            # choose best epoch by validation NRMSE from train_metrics.csv
            metrics_path = model_dir / "train_metrics.csv"
            if not metrics_path.exists():
                raise FileNotFoundError(
                    f"Missing metrics at {metrics_path} for {manuf} DRF {drf}"
                )
            metrics = np.loadtxt(metrics_path, delimiter=",")
            # assume val_nrmse is the second-to-last column
            val_nrmse = metrics[:, -2]
            epochs = metrics[:, 0].astype(int)
            best_row = int(np.nanargmin(val_nrmse)) + 1
            best_epoch = int(epochs[np.nanargmin(val_nrmse)])

            if best_row != best_epoch:
                raise ValueError(
                    f"Epoch mismatch in {metrics_path}: row {best_row} vs epoch {best_epoch}"
                )

            model_path = (model_dir / f"model_{best_epoch:04}_scripted.pt").resolve()

    model_dict[(manuf, drf)] = model_path
    print("configured model:", (manuf, drf), "->", model_path)

# save model_dict to out_dir for reference
with (out_dir / "used_models.json").open("w", encoding="utf-8") as f:
    json.dump(
        {f"{k[0]}_DRF{k[1]:03}": str(v) for k, v in model_dict.items()}, f, indent=4
    )

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
for i, row in input_df.iterrows():
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

    model_path = model_dict[(row["ManufacturerModelName"], row["DoseReductionFactor"])]

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
    subject["output"] = tio.Resample(subject["input"])(out_subject_resampled["output"])

    # write the output to nifti
    nii = nib.Nifti1Image(
        subject["output"].data.numpy().squeeze(), subject["output"].affine
    )
    nii.header["descrip"] = str(Path(model_path.parent.name[11:]) / model_path.name)[
        :80
    ]
    nii_file_path = out_dir / row["NiftiFileName"]
    nib.save(nii, str(nii_file_path))
    print(f"Wrote prediction to {nii_file_path}")

    ###
    if show:
        mip_out_dir = out_dir / f"{row['SubjectID']}"

        if not mip_out_dir.exists():
            mip_out_dir.mkdir(parents=True, exist_ok=True)

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
            mip_out_dir
            / f"{row['SubjectID']}_{row['DoseReductionFactor']:03}.mip1.png",
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
            mip_out_dir
            / f"logcompressed_{row['SubjectID']}_{row['DoseReductionFactor']:03}.mip2.png",
            dpi=100,
        )
        fig2.show()
        plt.close(fig2)
