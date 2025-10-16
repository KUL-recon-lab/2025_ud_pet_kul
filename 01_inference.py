import torch
import torchio as tio

from pathlib import Path
from data import SUVLogCompress, get_suv_factor_from_dicom, patch_inference

model_path = Path("denoising_models/crf_10-20_20251015_122023/model_0040_scripted.pt")
input_path = (
    Path(
        "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
    )
    / "Anonymous_ANO_20220223_1753176_134618"
    / "10"
)
input_nifti_file = input_path / "307_Body_PET_20211217134618_307.nii.gz"

################################################################################

# to be replaced by predictor
count_reduction_factor = 10

target_voxsize_mm: float = 1.65

resample_transform = tio.Resample(target=target_voxsize_mm)
intensity_norm_transform = SUVLogCompress()
inverse_intensity_norm_transform = intensity_norm_transform.inverse()
canonical_transform = tio.transforms.ToCanonical()

# get the SUV factor from the text file
suv_fac = get_suv_factor_from_dicom(input_path / "_sample.dcm")

subject = canonical_transform(
    tio.Subject({"input": tio.ScalarImage(input_nifti_file), "suv_fac": suv_fac})
)

subject_resampled = resample_transform(subject)

subject_resampled_normalized = intensity_norm_transform(subject_resampled)

####
####
output_tensor = patch_inference(
    subject_resampled_normalized,
    scripted_model_path=model_path,
    patch_size=96,
    patch_overlap=48,
    batch_size=20,
    verbose=True,
).cpu()
####
####

subject_resampled_normalized["output"] = tio.ScalarImage(
    tensor=output_tensor, affine=subject_resampled_normalized["input"].affine
)

# undo SUV compression
out_subject_resampled = inverse_intensity_norm_transform(subject_resampled_normalized)

# undo resampling
subject["output"] = tio.Resample(subject["input"])(out_subject_resampled["output"])
