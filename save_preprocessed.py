import argparse
import torchio as tio

from pathlib import Path
from data import SUVLogCompress, get_subject_dict, AddSamplingMap, psnr

target_voxsize_mm = 1.65


m_path = Path("/tmp/nifti_out")

# HACK only first 2 images
s_dirs = sorted([x for x in m_path.iterdir() if x.is_dir()])

img_keys = ["10", "100", "ref"]

subjects_list = [
    tio.Subject(get_subject_dict(s_dir, crfs=img_keys)) for s_dir in s_dirs
]

# setup preprocessing transforms
transform_list = [tio.transforms.ToCanonical()]
transform_list += [tio.Resample(target=target_voxsize_mm)]
transform_list += [AddSamplingMap()]

training_subjects_dataset = tio.SubjectsDataset(
    subjects_list, transform=tio.Compose(transform_list)
)

for i, sub in enumerate(training_subjects_dataset):
    for img_key in img_keys:
        img = sub[img_key]
        new_file = (
            sub[f"{img_key}_file"].parent / f"resampled_{target_voxsize_mm:.2f}.nii.gz"
        )
        print(sub[f"{img_key}_file"], "->", new_file)
        img.save(new_file)

    new_sample_file = sub["s_dir"] / f"sampling_map_{target_voxsize_mm:.2f}.nii.gz"
    sub["sampling_map"].save(new_sample_file)
