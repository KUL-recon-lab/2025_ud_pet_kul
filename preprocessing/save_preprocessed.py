import torch
import torchio as tio

from pathlib import Path
from data import get_subject_dict_old, AddSamplingMap, SUVLogCompress

target_voxsize_mm = 1.65


m_path = Path(
    "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
)

s_dirs = sorted([x for x in m_path.iterdir() if x.is_dir()])
img_keys = ["4", "10", "20", "50", "100", "ref"]

subjects_list = [
    tio.Subject(get_subject_dict_old(s_dir, crfs=img_keys)) for s_dir in s_dirs
]

# setup preprocessing transforms
transform_list = [tio.transforms.ToCanonical()]
transform_list += [tio.Resample(target=target_voxsize_mm)]
transform_list += [SUVLogCompress(), AddSamplingMap()]

training_subjects_dataset = tio.SubjectsDataset(
    subjects_list, transform=tio.Compose(transform_list)
)

for i, sub in enumerate(training_subjects_dataset):
    print(sub["s_dir"])
    try:
        for img_key in sub["crfs"]:
            img = sub[img_key]
            assert img.data.dtype == torch.float32
            new_file = (
                sub[f"{img_key}_file"].parent
                / f"resampled_{target_voxsize_mm:.2f}.nii.gz"
            )
            print(sub[f"{img_key}_file"], "->", new_file)
            img.save(new_file)

        new_sample_file = sub["s_dir"] / f"sampling_map_{target_voxsize_mm:.2f}.nii.gz"
        sub["sampling_map"].save(new_sample_file)

        suv_file = sub["s_dir"] / f"suv_factor.txt"
        with open(suv_file, "w") as f:
            f.write(f"{sub['suv_fac']}\n")

    except Exception as e:
        print(f"Could not process {sub['s_dir']}: {e}")
        # write sub['s_dir'] to a file for later inspection
        with open("failed_preprocessing.txt", "a") as f:
            f.write(str(sub["s_dir"]) + "\n")
        continue
