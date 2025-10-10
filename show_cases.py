import numpy as np
import array_api_compat.torch as torch
import torchio as tio

from pathlib import Path
from data import SUVLogCompress, get_subject_dict
from viewer import ThreeAxisViewer

m_path = Path("/tmp/nifti_out")

s_dirs = sorted([x for x in m_path.iterdir() if x.is_dir()])

subjects_list = [tio.Subject(get_subject_dict(s_dir, crfs=["ref"])) for s_dir in s_dirs]

# setup preprocessing transforms
transform_list = [tio.transforms.ToCanonical(), SUVLogCompress()]

training_subjects_dataset = tio.SubjectsDataset(
    subjects_list, transform=tio.Compose(transform_list)
)


for i, sub in enumerate(training_subjects_dataset):
    print(i, subjects_list[i])

    # compressed_vol = np.flip(sub["ref"].data.squeeze().numpy(), (0, 1))
    # vol = np.exp(compressed_vol) - 1

    compressed_vol = torch.flip(sub["ref"].data.squeeze(), axis=0)
    vol = torch.expm1(compressed_vol)

    v = ThreeAxisViewer(vol, affine=sub["ref"].affine, vmin=0, vmax=7, cmap="Greys")

    breakpoint()
    v.close()
