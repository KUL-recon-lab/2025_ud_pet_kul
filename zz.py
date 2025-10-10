import nibabel as nib
import random
import argparse
import json
import torch
import torchio as tio

from torchmetrics.image import PeakSignalNoiseRatio
from pathlib import Path
from data import (
    SUVLogCompress,
    get_subject_dict,
    AddSamplingMap,
)
from models import UNet3D
from datetime import datetime

import pydicom

def hack(s_dir: Path, crfs: list[str], **kwargs):
    dcm_file = s_dir / "ref" / "_sample.dcm"
    #suv_fac = get_suv_factor_from_dicom(dcm_file, **kwargs)
    suv_fac = 1.0

    subject_dict = {}
    subject_dict["suv_fac"] = suv_fac
    subject_dict["crfs"] = crfs

    for d in crfs:
        dfile = s_dir / f"{d}" / "zz.nii"
        subject_dict[f"{d}"] = tio.ScalarImage(dfile)
        subject_dict[f"dfile_{d}"] = dfile

    return subject_dict

parser = argparse.ArgumentParser(description="Train 3D UNet on PET data")
parser.add_argument(
    "--count_reduction_factor", type=int, default=10, help="Count reduction factor"
)
parser.add_argument("--patch_size", type=int, default=64, help="Patch size")
parser.add_argument("--queue_length", type=int, default=400, help="Queue length")
parser.add_argument(
    "--samples_per_volume", type=int, default=200, help="Samples per volume"
)
parser.add_argument("--n_sub", type=int, default=100, help="Number of subjects")
parser.add_argument("--batch_size", type=int, default=15, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
parser.add_argument(
    "--target_voxsize_mm",
    type=float,
    default=1.65,
    help="Target voxel size (mm), set to None for no resampling",
)

args = parser.parse_args()

count_reduction_factor = args.count_reduction_factor
patch_size = args.patch_size
queue_length = args.queue_length
samples_per_volume = args.samples_per_volume
n_sub = args.n_sub
batch_size = args.batch_size
lr = args.lr
num_epochs = args.num_epochs
target_voxsize_mm = args.target_voxsize_mm


model_kwargs = dict(in_channels=1, out_channels=1)


# %%
normalized_data_range = 3.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#### HACK
sub_dirs = [Path("/tmp/nifti_out/Anonymous_ANO_20230525_2002263_090039")]
subjects_list = [
    #tio.Subject(get_subject_dict(s_dir, crfs=[str(count_reduction_factor), "ref"]))
    tio.Subject(dict(ref=tio.ScalarImage(torch.asarray(nib.load(str(s_dir / "ref" / "zz.nii")).get_fdata()))))
    for s_dir in sub_dirs
]
### END HACK

# setup preprocessing transforms
transform_list = []
#transform_list = [tio.transforms.ToCanonical()]
#if target_voxsize_mm is not None:
#    transform_list += [tio.Resample(target=target_voxsize_mm)]
#transform_list += [SUVLogCompress(), AddSamplingMap()]

training_subjects_dataset = tio.SubjectsDataset(
    subjects_list, transform=tio.Compose(transform_list)
)
