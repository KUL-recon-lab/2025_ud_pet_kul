import torchio as tio
import torch
from pathlib import Path

from data import get_subject_dict, SUVLogCompress

count_reduction_factor = 10
patch_size = 64
patch_overlap = 32
batch_size = 20
target_voxsize_mm = 1.65

s_dirs = sorted(
    [
        x
        for x in Path(
            "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
        ).iterdir()
        if x.is_dir()
    ]
)

s_dir = s_dirs[421]

# %%
transform_list = [tio.transforms.ToCanonical()]
if target_voxsize_mm is not None:
    transform_list += [tio.Resample(target=target_voxsize_mm)]
transform_list += [SUVLogCompress()]
transform = tio.Compose(transform_list)

subject = transform(tio.Subject(get_subject_dict(s_dir, crfs=[count_reduction_factor])))

grid_sampler = tio.inference.GridSampler(
    subject,
    patch_size,
    patch_overlap,
)

patch_loader = tio.SubjectsLoader(grid_sampler, batch_size=batch_size)
aggregator = tio.inference.GridAggregator(grid_sampler)

# load the model from torch script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("run3/unet3d_epoch0020_scripted.pt", map_location=device)

model.eval()
with torch.no_grad():
    for patches_batch in patch_loader:
        input_tensor = patches_batch[f"crf{count_reduction_factor}"][tio.DATA].to(
            device
        )
        locations = patches_batch[tio.LOCATION].to(device)
        outputs = model(input_tensor)
        aggregator.add_batch(outputs, locations)

output_tensor = aggregator.get_output_tensor()

# %%
import numpy as np
import pymirc.viewer as pv

o = output_tensor.cpu().numpy().squeeze()
ref = subject["ref"].numpy().squeeze()
i = subject["crf10"].numpy().squeeze()

o2 = np.exp(o) - 1
i2 = np.exp(i) - 1
ref2 = np.exp(ref) - 1

vi = pv.ThreeAxisViewer([i2, o2, ref2], imshow_kwargs=dict(vmin=0, vmax=5))
