import nibabel as nib
import numpy as np
from pathlib import Path

with open("acquisitions.txt", "r") as f:
    s_dirs = [Path(line.strip()) for line in f.readlines()]

for i, s_dir in enumerate(s_dirs):
    samp_file = s_dir / "sampling_map_1.65.nii.gz"
    new_samp_file = s_dir / "new_sampling_map_1.65.nii.gz"
    ref_file = s_dir / "ref" / "resampled_1.65.nii.gz"

    if not new_samp_file.exists() and ref_file.exists():
        print(f"{i} {s_dir}")
        ref_nii = nib.load(ref_file)
        ref = ref_nii.get_fdata()
        weights = 3 * ((ref > 0.1) + 2 * (ref > np.log(5 + 1)))
        weights[..., :64] = 1  # ensure sampling of empty patches
        weights = weights.astype(np.int8)

        # save weights to new file
        new_samp_nii = nib.Nifti1Image(weights, affine=ref_nii.affine)
        nib.save(new_samp_nii, new_samp_file)
        print(f"Saved new sampling map to {new_samp_file}")
