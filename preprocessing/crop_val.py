import nibabel as nib
import numpy as np
import json
from pathlib import Path
from scipy.ndimage import find_objects

with open("../data_stats/config_uexp_nonfdg.json", "r") as f:
    data = json.load(f)

validation_s_dirs = [Path(data["mdir"]) / x for x in data["validation_s_dirs"]]

for s_dir in validation_s_dirs:
    orfile = s_dir / "ref" / "resampled_1.65_cropped.nii.gz"

    if not orfile.exists():
        print(s_dir.name)
        ref_nii = nib.load(s_dir / "ref" / "resampled_1.65.nii.gz")
        ref_vol = ref_nii.get_fdata().astype(np.float32)

        # find the bounding box
        bbox = find_objects((ref_vol > 0.1).astype(int))[0]

        # save ref volume cropped to bbox
        # Adjust the affine to account for cropping
        start = [bbox[i].start for i in range(3)]
        new_affine = ref_nii.affine.copy()
        new_affine[:3, 3] += ref_nii.affine[:3, :3] @ start
        new_affine = new_affine.astype(np.float32)

        nib.save(nib.Nifti1Image(ref_vol[bbox], new_affine), orfile)

        for d in ["100", "50", "20", "10", "4"]:
            print(d)
            odfile = s_dir / d / "resampled_1.65_cropped.nii.gz"
            if not odfile.exists():
                dfile = s_dir / d / "resampled_1.65.nii.gz"
                nii = nib.load(dfile)
                vol = nii.get_fdata().astype(np.float32)
                nib.save(nib.Nifti1Image(vol[bbox], new_affine), odfile)
