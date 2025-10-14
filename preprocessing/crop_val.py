import nibabel as nib
from pathlib import Path
from scipy.ndimage import find_objects

mdir = Path(
    "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
)

with open("../val.txt", "r") as f:
    validation_s_dirs = [mdir / Path(line.strip()) for line in f.readlines()]

for s_dir in validation_s_dirs:
    print(s_dir.name)

    ref_nii = nib.load(s_dir / "ref" / "resampled_1.65.nii.gz")
    ref_vol = ref_nii.get_fdata()

    # find the bounding box
    bbox = find_objects((ref_vol > 0.1).astype(int))[0]

    # save ref volume cropped to bbox
    # Adjust the affine to account for cropping
    start = [bbox[i].start for i in range(3)]
    new_affine = ref_nii.affine.copy()
    new_affine[:3, 3] += ref_nii.affine[:3, :3] @ start

    ofile = s_dir / "ref" / "resampled_1.65_cropped.nii"
    if not ofile.exists():
        nib.save(nib.Nifti1Image(ref_vol[bbox], new_affine), ofile)

    for d in ["100", "50", "20", "10", "4"]:
        print(d)
        ofile = s_dir / d / "resampled_1.65_cropped.nii"
        if not ofile.exists():
            dfile = s_dir / d / "resampled_1.65.nii.gz"
            nii = nib.load(dfile)
            vol = nii.get_fdata()
            nib.save(nib.Nifti1Image(vol[bbox], new_affine), ofile)
