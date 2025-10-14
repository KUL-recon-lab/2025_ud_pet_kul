from pathlib import Path
from data import val_subject_nrmse

model_path = Path("run_20251011_165522/model_0020_scripted.pt")
count_reduction_factor = 10
mdir = Path(
    "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
)

with open("val.txt", "r") as f:
    s_dirs = [mdir / Path(line.strip()) for line in f.readlines()]
s_dir = s_dirs[0]

met = val_subject_nrmse(
    s_dir,
    model_path,
    count_reduction_factor,
    save_path=Path("."),
    patch_size=96,
    patch_overlap=48,
    batch_size=20,
)
print(f"NRMSE: {met:.3f}")
