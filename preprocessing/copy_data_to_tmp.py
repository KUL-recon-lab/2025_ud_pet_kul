import shutil
from pathlib import Path

mdir = Path(
    "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
)

new_mdir = Path("/tmp/2025_ud_pet_challenge/nifti_out")

if not new_mdir.exists():
    new_mdir.mkdir(parents=True, exist_ok=True)

s_dirs = []

# read s_dirs from subjects.txt
with open("../train.txt", "r") as f:
    s_dirs += [line.strip() for line in f.readlines()]

with open("../val.txt", "r") as f:
    s_dirs += [line.strip() for line in f.readlines()]

for i, s_dir in enumerate(s_dirs):
    s_dir_path = mdir / s_dir
    new_s_dir_path = new_mdir / s_dir
    print(f"{(i+1):03d}/{len(s_dirs)}: {s_dir}, {new_s_dir_path}")

    if not new_s_dir_path.exists():
        shutil.copytree(s_dir_path, new_s_dir_path, dirs_exist_ok=True)
