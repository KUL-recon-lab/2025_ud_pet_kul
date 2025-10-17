import json

from pathlib import Path

# -------------------------------------------------------------------------------

cfg_path = Path("config_biograph_fdg.json")
# open config file containing, mdir, training_s_sdirs, validation_s_dirs
with open(cfg_path, "r") as f:
    cfg = json.load(f)

mdir = Path(cfg["mdir"])
training_s_dirs = [mdir / s for s in cfg["training_s_dirs"]]
validation_s_dirs = [mdir / s for s in cfg["validation_s_dirs"]]

all_s_dirs = training_s_dirs + validation_s_dirs
print(cfg_path)
print(f"Checking {len(all_s_dirs)} subjects...")

for s_dir in all_s_dirs:
    sampling_file = s_dir / "new_sampling_map_1.65.nii.gz"

    if not sampling_file.exists():
        print(f"{s_dir}: no sampling")
        continue

    for d in ["ref", "100", "50", "20", "10", "4"]:
        input_file = s_dir / d / "resampled_1.65.nii.gz"
        dcm_file = s_dir / d / "_sample.dcm"
        if not input_file.exists():
            print(f"{s_dir}: missing {d} input")
            continue
        if not dcm_file.exists():
            print(f"{s_dir}: missing {d} dcm")
            continue
