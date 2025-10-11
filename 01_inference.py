from pathlib import Path
from data import val_subject_nrmse

model_path = Path("run_20251010_180549/model_0020_scripted.pt")
count_reduction_factor = 10
with open("acquisitions.txt", "r") as f:
    s_dirs = [Path(line.strip()) for line in f.readlines()]
s_dir = s_dirs[0]

met = val_subject_nrmse(s_dir, model_path, count_reduction_factor, save_path=Path("."))
print(f"NRMSE: {met:.3f}")
