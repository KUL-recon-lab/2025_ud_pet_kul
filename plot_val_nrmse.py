import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# find all subdirs of ./dennoising_models
cfg_dirs = [
    Path("./denoising_models") / x for x in ["uexp_fdg", "biograph_fdg", "uexp_non_fdg"]
]
crfs = [100, 50, 20, 10, 4]

fig, ax = plt.subplots(
    5,
    3,
    figsize=(14, 8),
    layout="constrained",
)

for i, cfg_dir in enumerate(cfg_dirs):
    for j, crf in enumerate(crfs):
        model_dirs = sorted((cfg_dir / f"{crf}").glob("run_*"))
        for k, md in enumerate(model_dirs):
            metrics = np.loadtxt(md / "train_metrics.csv", delimiter=",")
            val_nrmse = metrics[:, -2]
            ax[j, i].semilogx(
                np.arange(1, val_nrmse.size + 1)[5:],
                val_nrmse[5:],
                label=md.stem[11:].split("_2025")[0],
            )
        ax[j, i].grid(ls=":")
        ax[j, i].set_xlim(None, 100)
        ax[j, i].legend(fontsize="xx-small")

        if j == 0:
            ax[j, i].set_title(cfg_dir.stem, fontsize="small")
        if i == 0:
            ax[j, i].set_ylabel(f"CRF {crf} - NRMSE", fontsize="small")

# ax.set_xlabel("epoch")
# ax.set_ylabel("Validation NRMSE")

fig.show()
