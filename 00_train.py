import os
import random
import argparse
import json
import torch
import torchio as tio
import numpy as np

from pathlib import Path
from data import get_subject_dict, nrmse, val_subject_nrmse
from losses import RobustL1Loss, l1_ssim_edge_loss_1, l1_ssim_edge_loss_2
from models import UNet3D
from datetime import datetime
from time import time

parser = argparse.ArgumentParser(description="Train 3D UNet on PET data")
parser.add_argument(
    "cfg_path",
    type=str,
    help="Path to json file containg training / validation subject directories",
)

# we need to run trainings for all 3 valid settings
parser.add_argument(
    "crf",
    type=int,
    help="Count reduction factor",
    choices=[100, 50, 20, 10, 4],
)
parser.add_argument("--patch_size", type=int, default=96, help="Patch size")
parser.add_argument("--queue_length", type=int, default=1725, help="Queue length")
parser.add_argument(
    "--samples_per_volume", type=int, default=40, help="Samples per volume"
)
parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")

# sweep parameters
# start_features in 16, 32
parser.add_argument(
    "--start_features",
    type=int,
    default=16,
    help="Features in first level of the UNet",
)  # lr in 1e-3, 3e-4, keep 4e-3
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")


# keep MSE
parser.add_argument(
    "--loss",
    type=str,
    default="MSE",
    choices=["MSE", "RobustL1", "L1SSIMEdge1", "L1SSIMEdge2"],
    help="Loss function to use",
)

# num_levels keep 3
parser.add_argument(
    "--num_levels", type=int, default=3, help="Number of levels in UNet"
)

# max_pool instead of down_conv in True, False
parser.add_argument(
    "--max_pool",
    action="store_true",
    help="use down convolution instead of max pooling UNET",
)
# final_softplus in False, True
parser.add_argument(
    "--final_softplus", action="store_true", help="Use final Softplus instead of ReLU"
)
parser.add_argument(
    "--num_train",
    type=int,
    default=-1,
    help="Number of training subjects, use -1 (default) for all",
)
parser.add_argument(
    "--num_val",
    type=int,
    default=-1,
    help="Number of validation subjects, use -1 (default) for all",
)


args = parser.parse_args()

# -------------------------------------------------------------------------------

cfg_path = Path(args.cfg_path)
crf = args.crf
patch_size = args.patch_size
queue_length = args.queue_length
samples_per_volume = args.samples_per_volume
batch_size = args.batch_size
lr = args.lr
loss = args.loss
num_epochs = args.num_epochs
num_train = args.num_train
num_val = args.num_val


down_conv = not args.max_pool
start_features = args.start_features
num_levels = args.num_levels
final_softplus = args.final_softplus

# open config file containing, mdir, training_s_sdirs, validation_s_dirs
with open(cfg_path, "r") as f:
    cfg = json.load(f)

mdir = Path(cfg["mdir"])
training_s_dirs = [mdir / s for s in cfg["training_s_dirs"]]
validation_s_dirs = [mdir / s for s in cfg["validation_s_dirs"]]

# seed all random number generators
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# %% create an output directory starting with run followed by a date-time stamp
# dont use tio for date time stamp
dt_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"run_{cfg_path.stem}_{crf}_{loss}_{start_features}_{dt_stamp}")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")

# save args to file output_dir/args.json
with open(output_dir / "args.json", "w") as f:
    json.dump(vars(args), f, indent=4)

# %%
try:
    nproc = len(os.sched_getaffinity(0))  # Linux: respects taskset/cgroups affinity
except AttributeError:
    import multiprocessing

    nproc = multiprocessing.cpu_count()  # fallback (Windows/macOS)

num_workers = min(nproc - 1, 15)
# norm factor for NRMS computed on log compressed SUV images
normalized_data_range = 1.0  # exp(1)-1 = 1.71 SUV for uncompressed images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# check whether there are no mutual subjects in training and validation
mutual = set(training_s_dirs).intersection(set(validation_s_dirs))
if len(mutual) > 0:
    raise ValueError(f"Mutual subjects in training and validation: {mutual}")

# shuffle s_dirs - the are stored in order of the categories
random.shuffle(training_s_dirs)
random.shuffle(validation_s_dirs)

if num_train > 0:
    training_s_dirs = training_s_dirs[:num_train]
if num_val > 0:
    validation_s_dirs = validation_s_dirs[:num_val]

subjects_list = []
for i, s_dir in enumerate(training_s_dirs):
    subjects_list.append(
        tio.Subject(get_subject_dict(s_dir, input_str=str(crf), ref_str="ref"))
    )

# save subset_dirs to file output_dir/subset_dirs.json
with open(output_dir / "training_dirs.json", "w") as f:
    json.dump([str(s) for s in training_s_dirs], f, indent=4)

with open(output_dir / "validation_dirs.json", "w") as f:
    json.dump([str(s) for s in validation_s_dirs], f, indent=4)

# setup preprocessing transforms
transform_list = [tio.transforms.ToCanonical()]

training_subjects_dataset = tio.SubjectsDataset(
    subjects_list, transform=tio.Compose(transform_list)
)

training_sampler = tio.data.WeightedSampler(
    probability_map="sampling_map", patch_size=patch_size
)

# %%
training_patches_queue = tio.Queue(
    training_subjects_dataset,
    queue_length,
    samples_per_volume,
    training_sampler,
    num_workers=num_workers,
    start_background=True,
    verbose=True,
)

training_patches_loader = tio.SubjectsLoader(
    training_patches_queue,
    batch_size=batch_size,
    num_workers=0,  # this must be 0
    pin_memory=True,
)

# %%
if num_epochs > 0:
    model = UNet3D(
        start_features=start_features,
        num_levels=num_levels,
        down_conv=down_conv,
        final_softplus=final_softplus,
    ).to(device)
    print(model)
    print(cfg_path)

    print(f"number of training subjects: {len(training_s_dirs)}")
    print(f"number of validation subjects: {len(validation_s_dirs)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if loss == "MSE":
        criterion = torch.nn.MSELoss()
    elif loss == "RobustL1":
        criterion = RobustL1Loss(eps=1e-2)
    elif loss == "L1SSIMEdge1":
        criterion = l1_ssim_edge_loss_1
    elif loss == "L1SSIMEdge2":
        criterion = l1_ssim_edge_loss_2
    else:
        raise ValueError(f"Unknown loss function: {loss}")

    train_loss_avg = torch.zeros(num_epochs)
    train_loss_std = torch.zeros(num_epochs)
    train_nrmse_avg = torch.zeros(num_epochs)
    train_nrmse_std = torch.zeros(num_epochs)

    val_nrmse_avg = torch.zeros(num_epochs)
    val_nrmse_std = torch.zeros(num_epochs)
    val_loss_avg = torch.zeros(num_epochs)
    val_loss_std = torch.zeros(num_epochs)

    for epoch in range(1, num_epochs + 1):
        ############################################################################
        # training loop
        t0 = time()
        model.train()
        batch_losses = torch.zeros(len(training_patches_loader))
        batch_nrmse = torch.zeros(len(training_patches_loader))
        for batch_idx, patches_batch in enumerate(training_patches_loader):
            inputs = patches_batch["input"][tio.DATA].to(torch.float32).to(device)
            targets = patches_batch["ref"][tio.DATA].to(torch.float32).to(device)

            output = model(inputs)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_losses[batch_idx] = loss.item()
            batch_nrmse[batch_idx] = nrmse(
                output, targets, data_range=normalized_data_range
            )

            print(
                f"Epoch [{epoch:04}/{num_epochs:04}] Batch [{(batch_idx+1):03}/{len(training_patches_loader):03}] - loss: {loss.item():.2E} - NRMSE: {batch_nrmse[batch_idx]:.2E}"
            )

        ########################################################################################
        # end of epoch
        train_loss_avg[epoch - 1] = batch_losses.mean().item()
        train_loss_std[epoch - 1] = batch_losses.std().item()

        train_nrmse_avg[epoch - 1] = batch_nrmse.mean().item()
        train_nrmse_std[epoch - 1] = batch_nrmse.std().item()
        print(
            f"\nEpoch [{epoch:04}/{num_epochs:04}] train loss: {train_loss_avg[epoch-1]:.2E} +- {train_loss_std[epoch-1]:.2E} train NRMSE: {train_nrmse_avg[epoch-1]:.4f} +- {train_nrmse_std[epoch-1]:.4f}"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss_avg": train_loss_avg[epoch - 1],
                "train_nrmse_avg": train_nrmse_avg[epoch - 1],
                "train_loss_std": train_loss_std[epoch - 1],
                "train_nrmse_std": train_nrmse_std[epoch - 1],
                "epoch": epoch,
            },
            output_dir / f"model_epoch_{epoch:04}.pth",
        )

        # save inputs, ouput, targets tensors to output_dir / last_batch_tensors.pt
        torch.save(
            {
                "inputs": inputs.cpu(),
                "output": output.detach().cpu(),
                "targets": targets.cpu(),
            },
            output_dir / "last_batch_tensors.pt",
        )

        # end of training loop
        ###########################################################################

        try:
            model.eval()
            scripted_model = torch.jit.script(model)
            scripted_model.save(output_dir / f"model_{epoch:04}_scripted.pt")
        except Exception as e:
            print(f"Could not export model to TorchScript: {e}")

        ############################################################################
        # validation loop
        val_batch_nrmse = torch.zeros(len(validation_s_dirs))
        val_batch_loss = torch.zeros(len(validation_s_dirs))
        for ivb, s_dir in enumerate(validation_s_dirs):
            print(f"Validating subject {ivb+1:03}/{len(validation_s_dirs):03}")

            val_batch_nrmse[ivb], val_batch_loss[ivb] = val_subject_nrmse(
                s_dir,
                output_dir / f"model_{epoch:04}_scripted.pt",
                crf,
                norm_factor=normalized_data_range,
                save_path=output_dir / f"val_sub_{ivb:03}",
                patch_size=patch_size,
                patch_overlap=patch_size // 2,
                batch_size=batch_size,
                loss_fct=criterion,
                dev=device,
            )

            # write val_batch_nrmse to file output_dir / val_nrmse_{epoch:04}.txt
            with open(output_dir / f"val_sub_{ivb:03}" / f"val_nrmse.csv", "a") as f:
                f.write(
                    f"{epoch:04}, {s_dir.name}, {val_batch_nrmse[ivb].item():.6f}\n"
                )

            with open(output_dir / f"val_sub_{ivb:03}" / f"val_loss.csv", "a") as f:
                f.write(f"{epoch:04}, {s_dir.name}, {val_batch_loss[ivb].item():.6f}\n")

        val_nrmse_avg[epoch - 1] += val_batch_nrmse.mean().item()
        val_nrmse_std[epoch - 1] += val_batch_nrmse.std().item()

        val_loss_avg[epoch - 1] += val_batch_loss.mean().item()
        val_loss_std[epoch - 1] += val_batch_loss.std().item()

        print(
            f"\nEpoch [{epoch:04}/{num_epochs:04}] val NRMSE: {val_nrmse_avg[epoch-1]:.6f} +- {val_nrmse_std[epoch-1]:.6f} val loss: {val_loss_avg[epoch-1]:.3E} +- {val_loss_std[epoch-1]:.3E}"
        )
        t1 = time()
        print(f" Epoch time: {((t1-t0)/60):.1f} min")

        #########################################################################
        with open(output_dir / "train_metrics.csv", "a") as f:
            f.write(
                f"{epoch}, {train_loss_avg[epoch-1]:.3E}, {train_loss_std[epoch-1]:.3E}, {train_nrmse_avg[epoch-1]:.6f}, {train_nrmse_std[epoch-1]:.6f}, {val_nrmse_avg[epoch-1]:.6f}, {val_nrmse_std[epoch-1]:.6f}\n"
            )

        with open(output_dir / "val_loss.csv", "a") as f:
            f.write(
                f"{epoch}, {val_loss_avg[epoch-1]:.3E}, {val_loss_std[epoch-1]:.3E}\n"
            )
