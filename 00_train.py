import random
import argparse
import json
import torch
import torchio as tio
import numpy as np

from pathlib import Path
from data import get_subject_dict, nrmse, val_subject_nrmse
from models import UNet3D
from datetime import datetime

parser = argparse.ArgumentParser(description="Train 3D UNet on PET data")
parser.add_argument(
    "--count_reduction_factor", type=int, default=10, help="Count reduction factor"
)
parser.add_argument("--patch_size", type=int, default=64, help="Patch size")
parser.add_argument("--queue_length", type=int, default=1000, help="Queue length")
parser.add_argument(
    "--samples_per_volume", type=int, default=100, help="Samples per volume"
)
parser.add_argument(
    "--num_sub_train", type=int, default=100, help="Number of subjects for training"
)
parser.add_argument(
    "--num_sub_val", type=int, default=10, help="Number of subjects for validation"
)
parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
parser.add_argument(
    "--target_voxsize_mm",
    type=float,
    default=1.65,
    help="Target voxel size (mm), set to None for no resampling",
)

# sweep parameters
# lr in 1e-3, 3e-4
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

# start_features in 16, 32
parser.add_argument(
    "--start_features",
    type=int,
    default=16,
    help="Features in first level of the UNet",
)
# num_levels in 3, 4
parser.add_argument(
    "--num_levels", type=int, default=3, help="Number of levels in UNet"
)
# donw_conv in True, False
parser.add_argument(
    "--down_conv",
    action="store_true",
    help="use down convolution instead of max pooling UNET",
)
# final_softplus in False, True
parser.add_argument(
    "--final_softplus", action="store_true", help="Use final Softplus instead of ReLU"
)

args = parser.parse_args()

count_reduction_factor = args.count_reduction_factor
patch_size = args.patch_size
queue_length = args.queue_length
samples_per_volume = args.samples_per_volume
num_sub_train = args.num_sub_train
num_sub_val = args.num_sub_val
batch_size = args.batch_size
lr = args.lr
num_epochs = args.num_epochs
target_voxsize_mm = args.target_voxsize_mm

down_conv = args.down_conv
start_features = args.start_features
num_levels = args.num_levels
final_softplus = args.final_softplus

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
output_dir = Path(f"run_{dt_stamp}")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")

# save args to file output_dir/args.json
with open(output_dir / "args.json", "w") as f:
    json.dump(vars(args), f, indent=4)

# %%
num_workers = 10
# norm factor for NRMS computed on log compressed SUV images
normalized_data_range = 1.0  # exp(1)-1 = 1.71 SUV for uncompressed images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read s_dirs from subjects.txt
with open("acquisitions.txt", "r") as f:
    s_dirs = [Path(line.strip()) for line in f.readlines()]

# shuffle s_dirs
random.shuffle(s_dirs)
# take first num_sub_train for training
training_s_dirs = s_dirs[:num_sub_train]
# take next num_sub_val for validation
validation_s_dirs = s_dirs[num_sub_train : (num_sub_train + num_sub_val)]

subjects_list = [
    tio.Subject(get_subject_dict(s_dir, crfs=[str(count_reduction_factor), "ref"]))
    for s_dir in training_s_dirs
]

# save subset_dirs to file output_dir/subset_dirs.json
with open(output_dir / "training_dirs.json", "w") as f:
    json.dump([str(s) for s in training_s_dirs], f, indent=4)


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
    verbose=True,
)

training_patches_loader = tio.SubjectsLoader(
    training_patches_queue,
    batch_size=batch_size,
    num_workers=0,  # this must be 0
)

# %%
if num_epochs > 0:
    model = UNet3D(
        start_features=start_features,
        num_levels=num_levels,
        down_conv=down_conv,
        final_softplus=final_softplus,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train_loss_avg = torch.zeros(num_epochs)
    train_loss_std = torch.zeros(num_epochs)
    train_nrmse_avg = torch.zeros(num_epochs)
    train_nrmse_std = torch.zeros(num_epochs)

    val_nrmse_avg = torch.zeros(num_epochs)
    val_nrmse_std = torch.zeros(num_epochs)

    for epoch in range(1, num_epochs + 1):
        ############################################################################
        # training loop
        model.train()
        batch_losses = torch.zeros(len(training_patches_loader))
        batch_nrmse = torch.zeros(len(training_patches_loader))
        for batch_idx, patches_batch in enumerate(training_patches_loader):
            inputs = patches_batch[str(count_reduction_factor)][tio.DATA].to(device)
            targets = patches_batch["ref"][tio.DATA].to(device)

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
                f"Epoch [{epoch:04}/{num_epochs:04}] Batch [{(batch_idx+1):03}/{len(training_patches_loader):03}] - loss: {loss.item():.2E} - NRMSE: {batch_nrmse[batch_idx]:.2f}",
                end="\r",
            )

        ########################################################################################
        # end of epoch
        train_loss_avg[epoch - 1] = batch_losses.mean().item()
        train_loss_std[epoch - 1] = batch_losses.std().item()

        train_nrmse_avg[epoch - 1] = batch_nrmse.mean().item()
        train_nrmse_std[epoch - 1] = batch_nrmse.std().item()
        print(
            f"\nEpoch [{epoch:04}/{num_epochs:04}] train loss: {train_loss_avg[epoch-1]:.2E} +- {train_loss_std[epoch-1]:.2E} train NRMSE: {train_nrmse_avg[epoch-1]:.2f} +- {train_nrmse_std[epoch-1]:.2f}"
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
        for ivb, s_dir in enumerate(validation_s_dirs):
            print(
                f"Validating subject {ivb+1:03}/{len(validation_s_dirs):03}", end="\r"
            )
            val_batch_nrmse[ivb] = val_subject_nrmse(
                s_dir,
                output_dir / f"model_{epoch:04}_scripted.pt",
                count_reduction_factor,
                norm_factor=normalized_data_range,
                save_path=output_dir / f"val_sub_{ivb:03}",
            )

        val_nrmse_avg[epoch - 1] += val_batch_nrmse.mean().item()
        val_nrmse_std[epoch - 1] += val_batch_nrmse.std().item()

        print(
            f"\nEpoch [{epoch:04}/{num_epochs:04}] val NRMSE: {val_nrmse_avg[epoch-1]:.4f} +- {val_nrmse_std[epoch-1]:.4f}"
        )

        #########################################################################
        with open(output_dir / "train_metrics.csv", "a") as f:
            f.write(
                f"{epoch}, {train_loss_avg[epoch-1]:.3E}, {train_loss_std[epoch-1]:.3E}, {train_nrmse_avg[epoch-1]:.4f}, {train_nrmse_std[epoch-1]:.4f}, {val_nrmse_avg[epoch-1]:.4f}, {val_nrmse_std[epoch-1]:.4f}\n"
            )
