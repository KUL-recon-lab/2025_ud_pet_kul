import argparse
import json
import torch
import torchio as tio

from pathlib import Path
from data import get_subject_dict, psnr
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
parser.add_argument("--n_sub", type=int, default=100, help="Number of subjects")
parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
parser.add_argument(
    "--target_voxsize_mm",
    type=float,
    default=1.65,
    help="Target voxel size (mm), set to None for no resampling",
)

args = parser.parse_args()

count_reduction_factor = args.count_reduction_factor
patch_size = args.patch_size
queue_length = args.queue_length
samples_per_volume = args.samples_per_volume
n_sub = args.n_sub
batch_size = args.batch_size
lr = args.lr
num_epochs = args.num_epochs
target_voxsize_mm = args.target_voxsize_mm


model_kwargs = dict(in_channels=1, out_channels=1)

m_path = Path(
    "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
)

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
normalized_data_range = 3.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

s_dirs = sorted([x for x in m_path.iterdir() if x.is_dir()])[:n_sub]

subjects_list = [
    tio.Subject(get_subject_dict(s_dir, crfs=[str(count_reduction_factor), "ref"]))
    for s_dir in s_dirs
]

# save subset_dirs to file output_dir/subset_dirs.json
with open(output_dir / "training_dirs.json", "w") as f:
    json.dump([str(s) for s in s_dirs], f, indent=4)


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
    model = UNet3D(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train_loss_avg = torch.zeros(num_epochs)
    train_psnr_avg = torch.zeros(num_epochs)

    for epoch in range(1, num_epochs + 1):
        ############################################################################
        # training loop
        model.train()
        batch_losses = torch.zeros(len(training_patches_loader))
        batch_psnr = torch.zeros(len(training_patches_loader))
        for batch_idx, patches_batch in enumerate(training_patches_loader):
            inputs = patches_batch[str(count_reduction_factor)][tio.DATA].to(device)
            targets = patches_batch["ref"][tio.DATA].to(device)

            output = model(inputs)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_losses[batch_idx] = loss.item()
            batch_psnr[batch_idx] = psnr(
                output, targets, data_range=normalized_data_range
            )

            print(
                f"Epoch [{epoch:04}/{num_epochs:04}] Batch [{(batch_idx+1):03}/{len(training_patches_loader):03}] - loss: {loss.item():.2E} - PSNR: {batch_psnr[batch_idx]:.2f}",
                end="\r",
            )

        ########################################################################################
        # end of epoch
        loss_avg = batch_losses.mean().item()
        train_loss_avg[epoch - 1] = loss_avg
        loss_std = batch_losses.std().item()

        psnr_avg = batch_psnr.mean().item()
        train_psnr_avg[epoch - 1] = psnr_avg
        psnr_std = batch_psnr.std().item()
        print(
            f"\nEpoch [{epoch:04}/{num_epochs:04}] train loss: {loss_avg:.2E} +- {loss_std:.2E} train PSNR: {psnr_avg:.2f} +- {psnr_std:.2f}"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss_avg[epoch - 1],
                "train_psnr": train_psnr_avg[epoch - 1],
                "epoch": epoch,
            },
            output_dir / f"model_epoch_{epoch:04}.pth",
        )

        with open(output_dir / "train_psnr.txt", "w") as f:
            for psnr_value in train_psnr_avg[:epoch]:
                f.write(f"{psnr_value}\n")

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
