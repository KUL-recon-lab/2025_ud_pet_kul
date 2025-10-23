import os
import argparse
import json
import torch
import torchio as tio

from pathlib import Path
from data import get_subject_dict, nrmse, val_subject_nrmse
from losses import RobustL1Loss, l1_ssim_edge_loss_1, l1_ssim_edge_loss_2
from models import UNet3D
from time import time

parser = argparse.ArgumentParser(description="Train 3D UNet on PET data")
# we need to run trainings for all 3 valid settings
parser.add_argument("run_folder", type=str)
parser.add_argument("num_epochs", type=int)
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

args = parser.parse_args()

# -------------------------------------------------------------------------------

output_dir: Path = Path(args.run_folder)
lr: float = args.lr
num_epochs: int = args.num_epochs

# read the json file output_dir / "args.json"
with open(output_dir / "args.json", "r") as f:
    args_dict = json.load(f)

patch_size = args_dict["patch_size"]
queue_length = args_dict["queue_length"]
samples_per_volume = args_dict["samples_per_volume"]
batch_size = args_dict["batch_size"]
start_features = args_dict["start_features"]
num_levels = args_dict["num_levels"]
down_conv = not args_dict["max_pool"]
final_softplus = args_dict["final_softplus"]
loss = args_dict["loss"]
crf = args_dict["crf"]

# read training_dirs.json into the list training_s_dirs
with open(output_dir / "training_dirs.json", "r") as f:
    training_s_dirs = json.load(f)
training_s_dirs = [Path(s) for s in training_s_dirs]

# read validation_dirs.json into the list validation_s_dirs
with open(output_dir / "validation_dirs.json", "r") as f:
    validation_s_dirs = json.load(f)
validation_s_dirs = [Path(s) for s in validation_s_dirs]

# find the last save ceckpoint in output_dir
checkpoint_files = list(output_dir.glob("model_epoch_*.pth"))
if len(checkpoint_files) == 0:
    raise ValueError(f"No checkpoint files found in {output_dir}")
checkpoint_files.sort()
last_checkpoint_file = checkpoint_files[-1]
print(f"Resuming training from checkpoint: {last_checkpoint_file}")

################################################################################
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

################################################################################
# %%
subjects_list = []
for i, s_dir in enumerate(training_s_dirs):
    subjects_list.append(
        tio.Subject(get_subject_dict(s_dir, input_str=str(crf), ref_str="ref"))
    )

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

    # load the checkpoint
    checkpoint = torch.load(last_checkpoint_file, map_location=device)
    start_epoch = checkpoint["epoch"]
    # set model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    print(model)

    print(f"number of training subjects: {len(training_s_dirs)}")
    print(f"number of validation subjects: {len(validation_s_dirs)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if loss == "MSE":
        criterion = torch.nn.MSELoss()
    elif loss == "RobustL1":
        criterion = RobustL1Loss(eps=1e-2)
    elif loss == "L1SSIMEdge1":
        criterion = l1_ssim_edge_loss_1
    elif loss == "L1SSIMEdge2":
        criterion = l1_ssim_edge_loss_2
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")

    train_loss_avg = torch.zeros(start_epoch + num_epochs)
    train_loss_std = torch.zeros(start_epoch + num_epochs)
    train_nrmse_avg = torch.zeros(start_epoch + num_epochs)
    train_nrmse_std = torch.zeros(start_epoch + num_epochs)

    val_nrmse_avg = torch.zeros(start_epoch + num_epochs)
    val_nrmse_std = torch.zeros(start_epoch + num_epochs)
    val_loss_avg = torch.zeros(start_epoch + num_epochs)
    val_loss_std = torch.zeros(start_epoch + num_epochs)

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
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
                f"Epoch [{epoch:04}/{(num_epochs+start_epoch):04}] Batch [{(batch_idx+1):03}/{len(training_patches_loader):03}] - loss: {loss.item():.2E} - NRMSE: {batch_nrmse[batch_idx]:.2E}",
                end="\r",
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
            print(
                f"Validating subject {ivb+1:03}/{len(validation_s_dirs):03}", end="\r"
            )

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
