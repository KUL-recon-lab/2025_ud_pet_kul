import torch
import torchio as tio

from torchmetrics.image import PeakSignalNoiseRatio
from pathlib import Path
from data import (
    SUVLogCompress,
    get_subject_dict,
    AddSamplingMap,
)
from models import UNet3D

count_reduction_factor = 10
patch_size: int = 64
queue_length: int = 10000
samples_per_volume: int = 100
n_sub: int = 100
batch_size: int = 20
lr: float = 1e-3
num_epochs: int = 100
target_voxsize_mm: float | None = 1.65
model_kwargs = dict(in_channels=1, out_channels=1)

# %%
num_workers = 12
normalized_data_range = 3.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m_path = Path(
    "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
)

s_dirs = sorted([x for x in m_path.iterdir() if x.is_dir()])
#### HACK FOR -nsub (needed until data structure is cleaned)
subjects_list = [
    tio.Subject(get_subject_dict(s_dir, crfs=[count_reduction_factor]))
    for s_dir in s_dirs[320 : (320 + n_sub)]
]
### END HACK

# setup preprocessing transforms
transform_list = [tio.transforms.ToCanonical()]
if target_voxsize_mm is not None:
    transform_list += [tio.Resample(target=target_voxsize_mm)]
transform_list += [SUVLogCompress(), AddSamplingMap()]

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
    psnr = PeakSignalNoiseRatio(data_range=(0, normalized_data_range)).to(device)

    train_loss_avg = torch.zeros(num_epochs)
    train_psnr_avg = torch.zeros(num_epochs)

    for epoch in range(1, num_epochs + 1):
        ############################################################################
        # training loop
        model.train()
        batch_losses = torch.zeros(len(training_patches_loader))
        batch_psnr = torch.zeros(len(training_patches_loader))
        for batch_idx, patches_batch in enumerate(training_patches_loader):
            inputs = patches_batch[f"crf{count_reduction_factor}"][tio.DATA].to(device)
            targets = patches_batch["ref"][tio.DATA].to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()

            optimizer.step()
            batch_losses[batch_idx] = loss.item()
            batch_psnr[batch_idx] = psnr(output, targets)

            print(
                f"Epoch [{epoch:04}/{num_epochs:04}] Batch [{(batch_idx+1):03}/{len(training_patches_loader):03}] - tr. loss: {loss.item():.2E}",
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
            f"unet3d_epoch{epoch:04}.pth",
        )

        with open("train_psnr.txt", "w") as f:
            for psnr_value in train_psnr_avg[:epoch]:
                f.write(f"{psnr_value}\n")

        # end of training loop
        ############################################################################

        try:
            model.eval()
            scripted_model = torch.jit.script(model)
            scripted_model.save(f"unet3d_epoch{epoch:04}_scripted.pt")
        except Exception as e:
            print(f"Could not export model to TorchScript: {e}")
