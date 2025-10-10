import torchio as tio
import torch
from pathlib import Path


def patch_inference(
    subject: tio.Subject,
    scripted_model_path: Path,
    count_reduction_factor=10,
    patch_size=64,
    patch_overlap=32,
    batch_size=20,
):

    grid_sampler = tio.inference.GridSampler(
        subject,
        patch_size,
        patch_overlap,
    )

    patch_loader = tio.SubjectsLoader(grid_sampler, batch_size=batch_size)
    aggregator = tio.inference.GridAggregator(grid_sampler)

    # load the model from torch script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(scripted_model_path, map_location=device)

    model.eval()
    with torch.no_grad():
        for patches_batch in patch_loader:
            input_tensor = patches_batch[f"{count_reduction_factor}"][tio.DATA].to(
                device
            )
            locations = patches_batch[tio.LOCATION].to(device)
            outputs = model(input_tensor)
            aggregator.add_batch(outputs, locations)

    return aggregator.get_output_tensor()


# %%

if __name__ == "__main__":
    import numpy as np
    import pymirc.viewer as pv
    from data import get_subject_dict, SUVLogCompress

    model_path = Path("run_20251010_180549/model_0020_scripted.pt")
    target_voxsize_mm = 1.65
    count_reduction_factor = 10

    m_path = Path(
        "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/nifti_out"
    )
    s_dirs = sorted([x for x in m_path.iterdir() if x.is_dir()])
    s_dir = s_dirs[101]

    transform = tio.Compose(
        [
            tio.transforms.ToCanonical(),
            tio.Resample(target=target_voxsize_mm),
        ]
    )

    subject = transform(
        tio.Subject(get_subject_dict(s_dir, crfs=[str(count_reduction_factor), "ref"]))
    )

    output_tensor = patch_inference(
        subject,
        model_path,
        count_reduction_factor=count_reduction_factor,
    )
    o = output_tensor.cpu().numpy().squeeze()
    ref = subject["ref"].numpy().squeeze()
    i = subject[f"{count_reduction_factor}"].numpy().squeeze()

    o2 = np.exp(o) - 1
    i2 = np.exp(i) - 1
    ref2 = np.exp(ref) - 1

    vi = pv.ThreeAxisViewer([i2, o2, ref2], imshow_kwargs=dict(vmin=0, vmax=5))
