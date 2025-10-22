# TODO: interpolation

import pydicom
import torch
import torch.nn.functional as F
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from time import strptime
from datetime import timedelta

from scipy.ndimage import gaussian_filter


def get_suv_factor_from_dicom(dcm_file: Path, use_series_time: bool = True) -> float:
    dcm = pydicom.dcmread(dcm_file)
    weight_g = float(dcm.PatientWeight) * 1000  # in g
    radiopharm_info = dcm.RadiopharmaceuticalInformationSequence[0]
    injected_act_Bq: float = float(radiopharm_info.RadionuclideTotalDose)
    halflife_s: float = float(radiopharm_info.RadionuclideHalfLife)
    injection_time: str = (
        radiopharm_info.RadiopharmaceuticalStartTime
    )  # '131313.000000'

    if use_series_time:
        ref_time: str = dcm.SeriesTime  # '141842.000000'
    else:
        ref_time: str = dcm.AcquisitionTime  # '141842.00000'

    t1 = strptime(ref_time.split(".")[0], "%H%M%S")
    t2 = strptime(injection_time.split(".")[0], "%H%M%S")

    # time difference ref_time - injection_time in seconds
    dt_s = (
        timedelta(hours=t1.tm_hour, minutes=t1.tm_min, seconds=t1.tm_sec)
        - timedelta(hours=t2.tm_hour, minutes=t2.tm_min, seconds=t2.tm_sec)
    ).seconds

    decay_factor = 0.5 ** (dt_s / halflife_s)
    suv_conv_factor = weight_g / (decay_factor * injected_act_Bq)

    return suv_conv_factor


class SUVLogCompress(tio.Transform):
    """
    Apply: img <- log1p(img * suv_fac), per subject.
    Optionally restrict to specific image keys (e.g., ['full_dose']).
    """

    def __init__(self, keys=None, invert: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.keys = keys
        self.invert_transform = bool(invert)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        suv_fac = float(subject["suv_fac"])
        images = (
            subject.get_images_dict(intensity_only=True)
            if self.keys is None
            else {k: subject[k] for k in self.keys if k in subject}
        )

        inverse = getattr(self, "invert_transform", False)

        for img in images.values():
            src = img.data
            x = src.to(torch.float32)

            if not inverse:
                y = torch.log1p(x * suv_fac)
            else:
                y = torch.expm1(x) / suv_fac

            # preserve dtype
            img.set_data(y.to(src.dtype))

        return subject


class AddSamplingMap(tio.Transform):
    def __init__(self, p: float = 1.0):
        super().__init__(p=p)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        ref = subject["ref"]
        weights = (ref.data > 0.1).to(torch.int8)
        # set the weight to one in the first 128 slice such that once in a while
        # we sample an empty patch
        weights[..., :129] = 1
        subject["sampling_map"] = tio.Image(
            tensor=weights,
            affine=ref.affine,
            type=tio.SAMPLING_MAP,
        )
        return subject


def get_subject_dict(
    s_dir: Path,
    input_str: str,
    ref_str: str | None = "ref",
    target_voxel_size: float = 1.65,
    crop=False,
    **kwargs,
):
    suv_fac_file = s_dir / "suv_factor.txt"
    if suv_fac_file.exists():
        with open(suv_fac_file, "r") as f:
            suv_fac = float(f.read().strip())
    else:
        dcm_file = s_dir / "ref" / "_sample.dcm"
        suv_fac = get_suv_factor_from_dicom(dcm_file, **kwargs)

    subject_dict = {}
    subject_dict["suv_fac"] = suv_fac
    subject_dict["input_str"] = input_str
    subject_dict["s_dir"] = s_dir

    if crop:
        crop_suffix = "_cropped"
    else:
        crop_suffix = ""

    # load the input
    dfile = s_dir / input_str / f"resampled_{target_voxel_size:.2f}{crop_suffix}.nii.gz"
    subject_dict["input_file"] = dfile
    subject_dict["input"] = tio.ScalarImage(dfile)

    # load the reference if specified
    if ref_str is not None:
        ref_file = (
            s_dir / ref_str / f"resampled_{target_voxel_size:.2f}{crop_suffix}.nii.gz"
        )
        subject_dict["ref_file"] = ref_file
        subject_dict["ref"] = tio.ScalarImage(ref_file)

    if not crop:
        subject_dict["sampling_map"] = tio.ScalarImage(
            s_dir / f"new_sampling_map_{target_voxel_size:.2f}.nii.gz"
        )

    return subject_dict


def nrmse(output, targets, data_range: float):
    mse_per_elem = F.mse_loss(output, targets, reduction="none")
    # spatial/channel dims to reduce to per-sample values
    dims = tuple(range(1, mse_per_elem.ndim))

    # per-sample MSE
    mse_per_sample = mse_per_elem.mean(dim=dims)
    nrmse_per_sample = torch.sqrt(mse_per_sample) / data_range

    return nrmse_per_sample.mean().item()


def patch_inference(
    subject: tio.Subject,
    scripted_model_path: Path,
    patch_size=96,
    patch_overlap=48,
    batch_size=20,
    verbose=False,
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
        for i, patches_batch in enumerate(patch_loader):
            if verbose:
                print(
                    f"Processing patch batch {(i+1):04}/{len(patch_loader):04}",
                    end="\r",
                )
            input_tensor = patches_batch["input"][tio.DATA].to(torch.float32).to(device)
            locations = patches_batch[tio.LOCATION].to(device)
            outputs = model(input_tensor)
            aggregator.add_batch(outputs, locations)

    if verbose:
        print("")  # newline after progress

    return aggregator.get_output_tensor()


# %%


def val_subject_nrmse(
    s_dir: Path,
    model_path: Path,
    count_reduction_factor: int,
    save_path: None | Path = None,
    norm_factor: float = 1.0,
    crop: bool = True,
    verbose: bool = True,
    loss_fct=None,
    dev: torch.device = torch.device("cpu"),
    **kwargs,
):
    transform = tio.Compose([tio.transforms.ToCanonical()])

    if verbose:
        print(f"Loading {s_dir.name}")

    subject = transform(
        tio.Subject(
            get_subject_dict(
                s_dir, input_str=str(count_reduction_factor), ref_str="ref", crop=crop
            )
        )
    )

    if verbose:
        print(f"Running inference on {s_dir.name}")

    output_tensor = patch_inference(
        subject,
        model_path,
        verbose=verbose,
        **kwargs,
    )

    if verbose:
        print(f"Computing NRMSE on {s_dir.name}")

    out = output_tensor.unsqueeze(0).to(dev)
    ref = subject["ref"].data.unsqueeze(0).to(dev)

    metric = nrmse(out, ref, norm_factor)

    if loss_fct is not None:
        loss_value = loss_fct(out, ref)
    else:
        loss_value = -1.0

    if verbose:
        print(f"writing outputs to {save_path}")

    # dump arrays to numpy files (for quick loading / viewing)
    if save_path is not None:
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        # calculate MIPs show in matplotlib figure and save
        mip1 = torch.max(out.squeeze(), 0)[0].cpu().numpy().T
        mip2 = np.flip(torch.max(out.squeeze(), 1)[0].cpu().numpy().T, 1)
        mip3 = torch.max(ref.squeeze(), 0)[0].cpu().numpy().T
        mip4 = np.flip(torch.max(ref.squeeze(), 1)[0].cpu().numpy().T, 1)

        ims1 = dict(vmin=0, vmax=4, origin="lower", cmap="Greys")
        fig, ax = plt.subplots(1, 4, figsize=(22, 12), layout="constrained")
        ax[0].imshow(mip1, **ims1)
        ax[1].imshow(mip2, **ims1)
        ax[2].imshow(mip3, **ims1)
        ax[3].imshow(mip4, **ims1)
        fig.savefig(save_path / "out_mips.png")
        plt.close(fig)

    if verbose:
        print(f"finished")

    return metric, loss_value


def noise_metric(vol):
    vol_sm = gaussian_filter(vol, 2.0)
    return float(np.abs(vol - vol_sm).sum() / (vol_sm > 0.1).sum())
