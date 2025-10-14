# TODO: interpolation

import pydicom
import torch
import torch.nn.functional as F
import torchio as tio
import nibabel as nib
import numpy as bp

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

    def __init__(self, keys=None, **kwargs):
        super().__init__(**kwargs)
        self.keys = keys

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        suv_fac = float(subject["suv_fac"])
        images = (
            subject.get_images_dict(intensity_only=True)
            if self.keys is None
            else {k: subject[k] for k in self.keys if k in subject}
        )
        for img in images.values():
            data = img.data.to(torch.float32)  # ensure float for log1p
            # PET should be >= 0; if needed, clamp tiny negatives from interpolation
            # data = data.clamp_min_(0)
            img.set_data(torch.log1p(data * suv_fac))
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
    s_dir: Path, crfs: list[str], target_voxel_size: float = 1.65, crop=False, **kwargs
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
    subject_dict["crfs"] = crfs
    subject_dict["s_dir"] = s_dir

    for d in crfs:
        if crop:
            dfile = s_dir / d / f"resampled_{target_voxel_size:.2f}_cropped.nii.gz"
        else:
            dfile = s_dir / d / f"resampled_{target_voxel_size:.2f}.nii.gz"
        # dfile = sorted(list(s_dir.glob(f"{d}/*.nii.gz")))[0]
        subject_dict[f"{d}_file"] = dfile
        subject_dict[f"{d}"] = tio.ScalarImage(dfile)

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
    count_reduction_factor=10,
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
            input_tensor = (
                patches_batch[f"{count_reduction_factor}"][tio.DATA]
                .to(torch.float32)
                .to(device)
            )
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
    **kwargs,
):
    transform = tio.Compose([tio.transforms.ToCanonical()])

    if verbose:
        print(f"Loading {s_dir.name}")

    subject = transform(
        tio.Subject(
            get_subject_dict(
                s_dir, crfs=[str(count_reduction_factor), "ref"], crop=crop
            )
        )
    )

    if verbose:
        print(f"Running inference on {s_dir.name}")

    output_tensor = patch_inference(
        subject,
        model_path,
        count_reduction_factor=count_reduction_factor,
        verbose=verbose,
        **kwargs,
    )

    if verbose:
        print(f"Computing NRMSE on {s_dir.name}")

    metric = nrmse(
        output_tensor.unsqueeze(0),
        subject["ref"].data.unsqueeze(0).to(output_tensor.device),
        norm_factor,
    )

    if verbose:
        print(f"writing outputs to {save_path}")

    # dump arrays to numpy files (for quick loading / viewing)
    if save_path is not None:
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        affine = subject["ref"].affine
        # nib.save(
        #    nib.Nifti1Image(
        #        subject[f"{count_reduction_factor}"].data.numpy().squeeze(), affine
        #    ),
        #    save_path / "input.nii",
        # )
        nib.save(
            nib.Nifti1Image(output_tensor.cpu().numpy().squeeze(), affine),
            save_path / "out.nii",
        )
        # nib.save(
        #    nib.Nifti1Image(subject["ref"].data.numpy().squeeze(), affine),
        #    save_path / "ref.nii",
        # )

    if verbose:
        print(f"finished")

    return metric


def noise_metric(vol):
    vol_sm = gaussian_filter(vol, 2.0)
    return float(np.abs(vol - vol_sm).sum() / (vol_sm > 0.1).sum())
