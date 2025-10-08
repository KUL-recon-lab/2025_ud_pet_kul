# TODO: interpolation, sampling mask

import pydicom
import torch
import torchio as tio

from pathlib import Path
from time import strptime
from datetime import timedelta


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
        subject["sampling_map"] = tio.Image(
            tensor=weights,
            affine=ref.affine,
            type=tio.SAMPLING_MAP,
        )
        return subject


def get_subject_dict(s_dir: Path, ds: list[int], **kwargs):
    ref_dir = sorted(list(s_dir.glob("*NORMAL")))[0]

    dcm_file = ref_dir / "_sample.dcm"
    suv_fac = get_suv_factor_from_dicom(dcm_file, **kwargs)

    ref_nii_file = sorted(list(ref_dir.glob("*.nii.gz")))[0]

    subject_dict = {}
    subject_dict["suv_fac"] = suv_fac
    subject_dict["ref"] = tio.ScalarImage(ref_nii_file)

    for d in ds:
        d_dir = sorted(list(s_dir.glob(f"*D{d}")))[0]
        dfile = sorted(list(d_dir.glob("*.nii.gz")))[0]
        subject_dict[f"d{d}"] = tio.ScalarImage(dfile)

    return subject_dict


if __name__ == "__main__":
    show = False
    ds = [10]

    m_path = Path(
        "/uz/data/Admin/ngeworkingresearch/schramm_lab/data/2025_ud_pet_challenge/Shanghai-Ruijin-Hospital-2023/nifti_out"
    )

    s_dirs = sorted([x for x in m_path.iterdir() if x.is_dir()])

    subjects_list = []
    refs = []

    for i, s_dir in enumerate(s_dirs[:16]):
        print(i, s_dir)
        subject_dict = get_subject_dict(s_dir, ds)
        subject = tio.Subject(subject_dict)
        subjects_list.append(subject)

    transform = tio.Compose(
        [tio.transforms.ToCanonical(), SUVLogCompress(), AddSamplingMap()]
    )
    subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform)

    patch_size = 128
    queue_length = 300
    samples_per_volume = 32
    batch_size = 32
    sampler = tio.data.WeightedSampler(
        probability_map="sampling_map", patch_size=patch_size
    )

    # %%
    patches_queue = tio.Queue(
        subjects_dataset,
        queue_length,
        samples_per_volume,
        sampler,
        num_workers=4,
    )

    patches_loader = tio.SubjectsLoader(
        patches_queue,
        batch_size=batch_size,
        num_workers=0,  # this must be 0
    )

    for ib, patches_batch in enumerate(patches_loader):
        print(ib)
        inputs = patches_batch["d10"][tio.DATA]  # key 't1' is in subject
        targets = patches_batch["ref"][tio.DATA]  # key 'brain' is in subject

    # %%
    if show:
        import pymirc.viewer as pv

        ims = dict(vmin=0, vmax=3.5)

        for s in subjects_dataset:
            vi = pv.ThreeAxisViewer(
                [s[x].data.numpy().squeeze() for x in ["ref"] + [f"d{d}" for d in ds]],
                imshow_kwargs=ims,
            )
            breakpoint()
