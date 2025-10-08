import pydicom
import numpy as np
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


if __name__ == "__main__":
    import nibabel as nib
    import pymirc.viewer as pv

    m_path = Path(
        "W:/schramm_lab/data/2025_ud_pet_challenge/Shanghai-Ruijin-Hospital-2023/nifti_out"
    )

    s_dirs = sorted([x for x in m_path.iterdir() if x.is_dir()])

    for i, s_dir in enumerate(s_dirs):
        if i >= 108:
            print(i, s_dir)
            ref_dir = sorted(list(s_dir.glob("*NORMAL")))[0]
            dcm_file = ref_dir / "_sample.dcm"
            suv_fac = get_suv_factor_from_dicom(dcm_file, use_series_time=True)

            nii_file = sorted(list(ref_dir.glob("*.nii.gz")))[0]
            nii_hdr = nib.as_closest_canonical(nib.load(nii_file))
            vol = nii_hdr.get_fdata()

            suv_vol = vol * suv_fac
            suv_vol_scaled = np.log(1 + suv_vol)

            # create an empty text file that shows that we have seen this case
            (s_dir / "seen.txt").touch()

            ims = [dict(vmin=0, vmax=7)] + [dict(vmin=0, vmax=3.5)]
            vi = pv.ThreeAxisViewer([suv_vol, suv_vol_scaled], imshow_kwargs=ims)
            breakpoint()
