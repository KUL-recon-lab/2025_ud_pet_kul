# Supervised Log-SUV U-NET Denoising of Low-Count PET Images

## Setup

Create and activate the conda environment using the provided `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate 2025_udpet_logsuv
```

## Prediction of 2025 UD-PET Challenge Test Data

```bash
python 03_test_predictions.py <path_to_udp_test_data>
```

where `<path_to_udp_test_data>` is the path to the directory containing the 2025 UD-PET challenge 
test data in NIfTI format and should look like:

```bash
.
|-- Anonymous-01_DRF-100.nii.gz
|-- Anonymous-01_DRF-10.nii.gz
|-- Anonymous-01_DRF-20.nii.gz
|-- Anonymous-01_DRF-4.nii.gz
|-- Anonymous-01_DRF-50.nii.gz
...
|-- Anonymous-50_DRF-100.nii.gz
|-- Anonymous-50_DRF-10.nii.gz
|-- Anonymous-50_DRF-20.nii.gz
|-- Anonymous-50_DRF-4.nii.gz
|-- Anonymous-50_DRF-50.nii.gz
```


`inference_model_config.json` contains the configuration of the models to be used for inference.
By default, the pre-trained models that we used for the 2025 UD-PET challenge are specified there.
`hf_epoch` can be used to specify a particular epoch to use for inference, if `null` the best epoch
as determined from the validation NRMSE is used.

You can you your own models by changing the `checkpoint` entry.

## Training of denoising models

```bash
python 01_train.py <input_data_config.json> <DRF>
```

Make sure to adjust the master directories in the data config files
`config_biograph_fdg.json` and `config_uexp_fdg.json`.

Those config files contain the master and sub directories for the training and validation data.

For training, we assume that the preprocesesd data (resampled to 1.65mm isotropic voxel size, converted
to SUV units and log(1 + SUV) compressed) are save for every data set.

Every directory sub-directory should contain the files:
```
CASE_ID/
|-- 10
|   |-- resampled_1.65.nii.gz
|   `-- _sample.dcm
|-- 100
|   |-- resampled_1.65.nii.gz
|   `-- _sample.dcm
|-- 20
|   |-- resampled_1.65.nii.gz
|   `-- _sample.dcm
|-- 4
|   |-- resampled_1.65.nii.gz
|   `-- _sample.dcm
|-- 50
|   |-- resampled_1.65.nii.gz
|   `-- _sample.dcm
|-- ref
|   |-- resampled_1.65.nii.gz
|   `-- _sample.dcm
|-- new_sampling_map_1.65.nii.gz
`-- suv_factor.txt
```

where `10`, `20`, `4`, `50`, and `100` are subfolders the different dose reduction factors (DRF)
each containg the corresponding preprocessed PET image in NIfTI format. 
The `ref` folder contains the full-dose reference image.
`new_sampling_map_1.65.nii.gz` contains the new sampling map used for the patch sampler.
`suv_factor.txt` contains the SUV conversion factor used for the conversion from Bq/ml to SUV.

Utilities to create the and save the preprocessed data are provided in `preprocessing` subfolder.