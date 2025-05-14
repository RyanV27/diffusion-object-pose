<div align="center">

# [Using Vision Transformers to Improve the Aggregation of Diffusion Features for Object Pose Estimation]

</div>

<div align="justify">

</div>

## Installation
<details><summary>Click to expand</summary>

### 1. Clone this repo.
```
git clone https://github.com/RyanV27/diffusion-object-pose.git
```
### 2. Install environments.
```
conda env create -f environment.yaml
conda activate diff-feats
```
</details>

## Data Preparation

<details><summary>Click to expand</summary>

### Final structure of folder dataset
```bash
./dataset
    ├── linemod 
        ├── models
        ├── opencv_pose
        ├── LINEMOD
        ├── occlusionLINEMOD
    ├── templates	
        ├── linemod
            ├── train
            ├── test
    ├── LINEMOD.json # query-template pairwise for LINEMOD
    ├── occlusionLINEMOD.json # query-template pairwise for Occlusion-LINEMOD
    └── crop_image512 # pre-cropped images for LINEMOD
```

### 1. Download datasets:
Download with following gdrive links and unzip them in ./dataset. I use the same data as [template-pose](https://github.com/nv-nguyen/template-pose).
- [LINEMOD and Occlusion-LINEMOD (3GB)](https://drive.google.com/file/d/1XkQBt01nlfCbFuBsPMfSHlcNIzShn7e7/view?usp=sharing)

### 2. Process ground-truth poses
Convert the coordinate system to [BOP datasets format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md) and save GT poses of each object separately:
```bash
python -m data.process_gt_linemod
```
### 3. Render templates
To render templates:
```bash
python -m data.render_templates --dataset linemod --disable_output --num_workers 4
```
### 4. Crop images (only for LINEMOD)
Crop images of LINEMOD, OcclusionLINEMOD and its templates with GT poses:
```bash
python -m data.crop_image_linemod
```
### 5. Compute neighbors with GT poses
```bash
python -m data.create_dataframe_linemod
```

</details>

## Launch a training

<details><summary>Click to expand</summary>

### 1. Launch a training on LINEMOD
```bash
python train_linemod.py --config_path config_run/LM_Diffusion_$split_name.json
```

### 2. Launch a training on T-LESS
```bash
python train_tless.py --config_path ./config_run/TLESS_Diffusion.json
```

</details>

## Reproduce the results
<details><summary>Click to expand</summary>

### 1. Download checkpoints
You can download it from this [link](https://drive.google.com/drive/folders/1CVyW7IDAZ0uGZSJIoN3ARRyP_wY2Ntk9?usp=sharing).

### 2. Reproduce the results on LINEMOD
```bash
python test_linemod.py --config_path config_run/LM_Diffusion_$split_name.json --checkpoint checkpoint_path
```
</details>
