# [ICCV 2025 - Hightlight] DynFaceRestore: Balancing Fidelity and Quality in Diffusion-Guided Blind Face Restoration with Dynamic Blur-Level Mapping and Guidance
This repository provides the official implementation of **DynFaceRestore**, a novel blind face restoration approach.  
The method dynamically adapts diffusion timesteps and guidance strength based on Gaussian-blurry priors, effectively balancing fidelity and detail. 

[📄 Read the Paper (arXiv)](https://arxiv.org/abs/2507.13797)  

## Ours Results + Checkpoints
[Google Drive (Download here)](https://drive.google.com/drive/folders/1bSC9s8p6SaWs8Q-zNLjo9XSlNm42204Y?usp=sharing)

## Requirements
A suitable [conda](https://conda.io/) environment named `DynFaceRestore` can be created and activated with:

```
conda env create -f environment.yaml
conda activate DynFaceRestore
```

## Inference
#### :boy: Blind Face Restoration
```
torchrun --nproc_per_node=8 inference_difface.py -i [image folder/image path] -i_hq [image folder/image path] --bs [GPU_NUM] -o [result folder] --task restoration --eta 0.5 --aligned --use_fp16
```
Note that the hyper-parameter eta controls the fidelity-realness trade-off. We use std/10 as eta in sample-wise manner. One can command the line of dynamic eta in sampler.py and use command line to assign eta.

## Training
#### :turtle: Train DBLM
```
cd DiffBIR
accelerate launch train_SDE.py --config configs/train/train_SDE.yaml
accelerate launch train_DBLM.py --config configs/train/train_DBLM_TM.yaml
accelerate launch train_DBLM.py --config configs/train/train_DBLM_RM.yaml
```
#### :dolphin: Make DSST
1. 
```
cd ..
cd models
```
2. Run make_DSST.ipynb
#### :whale: Train DGSA with 8 GPUS
##### Configuration
1. Modify the data path in data.train and data.val according to your own settings. 
2. Adjust the batch size based on your GPU devices.
    * train.batchsize: [A, B]    # A denotes the batch size for training,  B denotes the batch size for validation
    * train.microbatch: C        # C denotes the batch size on each GPU, A = C * num_gpus * num_grad_accumulation
##### Command
```
torchrun --nproc_per_node=8 main.py --cfg_path configs/training/uncertainty.yaml --save_dir [Logging Folder]  
```

<!-- ## Citation
If you find this work useful in your research, please consider citing:

```bibtex
@article{dynfacerestore2025,
  title   = {DynFaceRestore: Balancing Fidelity and Quality in Diffusion-Guided Blind Face Restoration with Dynamic Blur-Level Mapping and Guidance},
  author  = {Your Name and Co-authors},
  journal = {To appear},
  year    = {2025}
}
``` -->

## Acknowledgement

This project is based on [Improved Diffusion Model](https://github.com/openai/improved-diffusion) and [DifFace](https://github.com/zsyOAOA/DifFace).  Some codes are brought from [BasicSR](https://github.com/XPixelGroup/BasicSR) and [DiffBIR](https://github.com/XPixelGroup/DiffBIR). Thanks for their awesome works.

