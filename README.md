# DynFaceRestore: Balancing Fidelity and Quality in Diffusion-Guided Blind Face Restoration with Dynamic Blur-Level Mapping and Guidance

## Abstract
Blind Face Restoration aims to recover high-fidelity, detail-rich facial images from unknown degraded inputs, presenting significant challenges in preserving both identity and detail. Pre-trained diffusion models have been increasingly used as image priors to generate fine details. Still, existing methods often use fixed diffusion sampling timesteps and a global guidance scale, assuming uniform degradation. This limitation and potentially imperfect degradation kernel estimation frequently lead to under- or over-diffusion, resulting in an imbalance between fidelity and quality. We propose DynFaceRestore, a novel blind face restoration approach that learns to map any blindly degraded input to Gaussian blurry images. By leveraging these blurry images and their respective Gaussian kernels, we dynamically select the starting timesteps for each blurry image and apply closed-form guidance during the diffusion sampling process to maintain fidelity. Additionally, we introduce a dynamic guidance scaling adjuster that modulates the guidance strength across local regions, enhancing detail generation in complex areas while preserving structural fidelity in contours. This strategy effectively balances the trade-off between fidelity and quality. DynFaceRestore achieves state-of-the-art performance in both quantitative and qualitative evaluations, demonstrating robustness and effectiveness in blind face restoration.

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

## Acknowledgement

This project is based on [Improved Diffusion Model](https://github.com/openai/improved-diffusion) and [DifFace](https://github.com/zsyOAOA/DifFace).  Some codes are brought from [BasicSR](https://github.com/XPixelGroup/BasicSR) and [DiffBIR](https://github.com/XPixelGroup/DiffBIR). Thanks for their awesome works.

