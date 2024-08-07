# [Neurips 2023] Addressing Negative Transfer in Diffusion Models.


<!-- Arxiv Link, Project Link -->
<a href="https://arxiv.org/abs/2306.00354"><img src="https://img.shields.io/badge/arXiv-2306.00354-b31b1b.svg"></a>
<a href="https://openreview.net/forum?id=3G2ec833mW"><img src="https://img.shields.io/badge/OpenReview-NeurIPS2023-orange"></a>
<a href="https://gohyojun15.github.io/ANT_diffusion/"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>

<!-- Teasure image: ANT_UW_DITL_output.jpg -->
![Alt text](asset/ANT_UW_DITL_output.jpg)
*Generated image from ANT-UW (DiT-L) with guidance scale 3.0*

This repository contains the official PyTorch implementation of the following paper: "Addressing Negative Transfer in Diffusion Models" (Neurips 2023). To gain a better understanding of the paper, please visit our [project page](https://gohyojun15.github.io/ANT_diffusion/) and paper on [arXiv](https://arxiv.org/abs/2306.00354).


This code is to train DiT model with ANT on ImageNet dataset.
Our implementation is based on [DiT](https://github.com/facebookresearch/DiT), [LibMTL](https://github.com/median-research-group/LibMTL), [NashMTL](https://github.com/AvivNavon/nash-mtl).



## Updates
- **2024.07.04**: Our project [Switch Diffusion Transformer](https://github.com/byeongjun-park/Switch-DiT) utilizing ANT-UW has been accepted in ECCV2024!.
- **2024.05.31**: Uncertainty weighting is utilized in [EDM2](https://github.com/NVlabs/edm2), a concept we had previously explored in our work. We believe that rethinking diffusion models as multi-task learners is an effective direction to improve these models!
- **2023.10.07**: [DTR](https://arxiv.org/abs/2310.07138) uses ANT-UW and shows that ANT-UW outperforms [P2weight](https://arxiv.org/abs/2204.00227) and [MinSNR](https://arxiv.org/abs/2303.09556).
    - Please check our new project ["Denoising Task Routing for Diffusion Models"](https://github.com/byeongjun-park/DTR)
- **2023.11.05**: Initial release.


## Install pre-requisites
```bash
$ pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
$ pip install -r requirements.txt
```


## Training DiT with ANT.
- ### You can train DiT-S model with ANT 
    ```
    python train_ant_single_gpu.py \
    --data_path [Your Data Path] \
    --total_clusters [Your cluster size (Default: 5)] \
    --mtl_method [uw, nash, or pcgrad]
    ```

- ### Train DiT-L model with ANT-UW using multiple-gpus.
    ```
    torchrun --nproc_per_node=[number of gpus] train_uw_multi_gpu.py \
    --data_path [Your Data Path] \
    --total_clusters [Your cluster size (Default: 5)]
    ```


## Sampling DiT trained with ANT.

```
torchrun --nproc_per_node=[number of gpus] sample_ddp.py \
--model_config [Your config path (config/DiT-L.yaml, config/DiT-S.yaml)] \
--ckpt [Your ckpt path] \
--sample-dir [Your sample dir] 
```

## Interval clustering
The example codes for interval clustering are shown in `interval_clustering.py`


## Model card ($k$=5)
All models can be downloaded from [OneDrive link](https://1drv.ms/f/s!Aj1gDFRo3D7Jge1MfwcTk0nqxfbMEA?e=RWn9KL)

| Model | FID | IS | Precision | Recall |
| :---: | :---: | :---: | :---: | :---: |
| DiT-L + ANT-UW (Multi-GPU) | 5.695 | 186.661 | 0.811 | 0.491 |
| DiT-S + Nash | 44.65 | 33.48 | 0.4209 | 0.5272 |
| DiT-S + UW | 48.40 | 30.84 | 0.4196 | 0.5172 |

<!-- 
DiT-S with ANT-UW.
DIT-S with PCGRAD
DIT-S with Nash -->
