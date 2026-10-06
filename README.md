# PaLoRA: Paced Low-Rank Adaptation for Continual Learning

This repository contains the official implementation of our NeurIPS 2026 paper:

**PaLoRA: Paced Low-Rank Adaptation for Continual Learning**
Yuxuan Li, Fanhu Zeng, Hao Tang
**NeurIPS 2026** | [Paper](https://arxiv.org/abs/2610.04226)

## Overview

How should the magnitude of LoRA updates be controlled as knowledge accumulates in continual learning?

Existing LoRA-based continual learning methods commonly rely on a **fixed small learning rate** to mitigate catastrophic forgetting. However, the amount of previously learned knowledge continuously grows over time, suggesting that a fixed restriction may be insufficient for long task sequences.

**PaLoRA** provides a rank-aware solution. We use the effective rank of accumulated updates to measure the growth of learned knowledge and progressively restrict new updates accordingly.

PaLoRA combines three key components:

- **Adaptive SVD compression** to estimate and compact the effective subspace of accumulated knowledge;
- **Null-space gradient projection** to reduce interference with previously learned directions;
- **Rank-aware pacing** to adaptively control the magnitude of new LoRA updates as the effective rank grows.

Our theoretical analysis motivates a rank-dependent pacing rule, while experiments on **CIFAR-100, ImageNet-R, and ImageNet-A** demonstrate consistent improvements, especially in challenging long-horizon settings with up to **50 sequential tasks**.

> **Key idea:** the strength of update restriction should increase as learned knowledge accumulates, rather than remaining fixed throughout continual learning.

<p align="center">   <img src="fig/PaLoRA_Framework.png" alt="PaLoRA Framework"> </p>

## Requirements

The implementation is based on **PyTorch**.

Our experiments were conducted with:

```text
Python      3.11.4
PyTorch     2.0.1
torchvision 0.15.2
timm        0.6.7
```

The code was tested on Linux with an **NVIDIA GeForce RTX 4080 SUPER** GPU.

We recommend creating a dedicated environment before installing the dependencies.

For example:

```bash
conda create -n palora python=3.11.4
conda activate palora

pip install torch==2.0.1 torchvision==0.15.2
pip install timm==0.6.7
```

## Dataset Preparation

Create a `data/` directory in the project root:

```
mkdir data
```

Then prepare the datasets as follows:

- **CIFAR-100**: The dataset will be downloaded automatically.
- **ImageNet-R**: Download it from [here](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar). After extracting the archive, place the dataset under the `data/` directory.
- **ImageNet-A**: Download it from [here](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/ERYi36eg9b1KkfEplgFTW3gBg1otwWwkQPSml0igWBC46A?e=NiTUkL). After extracting the archive, place the dataset under the `data/` directory.

The resulting directory structure should look similar to:

```
data/
├── imagenet-r/
└── imagenet-a/
```

## Reproducing the Experiments

The configuration files under `configs/palora/` contain the settings used in our experiments.

### CIFAR-100: 10 Tasks

```
python main.py --config configs/palora/c10_palora.json
```

### ImageNet-R: 20 Tasks

```
python main.py --config configs/palora/ir20_palora.json
```

### ImageNet-A: 50 Tasks

```
python main.py --config configs/palora/ia50_palora.json
```

## Acknowledgements

Our implementation is built upon the following excellent open-source projects. We sincerely thank the authors for making their code publicly available:

- [LoRA-Sub-DRS](https://github.com/scarlet0703/LoRA-Sub-DRS/)
- [InfLoRA](https://github.com/liangyanshuo/InfLoRA)

## Citation

If you find this work useful for your research, please consider citing our paper:

```
@misc{li2026palorapacedlowrankadaptation,
  title        = {PaLoRA: Paced Low-Rank Adaptation for Continual Learning},
  author       = {Yuxuan Li and Fanhu Zeng and Hao Tang},
  year         = {2026},
  eprint       = {2610.04226},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2610.04226}
}
```
