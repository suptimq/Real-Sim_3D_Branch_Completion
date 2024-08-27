# Simulation-based Joint Branch Completion and Skeletonization Model

## Usage

### Requirements

Please follow the [PoinTr](https://github.com/yuxumin/PoinTr) repository to install the required packages.

### Dataset

We followed the PCN dataset to organize our simulated datasets (i.e., ***NB*** and ***FB*** branches). More details can be found in [DATASET.md](./DATASET.md).

### Pretrained Model

We pretrained the joint completion and skeletonization model on the PCN dataset. Please download the weights in [Google Drive](https://drive.google.com/file/d/14Dy-r6i2R83w0s0VbosIiyku_I-9D695/view?usp=sharing).

### Training and Inference

We provide a `ddp_ft.sh` and a `inference.sh` shell script as templates for the configuration. 

> Training has to be done on at least 2-GPUs. Otherwise, it would raise a DDP-related issue.

## Acknowledgements

Our code is inspired by [PoinTr](https://github.com/yuxumin/PoinTr).