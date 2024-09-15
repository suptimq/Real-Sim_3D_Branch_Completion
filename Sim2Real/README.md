# Simulation-based Joint Branch Completion and Skeletonization Model

## Usage

### Requirements

1. Create conda environment
```shell
conda create -n sim_joint_branch python=3.9
```

2. Install PyTorch

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
```

3. Install libraries

```shell
# Torch Scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
# Others
pip install -r requirements.txt
```

Please follow the [PoinTr](https://github.com/yuxumin/PoinTr) repository to install the `Chamfer_Distance` and `EMD` extensions.

### Dataset

We followed the PCN dataset to organize our simulated datasets (i.e., ***NB*** and ***FB*** branches). More details can be found in [DATASET.md](./DATASET.md).

### Pretrained Model

We pretrained the joint completion and skeletonization model on the PCN dataset. Please download the weights in [Google Drive](https://drive.google.com/file/d/14Dy-r6i2R83w0s0VbosIiyku_I-9D695/view?usp=sharing).

### Training and Inference

We provide a `ddp_ft.sh` and a `inference.sh` shell script as templates for the configuration. 

> Training has to be done on at least 2-GPUs. Otherwise, it would raise a DDP-related issue.

## Acknowledgements

Our code is inspired by [PoinTr](https://github.com/yuxumin/PoinTr).