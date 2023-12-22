## SPGroup3D

<p align="center"><img src="./images/SPGroup3D.png" alt="drawing" width="90%"/></p>

This project provides the code and results for 'SPGroup3D: Superpoint Grouping Network for Indoor 3D Object Detection', AAAI2024.

Anchors: [Yun Zhu](https://github.com/zyrant), [Le Hui](https://fpthink.github.io/), [Yaqi Shen](https://github.com/supersyq), [Jin Xie*](https://csjinxie.github.io/)

PaperLink: https://arxiv.org/abs/2312.13641


### Introduction
> Current 3D object detection methods for indoor scenes mainly follow the voting-and-grouping strategy to generate proposals. However, most methods utilize instance-agnostic groupings, such as ball query, leading to inconsistent semantic information and inaccurate regression of the proposals. To this end, we propose a novel superpoint grouping network for indoor anchor-free one-stage 3D object detection. Specifically, we first adopt an unsupervised manner to partition raw point clouds into superpoints, areas with semantic consistency and spatial similarity. Then, we design a geometry-aware voting module that adapts to the centerness in anchor-free detection by constraining the spatial relationship between superpoints and object centers. Next, we present a superpoint-based grouping module to explore the consistent representation within proposals. This module includes a superpoint attention layer to learn feature interaction between neighboring superpoints, and a superpoint-voxel fusion layer to propagate the superpoint-level information to the voxel level. Finally, we employ effective multiple matching to capitalize on the dynamic receptive fields of proposals based on superpoints during the training.  Experimental results demonstrate our method achieves state-of-the-art performance on ScanNet V2, SUN RGB-D, and S3DIS datasets in the indoor one-stage 3D object detection. 

### Preparation

- For installing the environment, we mainly follow [TR3D](https://github.com/SamsungLabs/tr3d). Besides, we also provide [our_env.yaml](our_env.yaml) for details check.

- Alternatively, you can install all required packages manually. This implementation is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) framework. Please refer to the installation guide [getting_started.md](docs/en/getting_started.md), including MinkowskiEngine installation.

```
# If you can not install MinkowskiEngine with pip successfully,
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas=openblas --force_cuda
```
- Install [torch_scatter](https://github.com/rusty1s/pytorch_scatter).

```
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.9+${CUDA}.html
```

- All the `SPGroup3D`-related code locates in the folder [projects](projects).


### Data Preparation

- We mainly follow the mmdetection3d data preparation protocol described in [scannet](data/scannet/README.md), [sunrgbd](data/sunrgbd/README.md), and [s3dis](data/s3dis/README.md), including superpoint generation.

- Since superpoint generation needs many dependencies. We recommend you to use the processed superpoint in [GoogleDrive](https://drive.google.com/drive/folders/1uj4Y5HgWaf3cpYrQ0-4pBAB0sMc2mNnQ?usp=sharing) / [BaiduDrive](https://pan.baidu.com/s/1AvOWuXQACEoK2NYc9fprMA?pwd=x52a) and you just need to re-generate `{}.pkl` of different datasets. 

- Please DON'T do any sampling operation in Data preparation, otherwise it will result in a mismatch with the superpoint we provide.

### Training

To start training, run [train](tools/train.py) with SPGroup3D [configs](projects/configs):

```shell
# Remember to modify the data_root
# scannet v2
CUDA_VISIBLE_DEVICE={} bash tools/dist_train.sh projects/configs/SPGroup_scannet.py 4 --work-dir work_dirs/{YOUR PATH}
# sunrgbd
CUDA_VISIBLE_DEVICE={} bash tools/dist_train.sh projects/configs/SPGroup_sunrgbd.py 4 --work-dir work_dirs/{YOUR PATH}
# s3dis
CUDA_VISIBLE_DEVICE={} bash tools/dist_train.sh projects/configs/SPGroup_s3dis.py 4 --work-dir work_dirs/{YOUR PATH}
```

### Testing

Test pre-trained model using [test](tools/dist_test.sh) with SPGroup3D [configs](projects/configs):
```shell
# scannet v2
python tools/test.py projects/configs/SPGroup_scannet.py \
    work_dirs/{YOUR PATH}.pth --eval mAP
# sunrgbd
python tools/test.py projects/configs/SPGroup_sunrgbd.py \
    work_dirs/{YOUR PATH}.pth --eval mAP
# s3dis
python tools/test.py projects/configs/SPGroup_s3dis.py \
    work_dirs/{YOUR PATH}.pth --eval mAP
```

### Visualization

Visualizations can be created with [test](tools/test.py) script. 
For better visualizations, you may set `score_thr` in configs to `0.25`:
```shell
# scannet v2
python tools/test.py projects/configs/SPGroup_scannet.py \
    work_dirs/{YOUR PATH} --eval mAP --show \
    --show-dir work_dirs/{YOUR PATH}
```

### Main Results
All models are trained with 4 3090 GPUs. 

| Dataset | mAP@0.25 | mAP@0.5 | Download | config |
|:-------:|:--------:|:-------:|:--------:|:--------:|
| ScanNet V2 | 74.3 (73.5) | 59.6 (58.3) | [GoogleDrive](https://drive.google.com/drive/folders/150aEBttNKodR7z63mI5h50raA1LyQrQ3?usp=sharing) / [BaiduDrive](https://pan.baidu.com/s/1S6q2Atu1AlT55n12ZAIjlQ?pwd=34x2) | [config](projects/configs/SPGroup_scannet.py) |
| SUN RGB-D | 65.4 (64.8) | 47.1 (46.4)| [GoogleDrive](https://drive.google.com/drive/folders/1wxQ7ZVp1WqsXUfX8_lNsYLUAFfxzPRgw?usp=sharing) / [BaiduDrive](https://pan.baidu.com/s/1rCKgQyuo5e9kHNZERwe1Lg?pwd=8mdv)|[config](projects/configs/SPGroup_sunrgbd.py) |
| S3DIS | 69.2 (67.7) | 47.2 (43.6) | [GoogleDrive](https://drive.google.com/drive/folders/1QK9sJj3PzEJvEtNBWW-2GsXuySgwYRPy?usp=sharing) / [BaiduDrive](https://pan.baidu.com/s/15FXU2H2UB3cAnjz-xt6XKg?pwd=7jn4) | [config](projects/configs/SPGroup_s3dis.py)|

Due to the size of these datasets and the randomness that inevitably exists in the model,  the results on these datasets fluctuate significantly. It's normal for results to fluctuate within a range.

### Citation

If you find this work useful for your research, please cite our paper:

```
@inproceedings{zhu2024spgroup,
  title={SPGroup3D: Superpoint Grouping Network for Indoor 3D Object Detection},
  author={Yun Zhu, Le Hui, Yaqi Shen, Jin Xie},
  booktitle={AAAI},
  year={2024}
}
```

### Acknowledgments

This project is based on the following codebases.
- [mmdetection3D](https://github.com/open-mmlab/mmdetection3d)
- [TR3D](https://github.com/SamsungLabs/tr3d)
- [CAGroup3D](https://github.com/Haiyang-W/CAGroup3D)
- [3D-WSIS](https://github.com/fpthink/3D-WSIS)
- [segmentator](https://github.com/Karbo123/segmentator)
- [SPG/SSP](https://github.com/loicland/superpoint_graph)
- [pclpy](https://github.com/davidcaron/pclpy)
 
If you find this project helpful, Please also cite the codebases above. Thanks.
