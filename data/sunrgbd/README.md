### Prepare SUN RGB-D Data

Almost all steps are the same as the [mmdet3d](https://github.com/open-mmlab/mmdetection3d/blob/main/data/sunrgbd/README.md).

superpoint generatation
- VCCS
1. Install [pclpy](https://github.com/davidcaron/pclpy).


``` shell
# python 3.8.16
conda install -c conda-forge/label/gcc7 qhull
conda install -c conda-forge -c davidcaron pclpy
```
2. In this directory, extract superpoint run `python  slic_function_multi_processing.py`.

or

- SPG/SSP
1. You can use [SPG/SSP](https://github.com/loicland/superpoint_graph) to generate superpoints like s3dis.

In our experiment, we use VCCS.

```
python tools/create_data.py sunrgbd --root-path ./data/sunrgbd  --out-dir ./data/sunrgbd --extra-tag sunrgbd
```

The directory structure after pre-processing should be as below

```
sunrgbd
├── README.md
├── matlab
│   ├── extract_rgbd_data_v1.m
│   ├── extract_rgbd_data_v2.m
│   ├── extract_split.m
├── OFFICIAL_SUNRGBD
│   ├── SUNRGBD
│   ├── SUNRGBDMeta2DBB_v2.mat
│   ├── SNRGBDMeta3DBB_v2.mat
│   ├── SUNRGBDtoolbox
├── sunrgbd_trainval
│   ├── calib
│   ├── depth
│   ├── image
│   ├── label
│   ├── label_v1
│   ├── seg_label
│   ├── train_data_idx.txt
│   ├── val_data_idx.txt
├── points
├── superpoints
├── sunrgbd_infos_train.pkl
├── sunrgbd_infos_val.pkl

```
