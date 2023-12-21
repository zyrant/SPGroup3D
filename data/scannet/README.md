### Prepare ScanNet Data for Indoor Detection

Almost all steps are the same as the [mmdet3d](https://github.com/open-mmlab/mmdetection3d/blob/main/data/scannet/README.md), except step 2.


2. Install segmentator from this [repo](https://github.com/Karbo123/segmentator) (segmentator is used for superpoint generation), In this directory, extract point clouds and annotations by running ` python batch_load_scannet_data_superpoint.py`. DON'T add `--max_num_point 50000`.

The directory structure after pre-processing should be as below

```
scannet
├── meta_data
├── batch_load_scannet_data.py
├── load_scannet_data.py
├── scannet_utils.py
├── README.md
├── scans
├── scans_test
├── scannet_instance_data
├── points
│   ├── xxxxx.bin
├── superpoints
│   ├── xxxxx.bin
├── instance_mask
│   ├── xxxxx.bin
├── semantic_mask
│   ├── xxxxx.bin
├── seg_info
│   ├── train_label_weight.npy
│   ├── train_resampled_scene_idxs.npy
│   ├── val_label_weight.npy
│   ├── val_resampled_scene_idxs.npy
├── posed_images
│   ├── scenexxxx_xx
│   │   ├── xxxxxx.txt
│   │   ├── xxxxxx.jpg
│   │   ├── intrinsic.txt
├── scannet_infos_train.pkl
├── scannet_infos_val.pkl
├── scannet_infos_test.pkl

```
