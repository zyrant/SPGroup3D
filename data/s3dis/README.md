### Prepare S3DIS Data

Almost all steps are the same as the [mmdet3d](hhttps://github.com/open-mmlab/mmdetection3d/blob/main/data/s3dis/README.md).

# Superpoint Generatation

we use SSP/SPG to generate superpoints, please see https://github.com/loicland/superpoint_graph or https://github.com/fpthink/3D-WSIS/blob/main/data/S3DIS/S3DIS.md.

Then

```bash
python tools/create_data.py s3dis --root-path ./data/s3dis --out-dir ./data/s3dis --extra-tag s3dis
```

The directory structure after pre-processing should be as below

```
s3dis
├── meta_data
├── indoor3d_util.py
├── collect_indoor3d_data.py
├── README.md
├── Stanford3dDataset_v1.2_Aligned_Version
├── s3dis_data
├── points
│   ├── xxxxx.bin
├── superpoints
│   ├── xxxxx.bin
├── instance_mask
│   ├── xxxxx.bin
├── semantic_mask
│   ├── xxxxx.bin
├── seg_info
│   ├── Area_1_label_weight.npy
│   ├── Area_1_resampled_scene_idxs.npy
│   ├── Area_2_label_weight.npy
│   ├── Area_2_resampled_scene_idxs.npy
│   ├── Area_3_label_weight.npy
│   ├── Area_3_resampled_scene_idxs.npy
│   ├── Area_4_label_weight.npy
│   ├── Area_4_resampled_scene_idxs.npy
│   ├── Area_5_label_weight.npy
│   ├── Area_5_resampled_scene_idxs.npy
│   ├── Area_6_label_weight.npy
│   ├── Area_6_resampled_scene_idxs.npy
├── s3dis_infos_Area_1.pkl
├── s3dis_infos_Area_2.pkl
├── s3dis_infos_Area_3.pkl
├── s3dis_infos_Area_4.pkl
├── s3dis_infos_Area_5.pkl
├── s3dis_infos_Area_6.pkl

```
