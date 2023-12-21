plugin = True
plugin_dir = "projects/spgroup/"
voxel_size = .02

model = dict(
    type='SPMinkSingleStage3DDetector',
    voxel_size=voxel_size,
    backbone=dict(
        type='BiResNet',
        in_channels = 3,
        out_channels = 64),
    voxel_encoder=dict(
        type='SSG', 
        in_channels=64,
        local_k = 8,
        voxel_size = voxel_size,
        latter_voxel_size = voxel_size * 2,
        feat_channels = (64, 128, 128),
        with_xyz = True,
        with_distance=False, # No used
        with_cluster_center=False, # No used
        with_superpoint_center = True,
        mode='max'),
    head=dict(
        type='SPHead',
        pred_layer_cfg=dict(
            in_channels = 390, 
            cls_linear_channels=(256, 256),
            reg_linear_channels=(256, 256),
            center_linear_channels=(256, 256)
            ),
        norm_cfg=dict(type='LN', eps=1e-3),
        n_reg_outs = 8,
        n_classes = 10,
        with_yaw = True,
        center_type = 'pow',
        pts_threshold = 18,
        center_loss=dict(type='CrossEntropyLoss', use_sigmoid=True),
        bbox_loss=dict(type='RotatedIoU3DLoss', mode='diou'),
        cls_loss=dict(type='FocalLoss'),
        vote_loss= dict(type='SmoothL1Loss', beta=0.04, reduction='sum', loss_weight=1),
        cls_cost=dict(type='FocalLossCost', weight=1),
        reg_cost=dict(type='IoU3DCost', iou_mode='diou', weight=1),
        # reg_cost=dict(type='BBox3DL1Cost', weight=1) 
        ),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=0.5, score_thr=0.01))


# dataset settings
dataset_type = 'SPSUNRGBDDataset'
data_root = '../all_data/sunrgbd/'
class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')
n_points = 100000

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadSuperPointsFromFile'), 
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True),
    dict(type='SPPointSample', num_points=n_points),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-.523599, .523599],
        scale_ratio_range=[.85, 1.15],
        translation_std=[.1, .1, .1],
        shift_height=False),
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type='SPDefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'superpoints', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadSuperPointsFromFile'), 
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        # pcd_horizontal_flip = True,
        transforms=[
            # dict(
            #     type='GlobalRotScaleTrans',
            #     rot_range=[0, 0],
            #     scale_ratio_range=[1., 1.],
            #     translation_std=[0, 0, 0]),
            # dict(
            #     type='RandomFlip3D',
            #     sync_2d=False,
            #     flip_ratio_bev_horizontal=0.5),
            dict(type='SPPointSample', num_points=n_points),
            # dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='SPDefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'superpoints'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            modality=dict(use_camera=False, use_lidar=True),
            data_root=data_root,
            ann_file=data_root + 'sunrgbd_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=True,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))



optimizer = dict(type='AdamW', lr=.001, weight_decay=.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[9, 12])
runner = dict(type='EpochBasedRunner', max_epochs=15)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

checkpoint_config = dict(interval=1, max_keep_ckpts=10)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

