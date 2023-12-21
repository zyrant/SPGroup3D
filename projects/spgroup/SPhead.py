# Copyright (c) OpenMMLab. All rights reserved.
# Modify from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/dense_heads/fcaf3d_neck_with_head.py # noqa
# and https://github.com/Haiyang-W/CAGroup3D/blob/main/pcdet/models/dense_heads/cagroup_head.py
# by --zyrant

import torch
from mmcv.ops import nms3d, nms3d_normal
from mmcv.runner.base_module import BaseModule
from torch import nn
from torch_scatter import scatter_mean
import torch.nn.functional as F

from mmcv.cnn import Scale
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.models import HEADS, build_loss
from mmdet.core import reduce_mean
from mmdet3d.ops import knn
from mmcv.cnn import build_norm_layer
from mmdet.core.bbox.match_costs import build_match_cost
from mmcv.cnn import bias_init_with_prob

@HEADS.register_module()
class SPHead(BaseModule):
    def __init__(self,
                 n_classes,
                 n_reg_outs,
                 with_yaw =False,
                 pts_threshold=18,
                 norm_cfg=dict(type='LN', eps=1e-3),
                 center_type = 'pow',
                 pred_layer_cfg=dict(
                    in_channels=256,
                    cls_linear_channels=(256, 256),
                    reg_linear_channels=(256, 256),
                    center_linear_channels=(64, 64),),
                 dropout_ratio = 0,   
                 center_loss=dict(type='CrossEntropyLoss', use_sigmoid=True),
                 bbox_loss=dict(type='AxisAlignedIoULoss'),
                 cls_loss=dict(type='FocalLoss'),
                 vote_loss = dict(
                     type='SmoothL1Loss', reduction='sum'),
                 cls_cost=dict(type='FocalLossCost', weight=2),
                 reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(SPHead, self).__init__(init_cfg)
        self.pts_threshold = pts_threshold
        self.center_type = center_type
        
        self.with_yaw = with_yaw 
        if self.with_yaw:
            self.gt_per_seed = 1

        self.center_loss = build_loss(center_loss)
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.vote_loss = build_loss(vote_loss)

        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.conv_center = self._make_fc_layers(
            fc_cfg=pred_layer_cfg.center_linear_channels,
            input_channels=pred_layer_cfg.in_channels,
            output_channels=1, 
            norm_cfg=norm_cfg, 
            dp_ratio = dropout_ratio)
        self.conv_reg = self._make_fc_layers(
            fc_cfg=pred_layer_cfg.reg_linear_channels,
            input_channels=pred_layer_cfg.in_channels,
            output_channels=n_reg_outs, 
            norm_cfg=norm_cfg, 
            dp_ratio = dropout_ratio)
        self.conv_cls = self._make_fc_layers(
            fc_cfg=pred_layer_cfg.cls_linear_channels,
            input_channels=pred_layer_cfg.in_channels,
            output_channels=n_classes, 
            norm_cfg=norm_cfg, 
            dp_ratio = dropout_ratio)
        
        self.num_classes = n_classes

    def _make_fc_layers(self, fc_cfg: dict, input_channels: int,
                        output_channels: int, norm_cfg, dp_ratio=0) -> nn.Sequential:
        """Make fully connect layers.

        Args:
            fc_cfg (dict): Config of fully connect.
            input_channels (int): Input channels for fc_layers.
            output_channels (int): Input channels for fc_layers.

        Returns:
            nn.Sequential: Fully connect layers.
        """
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            norm_name, norm_layer = build_norm_layer(norm_cfg, fc_cfg[k])
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                norm_layer,
                nn.ReLU(inplace=True),
            ])
            c_in = fc_cfg[k]
            if k != fc_cfg.__len__() - 1 and dp_ratio > 0:
                fc_layers.extend([nn.Dropout(dp_ratio)])
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)
    

    def _forward_single(self, feats, coors):
        """
        Forward pass per level.
        """
        center_pred = self.conv_center(feats)
        reg_final = self.conv_reg(feats) 

        reg_distance = torch.exp(reg_final[:, 3:6])  
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle), dim=1)

        cls_pred = self.conv_cls(feats)

        batch_size = coors[:, 0].max() + 1

        center_preds, bbox_preds, cls_preds, points = [], [], [], []
        for batch_ids in range(batch_size.int()): 
            batch_mask = coors[:,0].int()==batch_ids
            center_preds.append(center_pred[batch_mask])
            bbox_preds.append(bbox_pred[batch_mask]) 
            cls_preds.append(cls_pred[batch_mask])
            points.append(coors[batch_mask][:, 1:])

        return center_preds, bbox_preds, cls_preds, points

    def forward(self, feats_dict):
        """Forward pass.

        Args:
            x (list[Tensor]): Features from the backbone.

        Returns:
            list[list[Tensor]]: Predictions of the head.
        """
        feats = [feats_dict['voxel_feats']]
        coors = [feats_dict['voxel_coods']]
        vote_offsets = [feats_dict['vote_offsets']]
        vote_voxel_points = [feats_dict['vote_voxel_points']]
        orgin_superpoints = [feats_dict['orgin_superpoints']]

        center_preds, bbox_preds, cls_preds, points = [], [], [], []
        for i in range(len(feats)): 
            center_pred, bbox_pred, cls_pred, point = \
                self._forward_single(feats[i], coors[i])
            center_preds.append(center_pred)
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)
        return center_preds, bbox_preds, cls_preds, points, vote_offsets, vote_voxel_points, orgin_superpoints


    def forward_train(self, x, original_points, pts_semantic_mask, pts_instance_mask, gt_bboxes, gt_labels, img_metas):
        """Forward pass of the train stage.

        Args:
            x (list[SparseTensor]): Features from the backbone.
            gt_bboxes (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                bboxes of each sample.
            gt_labels(list[torch.Tensor]): Labels of each sample.
            img_metas (list[dict]): Contains scene meta info for each sample.

        Returns:
            dict: Centerness, bbox and classification loss values.
        """
        center_preds, bbox_preds, cls_preds, points, vote_offsets, vote_voxel_points, orgin_superpoints = self(x)
        return self._loss(center_preds, bbox_preds, cls_preds, points, vote_offsets, vote_voxel_points, orgin_superpoints, original_points,
                          pts_semantic_mask, pts_instance_mask, gt_bboxes, gt_labels, img_metas)

    def forward_test(self, x, img_metas):
        """Forward pass of the test stage.

        Args:
            x (list[SparseTensor]): Features from the backbone.
            img_metas (list[dict]): Contains scene meta info for each sample.

        Returns:
            list[list[Tensor]]: bboxes, scores and labels for each sample.
        """
        center_preds, bbox_preds, cls_preds, points, vote_offsets, vote_voxel_points, orgin_superpoints = self(x)
        return self._get_bboxes(center_preds, bbox_preds, cls_preds, points,
                                img_metas)


    def _loss_single(self, center_preds, bbox_preds, cls_preds, points,
                     vote_offsets, vote_voxel_points, orgin_superpoints, original_points,
                     pts_semantic_mask, pts_instance_mask, 
                     gt_bboxes, gt_labels, img_meta):
        """Per scene loss function.

        Args:
            center_preds (list[Tensor]): Centerness predictions for all levels.
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            points (list[Tensor]): Final location coordinates for all levels.
            gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.
            img_meta (dict): Scene meta info.

        Returns:
            tuple[Tensor]: Centerness, bbox, and classification loss values.
        """

        # CAGroup
        vote_offsets = torch.cat(vote_offsets)
        vote_targets, vote_mask  = self.get_voxel_vote_targets(vote_voxel_points,  
                                                                original_points,  
                                                                pts_instance_mask, 
                                                                pts_semantic_mask, 
                                                                gt_bboxes) 

        vote_mask = (vote_mask.float() / torch.ones_like(vote_mask).float().sum() + 1e-6).unsqueeze(1).repeat(1, 3)
        vote_loss = self.vote_loss(
            vote_offsets,
            vote_targets,
            weight = vote_mask)

        center_preds = torch.cat(center_preds)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        center_targets, bbox_targets, cls_targets = self._get_targets(
            original_points, points, center_preds, bbox_preds, cls_preds, gt_bboxes, gt_labels)

        # only compute forgrounds
        # cls loss
        pos_inds = torch.nonzero(cls_targets >= 0).squeeze(1)
        n_pos = points.new_tensor(len(pos_inds))
        n_pos = max(reduce_mean(n_pos), 1.)
        cls_loss = self.cls_loss(cls_preds, cls_targets, avg_factor=n_pos)

        # bbox and centerness losses
        pos_center_preds = center_preds[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds] # xyzwhl
        pos_center_targets = center_targets[pos_inds].unsqueeze(1)
        pos_bbox_targets = bbox_targets[pos_inds] # xyzwhl
        # reduce_mean is outside if / else block to prevent deadlock
        center_denorm = max(
            reduce_mean(pos_center_targets.sum().detach()), 1e-6)
        if len(pos_inds) > 0:
            pos_points = points[pos_inds]
            center_loss = self.center_loss(
                pos_center_preds, pos_center_targets, avg_factor=n_pos)
            pos_bboxes = self._bbox_pred_to_bbox(pos_points, pos_bbox_preds) # dxmin, dxmax, dymin,dymax,dzmin,dzmax
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(pos_bboxes),
                self._bbox_to_loss(pos_bbox_targets),
                weight=pos_center_targets.squeeze(1),
                avg_factor=center_denorm)
        else:
            center_loss = pos_center_preds.sum()
            bbox_loss = pos_bbox_preds.sum()
        return center_loss, bbox_loss, cls_loss, vote_loss

    def _loss(self, center_preds, bbox_preds, cls_preds, 
              points, vote_offsets, vote_voxel_points, orgin_superpoints,  original_points, 
              pts_semantic_mask, pts_instance_mask, gt_bboxes, gt_labels, img_metas):
        """Per scene loss function.

        Args:
            center_preds (list[list[Tensor]]): Centerness predictions for
                all scenes.
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes.
            points (list[list[Tensor]]): Final location coordinates for all
                scenes.
            gt_bboxes (list[BaseInstance3DBoxes]): Ground truth boxes for all
                scenes.
            gt_labels (list[Tensor]): Ground truth labels for all scenes.
            img_metas (list[dict]): Meta infos for all scenes.

        Returns:
            dict: Centerness, bbox, and classification loss values.
        """
        if pts_semantic_mask is None:
            pts_semantic_mask = [None for _ in range(len(center_preds[0]))]
            pts_instance_mask = pts_semantic_mask

        center_losses, bbox_losses, cls_losses, vote_losses = [], [], [], []

        for i in range(len(img_metas)):
            center_loss, bbox_loss, cls_loss,  vote_loss = self._loss_single(
                center_preds=[x[i] for x in center_preds],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                vote_offsets=[x[i] for x in vote_offsets],
                vote_voxel_points=[x[i] for x in vote_voxel_points],
                orgin_superpoints=[x[i] for x in orgin_superpoints],
                original_points=[x[i] for x in original_points],
                pts_semantic_mask = pts_semantic_mask[i],
                pts_instance_mask = pts_instance_mask[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i],
                img_meta=img_metas[i],)
            center_losses.append(center_loss)
            bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
            vote_losses.append(vote_loss)
        return dict(
            center_loss=torch.mean(torch.stack(center_losses)),
            bbox_loss=torch.mean(torch.stack(bbox_losses)),
            cls_loss=torch.mean(torch.stack(cls_losses)),
            vote_loss=torch.mean(torch.stack(vote_losses))
            )

    def _get_bboxes_single(self, center_preds, bbox_preds, cls_preds, points,
                           img_meta):
        """Generate boxes for a single scene.

        Args:
            center_preds (list[Tensor]): Centerness predictions for all levels.
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            points (list[Tensor]): Final location coordinates for all levels.
            img_meta (dict): Scene meta info.

        Returns:
            tuple[Tensor]: Predicted bounding boxes, scores and labels.
        """
        mlvl_bboxes, mlvl_scores = [], []
        for center_pred, bbox_pred, cls_pred, point in zip(
                center_preds, bbox_preds, cls_preds, points):
            scores = cls_pred.sigmoid() * center_pred.sigmoid()
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                point = point[ids]

            bboxes = self._bbox_pred_to_bbox(point, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        bboxes, scores, labels = self._single_scene_multiclass_nms(
            bboxes, scores, img_meta)
        return bboxes, scores, labels

    def _get_bboxes(self, center_preds, bbox_preds, cls_preds, points,
                    img_metas):
        """Generate boxes for all scenes.

        Args:
            center_preds (list[list[Tensor]]): Centerness predictions for
                all scenes.
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes.
            points (list[list[Tensor]]): Final location coordinates for all
                scenes.
            img_metas (list[dict]): Meta infos for all scenes.

        Returns:
            list[tuple[Tensor]]: Predicted bboxes, scores, and labels for
                all scenes.
        """
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                center_preds=[x[i] for x in center_preds],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results

    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.

        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).

        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.

        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).

        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2 
        y_center = points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2
        z_center = points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2

        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max -> x, y, z, w, l, h
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 0] + bbox_pred[:, 1],
            bbox_pred[:, 2] + bbox_pred[:, 3],
            bbox_pred[:, 4] + bbox_pred[:, 5],
        ], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 0] + bbox_pred[:, 1] + \
            bbox_pred[:, 2] + bbox_pred[:, 3]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)
    
    @staticmethod
    def _get_face_distances(points, boxes):
        """Calculate distances from point to box faces.

        Args:
            points (Tensor): Final locations of shape (N_points, N_boxes, 3).
            boxes (Tensor): 3D boxes of shape (N_points, N_boxes, 7)

        Returns:
            Tensor: Face distances of shape (N_points, N_boxes, 6),
                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).
        """
        shift = torch.stack(
            (points[..., 0] - boxes[..., 0], points[..., 1] - boxes[..., 1],
             points[..., 2] - boxes[..., 2]),
            dim=-1).permute(1, 0, 2)
        shift = rotation_3d_in_axis(
            shift, -boxes[0, :, 6], axis=2).permute(1, 0, 2)
        centers = boxes[..., :3] + shift
        dx_min = centers[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
        dx_max = boxes[..., 0] + boxes[..., 3] / 2 - centers[..., 0]
        dy_min = centers[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
        dy_max = boxes[..., 1] + boxes[..., 4] / 2 - centers[..., 1]
        dz_min = centers[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
        dz_max = boxes[..., 2] + boxes[..., 5] / 2 - centers[..., 2]
        return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max),
                           dim=-1)

    @staticmethod
    def _get_centerness(face_distances):
        """Compute point centerness w.r.t containing box.

        Args:
            face_distances (Tensor): Face distances of shape (B, N, 6),
                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).

        Returns:
            Tensor: Centerness of shape (B, N).
        """
        x_dims = face_distances[..., [0, 1]]
        y_dims = face_distances[..., [2, 3]]
        z_dims = face_distances[..., [4, 5]]
        centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
            y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
            z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
        return centerness_targets


    @torch.no_grad()
    def _get_targets(self, original_points, points, center_preds, bbox_preds, cls_preds, gt_bboxes, gt_labels):
        """Compute targets for final locations for a single scene.

        Args:
            points (list[Tensor]): Final locations for all levels.
            gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.

        Returns:
            Tensor: Centerness targets for all locations.
            Tensor: Bbox targets for all locations.
            Tensor: Classification targets for all locations.
        """
        float_max = points.new_tensor(1e8)
        gt_bboxes = gt_bboxes.to(points.device)
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.unsqueeze(0).expand(n_points, n_boxes)

        # condition 1: point inside box
        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
                          dim=1)
        boxes = boxes.expand(n_points, n_boxes, 7)
        points_for_mask = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        face_distances = self._get_face_distances(points_for_mask, boxes)
        inside_box_condition = face_distances.min(dim=-1).values > 0
        
        # coundition 2: Select the best samples in the bbox according to loss
        bbox_preds = self._bbox_pred_to_bbox(points, bbox_preds)
        boxes_for_cost = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
                          dim=1)
        boxes_for_cost = boxes_for_cost[:, :6] 
        bbox_preds_for_cost = bbox_preds[:, :6] 
        reg_cost = self.reg_cost(self._bbox_to_loss(bbox_preds_for_cost), self._bbox_to_loss(boxes_for_cost)) # IOU version 
        # original_points = torch.cat(original_points)
        # reg_cost = self.reg_cost(original_points[:,:3], bbox_preds, boxes_for_cost) # L1 version 
        cls_cost = self.cls_cost(cls_preds, gt_labels)
        cost = reg_cost + cls_cost
        cost = torch.where(inside_box_condition, cost, float_max)
        top_cost = torch.topk(
            cost,
            min(self.pts_threshold + 1, len(cost)),
            dim=0,
            largest=False).values[-1]
        topk_condition = cost < top_cost.unsqueeze(0)

        # centerness
        centerness = self._get_centerness(face_distances)
        if self.center_type == 'pow':
            centerness = torch.pow(centerness , 1 / 3)
        elif self.center_type == 'sqrt':
            centerness = torch.sqrt(centerness)
        else:
            raise NotImplementedError
        
        centerness = torch.where(inside_box_condition, centerness,
                                 torch.ones_like(centerness) * -1)
        centerness = torch.where(topk_condition, centerness,
                                 torch.ones_like(centerness) * -1)
        
        # condition 4: min volume box per point
        volumes = torch.where(inside_box_condition, volumes, float_max)
        volumes = torch.where(topk_condition, volumes, float_max)
        min_volumes, min_inds = volumes.min(dim=1)

        center_targets = centerness[torch.arange(n_points), min_inds]
        bbox_targets = boxes[torch.arange(n_points), min_inds]
        if not gt_bboxes.with_yaw:
            bbox_targets = bbox_targets[:, :-1]
        cls_targets = gt_labels[min_inds]
        cls_targets = torch.where(min_volumes == float_max, -1, cls_targets)

        return center_targets, bbox_targets, cls_targets
    
    
    @torch.no_grad()
    def get_voxel_vote_targets(self, voxel_points, scene_points, pts_instance_mask, pts_semantic_mask, gt_bboxes_3d):

        '''
        https://github.com/Haiyang-W/CAGroup3D/blob/main/pcdet/models/dense_heads/cagroup_head.py
        '''

        voxel_points = torch.cat(voxel_points)
        scene_points = torch.cat(scene_points)
        gt_bboxes = torch.cat((gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
                          dim=1)
        if self.with_yaw:
            num_points = voxel_points.shape[0]
            num_boxes = gt_bboxes.shape[0]
            vote_targets = voxel_points.new_zeros([num_points, 3 * self.gt_per_seed])
            vote_target_masks = voxel_points.new_zeros([num_points],
                                                dtype=torch.long)
            vote_target_idx = voxel_points.new_zeros([num_points], dtype=torch.long)
            gt_bboxes_insidemask = gt_bboxes.expand(num_points, num_boxes, 7).to(voxel_points.device)
            voxel_points_insidemask = voxel_points.unsqueeze(1).expand(num_points, num_boxes, 3)
            box_indices_all = self._get_face_distances(voxel_points_insidemask, gt_bboxes_insidemask) # n_points, n_boxes
            box_indices_all = box_indices_all.min(dim=-1).values > 0
            
            for i in range(gt_bboxes.shape[0]):
                box_indices = box_indices_all[:, i]
                indices = torch.nonzero(
                    box_indices, as_tuple=False).squeeze(-1)
                selected_points = voxel_points[indices]
                vote_target_masks[indices] = 1
                vote_targets_tmp = vote_targets[indices]
                votes = gt_bboxes[i, :3].unsqueeze(
                    0).to(selected_points.device) - selected_points[:, :3]

                for j in range(self.gt_per_seed):
                    column_indices = torch.nonzero(
                        vote_target_idx[indices] == j,
                        as_tuple=False).squeeze(-1)
                    vote_targets_tmp[column_indices,
                                    int(j * 3):int(j * 3 +
                                                    3)] = votes[column_indices]
                    if j == 0:
                        vote_targets_tmp[column_indices] = votes[
                            column_indices].repeat(1, self.gt_per_seed)

                vote_targets[indices] = vote_targets_tmp
                vote_target_idx[indices] = torch.clamp(
                    vote_target_idx[indices] + 1, max=2)
            offset_targets = []
            offset_masks = []
            offset_targets.append(vote_targets)
            offset_masks.append(vote_target_masks)
        else:
            # ScanNet V2 with mask annotations
            # compute original all points offsets and masks
            allp_offset_targets = torch.zeros_like(scene_points[:, :3]) 
            allp_offset_masks = scene_points.new_zeros(len(scene_points))
            instance_center = scene_points.new_zeros((pts_instance_mask.max()+1, 3))
            instance_match_gt_id = -scene_points.new_ones((pts_instance_mask.max()+1)).long()
            for i in torch.unique(pts_instance_mask):
                indices = torch.nonzero(
                    pts_instance_mask == i, as_tuple=False).squeeze(-1) 
                if pts_semantic_mask[indices[0]] < self.num_classes: 
                    selected_points = scene_points[indices, :3] 
                    center = 0.5 * (
                            selected_points.min(0)[0] + selected_points.max(0)[0]) # instance center
                    allp_offset_targets[indices, :] = center - selected_points 
                    allp_offset_masks[indices] = 1 

                    match_gt_id = torch.argmin(torch.cdist(center.view(1, 1, 3),
                                                            gt_bboxes[:, :3].unsqueeze(0).to(center.device)).view(-1)) 
                    instance_match_gt_id[i] = match_gt_id
                    instance_center[i] = gt_bboxes[:, :3][match_gt_id].to(center.device) 
                else:
                    instance_center[i] = torch.ones_like(instance_center[i]) * (-10000.) 
                    instance_match_gt_id[i] = -1 


            # compute points offsets of each scale seed points
            offset_targets = [] 
            offset_masks = [] 
            knn_number = 1

            idx = knn(knn_number, scene_points[None, :, :3].contiguous(), voxel_points[None, ::])[0].long() 
            instance_idx = pts_instance_mask[idx.view(-1)].view(idx.shape[0], idx.shape[1]) 

            # condition1: all the points must belong to one instance
            valid_mask = (instance_idx == instance_idx[0]).all(0)

            max_instance_num = pts_instance_mask.max()+1 
            arange_tensor = torch.arange(max_instance_num).unsqueeze(1).unsqueeze(2).to(instance_idx.device)
            arange_tensor = arange_tensor.repeat(1, instance_idx.shape[0], instance_idx.shape[1]) 
            instance_idx = instance_idx[None, ::].repeat(max_instance_num, 1, 1) 

            max_instance_idx = torch.argmax((instance_idx == arange_tensor).sum(1), dim=0)
            offset_t = instance_center[max_instance_idx] - voxel_points
            offset_m = torch.where(offset_t < -100., torch.zeros_like(offset_t), torch.ones_like(offset_t)).all(1)
            offset_t = torch.where(offset_t < -100., torch.zeros_like(offset_t), offset_t)
            
            offset_m *= valid_mask

            offset_targets.append(offset_t)
            offset_masks.append(offset_m)
    
        offset_targets = torch.cat(offset_targets) 
        offset_masks = torch.cat(offset_masks)

        return offset_targets, offset_masks

    def _single_scene_multiclass_nms(self, bboxes, scores, img_meta):
        """Multi-class nms for a single scene.

        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.

        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels

