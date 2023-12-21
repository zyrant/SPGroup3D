import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet3d.core.bbox import AxisAlignedBboxOverlaps3D
import math
from mmcv.ops import box_iou_rotated
from mmcv.ops.diff_iou_rotated import box2corners
from mmdet3d.core.visualizer.open3d_vis import show_pts_index_boxes


def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    https://github.com/facebookresearch/3detr/blob/main/utils/pc_util.py
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """

    if dst_range is None:
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
        ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
    ) + dst_range[0][:, None, :]
    return prop_xyz


@MATCH_COST.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, original_points, bbox_pred, gt_bboxes):
        src_range = [original_points.min(dim=0)[0].unsqueeze(0), original_points.max(dim=0)[0].unsqueeze(0)]
        bbox_pred[:,:3] = shift_scale_points(bbox_pred[:,:3].unsqueeze(0), src_range)
        bbox_pred[:,3:6] = bbox_pred[:,3:6] / (src_range[1] - src_range[0])
        gt_bboxes[:,:3] = shift_scale_points(gt_bboxes[:,:3].unsqueeze(0), src_range)
        gt_bboxes[:,3:6] = gt_bboxes[:,3:6] / (src_range[1] - src_range[0])
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight
    
@MATCH_COST.register_module()
class IoU3DCost:
    """IoUCost.
     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight
     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode='iou', weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape (num_query, 4).
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).
        Returns:
            torch.Tensor: iou_cost value with weight
        """
        # overlaps: [num_bboxes, num_gt]
        overlaps = AxisAlignedBboxOverlaps3D()(
        bboxes, gt_bboxes, is_aligned=False)
        if self.iou_mode =='diou':
            xp1, yp1, zp1, xp2, yp2, zp2 = torch.split(bboxes, 1, dim=-1)
            xt1, yt1, zt1, xt2, yt2, zt2 = torch.split(gt_bboxes, 1, dim=-1)

            xpc = (xp1 + xp2) / 2
            ypc = (yp1 + yp2) / 2
            zpc = (zp1 + zp2) / 2
            xtc = (xt1 + xt2) / 2
            ytc = (yt1 + yt2) / 2
            ztc = (zt1 + zt2) / 2

            r2 = (xpc[:, None] - xtc) ** 2 + (ypc[:, None] - ytc) ** 2 + (zpc[:, None] - ztc) ** 2

            x_min = torch.minimum(xp1[:, None], xt1)
            x_max = torch.maximum(xp2[:, None], xt2)
            y_min = torch.minimum(yp1[:, None], yt1)
            y_max = torch.maximum(yp2[:, None], yt2)
            z_min = torch.minimum(zp1[:, None], zt1)
            z_max = torch.maximum(zp2[:, None], zt2)
            c2 = (x_min - x_max) ** 2 + (y_min - y_max) ** 2 + (z_min - z_max) ** 2
            diou = overlaps - (r2 / c2).squeeze()
            # The 1 is a constant that doesn't change the matching, so omitted.
            iou_cost = - diou
        elif self.iou_mode == 'ciou':
            xp1, yp1, zp1, xp2, yp2, zp2 = torch.split(bboxes, 1, dim=-1)
            xt1, yt1, zt1, xt2, yt2, zt2 = torch.split(gt_bboxes, 1, dim=-1)

            xpc = (xp1 + xp2) / 2
            ypc = (yp1 + yp2) / 2
            zpc = (zp1 + zp2) / 2
            xtc = (xt1 + xt2) / 2
            ytc = (yt1 + yt2) / 2
            ztc = (zt1 + zt2) / 2

            r2 = (xpc[:, None] - xtc) ** 2 + (ypc[:, None] - ytc) ** 2 + (zpc[:, None] - ztc) ** 2

            x_min = torch.minimum(xp1[:, None], xt1)
            x_max = torch.maximum(xp2[:, None], xt2)
            y_min = torch.minimum(yp1[:, None], yt1)
            y_max = torch.maximum(yp2[:, None], yt2)
            z_min = torch.minimum(zp1[:, None], zt1)
            z_max = torch.maximum(zp2[:, None], zt2)
            c2 = (x_min - x_max) ** 2 + (y_min - y_max) ** 2 + (z_min - z_max) ** 2
            diou = overlaps - (r2 / c2).squeeze()

            wp = xp2 - xp1
            hp = yp2 - yp1
            lp = zp2 - zp1
            wt = xt2 - xt1
            ht = yt2 - yt1
            lt = zt2 - zt1

            eps = 1e-7  # to avoid division by zero
            vl = (4/(math.pi**2)) * ((torch.atan(lp/(wp+eps))[:, None] - torch.atan(lt/(wt+eps)))**2)
            vw = (4/(math.pi**2)) * ((torch.atan(wp/(hp+eps))[:, None] - torch.atan(wt/(ht+eps)))**2)
            vh = (4/(math.pi**2)) * ((torch.atan(hp/(lp+eps))[:, None] - torch.atan(ht/(lt+eps)))**2)
            v = ((vl + vw + vh)).squeeze(-1)
            
            with torch.no_grad():
                alpha = v / (1 - overlaps + v)
            # The 1 is a constant that doesn't change the matching, so omitted.
            ciou_loss = overlaps - (r2 / c2).squeeze() - alpha * v
            iou_cost = - ciou_loss
        else:
            iou_cost = - overlaps

        return iou_cost * self.weight
    

    
@MATCH_COST.register_module()
class RotatedIoU3DCost:
    """IoUCost.
     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight
     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode='iou', weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode


    def __call__(self, bboxes, gt_bboxes):
        """Calculate differentiable iou  cost of rotated 3d boxes.

        Args:
            box3d1 (Tensor): (B, N, 3+3+1) First box (x,y,z,w,h,l,alpha).
            box3d2 (Tensor): (B, M, 3+3+1) Second box (x,y,z,w,h,l,alpha).

        Returns:
            Tensor: (N, M).
        """

        rows = len(bboxes)
        cols = len(gt_bboxes)
        
        # bev overlap

        '''
        Tips: 
        1. rotation of DepthInstance3DBoxes is counterclockwise, 
            which is reverse to the definition of the yaw angle (clockwise).
        2. 'box_iou_rotated' need positive direction along y axis is top -> down, 
            while DepthInstance3DBoxes positive direction along y axis is down -> top
        '''

        bboxes_bev = bboxes[..., [0, 1, 3, 4, 6]]  # 2d box, x, y, w, h, r
        gt_bboxes_bev = gt_bboxes[..., [0, 1, 3, 4, 6]]
        gt_bboxes_bev[: , 1] = - gt_bboxes_bev[:, 1]
        bboxes_bev[: , 1] = - bboxes_bev[:, 1]
        
        iou2d = box_iou_rotated(bboxes_bev, gt_bboxes_bev, clockwise=False)
        areas1 = (bboxes_bev[:, 2] * bboxes_bev[:, 3]).unsqueeze(1).expand(
            rows, cols)
        areas2 = (gt_bboxes_bev[:, 2] * gt_bboxes_bev[:, 3]).unsqueeze(0).expand(
            rows, cols)
        overlaps_bev = iou2d * (areas1 + areas2) / (1 + iou2d)
        # print('overlaps_bev: ', overlaps_bev)
        # height overlap
        zmax1 = bboxes[..., 2] + bboxes[..., 5] * 0.5
        zmin1 = bboxes[..., 2] - bboxes[..., 5] * 0.5
        zmax2 = gt_bboxes[..., 2] + gt_bboxes[..., 5] * 0.5
        zmin2 = gt_bboxes[..., 2] - gt_bboxes[..., 5] * 0.5
        z_overlap = (torch.min(zmax1.view(-1, 1), zmax2.view(1, -1)) -
                    torch.max(zmin1.view(-1, 1), zmin2.view(1, -1))).clamp_(min=0.)
        
        # 3d overlaps
        overlaps_3d = overlaps_bev * z_overlap

        volume1 = (bboxes[..., 3] * bboxes[..., 4] * bboxes[..., 5]).view(-1, 1)
        volume2 = (gt_bboxes[..., 3] * gt_bboxes[..., 4] * gt_bboxes[..., 5]).view(1, -1)
        iou3d = overlaps_3d / torch.clamp(
                volume1 + volume2 - overlaps_3d, min=1e-8)

        if self.iou_mode =='diou':
            corners1 = box2corners(bboxes_bev.unsqueeze(0))
            corners2 = box2corners(gt_bboxes_bev.unsqueeze(0))
            x1_max = torch.max(corners1[..., 0], dim=2)[0]     # (N)
            x1_min = torch.min(corners1[..., 0], dim=2)[0]     # (N)
            y1_max = torch.max(corners1[..., 1], dim=2)[0]
            y1_min = torch.min(corners1[..., 1], dim=2)[0]
            
            x2_max = torch.max(corners2[..., 0], dim=2)[0]     # (M)
            x2_min = torch.min(corners2[..., 0], dim=2)[0]    # (M)
            y2_max = torch.max(corners2[..., 1], dim=2)[0]
            y2_min = torch.min(corners2[..., 1], dim=2)[0]

            x_max = torch.max(x1_max.view(-1, 1), x2_max)
            x_min = torch.min(x1_min.view(-1, 1), x2_min)
            y_max = torch.max(y1_max.view(-1, 1), y2_max)
            y_min = torch.min(y1_min.view(-1, 1), y2_min)

            z_max = torch.max(zmax1.view(-1, 1), zmax2.view(1, -1))
            z_min = torch.min(zmin1.view(-1, 1), zmin2.view(1, -1))

            # bug fix
            # it seems a bug it use xyw but we should use xyz
            # r2 = ((bboxes_bev[..., :3][:, None] - gt_bboxes_bev[..., :3]) ** 2).sum(dim=-1)
            r2 = ((bboxes[..., :3].unsqueeze(1) - gt_bboxes[..., :3].unsqueeze(0)) ** 2).sum(dim=-1)
            c2 = (x_min - x_max) ** 2 + (y_min - y_max) ** 2 + (z_min - z_max) ** 2
            diou = iou3d - r2 / c2
            iou_cost = - diou
        else:
            iou_cost = - iou3d

        return iou_cost * self.weight