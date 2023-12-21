import numpy as np
import tempfile
import warnings
from os import path as osp

from mmdet3d.core import show_result, show_seg_result
from mmdet3d.core.bbox import DepthInstance3DBoxes
# from mmdet.datasets import DATASETS
# from mmseg.datasets import DATASETS as SEG_DATASETS
from mmdet3d.datasets import DATASETS
from .custom_3d import Custom3DDataset
from .custom_3d_seg import Custom3DSegDataset
from .pipelines import Compose
import torch
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable


@DATASETS.register_module()
class ScanNetDatasetWithSeg(Custom3DDataset):
    r"""ScanNet Dataset for Detection Task.

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

    def __init__(self,
                 data_root,
                 ann_file,
                 semantic_threshold=0.1,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)
        self.semantic_threshold = semantic_threshold

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
                - axis_align_matrix (np.ndarray): Transformation matrix for \
                    global scene alignment.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.long)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.long)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        pts_instance_mask_path = osp.join(self.data_root,
                                          info['pts_instance_mask_path'])
        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        axis_align_matrix = self._get_axis_align_matrix(info)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path,
            axis_align_matrix=axis_align_matrix)
        return anns_results
    
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])

        input_dict = dict(
            pts_filename=pts_filename,
            sample_idx=sample_idx,
            file_name=pts_filename)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        return input_dict

    def prepare_test_data(self, index):
        """Prepare data for testing.

        We should take axis_align_matrix from self.data_infos since we need \
            to align point clouds.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)

        # take the axis_align_matrix from data_infos
        input_dict['ann_info'] = dict(
            axis_align_matrix=self._get_axis_align_matrix(
                self.data_infos[index]))
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @staticmethod
    def _get_axis_align_matrix(info):
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): one data info term.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info['annos'].keys():
            return info['annos']['axis_align_matrix'].astype(np.float32)
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(type='GlobalAlignment', rotation_axis=2),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points', load_annos=True).numpy()
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            gt_bboxes = gt_bboxes.corners.numpy() if len(gt_bboxes) else None
            gt_labels = self.get_ann_info(i)['gt_labels_3d']
            pred_bboxes = result['boxes_3d']
            pred_bboxes = pred_bboxes.corners.numpy() if len(pred_bboxes) else None
            pred_labels = result['labels_3d']
            show_result(points, gt_bboxes, gt_labels,
                        pred_bboxes, pred_labels, out_dir, file_name, False)


    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in detection and semantic segmentation indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str]): Metrics to be evaluated.
            iou_thr (list[float]): AP IoU thresholds.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """

        # evaluation in detection
        from mmdet3d.core.evaluation import indoor_eval
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        gt_annos = [info['annos'] for info in self.data_infos]
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
        ret_dict = indoor_eval(
            gt_annos,
            results,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d)
        if show:
            self.show(results, out_dir, pipeline=pipeline)

        # evaluation in bbox segmentation indoor protocol
        pred_sem_masks = [self.get_semantic_pred(result['semantic_preds'], gt_annos[i]['class']) for i, result in enumerate(results)]
        gt_sem_masks = [
            self.compute_semantic_bbox_gt(
                gt_anno=gt_annos[i], semantic_coords=results[i]['semantic_coords'])
            for i in range(len(self.data_infos))]

        self.label2cat = {i: name for i, name in enumerate(self.CLASSES)}
        self.label2cat[len(self.CLASSES)] = 'background'
        self.ignore_index = len(self.CLASSES) + 1

        seg_ret_dict = self.seg_eval(
            gt_sem_masks,
            pred_sem_masks,
            self.label2cat,
            self.ignore_index,
            logger=logger)

        return ret_dict
    """
    def get_semantic_pred(self, semantic_scores, gt_labels):

        semantic_pred = torch.zeros_like(semantic_scores).long().fill_(len(self.CLASSES))
        for cls_id in range(len(self.CLASSES)):
            if cls_id in gt_labels:
                cls_selected_id = torch.nonzero(semantic_scores[:, cls_id] > self.semantic_threshold).squeeze(1)
                semantic_pred[cls_selected_id, cls_id] = cls_id
            else:
                semantic_pred[:, cls_id] = len(self.CLASSES) + 1
        return semantic_pred.view(-1)
    """
    def get_semantic_pred(self, semantic_preds, gt_labels):

        for cls_id in range(len(self.CLASSES)):
            if cls_id in gt_labels:
                continue
            else:
                semantic_preds[:, cls_id] = len(self.CLASSES) + 1
        return semantic_preds.view(-1)

    def compute_semantic_bbox_gt(self, gt_anno, semantic_coords):
        # parse gt bboxes
        if gt_anno['gt_num'] != 0:
            gt_boxes = self.box_type_3d(
                gt_anno['gt_boxes_upright_depth'],
                box_dim=gt_anno['gt_boxes_upright_depth'].shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
            labels_3d = torch.LongTensor(gt_anno['class'])
        else:
            gt_boxes = self.box_type_3d(np.array([], dtype=np.float32))
            labels_3d = torch.LongTensor(np.array([], dtype=np.int64))

        sem_labels = self.assign_semantic(semantic_coords, gt_boxes, labels_3d)  # point_num
        sem_labels_set = sem_labels.unique()
        sem_class_labels = sem_labels.unsqueeze(1).repeat(1, len(self.CLASSES)).fill_(len(self.CLASSES))  # point_num, class_num
        for cls_id in range(len(self.CLASSES)):
            if cls_id in sem_labels_set:
                valid_idx = torch.nonzero(sem_labels == cls_id).squeeze(1)
                sem_class_labels[valid_idx, cls_id] = cls_id
            else:
                sem_class_labels[:, cls_id] = len(self.CLASSES) + 1

        return sem_class_labels.view(-1)

    def assign_semantic(self, points, gt_bboxes, gt_labels):
        float_max = 1e8

        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.to(points.device)
        volumes = volumes.expand(n_points, n_boxes).contiguous()
        gt_bboxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes, 7)
        expanded_points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        shift = torch.stack((
            expanded_points[..., 0] - gt_bboxes[..., 0],
            expanded_points[..., 1] - gt_bboxes[..., 1],
            expanded_points[..., 2] - gt_bboxes[..., 2]
        ), dim=-1).permute(1, 0, 2)
        shift = rotation_3d_in_axis(shift, -gt_bboxes[0, :, 6], axis=2).permute(1, 0, 2)
        centers = gt_bboxes[..., :3] + shift
        dx_min = centers[..., 0] - gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2
        dx_max = gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2 - centers[..., 0]
        dy_min = centers[..., 1] - gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2
        dy_max = gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2 - centers[..., 1]
        dz_min = centers[..., 2] - gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2
        dz_max = gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2 - centers[..., 2]
        bbox_targets = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max, gt_bboxes[..., 6]), dim=-1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets[..., :6].min(-1)[0] > 0  # skip angle

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        volumes = torch.where(inside_gt_bbox_mask, volumes, torch.ones_like(volumes) * float_max)
        min_area, min_area_inds = volumes.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels = torch.where(min_area == float_max, -1, labels)

        return labels




    def fast_hist(self, preds, labels, num_classes):
        """Compute the confusion matrix for every batch.

        Args:
            preds (np.ndarray):  Prediction labels of points with shape of
            (num_points, ).
            labels (np.ndarray): Ground truth labels of points with shape of
            (num_points, ).
            num_classes (int): number of classes

        Returns:
            np.ndarray: Calculated confusion matrix.
        """

        k = (labels >= 0) & (labels < num_classes)
        bin_count = np.bincount(
            num_classes * labels[k].astype(int) + preds[k],
            minlength=num_classes**2)
        return bin_count[:num_classes**2].reshape(num_classes, num_classes)


    def per_class_iou(self, hist):
        """Compute the per class iou.

        Args:
            hist(np.ndarray):  Overall confusion martix
            (num_classes, num_classes ).

        Returns:
            np.ndarray: Calculated per class iou
        """

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


    def get_acc(self, hist):
        """Compute the overall accuracy.

        Args:
            hist(np.ndarray):  Overall confusion martix
            (num_classes, num_classes ).

        Returns:
            float: Calculated overall acc
        """

        return np.diag(hist).sum() / hist.sum()


    def get_acc_cls(self, hist):
        """Compute the class average accuracy.

        Args:
            hist(np.ndarray):  Overall confusion martix
            (num_classes, num_classes ).

        Returns:
            float: Calculated class average acc
        """

        return np.nanmean(np.diag(hist) / hist.sum(axis=0))

    def per_get_acc_cls(self, hist):
        """Compute the per class accuracy.

        Args:
            hist(np.ndarray):  Overall confusion martix
            (num_classes, num_classes ).

        Returns:
            float: Calculated per class acc
        """

        return np.diag(hist) / hist.sum(axis=0)

    def get_recall_cls(self, hist):
        """Compute the class average recall.

        Args:
            hist(np.ndarray):  Overall confusion martix
            (num_classes, num_classes ).

        Returns:
            float: Calculated class average recall
        """

        return np.nanmean(np.diag(hist) / hist.sum(axis=1))

    def per_get_recall_cls(self, hist):
        """Compute the per class recall.

        Args:
            hist(np.ndarray):  Overall confusion martix
            (num_classes, num_classes ).

        Returns:
            float: Calculated per class recall
        """

        return np.diag(hist) / hist.sum(axis=1)


    def seg_eval(self, gt_labels, seg_preds, label2cat, ignore_index, logger=None):
        """Semantic Segmentation  Evaluation.

        Evaluate the result of the Semantic Segmentation.

        Args:
            gt_labels (list[torch.Tensor]): Ground truth labels.
            seg_preds  (list[torch.Tensor]): Predictions.
            label2cat (dict): Map from label to category name.
            ignore_index (int): Index that will be ignored in evaluation.
            logger (logging.Logger | str | None): The way to print the mAP
                summary. See `mmdet.utils.print_log()` for details. Default: None.

        Returns:
            dict[str, float]: Dict of results.
        """
        assert len(seg_preds) == len(gt_labels)
        num_classes = len(label2cat)

        hist_list = []
        for i in range(len(gt_labels)):
            gt_seg = gt_labels[i].clone().numpy().astype(np.int)
            pred_seg = seg_preds[i].clone().numpy().astype(np.int)

            # filter out ignored points
            pred_seg[gt_seg == ignore_index] = -1
            gt_seg[gt_seg == ignore_index] = -1

            # filter both background
            both_background_idx = ((gt_seg == (num_classes-1)) & (pred_seg == (num_classes-1)))
            pred_seg[both_background_idx] = -1
            gt_seg[both_background_idx] = -1

            # calculate one instance result
            hist_list.append(self.fast_hist(pred_seg, gt_seg, num_classes))

        iou = self.per_class_iou(sum(hist_list))
        miou = np.nanmean(iou)
        acc = self.get_acc(sum(hist_list))
        acc_cls = self.get_acc_cls(sum(hist_list))
        recall_cls = self.get_recall_cls(sum(hist_list))
        per_recall_cls = self.per_get_recall_cls(sum(hist_list))
        per_acc_cls = self.per_get_acc_cls(sum(hist_list))

        header = ['classes']
        table_columns = [[label2cat[label]
                          for label in range(len(label2cat))] + ['Overall']]
        header.extend(['IoU', 'Acc', 'Recall'])

        seg_ret_dict = dict()
        for label in range(len(label2cat)):
            seg_ret_dict[f'{label2cat[label]}_iou'] = float(iou[label])
        seg_ret_dict['miou'] = float(miou)
        table_columns.append(list(map(float, list(iou))))
        table_columns[-1] += [seg_ret_dict['miou']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

        for label in range(len(label2cat)):
            seg_ret_dict[f'{label2cat[label]}_acc'] = float(per_acc_cls[label])
        seg_ret_dict['macc'] = float(acc_cls)
        table_columns.append(list(map(float, list(per_acc_cls))))
        table_columns[-1] += [seg_ret_dict['macc']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

        for label in range(len(label2cat)):
            seg_ret_dict[f'{label2cat[label]}_recall'] = float(per_recall_cls[label])
        seg_ret_dict['mrecall'] = float(recall_cls)
        table_columns.append(list(map(float, list(per_recall_cls))))
        table_columns[-1] += [seg_ret_dict['mrecall']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]


        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)

        return seg_ret_dict


