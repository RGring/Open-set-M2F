# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
from typing import Optional, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_curve, auc
import h5py

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import SemSegEvaluator
from torchvision.utils import save_image
import torch.nn.functional as F
# from mask2former.utils import colorize_labels
import matplotlib.pyplot as plt
from .pixel_classification import MetricPixelClassification

_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False


def load_image_into_numpy_array(
    filename: str,
    copy: bool = False,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        array = np.array(Image.open(f), copy=copy, dtype=dtype)
    return array


def calculate_stat(conf, gt):
    fpr, tpr, threshold = roc_curve(gt, conf)
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    threshold = np.array(threshold)

    roc_auc = auc(fpr, tpr)
    fpr_best = fpr[tpr >= 0.95][0]
    tau = threshold[tpr >= 0.95][0]
    return roc_auc, fpr_best, tau

class DenseOODDetectionEvaluatorUNOOpenSet(SemSegEvaluator):
    """
    Evaluate OOD detection metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_classes=None,
        ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument
        """
        super().__init__(dataset_name, 
                         distributed=distributed, 
                         output_dir=output_dir,
                         sem_seg_loading_fn=sem_seg_loading_fn,
                         num_classes=num_classes,
                         ignore_label=ignore_label),


    def reset(self):
        super().reset()
        self._bc = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """

        for input, output in zip(inputs, outputs):
            seg_output = np.array(output["sem_seg"].argmax(dim=0).to(self._cpu_device), dtype=int)
            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int)
            
            # Anomaly map
            mask_pred = output['mask_pred']
            mask_cls_ = output['mask_cls']
            mask_pred = mask_pred.sigmoid()
            
            s_no = mask_cls_.softmax(-1)[..., :-2]
            s_unc = mask_cls_.softmax(-1)

            s_x = - s_unc[..., -2] + s_no.max(1)[0]

            v = (mask_pred * s_x.view(-1, 1, 1)).sum(0)
            ood_score = - v.to(self._cpu_device)
            # ood_score_normalized = (ood_score - ood_score.min()) / (ood_score.max() - ood_score.min())

            # Anomaly scores
            gt_anomaly = gt.copy()
            if self._dataset_name == 'road_anomaly':
                gt_anomaly[gt_anomaly == 2] = 1
            elif "cityscapesprivate" in self._dataset_name:
                gt_anomaly[gt_anomaly < 15] = 0
                gt_anomaly[gt_anomaly == 15] = 1
                
            bc = MetricPixelClassification.process_frame(gt_anomaly, ood_score.cpu().numpy(), 768, 'percentiles')
            self._bc.append(bc)
            
            # segmentation scores            
            gt[gt == self._ignore_label] = self._num_classes
            # ToDo:Paramaterize
            anomaly_thresh = 0.85
            denormalized_anomaly_thresh = ood_score.min() + anomaly_thresh * (ood_score.max() - ood_score.min())
            seg_output[ood_score > denormalized_anomaly_thresh] = self._num_classes - 1
            seg_output[gt == self._ignore_label] = self._num_classes
            pred = seg_output

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))


    def evaluate(self):
        if self._distributed:
            raise Exception('Not implemented.')
        
        # Anomaly metrics
        # ToDo: Extract component level values
        curves = MetricPixelClassification.aggregate(self._bc, 'percentiles')
        
        print(f"BEst f1 threshold: {curves['best_f1_threshold']}")
        res_odd = {}
        res_odd["AP"] = 100 * curves['area_PRC'] 
        res_odd["AUROC"] = 100 * curves['area_ROC'] 
        res_odd["FPR@TPR95"] = 100 * curves['tpr95_fpr']

        # Segmentation metrics
        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)
        tp = self._conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        iou_common = iou[:-1]
        m_iou_common = np.sum(iou_common[iou_valid[:-1]]) / np.sum(iou_valid[:-1])
        iou_private = iou[-1]
        h_score = 2/(1/m_iou_common + 1/iou_private)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)
        
        res = {}
        res["mIoU"] = 100 * miou
        res["mIoU_common"] = 100 * m_iou_common
        res["IoU_private"] = 100 * iou_private
        res["h_score"] = 100 * h_score
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
    
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}"] = 100 * acc[i]
        

        results = OrderedDict({"ood_detection": res_odd, "sem_seg": res})
        self._logger.info(results)
        return results