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
        self._gt = []
        self._ood_score = []

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
            
            
            # Anomaly scores
            gt_vec = torch.from_numpy(gt.copy()).view(-1)
            if self._dataset_name == 'road_anomaly':
                gt_vec[gt_vec == 2] = 1
            elif "cityscapesprivate" in self._dataset_name:
                gt_vec[gt_vec < 15] = 0
                gt_vec[gt_vec == 15] = 1

            self._ood_score += [ood_score.view(-1)[gt_vec != self._ignore_label]]
            self._gt += [gt_vec[gt_vec != self._ignore_label]]
            
            # score = [ood_score.view(-1)[gt_vec != self._ignore_label]]
            # gt_labels = [gt_vec[gt_vec != self._ignore_label]]
            
            # gt_labels = torch.cat(gt_labels, 0)
            # score = torch.cat(score, 0)
            # AUROC, FPR, _ = calculate_stat(score, gt_labels)
            # AP = average_precision_score(gt_labels, score)
            # self._gt.append(FPR)
            # self._ood_score.append(AP)
            
            # segmentation scores            
            gt[gt == self._ignore_label] = self._num_classes
            # ToDo:Paramaterize
            anomaly_thresh = 0.9
            threshold = torch.min(ood_score) + (torch.max(ood_score) - torch.min(ood_score)) * anomaly_thresh
            seg_output[ood_score > threshold] = self._num_classes - 1
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
        gt = torch.cat(self._gt, 0)
        score = torch.cat(self._ood_score, 0)

        AUROC, FPR, _ = calculate_stat(score, gt)
        AP = average_precision_score(gt, score)

        res_odd = {}
        res_odd["AP"] = AP #100 * np.nanmean(np.array(self._ood_score, dtype=np.float32))
        # res_odd["AUROC"] = 100 * AUROC
        res_odd["FPR@TPR95"] = AUROC # 100 * np.nanmean(np.array(self._gt, dtype=np.float32))

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