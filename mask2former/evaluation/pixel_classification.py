# Code taken and modified from: https://github.com/SegmentMeIfYouCan/road-anomaly-benchmark/blob/1c7804e0749686452eb02bbcfb81234e38795f8a/road_anomaly_benchmark/metrics/pixel_classification.py
from typing import List#, Literal

import numpy as np
from matplotlib import pyplot
from easydict import EasyDict

from .pixel_classification_curves import curves_from_cmats


def binary_confusion_matrix(
		prob : np.ndarray, gt_label_bool : np.ndarray, 
		num_bins : int = 1024, bin_strategy = 'uniform', # : Literal['uniform', 'percentiles'] = 'uniform',
		normalize : bool = False, dtype = np.float64):
	
	area = gt_label_bool.__len__()

	gt_area_true = np.count_nonzero(gt_label_bool)
	gt_area_false = area - gt_area_true

	prob_at_true = prob[gt_label_bool]
	prob_at_false = prob[~gt_label_bool]

	if bin_strategy == 'uniform':
		# bins spread uniforms in 0 .. 1
		bins = num_bins
		histogram_range = [-0.01, 1.01]

	elif bin_strategy == 'percentiles':
		# dynamic bins representing the range of occurring values
		# bin edges are following the distribution of positive and negative pixels

		# choose thresholds to surround the range of values
		vmin = np.min(prob)
		vmax = np.max(prob)
		vrange = vmax-vmin
		eps = np.maximum(vrange*1e-2, 1e-2) # make sure there is some separation between the thresholds

		bins = [
			[vmin - eps, vmax + eps]
		]

		if prob_at_true.size:
			bins += [
				np.quantile(prob_at_true, np.linspace(0, 1, min(num_bins//2, prob_at_true.size))),
			]
		if prob_at_false.size:
			bins += [
				np.quantile(prob_at_false, np.linspace(0, 1, min(num_bins//2, prob_at_false.size))),
			]

			
		bins = np.concatenate(bins)
		
		# sort and remove duplicates, duplicated cause an exception in np.histogram
		bins = np.unique(bins)
		

		histogram_range = None

	# the area of positive pixels is divided into
	#	- true positives - above threshold
	#	- false negatives - below threshold
	tp_rel, _ = np.histogram(prob_at_true, bins=bins, range=histogram_range)
	# the curve goes from higher thresholds to lower thresholds
	tp_rel = tp_rel[::-1]
	# cumsum to get number of tp at given threshold
	tp = np.cumsum(tp_rel)
	# GT-positives which are not TP are instead FN
	fn = gt_area_true - tp

	# the area of negative pixels is divided into
	#	- false positives - above threshold
	#	- true negatives - below threshold
	fp_rel, bin_edges = np.histogram(prob_at_false, bins=bins, range=histogram_range)
	# the curve goes from higher thresholds to lower thresholds
	bin_edges = bin_edges[::-1]
	fp_rel = fp_rel[::-1]
	# cumsum to get number of fp at given threshold
	fp = np.cumsum(fp_rel)
	# GT-negatives which are not FP are instead TN
	tn = gt_area_false - fp

	cmat_sum = np.array([
		[tp, fp],
		[fn, tn],
	]).transpose(2, 0, 1).astype(dtype)

	# cmat_rel = np.array([
	# 	[tp_rel, fp_rel],
	# 	[-tp_rel, -fp_rel],
	# ]).transpose(2, 0, 1).astype(dtype)
	
	if normalize:
		cmat_sum *= (1./area)
		# cmat_rel *= (1./area)

	return EasyDict(
		bin_edges = bin_edges,
		cmat_sum = cmat_sum,
		# cmat_rel = cmat_rel,
		tp_rel = tp_rel,
		fp_rel = fp_rel,
		num_pos = gt_area_true,
		num_neg = gt_area_false,
	)


def test_binary_confusion_matrix():
	pp = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	gt = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1], dtype=bool)
	cmat = binary_confusion_matrix(pp, gt, levels=20).cmat_sum
	cmat_all_p = np.sum(cmat[:, :, 0], axis=1)
	cmat_all_n = np.sum(cmat[:, :, 1], axis=1)

	print(cmat_all_p, cmat_all_n)

	pyplot.plot(cmat[:, 0, 1] / cmat_all_n, cmat[:, 0, 0] / cmat_all_p)



class MetricPixelClassification:

	configs = [
		EasyDict(
			name = 'PixBinaryClass-uniThr',
			# pixel scores in a given image are quantized into bins, 
			# so that big datasets can be stored in memory and processed in parallel
			num_bins = 4096,
			bin_strategy = 'uniform',
		),
		EasyDict(
			name = 'PixBinaryClass',
			num_bins = 768,
			bin_strategy = 'percentiles',
		)
	]

	@staticmethod
	def process_frame(label_pixel_gt : np.ndarray, anomaly_p : np.ndarray, num_bins: int, bin_strategy: str):
		"""
		@param label_pixel_gt: HxW uint8
			0 = road
			1 = obstacle
			255 = ignore
		@param anomaly_p: HxW float16
			heatmap of per-pixel anomaly detection, value from 0 to 1
		@param fid: frame identifier, for saving extra outputs
		@param dset_name: dataset identifier, for saving extra outputs
		"""
		try:
			mask_roi = label_pixel_gt < 255
		except TypeError:
			raise RuntimeError(f"No ground truth available for {fid}. Please check dataset path...")
		labels_in_roi = label_pixel_gt[mask_roi]
		predictions_in_roi = anomaly_p[mask_roi]

		bc = binary_confusion_matrix(
			prob = predictions_in_roi,
			gt_label_bool = labels_in_roi.astype(bool),
			num_bins = num_bins,
			bin_strategy = bin_strategy,
		)

		return bc


	@staticmethod
	def aggregate_fixed_bins(frame_results):
		
		# bin edges are the same in every frame
		bin_edges = frame_results[0].bin_edges
		thresholds = bin_edges[1:]

		# each frame has the same thresholds, so we can sum the cmats
		cmat_sum = np.sum([result.cmat_sum for result in frame_results], axis=0)

		return EasyDict(
			cmat = cmat_sum,
			thresholds = thresholds,
		)

	@staticmethod
	def aggregate_dynamic_bins(frame_results):

		thresholds = np.concatenate([r.bin_edges[1:] for r in frame_results])

		tp_relative = np.concatenate([r.tp_rel for r in frame_results], axis=0)
		fp_relative = np.concatenate([r.fp_rel for r in frame_results], axis=0)

		num_positives = sum(r.num_pos for r in frame_results)
		num_negatives = sum(r.num_neg for r in frame_results)


		threshold_order = np.argsort(thresholds)[::-1]

		# We start at threshold = 1, and lower it
		# Initially, prediction=0, all GT=1 pixels are false-negatives, and all GT=0 pixels are true-negatives.

		tp_cumu = np.cumsum(tp_relative[threshold_order].astype(np.float64))
		fp_cumu = np.cumsum(fp_relative[threshold_order].astype(np.float64))

		cmats = np.array([
			# tp, fp
			[tp_cumu, fp_cumu],
			# fn, tn
			[num_positives - tp_cumu, num_negatives - fp_cumu],
		]).transpose([2, 0, 1])

		return EasyDict(
			cmat = cmats,
			thresholds = thresholds[threshold_order],
		)
	@staticmethod
	def aggregate(frame_results : list, bin_strategy : str):
		# fuse cmats FIXED BINS

		if bin_strategy == 'uniform':
			ag = MetricPixelClassification.aggregate_fixed_bins(frame_results)
		else:
			ag = MetricPixelClassification.aggregate_dynamic_bins(frame_results)

		thresholds = ag.thresholds
		cmats = ag.cmat

		curves = curves_from_cmats(cmats, thresholds)

		return curves