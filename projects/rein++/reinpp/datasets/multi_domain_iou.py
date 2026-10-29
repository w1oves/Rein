# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Optional, Sequence
import numpy as np
from mmengine.logging import MMLogger, print_log
from PIL import Image
from mmseg.registry import METRICS
from mmseg.evaluation.metrics.iou_metric import IoUMetric
from collections import defaultdict
from mmengine.dist import get_dist_info
import logging
import copy

@METRICS.register_module(name='ReinPPDomainIoUMetric')
class DomainIoUMetric(IoUMetric):

    def __init__(self, mean_keys=None, **kwargs):
        super().__init__(**kwargs)
        rank, world_size = get_dist_info()
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info(f'Rank{rank} of {world_size}: init')
        self.mean_keys = mean_keys

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
                res1, res2, res3, res4 = self.intersect_and_union(pred_label, label, num_classes, self.ignore_index)
                dataset_key = data_sample['domain']
                self.results.append([dataset_key, res1, res2, res3, res4])
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(data_sample['img_path']))[0]
                png_filename = osp.abspath(osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        dataset_results = defaultdict(list)
        metrics = {}
        for result in results:
            for domain in result[0].split('|'):
                dataset_results[domain].append(copy.deepcopy(result[1:]))
        hierarchy_metrics = defaultdict(dict)
        print_log(f'----------metrics for all target dataset------------', logger)
        target_results = sum([v for k, v in dataset_results.items() if k in self.mean_keys], start=[])
        _metrics = super().compute_metrics(target_results)
        metrics.update(**_metrics)
        for dataset_name, results in dataset_results.items():
            print_log(f'----------metrics for {dataset_name}------------', logger)
            _metrics = super().compute_metrics(results)
            for metric_name, v in _metrics.items():
                metrics[f'{dataset_name}_{metric_name}'] = v
                hierarchy_metrics[metric_name][dataset_name] = v
        for metric_name, values in hierarchy_metrics.items():
            if self.mean_keys is not None:
                values = [v for k, v in values.items() if k in self.mean_keys]
            metrics[f'mean_{metric_name}'] = sum(values) / len(values)
        return metrics
