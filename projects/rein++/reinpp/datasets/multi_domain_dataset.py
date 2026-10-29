
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS
import numpy as np
from mmseg.utils import ConfigType
from mmengine.logging import MMLogger
from typing import List, Tuple, Set
from mmseg.structures import SegDataSample
from mmengine.dataset.base_dataset import force_full_init
import copy
import bisect

@DATASETS.register_module(name='ReinPPMultiDomainDataset')
class MultiDomainDataset:

    def __init__(self, dataset_names: List[str], datasets: List[ConfigType], selected_names: Set[str]=None, unified_pipeline=None, lazy_init: bool=False, debug=False, infer_mode=False) -> None:
        self.infer_mode = infer_mode
        if unified_pipeline is not None:
            for dataset in datasets:
                dataset.pipeline = copy.deepcopy(unified_pipeline)
        if selected_names is not None:
            _names, _datasets = ([], [])
            for name, dataset in zip(dataset_names, datasets):
                if name in selected_names:
                    _names.append(name)
                    _datasets.append(dataset)
            dataset_names = _names
            datasets = _datasets
        self.dataset_names = dataset_names
        self.datasets: List[BaseSegDataset] = [DATASETS.build(dataset) for dataset in datasets]
        self._metainfo = self.datasets[0].metainfo
        self.debug = debug
        logger = MMLogger.get_current_instance()
        for dataset, dataset_name in zip(self.datasets, self.dataset_names):
            logger.info(f'build dataset {dataset_name},len={len(dataset)}')
        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    def get_dataset_split_points(self):
        points, cur_idx = ([], 0)
        for name, dataset in zip(self.dataset_names, self.datasets):
            points.append(len(dataset) + cur_idx)
            cur_idx += len(dataset)
        logger = MMLogger.get_current_instance()
        for point, name in zip(points, self.dataset_names):
            logger.info(f'dataset {name} end in {point}')
        return points

    @property
    def metainfo(self) -> dict:
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        if self._fully_initialized:
            return
        for dataset in self.datasets:
            dataset.full_init()
        self._ori_len = sum((len(dataset) for dataset in self.datasets))
        self.split_points = self.get_dataset_split_points()
        self._fully_initialized = True

    @force_full_init
    def get_dataset_and_sample_indice(self, idx: int) -> Tuple[int, int]:
        if idx < 0:
            if -idx > len(self):
                raise ValueError(f'absolute value of index({idx}) should not exceed datasetlength({len(self)}).')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.split_points, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.split_points[dataset_idx - 1]
        return (dataset_idx, sample_idx)

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        dataset_idx, sample_idx = self.get_dataset_and_sample_indice(idx)
        return self.datasets[dataset_idx].get_data_info(sample_idx)

    @force_full_init
    def __len__(self):
        if self.debug:
            return 50
        return sum(map(len, self.datasets))

    def __getitem__(self, idx):
        if self.debug:
            idx = np.random.randint(0, sum(map(len, self.datasets)))
        if not self._fully_initialized:
            logger = MMLogger.get_current_instance()
            logger.info('Please call `full_init` method manually to accelerate the speed.')
            self.full_init()
        dataset_idx, sample_idx = self.get_dataset_and_sample_indice(idx)
        data = self.datasets[dataset_idx][sample_idx]
        if self.infer_mode:
            return data
        self.add_domain_info(data['data_samples'], self.dataset_names[dataset_idx])
        return data

    def add_domain_info(self, data_samples: SegDataSample, domain: str):
        domains = []
        if 'domain' in data_samples.metainfo_keys():
            domains.append(data_samples.get('domain'))
        domains.append(domain)
        data_samples.set_metainfo(dict(domain='|'.join(domains)))

    def get_subset_(self, indices) -> None:
        raise NotImplementedError()

    def get_subset(self):
        raise NotImplementedError()
