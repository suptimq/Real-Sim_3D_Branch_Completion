import torch
import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *

from typing import Any, List

# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

@DATASETS.register_module()
class PCNSkel(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.complete_point_segments_path = config.COMPLETE_SEGMENTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.npoints_partial = config.N_PARTIAL
        self.nskel_points = config.N_SKEL_POINTS
        self.subset = config.subset
        self.cars = config.CARS

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        self.n_renderings = config.N_RENDERINGS if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset, keys=['partial', 'gt', 'centers']):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': self.npoints_partial
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': keys
            }, {
                'callback': 'SkeletonFPS',
                'parameters': {
                    'n_points': self.nskel_points
                },
                'objects': ['centers', 'center_radii']
            }, {
                'callback': 'ToTensor',
                'objects': keys
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': self.npoints_partial
                },
                'objects': ['partial']
            }, {
                'callback': 'SkeletonFPS',
                'parameters': {
                    'n_points': self.nskel_points
                },
                'objects': ['centers', 'center_radii', 'center_directions']
            }, {
                'callback': 'ToTensor',
                'objects': keys
            }])

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
            samples = dc[subset]

            for s in samples:
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_path': [
                        self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path':
                    self.complete_points_path % (subset, dc['taxonomy_id'], s),
                    'gt_skeleton_path':
                    self.complete_point_segments_path % (subset, dc['taxonomy_id'], s),
                })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        if self.subset == 'test':
            rand_idx = 0

        transform_keys = []
        for ri in ['partial', 'gt', 'gt_skeleton']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            if ri == 'gt_skeleton':
                tmp_data = IO.get(file_path)
                for key in tmp_data.files:
                    data[key] = tmp_data[key].astype(np.float32)
                    transform_keys.append(key)
            else:
                data[ri] = IO.get(file_path).astype(np.float32)
                transform_keys.append(ri)

        assert data['gt'].shape[0] == self.npoints, f"{data['gt'].shape[0]} != {self.npoints}"

        if self.transforms is not None:
            data = self.transforms(data)

        # For collate_fn when len(segment_centers) could have different values
        # b = {
        #     'taxonomy_id': sample['taxonomy_id'],
        #     'model_id': sample['model_id'],
        #     'data': data   
        # }

        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)
    
    @staticmethod
    def _collate_fn(batch):
        """
        Custom collate function for batching data with varying sizes.

        Args:
            batch (list): List of samples, where each sample is a dictionary with keys 'taxonomy_id',
                        'model_id', and 'data'.

        Returns:
            batch (dict): A dictionary containing stacked tensors for 'taxonomy_id', 'model_id', and 'data'.
                        'data' is a dictionary containing the stacked tensors for 'partials', 'gts',
                        'padded_centers', 'padded_radii', and 'padded_directions'.
        """
        # Unzip the input batch into individual dictionaries
        taxonomy_ids, model_ids, data_list = zip(*[(sample['taxonomy_id'], sample['model_id'], sample['data']) for sample in batch])

        # Stack tensors along the batch dimension for partials, gts, centers, radii, and directions
        partials = torch.stack([data['partial'] for data in data_list], dim=0)
        gts = torch.stack([data['gt'] for data in data_list], dim=0)

        # Find the maximum lengths for centers, radii, and directions
        max_len = max([data['centers'].size(0) for data in data_list])

        # Initialize padded tensors for centers, radii, and directions with NaN
        # TODO when set the initial value as 0, all returned values are 0
        padded_centers = torch.full((len(batch), max_len, data_list[0]['centers'].size(1)), float('inf'))
        padded_radii = torch.full((len(batch), max_len, data_list[0]['center_radii'].size(1)), float('inf'))
        padded_directions = torch.full((len(batch), max_len, data_list[0]['center_directions'].size(1)), float('inf'))

        # Copy actual values to the padded tensors for centers, radii, and directions
        # TODO need to set the padded tensor requires_grad = False
        for i, data in enumerate(data_list):
            padded_centers[i, :data['centers'].size(0), :] = data['centers']
            padded_radii[i, :data['center_radii'].size(0), :] = data['center_radii']
            padded_directions[i, :data['center_directions'].size(0), :] = data['center_directions']


        # Create a dictionary for 'data' containing the stacked tensors
        data_dict = {
            'partial': partials,
            'gt': gts,
            'padded_centers': padded_centers,
            'padded_radii': padded_radii,
            'padded_directions': padded_directions,
        }

        # Return the batch as a dictionary containing 'taxonomy_id', 'model_id', and 'data'
        # batch_dict = {
        #     'taxonomy_id': taxonomy_ids,
        #     'model_id': model_ids,
        #     'data': data_dict,
        # }

        return taxonomy_ids, model_ids, data_dict
        