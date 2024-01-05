"""

Functions for DataLoader
"""

import os
import h5py
import numpy as np

import torch
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision import transforms, utils

class CustomDataset(Dataset):

    """Class for constructing a custom dataset"""

    def __init__(self, root_dir, pos_file_name, neg_file_name, dataset_type, transform=True):

        self.root_dir = root_dir
        self.pos_dir = os.path.join(root_dir, f'{pos_file_name}')
        self.neg_dir = os.path.join(root_dir, f'{neg_file_name}')

        with h5py.File(self.pos_dir) as f:
            self.valid_idx_pos = [x for x in range(len(f.attrs['dataset'])) if f.attrs['dataset'][x] == 'dataset_type']
            self.num_files_pos = len(valid_idx_pos)
        with h5py.File(self.neg_dir) as f:
            self.valid_idx_neg = [x for x in range(len(f.attrs['dataset'])) if f.attrs['dataset'][x] == 'dataset_type']
            self.num_files_neg = len(valid_idx_neg)

        self.num_files = self.num_files_pos + self.num_files_neg

        self.transform = transform

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):

        if idx < self.num_files_pos:
            new_idx = self.valid_idx_pos[idx]
            with h5py.File(self.pos_dir) as f:
                image = f['data'][new_idx, ...].astype(np.float64)
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5), (0.330,)),
                                            ])

            sample = transform(image)
            label = 1

        elif idx >= self.num_files_pos:
            with h5py.File(self.neg_dir) as f:
                new_idx = self.valid_idx_neg[idx - self.num_files_pos]
                image = f['data'][new_idx, ...].astype(np.float64)
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5), (0.330,)),
                                            ])

            sample = transform(image)
            label = 0

        return sample, label
