"""Based on https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer/blob/main/data/pascal.py
"""
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class DatasetColorization(Dataset):
    def __init__(self, datapath, image_transform, mask_transform, padding: bool = 1,
                 use_original_imgsize: bool = False, flipped_order: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False):
        self.padding = padding
        self.random = random
        self.use_original_imgsize = use_original_imgsize
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform
        self.ds = ImageFolder(os.path.join(datapath, 'val'))
        self.flipped_order = flipped_order
        np.random.seed(5)
        self.indices = np.random.choice(np.arange(0, len(self.ds)-1), size=1000, replace=False)


    def __len__(self):
        return 1000

    def create_grid_from_images(self, support_img, support_mask, query_img, query_mask):
        if self.reverse_support_and_query:
            support_img, support_mask, query_img, query_mask = query_img, query_mask, support_img, support_mask
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        if self.flipped_order:
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
        else:
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas

    def __getitem__(self, idx):
        support_idx = np.random.choice(np.arange(0, len(self)-1))
        idx = self.indices[idx]
        query, support = self.ds[idx], self.ds[support_idx]
        query_img, query_mask = self.mask_transform(query[0]), self.image_transform(query[0])
        support_img, support_mask = self.mask_transform(support[0]), self.image_transform(support[0])
        grid = self.create_grid_from_images(support_img, support_mask, query_img, query_mask)
        batch = {'query_img': query_img, 'query_mask': query_mask, 'support_img': support_img,
                 'support_mask': support_mask, 'grid': grid}

        return batch