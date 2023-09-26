import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

# dataset
class MyDataset(Dataset):
    def __init__(self, data_dir, phase, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.phase = phase

        data = os.listdir(os.path.join(self.data_dir, self.phase))
        
        data_images = [file for file in data if file.startswith('image')]
        data_labels = [file for file in data if file.startswith('label')]

        data_images.sort()
        data_labels.sort()

        self.data_images = data_images
        self.data_labels = data_labels


    def __len__(self):
        return len(self.data_labels)
    

    def __getitem__(self, index):
        images = np.load(os.path.join(self.data_dir, self.phase, self.data_images[index]))
        labels = np.load(os.path.join(self.data_dir, self.phase, self.data_labels[index]))

        images = images / 255.
        labels = labels / 255.

        if images.ndim == 2:
            images = images[:, :, np.newaxis]
        if labels.ndim == 2:
            labels = labels[:, :, np.newaxis]
        
        out = {'images':images, 'labels':labels}

        if self.transform:
            out = self.transform(out)

        return out

