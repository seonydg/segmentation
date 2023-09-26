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


class ToTensor(object):
    def __call__(self, data):
        image, label = data['image'], data['label']

        image = image.transpose((2, 0, 1)).astype(np.float32)
        label = label.transpose((2, 0, 1)).astype(np.float32)

        data = {'image':torch.from_numpy(image), 'label':torch.from_numpy(label)}

        return data


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
    
    def __call__(self, data):
        image, label = data['images'], data['labels']

        image = (image - self.mean) / self.std
        label = (label - self.mean) / self.std

        data = {'image':image, 'label':label}

        return data


class RandomFlip(object):
    def __call__(self, data):
        image, label = data['image'], data['label']

        if np.random.rand() > 0.5:
            image = np.fliplr(image)
            label = np.fliplr(label)

        if np.random.rand() > 0.5:
            image = np.flipud(image)
            label = np.flipud(label)

        data = {'image':image, 'label':label}

        return data