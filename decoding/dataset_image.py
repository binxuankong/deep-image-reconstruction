import os
import pickle
import glob
import torch
import warnings
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage import zoom
from torchvision import datasets, models, transforms


# Load image features of layer
class FeatureDataset(Dataset):
    def __init__(self, feat_dir, classes, layer, norm_dict):
        self.feat_dir = feat_dir
        self.features = []
        self.keys = []
        mean = norm_dict[layer]['mean']
        std = norm_dict[layer]['std']
        for cls in classes:
            class_dir = feat_dir + cls + '/'
            feat_file = class_dir + 'vgg19_' + layer + '.p'
            class_dict = pickle.load(open(feat_file, 'rb'))
            for i in range(len(class_dict)):
                key = list(class_dict.keys())[i]
                feat = list(class_dict.values())[i]
                # Normalize feature
                feat = (feat - mean) / std
                self.keys.append(cls + '*' + key)
                self.features.append(feat)

    def __getitem__(self, index):
        feat = self.features[index]
        key = self.keys[index]
        return {'feat': feat, 'key': key}

    def __len__(self):
        return len(self.features)


# Load fMRI data
class fMRIDataset(Dataset):
    def __init__(self, fmri_dict):
        self.fmri_dict = fmri_dict

    def __getitem__(self, index):
        key_id = list(self.fmri_dict)[index]
        fmri = self.fmri_dict[key_id].astype('float32')
        return fmri, key_id

    def __len__(self):
        return len(self.fmri_dict)


# Load EEG data
class EEGDataset(Dataset):
    def __init__(self, eeg_dict):
        self.eeg_dict = eeg_dict

    def __getitem__(self, index):
        key_id = list(self.eeg_dict)[index]
        fmri = self.eeg_dict[key_id].astype('float32')
        return fmri, key_id

    def __len__(self):
        return len(self.eeg_dict)

