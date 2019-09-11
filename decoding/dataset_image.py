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


class ImageDataFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        (img, label) = super(ImageDataFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        img_filename = path.split('/')[-1]
        img_id = img_filename.replace('.jpg', '')
        return label, img_id


class ImageDataset(Dataset):
    def __init__(self, root_dir, cls, transform=None):
        self.root_dir = root_dir
        self.cls = cls
        self.transform = transform
        self.img_list = [img for img in os.listdir(root_dir + cls + '/')]

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_id = img_name.replace('.jpg', '')
        image = Image.open(img_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, img_id

    def __len__(self):
        return len(self.img_list)


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


class ROIDataset(Dataset):
    def __init__(self, roi_dict):
        self.roi_dict = roi_dict

    def __getitem__(self, index):
        key_id = list(self.roi_dict)[index]
        key_split = key_id.split('*')
        cls = key_split[0]
        cls_id = key_split[1]
        fmri = self.roi_dict[key_id].astype('float32')

        return fmri, cls, cls_id

    def __len__(self):
        return len(self.roi_dict)


class EEGDataset(Dataset):
    def __init__(self, eeg_dict):
        self.eeg_dict = eeg_dict

    def __getitem__(self, index):
        key_id = list(self.eeg_dict)[index]
        key_split = key_id.split('*')
        cls = key_split[0]
        cls_id = key_split[1]
        fmri = self.eeg_dict[key_id].astype('float32')

        return fmri, cls, cls_id

    def __len__(self):
        return len(self.eeg_dict)


