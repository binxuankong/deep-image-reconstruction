import sys
import time
import datetime
import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from dataset_image import *
from autoencoder_model import *


# Image features directory
feat_dir = '../data/features/'
# Target directory
target_dir = '../data/encoded/'

cnn_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
              'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2',
              'conv5_3', 'conv5_4', 'fc6', 'fc7', 'fc8']

classes = ['bag', 'bulb', 'candle', 'cup', 'heels', 'jelly_fish', 'lamp', 'leaf', 'momento', 
           'perfume_bottle', 'pineapple', 'plate', 'sneakers', 'sound_box', 'spoon', 'sunflower',
           'table', 'wine_glass']

norm_dict = {}

for layer in cnn_layers:
    print('Processing for layer {}'.format(layer))
    norm_dict[layer] = {}

    all_feats = []
    for cls in classes:
        class_dir = feat_dir + cls + '/'
        feat_file = class_dir + 'vgg19_' + layer + '.p'
        class_dict = pickle.load(open(feat_file, 'rb'))
        for i in range(len(class_dict)):
            all_feats.append(list(class_dict.values())[i])
    all_feats = torch.stack(all_feats)
    print('All features torch size: {}'.format(all_feats.shape))

    if 'conv' in layer:
        mean = torch.mean(all_feats, (0, 2, 3)).unsqueeze(-1).unsqueeze(-1)
        std = torch.std(all_feats, (0, 2, 3)).unsqueeze(-1).unsqueeze(-1)
    else:
        mean = torch.mean(all_feats, 0)
        std = torch.std(all_feats, 0)

    print('Mean: {}, Std: {}'.format(mean.shape, std.shape))
    norm_dict[layer]['mean'] = mean
    norm_dict[layer]['std'] = std

filename = target_dir + 'mean_std.p'
print('Dumping dictionary to {}...'.format(filename))
pickle.dump(norm_dict, open(filename, 'wb'))


