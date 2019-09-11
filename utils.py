import copy
import numpy as np
import torch

from PIL import Image


def alpha_norm(x, alpha=2.):
    return torch.abs(x.view(-1)**alpha).sum()

def tv_norm(x, beta=2.):
    to_check = x[:,:-1,:-1]
    one_bottom = x[:,1:,:-1]
    one_right = x[:,:-1,1:]
    return (((to_check - one_bottom)**2 + (to_check - one_right)**2)**(beta/2)).sum()

def norm_loss(x, target):
    diff = target - x
    return torch.div(alpha_norm(diff, alpha=2.), alpha_norm(x, alpha=2.))


def estimate_cnn_feat_std(cnn_feat):
    feat_dim = len(cnn_feat.shape)
    feat_size = cnn_feat.shape
    if feat_dim == 1 or (feat_dim == 2 and feat_size[0] == 1) or (feat_dim == 3 and feat_size[1] == 1 and feat_size[2] == 1):
        cnn_feat_std = torch.std(cnn_feat)
    elif feat_dim == 3 and (feat_size[1] > 1 or feat_size[2] > 1):
        # std for each channel
        cnn_feat_std = torch.std(cnn_feat, (1, 2))
        # std averaged across channels
        cnn_feat_std = torch.mean(cnn_feat_std)
    return cnn_feat_std

