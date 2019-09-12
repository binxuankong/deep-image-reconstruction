import pickle
import torch
import numpy as np

from torch.utils.data import DataLoader
from dataset_image import EEGDataset
from feature_predict import *


# EEG directory
eeg_dir = '../data/eeg/'
# Encoded image features directory
encoded_dir = '../data/encoded_features/'
# Decoded EEG directory
target_dir = '../data/eeg_decoded/'

# Subject to be decoded (1/2/3/4/5)
subject = 1
# Number of voxels to decoded
num_voxel = 500
# Number of iterations for regression training
num_iter = 100

# Load EEG dictionary for presentations and imaginations
eeg_file = eeg_dir + 'sub' + str(subject) + '.p'
eeg_im1 = eeg_dir + 'sub' + str(subject) + '_im1.p'
eeg_im2 = eeg_dir + 'sub' + str(subject) + '_im2.p'
eeg_im3 = eeg_dir + 'sub' + str(subject) + '_im3.p'

eeg_dict = pickle.load(open(eeg_file, 'rb'))
eeg_im1_dict = pickle.load(open(eeg_im1, 'rb'))
eeg_im2_dict = pickle.load(open(eeg_im2, 'rb'))
eeg_im3_dict = pickle.load(open(eeg_im3, 'rb'))

# Dictionary containing the traning and testing ids
test_train = pickle.load(open('TRAIN_TEST.p', 'rb'))

cnn_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
              'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2',
              'conv5_3', 'conv5_4', 'fc6', 'fc7', 'fc8']

classes = ['bag', 'bulb', 'candle', 'cup', 'heels', 'jelly_fish', 'lamp', 'leaf', 'momento',
           'perfume_bottle', 'pineapple', 'plate', 'sneakers', 'sound_box', 'spoon',
           'sunflower', 'table', 'wine_glass']

# Imagine
testset1 = EEGDataset(eeg_im1_dict)
testset2 = EEGDataset(eeg_im2_dict)
testset3 = EEGDataset(eeg_im3_dict)

test1_loader = DataLoader(testset1, batch_size=1)
test2_loader = DataLoader(testset2, batch_size=1)
test3_loader = DataLoader(testset3, batch_size=1)

# Layers
for layer in cnn_layers:
    print('Processing for layer {}...'.format(layer))
    encoded_file = encoded_dir + 'vgg19_' + layer + '_n4.p'
    print('Reading pickle file {}...'.format(encoded_file))
    encoded_dict = pickle.load(open(encoded_file, 'rb'))

    x_train = []
    x_test = []
    y_train = []

    decoded_dict = {}
    keys = []

    # All classes train
    print('Loading training data...')
    for key_id in test_train['train']:
        if key_id in eeg_dict.keys():
            eeg = eeg_dict[key_id]
            x_train.append(eeg.flatten())
            encoded_feat = encoded_dict[key_id].detach().numpy()
            y_train.append(encoded_feat)
    # All classes valid
    print('Loading validation data...')
    for key_id in test_train['test']:
        if key_id in eeg_dict.keys():
            eeg = eeg_dict[key_id]
            x_test.append(eeg.flatten())
            keys.append(key_id)
    # Imagine
    print('Loading testing (imagine1) data...')
    for eeg, eeg_id in test1_loader:
        x_test.append(eeg.numpy().flatten())
        key_id = 'im1*' + eeg_id[0]
        keys.append(key_id)
    print('Loading testing (imagine2) data...')
    for eeg, eeg_id in test2_loader:
        x_test.append(eeg.numpy().flatten())
        key_id = 'im2*' + eeg_id[0]
        keys.append(key_id)
    print('Loading testing (imagine3) data...')
    for eeg, eeg_id in test3_loader:
        x_test.append(eeg.numpy().flatten())
        key_id = 'im3*' + eeg_id[0]
        keys.append(key_id)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)

    print('Training data (fmri) size: {}'.format(x_train.shape))
    print('Training label (image features) size: {}'.format(y_train.shape))
    print('Testing data(fmri) size: {}'.format(x_test.shape))

    pred_y = feature_prediction(x_train, y_train, x_test, num_voxel=num_voxel,
                                num_iter=num_iter)
    decoded_fmri = [y for y in pred_y]

    for i in range(len(decoded_fmri)):
        name = keys[i]
        decoded_dict[name] = decoded_fmri[i]

    file_name = target_dir + '/sub' + str(subject) + '/vgg19_' + layer + '_n5.p'
    print('Dumping decoded fmri dictionary to {}...'.format(file_name))
    pickle.dump(decoded_dict, open(file_name, 'wb'))

