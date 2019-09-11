import pickle
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset_image import *
from feature_predict import *


img_dir = '../../pan_data/image/all_class/'
roi_dir = '../data/rois/'
feat_dir = '../data/features/'
pca_dir = '../data/pca/'
ae_dir = '../data/encoded/'
target_dir = '../data/fmris/'

subject = 2
num_voxel = 500
num_iter = 100

roi_file = roi_dir + 'sub' + str(subject) + '_roi_avg.p'
roi_imagine1 = roi_dir + 'sub' + str(subject) + '_imagine_avg_1.p'
# roi_imagine2 = roi_dir + 'sub' + str(subject) + '_imagine_avg_2.p'
roi_imagine3 = roi_dir + 'sub' + str(subject) + '_imagine_avg_3.p'

roi_dict = pickle.load(open(roi_file, 'rb'))
roi_imagine1_dict = pickle.load(open(roi_imagine1, 'rb'))
# roi_imagine2_dict = pickle.load(open(roi_imagine2, 'rb'))
roi_imagine3_dict = pickle.load(open(roi_imagine3, 'rb'))

cnn_layers = ['conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2',
              'conv5_3', 'conv5_4', 'fc6', 'fc7', 'fc8']
'''
cnn_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
              'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2',
              'conv5_3', 'conv5_4', 'fc6', 'fc7', 'fc8']
'''
classes = ['bag', 'bulb', 'candle', 'cup', 'heels', 'jelly_fish', 'lamp', 'leaf', 'momento',
           'perfume_bottle', 'pineapple', 'plate', 'sneakers', 'sound_box', 'spoon',
           'sunflower', 'table', 'wine_glass']

# All classes
dataset = ROIDataset(roi_dict)
validation_split = 0.2
shuffle_dataset = True
random_seed = 42

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, valid_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=1, sampler=valid_sampler)

# Imagine
testset1 = ROIDataset(roi_imagine1_dict)
# testset2 = ROIDataset(roi_imagine2_dict)
testset3 = ROIDataset(roi_imagine3_dict)

test1_loader = DataLoader(testset1, batch_size=1)
# test2_loader = DataLoader(testset2, batch_size=1)
test3_loader = DataLoader(testset3, batch_size=1)

# Layers
for layer in cnn_layers:
    print('Processing for layer {}...'.format(layer))
    ae_file = ae_dir + 'vgg19_' + layer + '_n4.p'
    print('Reading pickle file {}...'.format(ae_file))
    ae_dict = pickle.load(open(ae_file, 'rb'))

    x_train = []
    x_test = []
    y_train = []

    decoded_dict = {}
    keys = []

    # All classes train
    print('Loading training data...')
    for fmri_vector, cls, cls_id in train_loader:
        x_train.append(fmri_vector.numpy().flatten())
        key_id = cls[0] + '*' + cls_id[0]
        encoded_feat = ae_dict[key_id].detach().numpy()
        y_train.append(encoded_feat)
    # All classes valid
    print('Loading validation data...')
    for fmri_vector, cls, cls_id in valid_loader:
        x_test.append(fmri_vector.numpy().flatten())
        key_id = 'all*' + cls[0] + '*' + cls_id[0]
        keys.append(key_id)
    # Imagine
    print('Loading testing (imagine1) data...')
    for fmri_vector, cls, cls_id in test1_loader:
        x_test.append(fmri_vector.numpy().flatten())
        key_id = 'im1*' + cls[0] + '*' + cls_id[0]
        keys.append(key_id)
    '''
    print('Loading testing (imagine2) data...')
    for fmri_vector, cls, cls_id in test2_loader:
        x_test.append(fmri_vector.numpy().flatten())
        key_id = 'im2*' + cls[0] + '*' + cls_id[0]
        keys.append(key_id)
    '''
    print('Loading testing (imagine3) data...')
    for fmri_vector, cls, cls_id in test3_loader:
        x_test.append(fmri_vector.numpy().flatten())
        key_id = 'im3*' + cls[0] + '*' + cls_id[0]
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

    file_name = target_dir + '/sub' + str(subject) + '/vgg19_' + layer + '_n4.p'
    print('Dumping decoded fmri dictionary to {}...'.format(file_name))
    pickle.dump(decoded_dict, open(file_name, 'wb'))


