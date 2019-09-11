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
ae_dir = '../data/encoded/'
target_dir = '../data/fmris/'

subject = 5
num_voxel = 500
num_iter = 100

test_train = pickle.load(open('TRAIN_TEST.p', 'rb'))

roi_file = roi_dir + 'sub' + str(subject) + '_roi_avg.p'
roi_imagine1 = roi_dir + 'sub' + str(subject) + '_imagine_avg_1.p'
roi_imagine2 = roi_dir + 'sub' + str(subject) + '_imagine_avg_2.p'
roi_imagine3 = roi_dir + 'sub' + str(subject) + '_imagine_avg_3.p'

roi_dict = pickle.load(open(roi_file, 'rb'))
roi_imagine1_dict = pickle.load(open(roi_imagine1, 'rb'))
roi_imagine2_dict = pickle.load(open(roi_imagine2, 'rb'))
roi_imagine3_dict = pickle.load(open(roi_imagine3, 'rb'))

# Imagine
testset1 = ROIDataset(roi_imagine1_dict)
testset2 = ROIDataset(roi_imagine2_dict)
testset3 = ROIDataset(roi_imagine3_dict)

test1_loader = DataLoader(testset1, batch_size=1)
test2_loader = DataLoader(testset2, batch_size=1)
test3_loader = DataLoader(testset3, batch_size=1)


cnn_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
              'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2',
              'conv5_3', 'conv5_4', 'fc6', 'fc7', 'fc8']

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
    for key_id in test_train['train']:
        if key_id in roi_dict.keys():
            fmri = roi_dict[key_id]
            x_train.append(fmri.flatten())
            encoded_feat = ae_dict[key_id].detach().numpy()
            y_train.append(encoded_feat)
    # All classes valid
    print('Loading validation data...')
    for key_id in test_train['test']:
        if key_id in roi_dict.keys():
            fmri = roi_dict[key_id]
            x_test.append(fmri.flatten())
            keys.append(key_id)
    # Imagine
    print('Loading testing (imagine1) data...')
    for fmri_vector, cls, cls_id in test1_loader:
        x_test.append(fmri_vector.numpy().flatten())
        key_id = 'im1*' + cls[0] + '*' + cls_id[0]
        keys.append(key_id)
    print('Loading testing (imagine2) data...')
    for fmri_vector, cls, cls_id in test2_loader:
        x_test.append(fmri_vector.numpy().flatten())
        key_id = 'im2*' + cls[0] + '*' + cls_id[0]
        keys.append(key_id)
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

    file_name = target_dir + '/sub' + str(subject) + '/vgg19_' + layer + '_n5.p'
    print('Dumping decoded fmri dictionary to {}...'.format(file_name))
    pickle.dump(decoded_dict, open(file_name, 'wb'))


