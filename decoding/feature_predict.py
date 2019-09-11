import numpy as np

from time import time
from slir import SparseLinearRegression
from utils import *


def feature_prediction(x_train, y_train, x_test, num_voxel=500, num_iter=100):
    n_unit = y_train.shape[1]
    print('Number of units in this layer: {}'.format(n_unit))
    print_iter = n_unit // 10

    # Normalize brain data x
    norm_mean_x = np.mean(x_train, axis=0)
    norm_scale_x = np.std(x_train, axis=0, ddof=1)

    x_train = (x_train - norm_mean_x) / norm_scale_x
    x_test = (x_test - norm_mean_x) / norm_scale_x

    y_pred_list = []

    for i in range(n_unit):
        if (i+1) % print_iter == 1:
            start_time = time()
        if (i+1) % print_iter == 0:
            print('Unit {}'.format(i+1))

        # Get unit features
        y_train_unit = y_train[:, i]

        # Normalize image features for training y_train_unit
        norm_mean_y = np.mean(y_train_unit, axis=0)
        std_y = np.std(y_train_unit, axis=0, ddof=1)
        norm_scale_y = 1 if std_y == 0 else std_y

        y_train_unit = (y_train_unit - norm_mean_y) / norm_scale_y

        # Voxel selection
        corr = coef_corr(y_train_unit, x_train, var='col')

        x_train_unit, voxel_index = select_top(x_train, np.abs(corr), num_voxel, axis=1)
        x_test_unit = x_test[:, voxel_index]

        # Add bias terms
        x_train_unit = add_bias(x_train_unit, axis=1)
        x_test_unit = add_bias(x_test_unit, axis=1)

        # Setup regression
        model = SparseLinearRegression(n_iter=num_iter, prune_mode=1)

        # Training and test
        try:
            # Training
            model.fit(x_train_unit, y_train_unit)
            # Testing
            y_pred = model.predict(x_test_unit)
        except:
            # When SLR failed, returns zero-filled array as predicted features
            y_pred = np.zeros(x_test_unit.shape[0])

        # Denormalize predicted features
        y_pred = y_pred * norm_scale_y + norm_mean_y

        y_pred_list.append(y_pred.astype('float32'))

        if (i+1) % print_iter == 0:
            completion = (i+1) * 100 / n_unit
            print('{}% done... Time: {} sec'.format(completion, (time() - start_time)))

    y_predicted = np.vstack(y_pred_list).T
    print('Returning predicted features of size:{}'.format(y_predicted.shape))

    return y_predicted


def get_averaged_feature(pred_y, true_y, labels):
    labels_set = np.unique(labels)

    pred_y_av = np.array([np.mean(pred_y[labels == c, :], axis=0) for c in labels_set])
    true_y_av = np.array([np.mean(true_y[labels == c, :], axis=0) for c in labels_set])

    return pred_y_av, true_y_av, labels_set


