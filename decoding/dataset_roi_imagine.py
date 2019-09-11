import pickle
import numpy as np

from operator import add

dict_dir = '../../pan_data/imagine/imagery-data/'
target_dir = '../data/rois/'

subjects = [1, 2, 3, 4, 5]
sub_num_dict = [2, 3, 3, 3, 3]

rois= ['Fusiform_L', 'Fusiform_R', 'Occipital_inf_L', 'Occipital_inf_R', 'Occipital_Mid_L',
       'Occipital_Mid_R', 'Occipital_Sup_L', 'Occipital_Sup_R', 'ParaHippocampus_L', 'ParaHippocampus_R']

sub = 2

for i in range(sub_num_dict[sub-1]):
    dict_index = i + 3
    print('Creating dict {} for subject {}...'.format(dict_index, sub))
    sub_dict = {}
    all_fmris = []

    # Region of interest
    for region in rois:
        roi_fmris = []
        keys = []
        print('Reading region: {}...'.format(region))
        dict_file = dict_dir + str(sub) + '/' + region + '/dictionary' + str(dict_index) + '.p'
        with open(dict_file, 'rb') as f:
            train = pickle.load(f)
        # Class
        for cls in train.keys():
            print('Processing class: {}...'.format(cls))
            fmri = train[cls][0]
            # Image
            for i in range(len(fmri)):
                name = list(fmri.keys())[i]
                img_roi = []
                #fMRI
                for j in range(len(fmri[name])):
                    '''
                    roi_fmris.append(list(fmri[name][j].values())[0])
                    key_id = cls + '*' + name + '*' + str(j)
                    keys.append(key_id)
                    '''
                    img_roi.append(list(fmri[name][j].values())[0])
                # Get average of the fmris
                avg_roi = np.mean(np.asarray(img_roi), 0)
                roi_fmris.append(avg_roi)
                key_id = cls + '*' + name
                keys.append(key_id)

        if region == 'Fusiform_L':
            all_fmris = roi_fmris
        else:
            all_fmris = list(map(add, all_fmris, roi_fmris))

    # Flatten and remove zeros
    print(len(all_fmris))
    code_list = []
    record_length = 10000000000
    for i, fmri in enumerate(all_fmris):
        a = np.nonzero(fmri)
        code_list.append(((fmri[a])[:4500]).reshape(4500))
        if record_length > len(a[0]):
            record_length = len(a[0])
    print(code_list[0].shape)
    print(record_length)
    '''
        x = np.trim_zeros(fmri.flatten())
        # Downsample the fmri
        all_fmris[i] = np.interp(np.arange(0, len(x), 30), np.arange(0, len(x)), x)
    '''
    
    for i in range(len(code_list)):
        name = keys[i]
        sub_dict[name] = code_list[i]

    filename = target_dir + 'sub' + str(sub) + '_imagine_avg_' + str(dict_index) + '.p'
    print('Dumping roi dictionary to {}...'.format(filename))
    pickle.dump(sub_dict, open(filename, 'wb'))
