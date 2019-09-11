import pickle
import numpy as np

from operator import add


# Directory of dictionary
dict_dir = '/ssd3/Workspace/pan/pan_data/roidict/'
dict_name = 'all_class_dict.p'
# Target directory
target_dir = '../data/rois/'

subjects = [1, 2, 3, 4, 5]
rois= ['fusiform_l', 'fusiform_r', 'occipital_inf_l', 'occipital_inf_r', 'occipital_mid_l',
       'occipital_mid_r', 'occipital_sup_l', 'occipital_sup_r', 'parahippocampus_l', 'parahippocampus_r']
classes = ['bag', 'bulb', 'candle', 'cup', 'heels', 'jelly_fish', 'lamp', 'leaf', 'momento', 'perfume_bottle',
           'pineapple', 'plate', 'sneakers', 'sound_box', 'spoon', 'sunflower', 'table', 'wine_glass']

# Subjects
for sub in subjects:
    print('Creating dictionary for subject {}...'.format(sub))
    sub_dict = {}
    all_fmris = []

    # Region of interest
    for region in rois:
        roi_fmris = []
        keys = []
        print('Reading region: {}...'.format(region))
        with open(dict_dir + str(sub) + '/' + region + '/' + dict_name, 'rb') as f:
            train = pickle.load(f)
        # Class
        for cls in classes:
            print('Processing class: {}...'.format(cls))
            if cls == 'sneakers':
                fmri = train['boots']
            else:
                fmri = train[cls]
            # Image
            for i in range(len(fmri)):
                name = str(','.join(fmri[i].keys()))
                # fMRI
                img_roi = []
                for j in range(len(fmri[i][name])):
                    img_roi.append(fmri[i][name][j])
                # Get average of the fmris
                avg_roi = np.mean(np.asarray(img_roi), 0)
                roi_fmris.append(avg_roi)
                key_id = cls + '*' + name
                keys.append(key_id)

        if region == 'fusiform_l':
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

    # Save to dictionary
    for i in range(len(code_list)):
        name = keys[i]
        sub_dict[name] = code_list[i]
    
    filename = target_dir + 'sub' + str(sub) + '_roi_avg.p'
    print('Dumping roi dictionary to {}...'.format(filename))
    pickle.dump(sub_dict, open(filename, 'wb'))


