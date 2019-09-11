import pickle
import numpy as np

from operator import add


# Directory of dictionary
dict_dir = '/ssd3/Workspace/pan/pan_data/imagine/eeg_img/imagine/'
# Target directory to be saved
target_dir = '../data/eeg/'

# Subject (1/2/3/4/5)
sub = '1'
# Subject dictionary file name
dict_name = dict_dir + sub + '_3.p'

classes = ['bag', 'bulb', 'candle', 'cup', 'heels', 'jelly_fish', 'lamp', 'leaf', 'momento', 
           'perfume_bottle', 'pineapple', 'plate', 'sneakers', 'sound_box', 'spoon', 'sunflower', 
           'table', 'wine_glass']

print('Creating dictionary for subject {}...'.format(sub))
sub_dict = {}


# Presentation
for cls in classes:
    print('Processing class: {}...'.format(cls))
    eeg = train[cls]
    for i in range(len(eeg)):
        name = str(','.join(eeg[i].keys()))
        # EEG
        img_eeg = []
        for j in range(len(eeg[i][name])):
            img_eeg.append(eeg[i][name][j])
        # Get average of the eegs
        avg_eeg = np.mean(np.asarray(img_eeg), 0)
        # Reduce the eeg channel size by taking the mean
        avg_eeg = np.mean(avg_eeg, 0)
        key_id = cls + '*' + name
        sub_dict[key_id] = avg_eeg

# Imagine
classes = train['0'][0].keys()
for cls in classes:
    print('Processing class: {}...'.format(cls))
    eeg = train['0'][0][cls]
    for i in range(len(eeg)):
        name = list(eeg.keys())[i]
        # EEG
        img_eeg = []
        for key in train.keys():
            img_eeg.append(train[key][0][cls][name][0])
        # Get average of the eegs
        avg_eeg = np.mean(np.asarray(img_eeg), 0)
        # Reduce the eeg channel size by taking the mean
        avg_eeg = np.mean(avg_eeg, 0)
        key_id = cls + '*' + name
        sub_dict[key_id] = avg_eeg
        print(key_id)

# Imagine Sub 5 Dict 3
for cls in train.keys():
    print('Processing class: {}...'.format(cls))
    eeg = train[cls][0]
    for name in eeg.keys():
        # EEG
        img_eeg = [] 
        for i in range(4):
            img_eeg.append(list(eeg[name][i].values())[0])
        # Get average of the eegs
        avg_eeg = np.mean(np.asarray(img_eeg), 0)
        # Reduce the eeg channel size by taking the mean
        avg_eeg = np.mean(avg_eeg, 0)
        key_id = cls + '*' + name
        sub_dict[key_id] = avg_eeg
        print(key_id)

print(len(sub_dict))
filename = target_dir + 'sub' + str(sub) + '_imagine_3.p'
print('Dumping eeg dictionary to {}...'.format(filename))
pickle.dump(sub_dict, open(filename, 'wb'))


