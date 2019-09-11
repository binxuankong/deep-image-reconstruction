import torch
from autoencoder_model import *


# Image features directory
feat_dir = '../data/features/'
# Target directory
target_dir = '../data/encoded/'

cnn_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
              'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2',
              'conv5_3', 'conv5_4', 'fc6', 'fc7']

classes = ['bag', 'bulb', 'candle', 'cup', 'heels', 'jelly_fish', 'lamp', 'leaf', 'momento', 
           'perfume_bottle', 'pineapple', 'plate', 'sneakers', 'sound_box', 'spoon', 'sunflower',
           'table', 'wine_glass']

# GPU configurations
use_gpu = False
gpu_id = 0
if torch.cuda.is_available():
    use_gpu = True

encoded_dict = {}

# Mean and standard deviation of each layer
mean_file = target_dir + 'mean_std.p'
norm_dict = pickle.load(open(mean_file, 'rb'))

for layer in cnn_layers:
    print('Encoding features for {} layer...'.format(layer))

    if 'conv1' in layer:
        model = conv1_autoencoder()
    elif 'conv2' in layer:
        model = conv2_autoencoder()
    elif 'conv3' in layer:
        model = conv3_autoencoder()
    elif 'conv4' in layer:
        model = conv4_autoencoder()
    elif 'conv5' in layer:
        model = conv5_autoencoder()
    elif 'fc' in layer:
        model = fc_autoencoder()

    # Load trained model
    modelfile = target_dir + 'models/vgg19_' + layer + '_n4.pth'
    model.load_state_dict(torch.load(modelfile))
    if use_gpu:
        model = model.cuda(gpu_id)
    model.eval()

    filename = target_dir + 'vgg19_' + layer +'_n4.p'
    print('Opening {} encoded dictionary {}...'.format(layer, filename))
    encoded_dict = pickle.load(open(filename, 'rb'))

    mean = norm_dict[layer]['mean']
    std = norm_dict[layer]['std']

    # Class
    for cls in classes:
        print('Encoding features of class {}'.format(cls))
        class_dir = feat_dir + cls + '/'
        feat_file = class_dir + 'vgg19_' + layer + '.p'
        class_dict = pickle.load(open(feat_file, 'rb'))
        for i in range(len(class_dict)):
            key = list(class_dict.keys())[i]
            feat = list(class_dict.values())[i]
            # De-normalize the features
            feat = (feat - mean) / std
            if use_gpu:
                feat = feat.cuda(gpu_id)
            encoded = model.encode(feat.unsqueeze(0))
            encoded_dict[cls + '*' + key] = encoded[0].cpu()

    # Save to dictionary
    print('Dictionary size: {}'.format(len(encoded_dict)))
    filename = target_dir + 'vgg19_' + layer +'_n4.p'
    print('Dumping {} encoded dictionary to {}...'.format(layer, filename))
    pickle.dump(encoded_dict, open(filename, 'wb'))


