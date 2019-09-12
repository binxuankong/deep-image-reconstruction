import pickle
import torch

from torchvision import models, transforms
from PIL import Image


# Image directory
img_dir = '../data/images/'
# Image features directory
target_dir = '../data/img_features/'

cnn_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
              'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2',
              'conv5_3', 'conv5_4', 'fc6', 'fc7', 'fc8']

cnn_layers_dict = {'conv1_1':0, 'conv1_2':2, 'conv2_1':5, 'conv2_2':7, 'conv3_1':10,
                   'conv3_2':12, 'conv3_3':14, 'conv3_4':16, 'conv4_1':19, 'conv4_2':21,
                   'conv4_3':23, 'conv4_4':25, 'conv5_1':28, 'conv5_2':30, 'conv5_3':32,
                   'conv5_4':34, 'fc6':0, 'fc7':3, 'fc8':6}

classes = ['bag', 'bulb', 'candle', 'cup', 'heels', 'jelly_fish', 'lamp', 'leaf', 'momento', 
           'perfume_bottle', 'pineapple', 'plate', 'sneakers', 'sound_box', 'spoon', 'sunflower', 
           'table', 'wine_glass']

img_size = 224

data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

use_gpu = False
gpu_id = 1
if torch.cuda.is_available():
    use_gpu = True

model = models.vgg19(pretrained=True)
if use_gpu:
    model = model.cuda(gpu_id)

model.eval()

for cnn_layer in cnn_layers:
    print('Extracting image features from {} layer...'.format(cnn_layer))
    # Class
    for cls in classes:
        class_dir = target_dir + cls + '/'
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        class_dict = {}
        images = [img for img in os.listdir(img_dir + cls + '/')]

        for image in images:
            img = Image.open(img_dir + cls + '/' + image).convert('RGB')
            img_id = image.replace('.jpg', '')

            x = data_transform(img).unsqueeze(0)
            if use_gpu:
                x = x.cuda(gpu_id)

            # Convolutional layers
            for i, layer in enumerate(model.features):
                x = layer(x)
                if i == cnn_layers_dict[cnn_layer] and 'conv' in cnn_layer:
                    x_feat = x[0].cpu().detach()
                    class_dict[img_id] = x_feat
            # Average pool
            x = model.avgpool(x)
            x = x.view(x.size(0), -1)
            # Fully connected layers
            for i, layer in enumerate(model.classifier):
                x = layer(x)
                if i == cnn_layers_dict[cnn_layer] and 'fc' in cnn_layer:
                    x_feat = x[0].cpu().detach()
                    class_dict[img_id] = x_feat

        filename = class_dir + 'vgg19_' + cnn_layer + '.p'
        print('Dumping {} feature dictionary to {}...'.format(cnn_layer, filename))
        pickle.dump(class_dict, open(filename, 'wb'))

