import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import models, transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
from PIL import Image
from torchvision.utils import save_image
from utils import *

plt.style.use('seaborn-white')


# img_dir = 'data/images/'
img_dir = '../pan_data/image/all_class/'
feat_dir = 'data/features/'
clss = 'sunflower'
key = '401'
img_size = 224

test_case = ''
num_epochs = 1000
print_iter = 100
save_iter = 500

# GPU config
use_gpu = False
gpu_id = 0
if torch.cuda.is_available():
    use_gpu = True

cnn_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
              'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2',
              'conv5_3', 'conv5_4', 'fc6', 'fc7', 'fc8']
cnn_layers_dict = {'conv1_1':0, 'conv1_2':2, 'conv2_1':5, 'conv2_2':7, 'conv3_1':10,
                   'conv3_2':12, 'conv3_3':14, 'conv3_4':16, 'conv4_1':19, 'conv4_2':21,
                   'conv4_3':23, 'conv4_4':25, 'conv5_1':28, 'conv5_2':30, 'conv5_3':32,
                   'conv5_4':34, 'fc6':0, 'fc7':3, 'fc8':6}

original_feats = []
'''
for layer in cnn_layers:
    feat_file = feat_dir + clss + '/' + 'vgg19_' + layer + '.p'
    print('Loading image features dictionary from {}...'.format(feat_file))
    layer_dict = pickle.load(open(feat_file, 'rb'))
    feat = layer_dict[key]
    if use_gpu:
        feat = feat.cuda(gpu_id)
    original_feats.append(feat)
'''

def get_features_from_layers(model, x):
    img_features = []
    layer_id = 0

    # Convolutional layers
    for i, layer in enumerate(model.features):
        this_layer = cnn_layers[layer_id]
        x = layer(x)
        if i == cnn_layers_dict[this_layer] and 'conv' in this_layer:
            x_feat = x[0]
            img_features.append(x_feat)
            layer_id += 1
    # Average pool
    x = model.avgpool(x)
    x = x.view(x.size(0), -1)
    # Fully connected layers
    for i, layer in enumerate(model.classifier):
        this_layer = cnn_layers[layer_id]
        x = layer(x)
        if i == cnn_layers_dict[this_layer] and 'fc' in this_layer:
            x_feat = x[0]
            img_features.append(x_feat)
            layer_id += 1

    return img_features


def get_total_loss(recon_img, output, target):
    # Alpha regularization parametrs
    # Parameter alpha, which is actually sixth norm
    alpha = 6
    # The multiplier, lambda alpha
    lambda_alpha = 1e-7

    # Total variation regularization parameters
    # Parameter beta, which is actually second norm
    tv_beta = 2
    # The multiplier, lambda beta
    lambda_beta = 1e-7

    euc_loss = 0
    alpha_loss = 0
    tv_loss = 0

    for i in range(len(output)):
        # Calculate euclidian loss
        euc_loss += 1e-1 * norm_loss(target[i].detach(), output[i])
        # Calculate alpha regularization
        alpha_loss += lambda_alpha * alpha_norm(recon_img, alpha)
        # Calculate total variation regularization
        tv_loss += lambda_beta * tv_norm(recon_img, tv_beta)

    # Sum all to optimize
    total_loss = euc_loss + alpha_loss + tv_loss

    return euc_loss, alpha_loss, tv_loss, total_loss


def reconstruct_image(image, img_size=224, test_case=1, num_epochs=200, print_iter=10, save_iter=50):
    data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    img = data_transform(image).unsqueeze(0)

    model = models.vgg19(pretrained=True)
    if use_gpu:
        model = model.cuda(gpu_id)
    model.eval()

    # Generate a random image which we will optimize
    if use_gpu:
        recon_img = Variable(1e-1 * torch.randn(1, 3, img_size, img_size).cuda(gpu_id), requires_grad=True)
    else:
        recon_img = Variable(1e-1 * torch.randn(1, 3, img_size, img_size), requires_grad=True)

    # Define optimizer for previously created image
    optimizer = optim.SGD([recon_img], lr=1e2, momentum=0.9)

    # Get the features from the model of the original image
    if use_gpu:
        img = img.cuda(gpu_id)
    img_features = get_features_from_layers(model, img)
    '''
    # Show feature loss of saved features and original image
    for i, layer in enumerate(cnn_layers):
        x = img_features[i]
        y = original_feats[i]
        feat_loss = nn.MSELoss()
        err = feat_loss(x, y)
        print('Feature loss in layer {}: {}'.format(layer, err))
    '''
    # Decay learning rate by a factor of 0.1 every x epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    euc_losses = []
    alpha_losses = []
    tv_losses = []
    total_losses = []

    for epoch in range(num_epochs):
        scheduler.step()
        optimizer.zero_grad()

        # Get the features from the model of the generated image
        output_features = get_features_from_layers(model, recon_img)

        # Calculate the losses
        euc_loss, alpha_loss, tv_loss, total_loss = get_total_loss(recon_img, output_features, img_features)

        # Step
        total_loss.backward()
        optimizer.step()
        # Generate image every x iterations
        if (epoch+1) % print_iter == 0:
            print('Epoch %d:\tAlpha: %.5f\tTV: %.5f\tEuc: %.5f\tLoss: %.5f' % (epoch+1,
                alpha_loss.data.cpu().numpy(), tv_loss.data.cpu().numpy(),
                euc_loss.data.cpu().numpy(), total_loss.data.cpu().numpy()))

        if (epoch+1) % save_iter == 0:
            img_sample = torch.squeeze(recon_img.cpu())
            im_path = clss + '_' + key + '_i' + str(epoch+1) + '_' + test_case + '.jpg'
            save_image(img_sample, im_path, normalize=True)

        euc_losses.append(euc_loss.data.cpu().numpy())
        alpha_losses.append(alpha_loss.data.cpu().numpy())
        tv_losses.append(tv_loss.data.cpu().numpy())
        total_losses.append(total_loss.data.cpu().numpy())
    '''
    epoches = np.arange(num_epochs)
    plt.figure
    plt.plot(epoches, euc_losses, 'r-', label='euclidean loss')
    plt.plot(epoches, alpha_losses, 'b-', label='alpha loss')
    plt.plot(epoches, tv_losses, 'g-', label='tv loss')
    plt.plot(epoches, total_losses, 'k-', label='total loss')
    plt.title('Image Reconstruction Loss', fontsize=20)
    plt.xlabel('Number of epochs', fontsize='16')
    leg = plt.legend(fontsize=14)
    leg.get_frame().set_edgecolor('black')
    plt.show()
    '''

image_file = img_dir + clss + '/' + key + '.jpg'
image = Image.open(image_file)

reconstruct_image(image, test_case=test_case, num_epochs=num_epochs, print_iter=print_iter, save_iter=save_iter)

