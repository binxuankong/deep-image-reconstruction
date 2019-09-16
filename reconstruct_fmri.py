import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import matplotlib.pyplot as plt

from torchvision import models, transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
from PIL import Image
from torchvision.utils import save_image
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
from utils import *
from fmri_decoding.autoencoder_model import *


# img_dir = 'data/images/'
img_dir = '../pan_data/image/all_class/'
ae_dir = 'data/encoded/'
feat_dir = 'data/features/'
fmri_dir = 'data/fmris/'
target_dir = 'images/autoencoder/for_report/'

subject = '2'
img_size = 224

# im1 classes: 1#jellyfish, 2#soundbox, 3#pineapple1, 4#sunflower, 5#wineglass, 6#bulb, 7#leaf, 8#momento, 9#pineapple2, 10#heels
# im2 classes: 1#bag, 2#redbag, 3#plate, 4#greenplate, 5#greenroundplate, 6#heels, 7#boots, 8#dailyboots, 9#redheels, 10#partyheels
# im3 classes: 1#1, 2#2, 3#3, 4#4, 5#5, 6#6, 7#7, 8#8, 9#9, 10#10

key_id = 'all*sunflower*2'
type_ = key_id.split('*')[0]
clss = key_id.split('*')[-2]
key = key_id.split('*')[-1]
print('Processing: class {}, key {}'.format(clss, key))

test_case = 'dgn'
num_epochs = 1000
print_iter = 100
save_iter = 1000

# Use BigGAN deep generator network as natural image prior
use_dgn = True

# GPU config
use_gpu = False
gpu_id = 1
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
layer_weights = []
mean_file = ae_dir + 'mean_std.p'
norm_dict = pickle.load(open(mean_file, 'rb'))

for layer in cnn_layers:
    # fMRI
    fmri_file = fmri_dir + 'sub' + subject + '/vgg19_' + layer + '_n4.p'
    print('Loading fmri dictionary from {}...'.format(fmri_file))
    fmri_dict = pickle.load(open(fmri_file, 'rb'))
    encoded_feat = fmri_dict[key_id]
    encoded_feat = torch.from_numpy(encoded_feat)
    if 'fc8' in layer:
        original_feat = encoded_feat
    else:
        if 'conv1' in layer:
            decoder = conv1_autoencoder()
        elif 'conv2' in layer:
            decoder = conv2_autoencoder()
        elif 'conv3' in layer:
            decoder = conv3_autoencoder()
        elif 'conv4' in layer:
            decoder = conv4_autoencoder()
        elif 'conv5' in layer:
            decoder = conv5_autoencoder()
        elif 'fc' in layer:
            decoder = fc_autoencoder()
        decoder_pth = ae_dir + 'models/vgg19_' + layer + '_n4.pth'
        decoder.load_state_dict(torch.load(decoder_pth, map_location=lambda storage, loc:storage))
        decoder.eval()
        original_feat = decoder.decode(encoded_feat.unsqueeze(0))[0]
        # Denormalize the features
        mean = norm_dict[layer]['mean']
        std = norm_dict[layer]['std']
        original_feat = (original_feat * std) + mean

    # Weight of each layer
    feat_norm = np.linalg.norm(original_feat.detach().numpy())
    weight = 1. / (feat_norm ** 2)
    layer_weights.append(weight)
    
    print('Feature size: {}'.format(original_feat.shape))
    if use_gpu:
        original_feat = original_feat.cuda(gpu_id)
    original_feats.append(original_feat)

layer_weights = np.asarray(layer_weights)
layer_weights = layer_weights / layer_weights.sum()


# Extract image features from each layer
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
        data_feat = []
        x = layer(x)
        if i == cnn_layers_dict[this_layer] and 'fc' in this_layer:
            x_feat = x[0]
            img_features.append(x_feat)
            layer_id += 1

    return img_features


# Get the sum of alpha loss, tv loss and norm loss
def get_total_loss(recon_img, output, target):
    # Alpha regularization parametrs
    # Parameter alpha, which is actually sixth norm
    alpha = 6
    # The multiplier, lambda alpha
    lambda_alpha = 1e-6 if use_dgn else 1e-5

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


# Reconstruct the image
def reconstruct_image(img_size=224, test_case=1, num_epochs=200, print_iter=10, save_iter=50):
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

    # Use deep generator network to get initial image
    if use_dgn:
        print('Loading deep generator network...')
        # Load pre-trained model tokenizer(vocabulary)
        dgn = BigGAN.from_pretrained('biggan-deep-256')
        # Prepare an input
        truncation = 0.4
        class_vector = np.zeros((1, 1000), dtype='float32')
        noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)
        # All in tensors
        class_vector = torch.from_numpy(class_vector)
        noise_vector = torch.from_numpy(noise_vector)
        if use_gpu:
            class_vector = class_vector.cuda(gpu_id)
            noise_vector = noise_vector.cuda(gpu_id)
            dgn.cuda(gpu_id)
        # Generate image
        with torch.no_grad():
            output = dgn(noise_vector, class_vector, truncation)
            output = output.cpu()
            output = nn.functional.interpolate(output, size=(img_size, img_size), mode='bilinear', align_corners=True)
        if use_gpu:
            recon_img = Variable(output.cuda(gpu_id), requires_grad=True)
        else:
            recon_img = Variable(output, requires_grad=True)

    # Define optimizer for previously created image
    optimizer = optim.SGD([recon_img], lr=1e2, momentum=0.9)

    # Decay learning rate by a factor of 0.1 every x epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    for epoch in range(num_epochs):
        scheduler.step()
        optimizer.zero_grad()

        # Get the features from the model of the generated image
        output_features = get_features_from_layers(model, recon_img)

        # Calculate the losses
        euc_loss, alpha_loss, tv_loss, total_loss = get_total_loss(recon_img, output_features, original_feats)

        # Step
        total_loss.backward()
        optimizer.step()

        # Generate image every x iterations
        if (epoch+1) % print_iter == 0:
            print('Epoch %d:\tAlpha: %.6f\tTV: %.6f\tEuc: %.6f\tLoss: %.6f' % (epoch+1,
                alpha_loss.data.cpu().numpy(), tv_loss.data.cpu().numpy(),
                euc_loss.data.cpu().numpy(), total_loss.data.cpu().numpy()))

        if (epoch+1) % save_iter == 0:
            img_sample = torch.squeeze(recon_img.cpu())
            im_path = target_dir + clss + '*' + key + '_sub' + subject + '_' + test_case + '.jpg'
            save_image(img_sample, im_path, normalize=True)


reconstruct_image(test_case=test_case, num_epochs=num_epochs, print_iter=print_iter, save_iter=save_iter)

