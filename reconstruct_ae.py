import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models, transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
from PIL import Image
from torchvision.utils import save_image
from utils import *
from fmri_decoding.autoencoder_model import *


# img_dir = 'data/images/'
img_dir = '../pan_data/image/all_class/'
ae_dir = 'data/encoded/'
feat_dir = 'data/features/'
fmri_dir = 'data/fmris/'
clss = 'sunflower'
key = '401'
img_size = 224

test_case = 'ae'
num_epochs = 1000
print_iter = 100
save_iter = 500

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
mean_file = ae_dir + 'mean_std.p'
norm_dict = pickle.load(open(mean_file, 'rb'))

for layer in cnn_layers:
    encoded_file = ae_dir + 'vgg19_' + layer + '_n4.p'
    print('Loading encoded dictionary from {}...'.format(encoded_file))
    encoded_dict = pickle.load(open(encoded_file, 'rb'))
    key_id = clss + '*' + key
    encoded_feat = encoded_dict[key_id]
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
        if use_gpu:
            decoder = decoder.cuda(gpu_id)
            encoded_feat = encoded_feat.cuda(gpu_id)
        decoder.load_state_dict(torch.load(decoder_pth, map_location=lambda storage, loc:storage))
        decoder.eval()
        original_feat = decoder.decode(encoded_feat.unsqueeze(0))[0].cpu()
        # Denormalize the features
        mean = norm_dict[layer]['mean']
        std = norm_dict[layer]['std']
        original_feat = original_feat * std + mean

    print('Feature size: {}'.format(original_feat.shape))
    if use_gpu:
        original_feat = original_feat.cuda(gpu_id)
    original_feats.append(original_feat)


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


# Reconstruct the image
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

    # Show feature loss of saved features and original image
    for i, layer in enumerate(cnn_layers):
        x = img_features[i]
        y = original_feats[i]
        feat_loss = nn.MSELoss()
        err = feat_loss(x, y).data.cpu().numpy()
        print('Feature loss in layer {}: {}'.format(layer, err))

    # Decay learning rate by a factor of 0.1 every x epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

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
            im_path = clss + '_' + key + '_i' + str(epoch+1) + '_' + test_case + '.jpg'
            save_image(img_sample, im_path, normalize=True)


image_file = img_dir + clss + '/' + key + '.jpg'
image = Image.open(image_file)

reconstruct_image(image, test_case=test_case, num_epochs=num_epochs, print_iter=print_iter, save_iter=save_iter)

