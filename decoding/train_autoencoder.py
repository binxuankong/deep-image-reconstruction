import time
import datetime
import torch

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset_image import FeatureDataset
from autoencoder_model import *


# Image features directory
feat_dir = '../data/img_features/'
# Trained models directory
target_dir = '../data/trained_models/'

cnn_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
              'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2',
              'conv5_3', 'conv5_4', 'fc6', 'fc7']

classes = ['bag', 'bulb', 'candle', 'cup', 'heels', 'jelly_fish', 'lamp', 'leaf', 'momento', 
           'perfume_bottle', 'pineapple', 'plate', 'sneakers', 'sound_box', 'spoon', 'sunflower', 
           'table', 'wine_glass']

# Training configurations
num_epochs = 100
batch_size = 32
learning_rate = 5e-4
print_interval = 5
validation_split = 0.1
shuffle_dataset = True
random_seed = 42
patience = 5

# GPU configurations
use_gpu = False
gpu_id = 1
if torch.cuda.is_available():
    use_gpu = True

# Mean and standard devation feature layers of images
mean_file = target_dir + 'mean_std.p'
norm_dict = pickle.load(open(mean_file, 'rb'))

for layer in cnn_layers:
    print('Training autoencoder for {} layer...'.format(layer))
    # Configure data loader
    encoded_dict = {}
    print('Loading data...')
    dataset = FeatureDataset(feat_dir, classes, layer, norm_dict)

    # Split data into training and validation set
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

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

    # Loss function
    criterion = nn.MSELoss()
    if use_gpu:
        model = model.cuda(gpu_id)
        criterion = criterion.cuda(gpu_id)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    prev_time = time.time()
    best_loss = 10000
    counter = 0
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for i, data in enumerate(train_loader):
            feat = data['feat'].cuda(gpu_id) if use_gpu else data['feat']
            optimizer.zero_grad()
            # Forward
            output = model(feat)
            loss = criterion(output, feat)
            # Step
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        valid_loss = 0
        for i, data in enumerate(valid_loader):
            feat = data['feat'].cuda(gpu_id) if use_gpu else data['feat']
            output = model(feat)
            loss = criterion(output, feat)
            valid_loss += loss.item()

        # Early stopping if validation loss does not decrease any further
        if valid_loss < best_loss:
            best_loss = valid_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping: [Epoch %d/%d] [Train Loss: %f] [Valid Loss: %f] [Best Loss: %f]"
                  % (epoch+1, num_epochs, train_loss/len(train_loader), valid_loss/len(valid_loader),
                     best_loss/len(valid_loader)))
            break

        prev_loss = valid_loss

        if (epoch+1) % print_interval == 0:
            # Log progress
            # Determine approximate time left
            epoches_left = num_epochs - epoch
            time_left = datetime.timedelta(seconds = epoches_left * (time.time() - prev_time) / print_interval)
            prev_time = time.time()
            print("[Epoch %d/%d] [Train Loss: %f] [Valid Loss: %f] ETA: %s"
                  % (epoch+1, num_epochs, train_loss/len(train_loader), valid_loss/len(valid_loader),
                     str(time_left).split('.')[0]))


    modelfile = target_dir + 'models/vgg19_' + layer + '.pth'
    print('Saving model to {}...'.format(modelfile))
    torch.save(model.state_dict(), modelfile)

