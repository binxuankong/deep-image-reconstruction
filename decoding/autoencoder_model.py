import torch
import torch.nn as nn


# CONV 1
class conv1_autoencoder(nn.Module):
    def __init__(self):
        super(conv1_autoencoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            # Input size: 64x224x224
            nn.Conv2d(64, 32, 3, stride=1, padding=1),     # 32x224x224
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),          # 32x112x112
            nn.Conv2d(32, 16, 3, stride=1, padding=1),     # 16x112x112
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),          # 16x56x56
            nn.Conv2d(16, 8, 3, stride=1, padding=1),      # 8x56x56
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),          # 8x28x28
        )
        self.encoder_linear = nn.Linear(8 * 28 * 28, 1000)
        self.decoder_linear = nn.Linear(1000, 8 * 28 * 28)
        self.decoder_conv = nn.Sequential(
            # Input size: 8x28x28
            nn.ConvTranspose2d(8, 16, 4, stride=2, padding=1),  # 16x56x56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1), # 32x112x112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), # 32x224x224
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 64, 3, stride=1, padding=1), # 64x224x224
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_linear(x)
        return x

    def decode(self, x):
        x = self.decoder_linear(x)
        x = x.view(x.size(0), 8, 28, 28)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


# CONV2
class conv2_autoencoder(nn.Module):
    def __init__(self):
        super(conv2_autoencoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            # Input size: 128x112x112
            nn.Conv2d(128, 64, 3, stride=1, padding=1),    # 64x112x112
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),          # 64x56x56
            nn.Conv2d(64, 32, 3, stride=1, padding=1),     # 32x56x56
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),          # 32x28x28
            nn.Conv2d(32, 16, 3, stride=1, padding=1),     # 16x28x28
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),          # 16x14x14
        )
        self.encoder_linear = nn.Linear(16 * 14 * 14, 1000)
        self.decoder_linear = nn.Linear(1000, 16 * 14 * 14)
        self.decoder_conv = nn.Sequential(
            # Input size: 16x14x14
            nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1),         # 32x28x28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 64, 4, stride=2, padding=1),         # 64x56x56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),         # 64x112x112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1),        # 128x112x112
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_linear(x)
        return x

    def decode(self, x):
        x = self.decoder_linear(x)
        x = x.reshape(x.size(0), 16, 14, 14)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


# CONV3
class conv3_autoencoder(nn.Module):
    def __init__(self):
        super(conv3_autoencoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            # Input size: 256x56x56
            nn.Conv2d(256, 128, 3, stride=1, padding=1),   # 128x56x56
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),          # 128x28x28
            nn.Conv2d(128, 64, 3, stride=1, padding=1),    # 64x28x28
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),          # 64x14x14
            nn.Conv2d(64, 32, 3, stride=1, padding=1),     # 32x14x14
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),          # 32x7x7
        )
        self.encoder_linear = nn.Linear(32 * 7 * 7, 1000)
        self.decoder_linear =  nn.Linear(1000, 32 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            # Input size: 32x7x7
            nn.ConvTranspose2d(32, 64, 4, stride=2, padding=1),         # 64x14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),        # 128x28x28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),       # 128x56x56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 256, 3, stride=1, padding=1),       # 256x56x56
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_linear(x)
        return x

    def decode(self, x):
        x = self.decoder_linear(x)
        x = x.reshape(x.size(0), 32, 7, 7)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


# CONV4
class conv4_autoencoder(nn.Module):
    def __init__(self):
        super(conv4_autoencoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            # Input size: 512x28x28
            nn.Conv2d(512, 256, 3, stride=1, padding=1),   # 256x28x28
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),          # 256x14x14
            nn.Conv2d(256, 128, 3, stride=1, padding=1),   # 128x14x14
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),          # 128x7x7
            nn.Conv2d(128, 64, 3, stride=1, padding=1),    # 64x7x7
            nn.LeakyReLU(0.2),
        )
        self.encoder_linear = nn.Linear(64 * 7 * 7, 1000)
        self.decoder_linear = nn.Linear(1000, 64 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            # Input size: 64x7x7
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),        # 128x14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),       # 128x28x28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 256, 3, stride=1, padding=1),       # 256x28x28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1),       # 512x28x28
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_linear(x)
        return x

    def decode(self, x):
        x = self.decoder_linear(x)
        x = x.reshape(x.size(0), 64, 7, 7)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


# CONV5
class conv5_autoencoder(nn.Module):
    def __init__(self):
        super(conv5_autoencoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            # Input size: 512x14x14
            nn.Conv2d(512, 256, 3, stride=1, padding=1),   # 256x14x14
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),          # 256x7x7
            nn.Conv2d(256, 128, 3, stride=1, padding=1),   # 128x7x7
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),    # 64x7x7
            nn.LeakyReLU(0.2),
        )
        self.encoder_linear = nn.Linear(64 * 7 * 7, 1000)
        self.decoder_linear = nn.Linear(1000, 64 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            # Input size: 64x7x7
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),         # 64x14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1),        # 128x14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 256, 3, stride=1, padding=1),       # 256x14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1),       # 512x14x14
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_linear(x)
        return x

    def decode(self, x):
        x = self.decoder_linear(x)
        x = x.reshape(x.size(0), 64, 7, 7)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


# FULLY CONNECTED
class fc_autoencoder(nn.Module):
    def __init__(self):
        super(fc_autoencoder, self).__init__()
        self.encoder_linear = nn.Sequential(
            # Input size: 4096
            nn.Linear(4096, 2048),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 1000),
        )
        self.decoder_linear = nn.Sequential(
            # Input size: 1000
            nn.Linear(1000, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 4096),
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.encoder_linear(x)
        return x

    def decode(self, x):
        x = self.decoder_linear(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


