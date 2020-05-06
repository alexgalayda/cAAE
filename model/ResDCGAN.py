import os
from tqdm import tqdm
import numpy as np
import itertools
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter

from model.basic import BasicModel


class ResDCGAN(BasicModel):
    def __init__(self, config, train_flg=True):
        super(ResDCGAN, self).__init__(config, train_flg)

        # Use binary cross-entropy loss
        self.adversarial_loss = torch.nn.BCELoss()
        self.pixelwise_loss = torch.nn.L1Loss()
        self.latent_loss = torch.nn.MSELoss()

        # Initialize generator and discriminator
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.discriminator = Discriminator(self.config)

        if self.cuda:
            self.encoder.cuda()
            self.decoder.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()
            self.pixelwise_loss.cuda()

    def train(self, dataset):
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters()
            ),
            lr=self.config.train.lr,
            betas=(self.config.train.b1, self.config.train.b2)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.train.lr,
            betas=(self.config.train.b1, self.config.train.b2))

        # tensorboard callback
        self.writer = SummaryWriter(os.path.join(self.output, 'log'))

        self.running_loss_g = 0
        self.running_loss_d = 0

        dataloader = dataset.dataloader()
        for epoch in tqdm(range(self.config.train.n_epochs), total=self.config.train.n_epochs, desc='Epoch',
                          leave=True):
            for batch in tqdm(dataloader, total=len(dataloader), desc='Bath'):
                imgs = batch.reshape(-1, self.img_shape[-2], self.img_shape[-1])
                # Adversarial ground truths
                valid = Variable(self.Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(self.Tensor))
                self.optimizer_G.zero_grad()

                encoded_imgs = self.encoder(real_imgs)
                decoded_imgs = self.decoder(encoded_imgs)

                # Loss measures generator's ability to fool the discriminator
                g_loss = \
                    0.01 * self.adversarial_loss(self.discriminator(encoded_imgs), valid) + \
                    0.99 * self.pixelwise_loss(decoded_imgs, real_imgs)
                g_loss.backward()
                self.optimizer_G.step()
                self.running_loss_g += g_loss.item()

                self.optimizer_D.zero_grad()

                # Sample noise as discriminator ground truth
                z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.config.struct.latent_dim))))

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(z), valid)
                fake_loss = self.adversarial_loss(self.discriminator(encoded_imgs.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)

                d_loss.backward()
                self.optimizer_D.step()
                self.running_loss_d += d_loss.item()
            if epoch % 10 == 0:
                save_path = '/root/weights'
                torch.save(self.encoder.state_dict(), os.path.join(save_path, f'encoder_{self.name}_{epoch}'))
                torch.save(self.decoder.state_dict(), os.path.join(save_path, f'decoder_{self.name}_{epoch}'))
                torch.save(self.discriminator.state_dict(),
                           os.path.join(save_path, f'discriminator_{self.name}_{epoch}'))
            self.tensorboard_callback(epoch, len(dataloader))
        self.writer.close()


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlockDown(32, 64),
            ResBlockDown(64, 128),
            ResBlockDown(128, 256),
            ResBlockDown(256, 512),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(512, self.config.struct.latent_dim),
            nn.Tanh()
        )

    def forward(self, img):
        x = self.model(img.unsqueeze(1))
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.img_shape = config.transforms.img_shape[-2:]

        self.model_prep = nn.Sequential(
            nn.Linear(self.config.struct.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlockUp(512, 256),
            ResBlockUp(256, 128),
            ResBlockUp(128, 64),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        img_conv = self.model_prep(z).unsqueeze(2).unsqueeze(3)
        img = self.model(img_conv).view(-1, *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config

        self.model = nn.Sequential(
            nn.Linear(self.config.struct.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


class ResBlockDown(nn.Module):
    def __init__(self, channels_in, channels_out=None):
        super(ResBlockDown, self).__init__()
        kernel_size, stride, padding = 4, 2, 1
        if channels_out is None:
            channels_out = channels_in
        if channels_out != channels_in:
            self.downsample = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(channels_out))
        else:
            kernel_size, stride, padding = 3, 1, 1
            self.downsample = None

        self.module = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels_out, channels_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels_out),
        )

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.module(x)
        if self.downsample:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResBlockUp(nn.Module):
    def __init__(self, channels_in, channels_out=None):
        super(ResBlockUp, self).__init__()
        kernel_size, stride, padding = 4, 2, 1
        if channels_out is None:
            channels_out = channels_in
        if channels_out != channels_in:
            self.downsample = nn.Sequential(
                nn.ConvTranspose2d(channels_in, channels_out, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(channels_out))
        else:
            kernel_size, stride, padding = 3, 1, 1
            self.downsample = None

        self.module = nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(channels_out, channels_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels_out),
        )

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.module(x)
        if self.downsample:
            residual = self.downsample(x)
        return self.relu(out + residual)
