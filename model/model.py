import os

from tqdm import tqdm
import numpy as np
import itertools

from torchvision.utils import save_image
from torch.autograd import Variable

import torch.nn as nn
import torch

from torch.utils.tensorboard import SummaryWriter

#TODO: объеденить классы сетей
class AAE:
    def __init__(self, config):
        self.config = config.train
        self.output = config.train_path
        self.img_shape = (config.train.channels, config.transforms.img_size, config.transforms.img_size)
        self.cuda = config.cuda and torch.cuda.is_available()

        self.data = config.data
        # Use binary cross-entropy loss
        self.adversarial_loss = torch.nn.BCELoss()
        self.pixelwise_loss = torch.nn.L1Loss()

        # Initialize generator and discriminator
        self.encoder = Encoder(self.config, self.img_shape)
        self.decoder = Decoder(self.config, self.img_shape)
        self.discriminator = Discriminator(self.config, self.img_shape)

        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2)
        )

        #tensorboard callback
        self.writer = SummaryWriter(os.path.join(config.mount_output, 'log'))

    def train(self, dataset):
        if self.cuda:
            self.encoder.cuda()
            self.decoder.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()
            self.pixelwise_loss.cuda()

            Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        for epoch in tqdm(range(self.config.n_epochs), total=self.config.n_epochs, desc='Epoch', leave=True):
            self.running_loss_g = 0
            self.running_loss_d = 0
            for i, imgs in tqdm(enumerate(dataset), total=len(dataset), desc='Bath'):
                if self.data == 'mnist':
                    imgs, _ = imgs

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                encoded_imgs = self.encoder(real_imgs)
                decoded_imgs = self.decoder(encoded_imgs)

                # Loss measures generator's ability to fool the discriminator
                g_loss = 0.001 * self.adversarial_loss(self.discriminator(encoded_imgs), valid) + 0.999 * self.pixelwise_loss(
                    decoded_imgs, real_imgs
                )

                g_loss.backward()
                self.optimizer_G.step()
                self.running_loss_g += g_loss.item()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Sample noise as discriminator ground truth
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.config.latent_dim))))

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(z), valid)
                fake_loss = self.adversarial_loss(self.discriminator(encoded_imgs.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)

                d_loss.backward()
                self.optimizer_D.step()
                self.running_loss_d += d_loss.item()
                # self.print_st(epoch, i, len(dataset), gen_imgs)
            self.tensorboard_callback(epoch, len(dataset))
        self.writer.close()

    def tensorboard_callback(self, i, dlen):
        self.writer.add_scalar('Loss/d_loss', self.running_loss_d / dlen, i)
        self.writer.add_scalar('Loss/g_loss', self.running_loss_g / dlen, i)

    def print_st(self, epoch, i, ll, gen_imgs):
        batches_done = epoch * ll + i
        if batches_done % self.config.sample_interval == 0:
            save_image(gen_imgs.data[:25], os.path.join(self.output, f"{batches_done}.png"), nrow=5, normalize=True)

        def sample_image(n_row, batches_done):
            """Saves a grid of generated digits"""
            # Sample noise
            z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
            gen_imgs = decoder(z)
            save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

class Encoder(nn.Module):
    def __init__(self, config, img_shape):
        super(Encoder, self).__init__()
        self.config = config
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512,  self.config.latent_dim)
        self.logvar = nn.Linear(512,  self.config.latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterization(mu, logvar)
        return z

    def reparameterization(self, mu, logvar):
        Tensor = torch.cuda.FloatTensor if self.config.cuda and torch.cuda.is_available() else torch.FloatTensor
        std = torch.exp(logvar / 2)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
        z = sampled_z * std + mu
        return z


class Decoder(nn.Module):
    def __init__(self, config, img_shape):
        super(Decoder, self).__init__()
        self.config = config
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(self.config.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(self.img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, config, img_shape):
        super(Discriminator, self).__init__()
        self.config = config
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(self.config.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity
