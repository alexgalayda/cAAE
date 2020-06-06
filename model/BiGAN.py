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


class BiGAN(BasicModel):
    def __init__(self, config, train_flg=True):
        super(BiGAN, self).__init__(config, train_flg)

        # self.discriminator_loss = torch.nn.BCELoss()
        self.discriminator_loss = torch.nn.BCEWithLogitsLoss()
        # self.adversarial_loss = torch.nn.BCELoss()
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.discriminator = Discriminator(self.config)

        
        if self.cuda:
            self.discriminator_loss.cuda()
            self.adversarial_loss.cuda()
        
            self.encoder.cuda()
            self.decoder.cuda()
            self.discriminator.cuda()

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

        dataloader = dataset.dataloader(tensor=self.Tensor)
        for epoch in tqdm(range(self.config.train.n_epochs), total=self.config.train.n_epochs, desc='Epoch', leave=True):
            for batch in tqdm(dataloader, total=len(dataloader), desc='Bath'):
                imgs = batch.reshape(-1, self.img_shape[-2], self.img_shape[-1])
                # Adversarial ground truths
                
                valid = Variable(self.Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
                real_imgs = Variable(imgs.type(self.Tensor))
                
                self.optimizer_G.zero_grad()
                
                encoded_imgs = self.encoder(real_imgs)
                decoded_imgs = self.decoder(encoded_imgs)
                
                g_loss = self.adversarial_loss(self.discriminator(encoded_imgs.detach(), real_imgs), valid)
                g_loss.backward()
                self.running_loss_g += g_loss.item()
                
                self.optimizer_D.zero_grad()
                
                z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.config.struct.latent_dim))))
                
                real_loss = self.adversarial_loss(self.discriminator(z, self.decoder(z).detach()), valid)
                fake_loss = self.adversarial_loss(self.discriminator(encoded_imgs.detach(), decoded_imgs.detach()), fake)
                
                d_loss = 0.5 * (real_loss + fake_loss)
                d_loss.backward()
                self.optimizer_G.step()
                self.running_loss_g += g_loss.item()
                
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
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )
        self.mu = nn.Linear(512, self.config.struct.latent_dim)
        self.logvar = nn.Linear(512, self.config.struct.latent_dim)

    def forward(self, img):
        x = self.model(img.unsqueeze(1))
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterization(mu, logvar)
        return z

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        sampled_z = Variable(self.config.Tensor(np.random.normal(0, 1, (mu.size(0), self.config.struct.latent_dim))))
        z = sampled_z * std + mu
        return z


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.img_shape = self.config.transforms.img_shape[-2:]

        self.model_prep = nn.Sequential(
            nn.Linear(self.config.struct.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
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

        self.img_dis = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, config.struct.latent_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.struct.latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(4),
            nn.Flatten()
        )

        self.model = nn.Sequential(
            nn.Linear(2*config.struct.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            # nn.Sigmoid()
        )

    def forward(self, z, img):
        x = self.img_dis(img.unsqueeze(1))
        validity = self.model(torch.cat((z, x), 1))
        return validity
