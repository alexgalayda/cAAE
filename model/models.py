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
class AAE():
    def __init__(self, config, train_flg=True):
        self.config = config.train if train_flg else config.test
        self.output = config.result
        self.img_shape = config.transforms.img_shape
        self.img_shape[0] *= self.config.batch_size
        self.cuda = config.cuda and torch.cuda.is_available()
        print(f'\033[3{2 if self.cuda else 1}m[Cuda: {self.cuda}]\033[0m')
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.config += {'Tensor': self.Tensor}

        # Use binary cross-entropy loss
        self.adversarial_loss = torch.nn.BCELoss()
        self.pixelwise_loss = torch.nn.L1Loss()

        # Initialize generator and discriminator
        self.encoder = Encoder(self.config, self.img_shape)
        self.decoder = Decoder(self.config, self.img_shape)
        self.discriminator = Discriminator(self.config, self.img_shape)

        if self.cuda:
            self.encoder.cuda()
            self.decoder.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()
            self.pixelwise_loss.cuda()

    def __repr__(self):
        return f'cuda: {self.cuda}\n' + \
               f'config: {self.config}'

    def __str__(self):
        return f'{self.__repr__()}\n' + \
               f'{self.encoder}\n{self.decoder}\n{self.discriminator}'

    def sample_image(self, n_row=5, batches_done='AAE_image'):
        z = Variable(self.Tensor(np.random.normal(0, 1, (n_row**2, self.config.latent_dim))))
        gen_imgs = self.decoder(z)
        save_image(gen_imgs.unsqueeze(1), os.path.join(self.output, f"{batches_done}.png"), nrow=n_row, normalize=True)

    def train(self, dataset):
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters()
            ),
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2))

        # tensorboard callback
        self.writer = SummaryWriter(os.path.join(self.output, 'log'))

        self.running_loss_g = 0
        self.running_loss_d = 0

        dataloader = dataset.dataloader()
        for epoch in tqdm(range(self.config.n_epochs), total=self.config.n_epochs, desc='Epoch', leave=True):
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Bath'):

                imgs = batch.reshape(-1, self.img_shape[1], self.img_shape[2])
                # imgs = batch.permute(0, 3, 1, 2).reshape(-1, self.img_shape[0], self.img_shape[1])
                # Adversarial ground truths
                valid = Variable(self.Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(self.Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                encoded_imgs = self.encoder(real_imgs)
                decoded_imgs = self.decoder(encoded_imgs)

                # Loss measures generator's ability to fool the discriminator
                g_loss = \
                    0.001 * self.adversarial_loss(self.discriminator(encoded_imgs), valid) + \
                    0.999 * self.pixelwise_loss(decoded_imgs, real_imgs)
                g_loss.backward()
                self.optimizer_G.step()
                self.running_loss_g += g_loss.item()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Sample noise as discriminator ground truth
                z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.config.latent_dim))))

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(z), valid)
                fake_loss = self.adversarial_loss(self.discriminator(encoded_imgs.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)

                d_loss.backward()
                self.optimizer_D.step()
                self.running_loss_d += d_loss.item()
            self.tensorboard_callback(epoch, len(dataloader))
        self.writer.close()

    def test(self, dataloader):
        for i, ants_img in tqdm(enumerate(dataloader), total=len(dataloader), desc='Bath'):
            brain_img = ants_img
            tumor_img = ants_img
            imgs = img.reshape(-1, self.img_shape[1], self.img_shape[2])

            real_imgs = Variable(imgs.type(self.Tensor))

            encoded_imgs = self.encoder(real_imgs)
            decoded_imgs = self.decoder(encoded_imgs)

            g_loss = self.pixelwise_loss(decoded_imgs, real_imgs)

    def test_one(self, dataset, trf=None, idx=None):
        test_person = dataset.person_list[idx] if idx else dataset.get_random()
        test_brain = test_person.get_ants(np_flg=False)
        test_tumor = test_person.get_tumor(np_flg=False)

        real_img = Variable(test_brain(self.transform).type(self.Tensor))
        decoded_img = self.decoder(self.encoder(real_img))
        decoded_img_np = decoded_img.cpu().detach().permute(1, 2, 0).numpy()

        scale_test_brain = trf(test_brain) if trf else test_brain
        scale_test_tumor = trf(test_tumor) if trf else test_tumor
        # scale_test_brain = ants.iMath_normalize(test_brain).resample_image(decoded_img_np.shape, 1, 0)
        # scale_test_tumor = ants.iMath_normalize(test_tumor).resample_image(decoded_img_np.shape, 1, 0)

        # g_loss = self.pixelwise_loss(decoded_imgs, real_imgs)
        return abs(scale_test_brain - decoded_img_np), scale_test_tumor

    def save(self, save_path):
        torch.save(self.encoder.state_dict(), os.path.join(save_path, 'encoder'))
        torch.save(self.decoder.state_dict(), os.path.join(save_path, 'decoder'))
        torch.save(self.discriminator.state_dict(), os.path.join(save_path, 'discriminator'))

    def load(self, load_path):
        self.encoder.load_state_dict(torch.load(os.path.join(load_path, 'encoder')))
        self.decoder.load_state_dict(torch.load(os.path.join(load_path, 'decoder')))
        self.discriminator.load_state_dict(torch.load(os.path.join(load_path, 'discriminator')))

    def tensorboard_callback(self, i, dlen):
        self.writer.add_scalar('Loss/d_loss', self.running_loss_d / dlen, i)
        self.writer.add_scalar('Loss/g_loss', self.running_loss_g / dlen, i)


class Encoder(nn.Module):
    def __init__(self, config, img_shape):
        super(Encoder, self).__init__()
        self.config = config
        self.img_shape = img_shape[1:]
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, self.config.latent_dim)
        self.logvar = nn.Linear(512, self.config.latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterization(mu, logvar)
        return z

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        sampled_z = Variable(self.config.Tensor(np.random.normal(0, 1, (mu.size(0), self.config.latent_dim))))
        z = sampled_z * std + mu
        return z


class Decoder(nn.Module):
    def __init__(self, config, img_shape):
        super(Decoder, self).__init__()
        self.config = config
        self.img_shape = img_shape[1:]

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
        self.img_shape = img_shape[1:]

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