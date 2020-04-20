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

class cAAE(BasicModel):
    def __init__(self, config, train_flg=True):
        super(cAAE, self).__init__(config, train_flg)

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
        for epoch in tqdm(range(self.config.train.n_epochs), total=self.config.train.n_epochs, desc='Epoch', leave=True):
            for batch in tqdm(dataloader, total=len(dataloader), desc='Bath'):
                imgs = batch.reshape(-1, self.img_shape[1], self.img_shape[2])
                # Adversarial ground truths
                valid = Variable(self.Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
                # Configure input
                real_imgs = Variable(imgs.type(self.Tensor))
                encoded_imgs = self.encoder(real_imgs)
                decoded_imgs = self.decoder(encoded_imgs)
        
                self.optimizer_D.zero_grad()    
                # Sample noise as discriminator ground truth
                z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.config.struct.latent_dim))))            
                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(encoded_imgs.data, z.data)
                # Adversarial loss
                d_loss = \
                    torch.mean(self.discriminator(encoded_imgs.detach())) - \
                    torch.mean(self.discriminator(z)) + \
                    self.config.struct.lambda_gp * gradient_penalty
                self.optimizer_D.step()
                self.running_loss_d += d_loss.item()
            
                self.optimizer_G.zero_grad()
                # if i % self.config.train.step_dis == 0:
                g_loss = \
                    self.pixelwise_loss(decoded_imgs, real_imgs) + \
                    self.config.struct.lambda_gp * self.latent_loss(self.encoder(decoded_imgs), encoded_imgs)
                g_loss.backward()
                self.optimizer_G.step()
                self.running_loss_g += g_loss.item()
            if epoch % 10 == 0:
                save_path = '/root/weights'
                torch.save(self.encoder.state_dict(), os.path.join(save_path, f'encoder_{self.name}_{epoch}'))
                torch.save(self.decoder.state_dict(), os.path.join(save_path, f'decoder_{self.name}_{epoch}'))
                torch.save(self.discriminator.state_dict(), os.path.join(save_path, f'discriminator_{self.name}_{epoch}'))
            self.tensorboard_callback(epoch, len(dataloader))
        self.writer.close()
        
    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = self.Tensor(np.random.random((real_samples.size(0), 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = Variable(self.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    def calc_metric(self, brain_tensor, recovered_brain, restore_tumor, tumor_tensor):
        acc_loss = self.pixelwise_loss.cpu()(recovered_brain, brain_tensor).item()
        ttn = (tumor_tensor != 0).sum().item()
        rtn = (restore_tumor != 0).sum().item()
        tn = (restore_tumor * tumor_tensor != 0).sum().item()
        pre_loss = 2 * tn / (ttn + rtn) if ttn + rtn else 1
        return acc_loss, pre_loss

    def test(self, dataset, acc=0.3):
        acc_loss, pre_loss = 0, 0
        for idx in tqdm(range(len(dataset)), desc='Testing'):
            test_person = dataset.get_person(idx)
            recovered_brain, restore_tumor = self.recover(test_person, dataset.transform, acc)
            test_tumor_tensor = test_person.get_tumor(dataset.transform)
            test_brain_tensor = test_person(dataset.transform)
            acc, pre = self.calc_metric(test_brain_tensor, recovered_brain, restore_tumor, test_tumor_tensor)
            acc_loss += acc
            pre_loss += pre
        print(f'pixelwise loss on brain: {acc_loss / len(dataset)}')
        print(f'tumor coverage: {pre_loss / len(dataset)}')

    def test_show(self, dataset, acc=0.3, idx=None, show_flg=False):
        test_person = dataset.get_person(idx) if idx else dataset.get_random()
        recovered_brain, restore_tumor = self.recover(test_person, dataset.transform, acc)
        fig = self.get_graph(test_person, dataset.transform, recovered_brain, restore_tumor)
        if show_flg:
            try:
                fig.show()
            except Exception as e:
                print(f'Cann\'t show result\n{e}')


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
            nn.Linear(512, self.config.struct.latent_dim)
        )

    def forward(self, img):
        x = self.model(img.unsqueeze(1))
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config

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
        img = self.model(img_conv).view(-1, *self.config.transforms.img_shape[1:])
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


def make_layer(config, channels, channels_in, stride):
    return nn.Sequential(
        ResBlock(channels, channels_in, stride, res_option=config.struct.res_option, use_dropout=config.struct.use_dropout),
        *[ResBlock(channels) for _ in range(config.struct.layer_count-1)])


# +
# various projection options to change number of filters in residual connection
# option A from paper
class IdentityPadding(nn.Module):
    def __init__(self, num_filters, channels_in, stride):
        super(IdentityPadding, self).__init__()
        # with kernel_size=1, max pooling is equivalent to identity mapping with stride
        self.identity = nn.MaxPool2d(1, stride=stride)
        self.num_zeros = num_filters - channels_in
    
    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out

# option B from paper
class ConvProjection(nn.Module):

    def __init__(self, num_filters, channels_in, stride):
        super(ConvProjection, self).__init__()
        self.conv = nn.Conv2d(channels_in, num_filters, kernel_size=1, stride=stride)
    
    def forward(self, x):
        out = self.conv(x)
        return out

# experimental option C
class AvgPoolPadding(nn.Module):

    def __init__(self, num_filters, channels_in, stride):
        super(AvgPoolPadding, self).__init__()
        self.identity = nn.AvgPool2d(stride, stride=stride)
        self.num_zeros = num_filters - channels_in
    
    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out
# -


