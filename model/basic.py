import os
from tqdm import tqdm
import numpy as np
import itertools
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from torch.autograd import Variable

import torch

from torch.utils.tensorboard import SummaryWriter

class Model():
    def __init__(self, config, train_flg=True):
        self.name = 'AAE'
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
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config, self.img_shape)
        self.discriminator = Discriminator(self.config)

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
        z = Variable(self.Tensor(np.random.normal(0, 1, (n_row ** 2, self.config.latent_dim))))
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
            for batch in tqdm(dataloader, total=len(dataloader), desc='Bath'):
                imgs = batch.reshape(-1, self.img_shape[1], self.img_shape[2])
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
                    0.01 * self.adversarial_loss(self.discriminator(encoded_imgs), valid) + \
                    0.99 * self.pixelwise_loss(decoded_imgs, real_imgs)
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

    def recover(self, person, transform, acc=0.3):
        # get brain
        test_brain_tensor = Variable(person(transform).type(self.Tensor))
        recovered_brain = self.decoder(self.encoder(test_brain_tensor)).data.cpu()
        # get mask
        mask = person.get_mask()
        recovered_brain *= transform(mask)
        recovered_brain = torch.clamp(recovered_brain, 0, 1)
        # get tumor
        restore_tumor = abs(recovered_brain - test_brain_tensor.cpu())
        restore_tumor[restore_tumor < acc] = 0
        return recovered_brain, restore_tumor

    def calc_metric(self, brain_tensor, recovered_brain, restore_tumor, tumor_tensor):
        acc_loss = self.pixelwise_loss.cpu()(recovered_brain, brain_tensor).item()
        ttn = (tumor_tensor != 0).sum().item()
        rtn = (restore_tumor != 0).sum().item()
        tn = (restore_tumor * tumor_tensor != 0).sum().item()
        pre_loss = 2 * tn / (ttn + rtn) if ttn + rtn else 1
        return acc_loss, pre_loss

    def test(self, dataset, acc=0.3):
        acc_loss, pre_loss = 0, 0
        print('lol')
        for idx in tqdm(range(len(dataset))[:5], desc='Testing'):
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

    def get_graph(self, person, transform, recovered_brain, restore_tumor):
        n = int(recovered_brain.shape[0] * 2 / 3)
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(recovered_brain[n, :, :])
        axs[0, 0].set_title('Recovered brain')
        axs[0, 1].imshow(restore_tumor[n, :, :])
        axs[0, 1].set_title('Founded tumor')
        axs[1, 0].imshow(person(transform)[n, :, :])
        axs[1, 0].set_title('Original brain')
        axs[1, 1].imshow(person.get_tumor(transform)[n, :, :])
        axs[1, 1].set_title('Original tumor')
        fig.savefig(os.path.join(self.output, f'random_tumor_{self.name}.png'))
        return fig

    def save(self, save_path):
        torch.save(self.encoder.state_dict(), os.path.join(save_path, f'encoder_{self.name}'))
        torch.save(self.decoder.state_dict(), os.path.join(save_path, f'decoder_{self.name}'))
        torch.save(self.discriminator.state_dict(), os.path.join(save_path, f'discriminator_{self.name}'))

    def load(self, load_path):
        self.encoder.load_state_dict(torch.load(os.path.join(load_path, f'encoder_{self.name}')))
        self.decoder.load_state_dict(torch.load(os.path.join(load_path, f'decoder_{self.name}')))
        self.discriminator.load_state_dict(torch.load(os.path.join(load_path, f'discriminator_{self.name}')))

    def tensorboard_callback(self, i, dlen):
        self.writer.add_scalar('Loss/d_loss', self.running_loss_d / dlen, i)
        self.writer.add_scalar('Loss/g_loss', self.running_loss_g / dlen, i)
