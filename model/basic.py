import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from torch.autograd import Variable

import torch

class BasicModel:
    def __init__(self, config, train_flg=True):
        self.name = config.struct.name
        self.config = config
        self.output = config.result
        self.img_shape = config.transforms.img_shape
        self.img_shape[0] *= self.config.train.batch_size if train_flg else self.config.test.batch_size
        self.cuda = config.cuda and torch.cuda.is_available()
        print(f'\033[3{2 if self.cuda else 1}m[Cuda: {self.cuda}]\033[0m')
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.config += {'Tensor': self.Tensor, 'train_flg': train_flg}

    def __repr__(self):
        return f'cuda: {self.cuda}\n' + \
               f'config: {self.config}'

    def __str__(self):
        return f'{self.__repr__()}\n' + \
               f'{self.encoder}\n{self.decoder}\n{self.discriminator}'

    def sample_image(self, n_row=5, batches_done=f'image'):
        z = Variable(self.Tensor(np.random.normal(0, 1, (n_row ** 2, self.config.struct.latent_dim))))
        gen_imgs = self.decoder(z)
        save_image(gen_imgs.unsqueeze(1), os.path.join(self.output, f"{batches_done}.png"), nrow=n_row, normalize=True)

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
