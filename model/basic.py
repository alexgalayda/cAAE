# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn as nn

import torch

class BasicModel(nn.Module):
    def __init__(self, config, train_flg=True):
        super(BasicModel, self).__init__()
        self.name = config.struct.name
        self.config = config
        self.output = config.result
        self.img_shape = config.transforms.img_shape
        self.img_shape[0] *= self.config.train.batch_size if train_flg else self.config.test.batch_size
        self.cuda_flg = config.cuda and torch.cuda.is_available()
        print(f'\033[3{2 if self.cuda_flg else 1}m[Cuda: {self.cuda_flg}]\033[0m')
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.config += {'Tensor': self.Tensor, 'train_flg': train_flg}

    def __repr__(self):
        return f'cuda: {self.cuda_flg}\n' + \
               f'config: {self.config}'

    def __str__(self):
        return f'{self.__repr__()}\n' + \
               f'{self.encoder}\n{self.decoder}\n{self.discriminator}'

    def sample_image(self, n_row=5, batches_done=f'image'):
        z = Variable(self.Tensor(np.random.normal(0, 1, (n_row ** 2, self.config.struct.latent_dim))))
        gen_imgs = self.decoder(z)
        save_image(gen_imgs.unsqueeze(1), os.path.join(self.output, f"{batches_done}.png"), nrow=n_row, normalize=True)

    def recover(self, person, transform, acc=None):
        if acc is None:
            acc = self.config.test.acc
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
        restore_tumor[restore_tumor >= acc] = 1
        return recovered_brain, self.del_border(restore_tumor, self.config.test.thickness)
    
    def del_border(self, tumor, tk):
        tumor_temp = tumor.clone()
        if tk:
            cent = tumor[:, tk:-tk, tk:-tk] > 0
            left = tumor[:, :-2*tk, 2:-2] > 0
            right = tumor[:, 2*tk:, 2:-2] > 0
            top = tumor[:, 2:-2, 2*tk:] > 0
            down = tumor[:, 2:-2, :-2*tk] > 0
            tumor_temp[:, tk:-tk, tk:-tk] = (cent & (left | right) & (top | down))#.type(config.Tensor)
        return tumor_temp
    
    def iou(self, restore_tumor, tumor_tensor):
        SMOOTH = 1e-6
        intersection = ((restore_tumor>0) & (tumor_tensor>0)).float().sum((1, 2))
        union = ((restore_tumor>0) | (tumor_tensor>0)).float().sum((1, 2))
        return (intersection + SMOOTH) / (union + SMOOTH)
        
    def sd(self, restore_tumor, tumor_tensor):
        SMOOTH = 1e-6
        intersection = ((restore_tumor>0) & (tumor_tensor>0)).float().sum((1, 2))
        union = ((restore_tumor>0).float() + (tumor_tensor>0).float()).sum((1, 2))
        return 2*(intersection + SMOOTH) / (union + SMOOTH)
    
    def calc_metric(self, restore_tumor, tumor_tensor, bound=None):
        if self.config.test.metric == 'iou':
            mt = self.iou(restore_tumor, tumor_tensor)
        elif self.config.test.metric == 'sd':
            mt = self.sd(restore_tumor, tumor_tensor)
        else:
            raise ValueError(f'Ждал sd или iou, а получил {self.config.test.metric}')
            
        tt = (tumor_tensor>0).float().sum((1, 2))
        
        if self.config.test.tumor_min:
            rt = (restore_tumor>0).float().sum((1, 2))
            mt[rt<self.config.test.tumor_min] = 0.
            tt[tt<=self.config.test.tumor_min] = 0.
            tt[tt>self.config.test.tumor_min] = 1.
            
        if bound:
            mt[mt<bound] = 0.
            mt[mt>=bound] = 1.
        return mt, tt
            
    def test(self, dataset, acc=None, bound=None):
        if bound is None:
            bound = self.config.test.bound
        test, _ = self.calc_metric(dataset, acc, bound)
        return test.mean()
    
    def calc_metric_all(self, dataset, acc=None, bound=None):
        loss = []
        target = []
        for idx in tqdm(range(len(dataset))[5:10], desc='Testing'):
            test_person = dataset.get_person(idx)
            _, restore_tumor = self.recover(test_person, dataset.transform, acc)
            test_tumor_tensor = test_person.get_tumor(dataset.transform)
            test, utarget = self.calc_metric(restore_tumor, test_tumor_tensor, bound)
            loss.extend(test)
            target.extend(utarget)
        return self.Tensor(loss), self.Tensor(target)

    def test_show(self, dataset, acc=0.3, idx=None, show_flg=False):
        test_person = dataset.get_person(idx) if idx else dataset.get_random()
        recovered_brain, restore_tumor = self.recover(test_person, dataset.transform, acc)
        print(f'tumor loss: {self.calc_metric(restore_tumor, test_person.get_tumor(dataset.transform), self.config.test.bound)[0].mean()}')
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
