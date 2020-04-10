import argparse

import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import ants

from tools.config import Config, read_conf
from tools.dataset import MRTDataset
from models import AAE


def generator(config, health_flg, shuffle=True, dat_flg=False):
    trf = []
    if config.transforms.norm:
        trf.append(ants.iMath_normalize)
    if config.transforms.resize:
        def resize(obj, img_size=config.transforms.img_size):
            return obj.resample_image((img_size, img_size, obj.shape[2]), 1, 0)
        trf.append(resize)
    trf.append(lambda x: np.flip(x.numpy(), 1).astype(np.float32) if health_flg else x.numpy().astype(np.float32))
    if config.transforms.to_tensor:
        trf.append(transforms.ToTensor())

    dataset = MRTDataset(config=config, health_flg=health_flg, transform=transforms.Compose(trf))
    if dat_flg:
        return dataset, dataset.get_img_shape()
    else:
        dataloader = DataLoader(dataset,
                                batch_size=config.train.batch_size if health_flg else config.test.batch_size,
                                shuffle=shuffle)
        return dataloader, dataset.get_img_shape()


def train(config, save_path):
    dataset, img_shape = generator(config, health_flg=True)
    config.transforms += {'img_shape': img_shape.copy()}
    model = AAE(config)
    model.train(dataset)
    model.save(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_name', dest='config_name', type=str, default='config_example')
    parser.add_argument('-w', '--weights_path', dest='weights_path', type=str, default='/root/weights')
    args = parser.parse_args()
    config = read_conf(f'./config/{args.config_name}.json')
    train(config, args.weights_path)
