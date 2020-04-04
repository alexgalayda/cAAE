import sys
import argparse

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import ants

from tools.config import Config, read_conf
from tools.dataset import NibDataset
# from models import AAE


def generator(config, shuffle=True):
    trf = []
    if config.transforms.norm:
        trf.append(ants.iMath_normalize)
    if config.transforms.resize:
        def resize(obj, img_size=config.transforms.img_size):
            return obj.resample_image((img_size, obj.shape[1], img_size), 1, 0)
        trf.append(resize)
    trf.append(lambda x: x.numpy())
    if config.transforms.to_tensor:
        trf.append(transforms.ToTensor())

    dataset = NibDataset(config=config, transform=transforms.Compose(trf))

    dataloader = DataLoader(dataset, batch_size=config.train.batch_size,
                            shuffle=shuffle, num_workers=4, drop_last=True)
    return dataloader


def train(config):
    dataset = generator(config)
    # model = AAE(config)
    # model.train(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_name', dest='config_name', type=str, default='config_example')
    args = parser.parse_args()
    config = read_conf(f'./config/{args.config_name}.json')
    train(config)
