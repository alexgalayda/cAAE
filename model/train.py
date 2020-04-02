import sys
import argparse

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tools.config import Config, read_conf
from tools.dataset import NibDataset
# from models import AAE


def generator(config, shuffle=True):
    dataset = NibDataset(
        config=config, transform=transforms.Compose(
            ([transforms.ToTensor()] if config.train.to_tensor else [])
        )
    )
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


