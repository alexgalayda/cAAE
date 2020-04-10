import argparse

import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import ants

from tools.config import Config, read_conf
from tools.dataset import MRTDataset
from models import AAE

from train import generator


def test(config, load_path):
    dataset, img_shape = generator(config, health_flg=False)
    config.transforms += {'img_shape': img_shape.copy()}
    model = AAE(config)
    model.load(load_path)
    model.test(dataset)


def test_one(config, load_path):
    dataset, img_shape = generator(config, health_flg=False, dat_flg=True)
    config.transforms += {'img_shape': img_shape.copy()}
    model = AAE(config, train_flg=False)
    model.load(load_path)
    decod_tumor, test_tumor = model.test_one(dataset)
    return decod_tumor, test_tumor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_name', dest='config_name', type=str, default='config_example')
    parser.add_argument('-w', '--weights_path', dest='weights_path', type=str, default='/root/weights')
    args = parser.parse_args()
    config = read_conf(f'./config/{args.config_name}.json')
    test(config, args.weights_path)
