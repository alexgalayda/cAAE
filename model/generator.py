import numpy as np
import torchvision.transforms as transforms
import ants

from model.tools.dataset import MRTDataset
from model.AAE import AAE
from model.cAAE import cAAE
from model.ResDCGAN import ResDCGAN
from model.BiGAN import BiGAN

net = {"AAE": AAE, "cAAE": cAAE, "ResDCGAN": ResDCGAN, "BiGAN": BiGAN}


def generator(config, train_flg):
    trf = []
    if config.transforms.norm:
        trf.append(ants.iMath_normalize)
    if config.transforms.resize:
        def resize(obj, img_size=config.transforms.img_size):
            return obj.resample_image((img_size, img_size, obj.shape[2]), 1, 0)
        trf.append(resize)
    trf.append(lambda x: np.flip(x.numpy(), 1).astype(np.float32) if train_flg else x.numpy().astype(np.float32))
    if config.transforms.to_tensor:
        trf.append(transforms.ToTensor())

    return MRTDataset(config=config, health_flg=train_flg, transform=transforms.Compose(trf))
