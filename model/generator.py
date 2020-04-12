import numpy as np
import torchvision.transforms as transforms
import ants

from tools.dataset import MRTDataset

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

    dataset = MRTDataset(config=config, train_flg=train_flg, transform=transforms.Compose(trf))

    return dataset