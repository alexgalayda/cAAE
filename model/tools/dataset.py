import os
import uuid

import torch
from torch.utils.data import Dataset
import ants

class NibDataset(Dataset):
    def __init__(self, config, transform=None):
        self.config = config
        self.transform = transform
        self.person_dict = {}
        self.person_list = []
        for address, dirs, files in os.walk(os.path.join(self.config.path, self.config.local)):
            for file in files:
                if file.endswith('.nii.gz'):
                    person = Person(os.path.join(address, file))
                    if (not config.slice) or (isinstance(config.slice, int) and config.slice >= person.num) or (
                            isinstance(config.slice, list) and (len(config.slice) == 2) and (
                            config.slice[0] <= person.num < config.slice[1])):
                        self.person_dict[person.uuid] = person
                        self.person_list.append(person)
                if len(self.person_list) >= self.config.train.max_batch:
                    break
            if len(self.person_list) >= self.config.train.max_batch:
                break

    def __str__(self):
        ans = ''
        for person in list(self.person_dict.values()):
            ans += f'{person.__repr__()}\n'
        return ans[:-1] if ans else ''

    def __repr__(self):
        ans = ''
        for person in list(self.person_dict.values()):
            ans += f'{person}\n'
        return ans[:-1] if ans else ''

    def __len__(self):
        return len(self.person_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, str):
            return self.person_dict[idx](self.transform)
        elif isinstance(idx, int):
            return self.person_list[idx](self.transform)
        else:
            raise TypeError(f'Неверный тип. Хотел: int или str, а получил {idx.__type__}')

    def get_img_shape(self):
        if len(self.person_list) == 0:
            return 0
        else:
            return list(self[0].shape)


class Person:
    def __init__(self, path):
        self.uuid = str(uuid.uuid4())
        self.path = path
        self.name = path.split('/')[-1].split('.')[0]
        self.num = int(self.name.split('_')[-1])

    def __call__(self, transform=None):
        img = ants.image_read(self.path)
        return transform(img) if transform else img.numpy()

    def __str__(self):
        return f"{self.uuid}: {self.path}"

    def __repr__(self):
        return f"{self.uuid[:-6]}: {self.name}"

    def plot(self):
        ants.plot(ants.image_read(self.path))

