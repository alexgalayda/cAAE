# -*- coding: utf-8 -*-
import os
import uuid

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import ants


class MRTDataset(Dataset):
    def __init__(self, config, health_flg, transform=None):
        self.config = config
        self.transform = transform
        self.person_dict = {}
        self.person_list = []
        self.health_flg = health_flg
        self._init_dataset()

    def _init_dataset(self):
        max_batch = self.config.train.max_batch if self.health_flg else self.config.test.max_batch
        for address, dirs, files in os.walk(os.path.join(
                self.config.train_path if self.health_flg else self.config.test_path,
                self.config.train_local if self.health_flg else self.config.test_local
        )):
            for file in files:
                if file.endswith(('.nii.gz', '_brain.mha')):
                    person = Person(os.path.join(address, file))
                    if not self.health_flg:
                        person.set_tumor()
                    if (not self.config.slice) or (
                            isinstance(self.config.slice, int) and self.config.slice >= person.num) or (
                            isinstance(self.config.slice, list) and (len(self.config.slice) == 2) and (
                            self.config.slice[0] <= person.num < self.config.slice[1])):
                        self.person_dict[person.uuid] = person
                        self.person_list.append(person)
                if len(self.person_list) >= max_batch:
                    break
            if len(self.person_list) >= max_batch:
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
        return self.get_person(idx)(self.transform)

    def dataloader(self, shuffle=True):
        return DataLoader(self,
                          batch_size=self.config.train.batch_size if self.health_flg else self.config.test.batch_size,
                          shuffle=shuffle)

    def get_img_shape(self):
        return list(self.get_random()(self.transform).shape) if len(self.person_list) != 0 else 0

    def get_random(self):
        return np.random.choice(self.person_list, 1)[0]

    def get_person(self, idx):
        if isinstance(idx, str):
            return self.person_dict[idx]
        elif isinstance(idx, int):
            return self.person_list[idx]
        else:
            raise TypeError(f'Неверный тип. Хотел: int или str, а получил {idx.__type__}')


class Person:
    def __init__(self, path):
        self.uuid = str(uuid.uuid4())
        self.path = path
        self.tumor = None
        self.name = path.split('/')[-1].split('.')[0]
        if self.path.endswith('.nii.gz'):
            self.num = int(self.name.split('_')[-1])
        if self.path.endswith('.mha'):
            self.num = int(self.name.split('_')[0][3:])

    def __str__(self):
        return f"{self.uuid}: {self.path}"

    def __repr__(self):
        return f"{self.uuid[:-6]}: {self.name}"

    def get_tumor(self, transform=None):
        if self.tumor:
            img = ants.image_read(self.tumor)
            return transform(img) if transform else img
        else:
            return None

    def set_tumor(self):
        self.tumor = f'{self.path[:-10]}_tumor.mha'

    def plot(self, axis=0):
        self.get_brain().plot(axis=axis)

    def get_brain(self, np_flg=False):
        img = ants.image_read(self.path)
        return img.numpy() if np_flg else img

    def get_mask(self):
        mask = ants.get_mask(self.get_brain())
        mask = ants.iMath(mask, 'ME', 2)
        return mask

    def __call__(self, transform=None):
        return transform(self.get_brain()) if transform else self.get_brain(np_flg=True).astype(np.float32)
