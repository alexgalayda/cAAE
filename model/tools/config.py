import os
import json
import copy
from tools.tools import get_attr


def read_conf(path):
    try:
        with open(path, 'r') as read_file:
            conf = json.load(read_file)
    except Exception:
        print(f'Не удалось прочитать {path}')
        return None
    else:
        conf = Config(conf)
        conf.root = os.getcwd()
        return conf


class Config:
    root = None

    def __init__(self, dict_json):
        self.__dict__.update(dict_json)
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = Config(value)

    def __repr__(self):
        return '\n_______\n' + str(self.__dict__)[1:-1].replace(', ', '\n') + '\n_______'

    def __add__(self, other):
        tmp = copy.copy(self)
        if other is None:
            return tmp

        if isinstance(other, dict):
            other = Config(other)
        elif not isinstance(other, Config):
            raise TypeError(f'Неверный тип. Хотел: Config или dict с маршрутами, а получил {other.__type__}')

        for key in get_attr(other):
            setattr(tmp, key, getattr(other, key))
        return tmp

    def __radd__(self, other):
        return other + self

    def __iadd__(self, other):
        if other is not None:
            if isinstance(other, dict):
                other = Config(other)
            elif not isinstance(other, Config):
                raise TypeError(f'Неверный тип. Хотел: Config или dict с маршрутами, а получил {other.__type__}')
            for key in get_attr(other):
                setattr(self, key, getattr(other, key))
        return self
