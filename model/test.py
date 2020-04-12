import argparse

from tools.config import Config, read_conf
from models import AAE
from generator import generator


#TODO: добавить что-то после теста
def test(config, load_path):
    dataset = generator(config, train_flg=False)
    config.transforms += {'img_shape': dataset.get_img_shape()}
    model = AAE(config, train_flg=False)
    model.load(load_path)
    model.test(dataset)


def test_show(config, load_path):
    dataset, img_shape = generator(config, health_flg=False)
    config.transforms += {'img_shape': dataset.get_img_shape()}
    model = AAE(config, train_flg=False)
    model.load(load_path)
    decod_tumor, test_tumor = model.test_show(dataset)
    return decod_tumor, test_tumor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_name', dest='config_name', type=str, default='config_example')
    parser.add_argument('-w', '--weights_path', dest='weights_path', type=str, default='/root/weights')
    args = parser.parse_args()
    config = read_conf(f'./config/{args.config_name}.json')
    test(config, args.weights_path)
