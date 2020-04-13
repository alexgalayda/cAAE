import argparse

from tools.config import Config, read_conf
from AAE import AAE
from generator import generator


#TODO: сохранить сгенегированные изображение в results
def train(config, save_path):
    dataset = generator(config, train_flg=True)
    config.transforms += {'img_shape': dataset.get_img_shape()}
    net = {"AAE": AAE}
    model = net['net'](config)
    model.train(dataset)
    model.save(save_path)
    model.sample_image()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_name', dest='config_name', type=str, default='config_example')
    parser.add_argument('-w', '--weights_path', dest='weights_path', type=str, default='/root/weights')
    args = parser.parse_args()
    config = read_conf(f'./config/{args.config_name}.json')
    train(config, args.weights_path)
