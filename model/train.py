import argparse, sys
sys.path.append('/root')
# sys.path.append('/root/model')
from model.tools.config import read_conf
from model.generator import generator, net


def train(config, save_path):
    dataset = generator(config, train_flg=True)
    config.transforms += {'img_shape': dataset.get_img_shape()}
    model = net[config.struct.name](config)
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
