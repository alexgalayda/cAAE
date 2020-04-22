import argparse, sys
sys.path.append('/root')
# sys.path.append('/root/model')
from model.tools.config import read_conf
from model.generator import generator, net


def test(config, load_path, acc=0.3):
    dataset = generator(config, train_flg=False)
    config.transforms += {'img_shape': dataset.get_img_shape()}
    model = net[config.struct.name](config, train_flg=False)
    model.load(load_path)
    model.test(dataset, acc)


def test_show(config, load_path, acc=0.3, idx=None):
    dataset = generator(config, train_flg=False)
    config.transforms += {'img_shape': dataset.get_img_shape()}
    model = net[config.struct.name](config, train_flg=False)
    model.load(load_path)
    model.test_show(dataset, acc, idx=idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_name', dest='config_name', type=str, default='config_example')
    parser.add_argument('-w', '--weights_path', dest='weights_path', type=str, default='/root/weights')
    args = parser.parse_args()
    config = read_conf(f'./config/{args.config_name}.json')
    test(config, args.weights_path)
