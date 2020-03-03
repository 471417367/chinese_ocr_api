import os

print(os.getcwd())
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from lib.fast_rcnn.train import get_training_roidb, train_net
from lib.fast_rcnn.config import cfg_from_file, get_output_dir, get_log_dir
from lib.datasets.factory import get_imdb
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg

if __name__ == '__main__':
    # 将text.yml的配置与默认config中的默认配置进行合并
    cfg_from_file('text.yml')
    print('Using config:~~~~~~~~~~~~~~~~')
    # 根据给定的名字，得到要加载的数据集
    imdb = get_imdb('voc_2007_trainval')
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    # 准备训练数据
    roidb = get_training_roidb(imdb)
    # 模型输出的路径
    # output_dir = get_output_dir(imdb, None)
    output_dir = './ckpt'
    # summary的输出路径
    log_dir = get_log_dir(imdb)
    device_name = '/gpu:0'
    print(device_name)

    network = get_network('VGGnet_train')

    train_net(
        network,
        imdb,
        roidb,
        output_dir=output_dir,
        log_dir=log_dir,
        pretrained_model='./pretrain_vggmodel/VGG_imagenet.npy',
        # pretrained_model='/home/xiaofeng/data/ctpn/pretrainde_vgg',
        # pretrained_model=None,
        max_iters=180000,
        restore=bool(int(1)))
