import os

from model import PreResNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np
from trainer import Trainer
import sys
from utils import *
import argparse


parser = argparse.ArgumentParser(description='Incremental Learning BIC')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--epoch', default=250, type=int)
parser.add_argument('--lr', default=0.1, type=int)
parser.add_argument('--max_size', default=2000, type=int)  # 旧类样本保存的最大数量
parser.add_argument('--total_cls', default=100, type=int)  # 总类别数目
args = parser.parse_args()


if __name__ == "__main__":
    model = PreResNet(32, 100).cuda()
    print(model)
    #trainer = Trainer(args.total_cls)
    #trainer.train(args.batch_size, args.epoch, args.lr, args.max_size)
