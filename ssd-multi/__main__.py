#!/usr/bin/env python3
import os
import argparse
import sys
import logging
import torch
# Import matplotlib and choose a different backend that doesn't need the X server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastai.conv_learner import ConvLearner
from fastai.model import resnet34
from fastai.transforms import tfms_from_model, transforms_side_on
from fastai.dataset import ImageClassifierData

IMAGE_BASE_DIR = '/Users/g6714/Data/fastai/dogscats'
EPOCH_COUNT = 25
IMAGE_SIZE = 224

log = logging.getLogger('transfer')
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


def main(imgdir, epochs):
    log.info(f'Cuda available: {torch.cuda.is_available()}')
    log.info(f'Cuda backend enabled: {torch.backends.cudnn.enabled}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a network with transfer learning')
    parser.add_argument('--image_dir', dest='imgdir', default=IMAGE_BASE_DIR, help='Location of image training data')
    parser.add_argument('--epochs', dest='epochs', type=int, default=EPOCH_COUNT, help='Number of epochs')
    args = parser.parse_args()

    main(args.imgdir, args.epochs)
