#!/usr/bin/env python3
import os
import argparse
import sys
import logging
import torch
from fastai.conv_learner import ConvLearner
from fastai.model import resnet34, resnet50
from fastai.transforms import tfms_from_model
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

    model = resnet34
    data = ImageClassifierData.from_paths(imgdir, tfms=tfms_from_model(model, IMAGE_SIZE))
    learn = ConvLearner.pretrained(model, data, precompute=True)

    # lrf = learn.lr_find()
    learn.fit(0.01, 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a network with transfer learning')
    parser.add_argument('--image_dir', dest='imgdir', default=IMAGE_BASE_DIR, help='Location of image training data')
    parser.add_argument('--epochs', dest='epochs', type=int, default=EPOCH_COUNT, help='Number of epochs')
    args = parser.parse_args()

    main(args.imgdir, args.epochs)
