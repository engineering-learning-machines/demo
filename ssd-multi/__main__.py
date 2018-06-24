#!/usr/bin/env python3
import os
import argparse
import sys
import logging
import torch
import json
import collections
from pathlib import Path
import pandas as pd
import numpy as np
# Import matplotlib and choose a different backend that doesn't need the X server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Project-level imports
from util import bb_hw
from fastai.conv_learner import resnet34, ConvLearner
from fastai.dataset import ImageClassifierData
from fastai.transforms import tfms_from_model, CropType
from torch import optim
# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

BASE_DIR = '/Users/g6714/Data/pascal'
TRAINING_METADATA_DIR = 'PASCAL_VOC/pascal_train2007.json'
TRAINING_IMAGE_SUBDIR = 'train/VOC2007/JPEGImages'
EPOCH_COUNT = 25
IMAGE_SIZE = 224

log = logging.getLogger('transfer')
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------


class TrainingMetaData(object):
    def __init__(self, basedir_path):
        # The training images are stored here in JPEG format
        # self.image_dir = basedir_path / TRAINING_IMAGE_DIR
        self.image_subdir = TRAINING_IMAGE_SUBDIR
        # Load the training json metadata and extract all relevant information
        pascal_metadata = json.load((basedir_path / TRAINING_METADATA_DIR).open())
        self.id_category_map = dict((o['id'], o['name']) for o in pascal_metadata['categories'])
        self.id_filename_map = {o['id']: o['file_name'] for o in pascal_metadata['images']}
        self.image_ids = [o['id'] for o in pascal_metadata['images']]
        self.annotations = self.get_annotations(pascal_metadata)

    def get_annotations(self, pascal_metadata):
        anno = collections.defaultdict(lambda: [])
        for o in pascal_metadata['annotations']:
            if not o['ignore']:
                bb = o['bbox']
                bb = np.array([bb[1], bb[0], bb[3] + bb[1] - 1, bb[2] + bb[0] - 1])
                anno[o['image_id']].append((bb, o['category_id']))
        return anno


class MetaData(object):
    """
    Parses the metadata
    """
    def __init__(self, basedir):
        self.basedir_path = Path(basedir)
        self.train = TrainingMetaData(self.basedir_path)


    @staticmethod
    def create(basedir):
        """
        Factory method
        :param basedir:
        :return:
        """
        return MetaData(basedir)


class MultiClassifier(object):
    def __init__(self, metadata):
        # The temp folder does not exist initially
        tmp_dir = metadata.basedir_path / 'tmp'
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        mc_csv = tmp_dir / 'mc.csv'
        # Example annotation: bounding box + category id
        # print(metadata.training.annotations[12])

        # Prepare the model
        self.model = resnet34
        self.image_size = 224
        self.batch_size = 64
        # Non-agumented transforms
        self.non_aug_transforms = tfms_from_model(self.model, self.image_size, crop_type=CropType.NO)
        # For some dumb reason this requires relative paths
        md = ImageClassifierData.from_csv(
            metadata.basedir_path,
            metadata.train.image_subdir,
            mc_csv, tfms=self.non_aug_transforms,
            bs=self.batch_size
        )
        self.learner = ConvLearner.pretrained(self.model, md)
        self.learner.opt_fn = optim.Adam

    def find_learning_rate(self):
        lr_finder = self.learner.lr_find(1e-5, 100)
        self.learner.sched.plot(0)
        plt.savefig('lr_find.png')

    def save_csv(self, csv_path, id_category_map, id_filename_map, annotations, image_ids):
        mc = [set([id_category_map[p[1]] for p in annotations[o]]) for o in image_ids]
        mcs = [' '.join(str(p) for p in o) for o in mc]
        df = pd.DataFrame({'fn': [id_filename_map[o] for o in image_ids], 'clas': mcs}, columns=['fn', 'clas'])
        df.to_csv(csv_path, index=False)



# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------


def main(basedir, epochs):
    log.info(f'Cuda available: {torch.cuda.is_available()}')
    log.info(f'Cuda backend enabled: {torch.backends.cudnn.enabled}')

    metadata = MetaData.create(basedir)
    multi_classifier = MultiClassifier(metadata)
    multi_classifier.find_learning_rate()



    # parent_dir = Path(basedir)
    # trn_j = json.load((parent_dir / 'PASCAL_VOC' / 'pascal_train2007.json').open())
    # # IMAGES,ANNOTATIONS,CATEGORIES = ['images', 'annotations', 'categories']
    # # FILE_NAME,ID,IMG_ID,CAT_ID,BBOX = 'file_name','id','image_id','category_id','bbox'
    #
    # cats = dict((o['id'], o['name']) for o in trn_j['categories'])
    # trn_fns = dict((o['id'], o['file_name']) for o in trn_j['images'])
    # trn_ids = [o['id'] for o in trn_j['images']]
    #
    # jpeg_path = 'VOCdevkit/VOC2007/JPEGImages'
    # img_path = parent_dir / jpeg_path
    # # Annotations object
    # trn_anno = get_trn_anno(trn_j)






    # mc_csv = parent_dir / 'tmp/mc.csv'
    # print(trn_anno[12])
    #
    # mc = [set([cats[p[1]] for p in trn_anno[o]]) for o in trn_ids]
    # mcs = [' '.join(str(p) for p in o) for o in mc]
    #
    # df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids], 'clas': mcs}, columns=['fn', 'clas'])
    # df.to_csv(mc_csv, index=False)
    #
    #
    # f_model=resnet34
    # sz=224
    # bs=64
    #
    #
    # tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO)
    # md = ImageClassifierData.from_csv(PATH, JPEGS, MC_CSV, tfms=tfms, bs=bs)
    #
    # learn = ConvLearner.pretrained(f_model, md)
    # learn.opt_fn = optim.Adam
    #
    # lrf=learn.lr_find(1e-5,100)
    #
    # lrf=learn.lr_find(1e-5,100)

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a network with transfer learning')
    parser.add_argument('--image_dir', dest='imgdir', default=BASE_DIR, help='Location of image training data')
    parser.add_argument('--epochs', dest='epochs', type=int, default=EPOCH_COUNT, help='Number of epochs')
    args = parser.parse_args()

    main(args.imgdir, args.epochs)
