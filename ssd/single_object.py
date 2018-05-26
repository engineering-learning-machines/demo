#!/usr/bin/env python3
import torch
from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path
import json
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects

# ----------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------
IMAGES, ANNOTATIONS, CATEGORIES = ['images', 'annotations', 'categories']
FILE_NAME, ID, IMG_ID, CAT_ID, BBOX = 'file_name', 'id', 'image_id', 'category_id', 'bbox'
PATH = Path('/home/ubuntu/data/pascal')
JPEGS = 'train/VOC2007/JPEGImages'
# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------


def hw_bb(bb):
    return np.array([bb[1], bb[0], bb[3]+bb[1]-1, bb[2]+bb[0]-1])


def bb_hw(a):
    return np.array([a[1],a[0],a[3]-a[1]+1,a[2]-a[0]+1])


def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(linewidth=lw, foreground='black'), patheffects.Normal()])


def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)


def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt, verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)


def draw_im(im, ann):
    ax = show_img(im, figsize=(16,8))
    for b,c in ann:
        b = bb_hw(b)
        draw_rect(ax, b)
        draw_text(ax, b[:2], cats[c], sz=16)


def draw_idx(i):
    im_a = trn_anno[i]
    im = open_image(IMG_PATH/trn_fns[i])
    print(im.shape)
    draw_im(im, im_a)

def get_lrg(b):
    if not b: raise Exception()
    b = sorted(b, key=lambda x: np.product(x[0][-2:]-x[0][:2]), reverse=True)
    return b[0]

# ----------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    #
    acceleration = torch.cuda.is_available()
    print('---- GPU Accelerion: {} ----'.format(acceleration))
    #
    trn_j = json.load((PATH/'PASCAL_VOC/pascal_train2007.json').open())
    cats = {o[ID]:o['name'] for o in trn_j[CATEGORIES]}
    trn_fns = {o[ID]:o[FILE_NAME] for o in trn_j[IMAGES]}
    trn_ids = [o[ID] for o in trn_j[IMAGES]]
    #
    IMG_PATH = PATH/JPEGS
    im0_d = trn_j[IMAGES][0]
    #
    trn_anno = collections.defaultdict(lambda: [])
    for o in trn_j[ANNOTATIONS]:
        if not o['ignore']:
            bb = o[BBOX]
            bb = hw_bb(bb)
            trn_anno[o[IMG_ID]].append((bb, o[CAT_ID]))

    len(trn_anno)
    im_a = trn_anno[im0_d[ID]]
    im0_a = im_a[0]

    im = open_image(IMG_PATH/im0_d[FILE_NAME])

    # Example: single bounding box + label
    # ax = show_img(im)
    # b = bb_hw(im0_a[0])
    # draw_rect(ax, b)
    # draw_text(ax, b[:2], cats[im0_a[1]])
    #
    # Example: Multiple bounding boxes with labels
    # draw_idx(17)
    #
    # plt.show()

    # ===== Largest Item Classifier ======= #
    # image id -> largest bounding box
    trn_lrg_anno = {a: get_lrg(b) for a,b in trn_anno.items()}

    # Example
    # b, c = trn_lrg_anno[23]
    # b = bb_hw(b)
    # ax = show_img(open_image(IMG_PATH/trn_fns[23]), figsize=(5,10))
    # draw_rect(ax, b)
    # draw_text(ax, b[:2], cats[c], sz=16)
    # plt.show()

    # Prepare the dataset for the classification of the largest object only
    (PATH/'tmp').mkdir(exist_ok=True)
    CSV = PATH/'tmp/lrg.csv'
    df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids],
                       'cat': [cats[trn_lrg_anno[o][1]] for o in trn_ids]}, columns=['fn', 'cat'])
    df.to_csv(CSV, index=False)

    # Some hyperparams, also images get resized to 224
    f_model = resnet34
    sz=224
    bs=64
    # Create some augmented data
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_side_on, crop_type=CropType.NO)
    md = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms=tfms, bs=bs)

    # x, y = next(iter(md.val_dl))
    # show_img(md.val_ds.denorm(to_np(x))[0])
    # plt.show()

    learn = ConvLearner.pretrained(f_model, md, metrics=[accuracy])
    learn.opt_fn = optim.Adam

    # lrf = learn.lr_find(1e-5, 100)

    # learn.sched.plot(n_skip=5, n_skip_end=1)
    # plt.show()

    # Choose the learning rate
    lr = 2e-2

    # Now train
    learn.fit(lr, 1, cycle_len=1)

    # Choose a learning rate for the different layers
    lrs = np.array([lr/1000, lr/100, lr])
    learn.freeze_to(-2)

    lrf=learn.lr_find(lrs/1000)

    # Retrain with different learning rates
    learn.fit(lrs/5, 1, cycle_len=1)

    learn.unfreeze()

    learn.fit(lrs/5, 1, cycle_len=2)

    learn.save('clas_one')




