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
PATH = Path('/home/rseed42/Data/pascal')
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


def detn_loss(input, target):
    bb_t,c_t = target
    bb_i,c_i = input[:, :4], input[:, 4:]
    bb_i = F.sigmoid(bb_i)*224
    # I looked at these quantities separately first then picked a multiplier
    #   to make them approximately equal
    return F.l1_loss(bb_i, bb_t) + F.cross_entropy(c_i, c_t)*20


def detn_l1(input, target):
    bb_t,_ = target
    bb_i = input[:, :4]
    bb_i = F.sigmoid(bb_i)*224
    return F.l1_loss(V(bb_i),V(bb_t)).data


def detn_acc(input, target):
    _,c_t = target
    c_i = input[:, 4:]
    return accuracy(c_i, c_t)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ConcatLblDataset(Dataset):
    def __init__(self, ds, y2):
        self.ds,self.y2 = ds,y2

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x,y = self.ds[i]
        return (x, ( y, self.y2[i] ) )


# ----------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
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
    sz = 224
    bs = 64

    BB_CSV = PATH/'tmp/bb.csv'
    bb = np.array([trn_lrg_anno[o][0] for o in trn_ids])
    bbs = [' '.join(str(p) for p in o) for o in bb]

    df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids], 'bbox': bbs}, columns=['fn','bbox'])
    df.to_csv(BB_CSV, index=False)


    tfm_y = TfmType.COORD
    augs = [RandomFlip(tfm_y=tfm_y),
            RandomRotate(3, p=0.5, tfm_y=tfm_y),
            RandomLighting(0.05,0.05, tfm_y=tfm_y)]

    val_idxs = get_cv_idxs(len(trn_fns))
    tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, tfm_y=TfmType.COORD, aug_tfms=augs)

    # Training Data
    md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms=tfms, bs=bs, continuous=True, val_idxs=val_idxs)
    md2 = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms=tfms_from_model(f_model, sz))

    trn_ds2 = ConcatLblDataset(md.trn_ds, md2.trn_y)
    val_ds2 = ConcatLblDataset(md.val_ds, md2.val_y)

    md.trn_dl.dataset = trn_ds2
    md.val_dl.dataset = val_ds2

    # x, y = next(iter(md.val_dl))
    # idx=3
    # ima=md.val_ds.ds.denorm(to_np(x))[idx]
    # b = bb_hw(to_np(y[0][idx]))
    # print(b)

    # ax = show_img(ima)
    # draw_rect(ax, b)
    # draw_text(ax, b[:2], md2.classes[y[1][idx]])
    # plt.show()

    head_reg4 = nn.Sequential(
        Flatten(),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(25088, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, 4+len(cats)),
    )
    models = ConvnetBuilder(f_model, 0, 0, 0, custom_head=head_reg4)

    learn = ConvLearner(md, models)
    learn.opt_fn = optim.Adam
    learn.crit = detn_loss
    learn.metrics = [detn_acc, detn_l1]

    lr = 1e-2
    learn.fit(lr, 1, cycle_len=3, use_clr=(32, 5))

    learn.save('reg1_0')
    learn.freeze_to(-2)
    lrs = np.array([lr/100, lr/10, lr])

    learn.lr_find(lrs/1000)

    learn.fit(lrs/5, 1, cycle_len=5, use_clr=(32, 10))
    learn.save('reg1_1')

    learn.load('reg1_1')
    learn.unfreeze()

    learn.fit(lrs/10, 1, cycle_len=10, use_clr=(32, 10))
    learn.save('reg1')

