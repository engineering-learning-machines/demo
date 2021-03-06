import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects


def show_img(im, figsize=None, ax=None):
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.set_xticks(np.linspace(0, 224, 8))
    ax.set_yticks(np.linspace(0, 224, 8))
    ax.grid()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    return ax


def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


def draw_rect(ax, b, color='white'):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
    draw_outline(patch, 4)


def draw_text(ax, xy, txt, sz=14, color='white'):
    text = ax.text(*xy, txt,
                   verticalalignment='top', color=color, fontsize=sz, weight='bold')
    draw_outline(text, 1)


# def draw_im(im, ann):
#     ax = show_img(im, figsize=(16,8))
#     for b,c in ann:
#         b = bb_hw(b)
#         draw_rect(ax, b)
#         draw_text(ax, b[:2], cats[c], sz=16)
#
# def draw_idx(i):
#     im_a = trn_anno[i]
#     im = open_image(IMG_PATH/trn_fns[i])
#     draw_im(im, im_a)


def plot_multiclass_predict(x, y, md):
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        ima = md.val_ds.denorm(x)[i]
        ya = np.nonzero(y[i] > 0.4)[0]
        b = '\n'.join(md.classes[o] for o in ya)
        ax = show_img(ima, ax=ax)
        draw_text(ax, (0, 0), b)
    plt.tight_layout()
