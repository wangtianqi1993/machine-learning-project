# !/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'wtq'

from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                      discriminant_analysis, random_projection)

digits = datasets.load_digits(n_class=6)
x = digits.data
y = digits.target
n_samples, n_features = x.shape
n_neighbors = 30


def plot_embedding(x, title=None):
    """
    scale and visualize the embedding vectors
    :param x:
    :param title:
    :return:
    """
    # 获取矩阵x每列的最小大值
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = (x - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(x.shape[0]):
        plt.text(x[i, 0], x[i, 1], str(digits.target[i]),
                 color='r',
                 fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])
        for i in range(digits.data.shape[0]):
            dist = np.sum((x[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                continue
            shown_images = np.r_[shown_images, [x[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap='Greys_r'),
                x[i]
            )
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

# plot images of the digits
n_img_per_row = 20
img = np.zeros((10 * n_img_per_row, 10*n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = x[i * n_img_per_row + j].reshape((8, 8))
plt.imshow(img, cmap='Greys_r')
plt.xticks([])
plt.yticks([])
plt.title('A selection from the 64-dimensional digits dataset')


def t_sne():
    """

    :return:
    """
    print('computing t-sne embedding')
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    x_tsne = tsne.fit_transform(x)

    plot_embedding(x_tsne, "t-SNE embedding of the digits (time %.2fs)" %
                   (time() - t0))
    plt.show()

if __name__ == "__main__":
    t_sne()
