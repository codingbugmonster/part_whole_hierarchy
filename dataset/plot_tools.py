#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


def plot_groups(groups, ax):
    mask = (groups == 0)

    colors = ['gold', 'royalblue', 'limegreen', 'coral', 'forestgreen', 'dodgerblue', 'firebrick', 'palevioletred', 'darkgreen', 'chocolate', 'purple',
     'darkorange', 'hotpink', 'c', 'indigo', 'teal', 'navy', 'gray']


    # colors = ['white', 'coral', 'forestgreen', 'powderblue']
    cmap_obj = mpl.colors.ListedColormap(colors)
    # plt.imshow(label, cmap=cmap_obj)

    sns.heatmap(groups, mask=mask, square=True, cmap=cmap_obj,
                xticklabels=False, yticklabels=False, cbar=False, ax=ax)
    fig = plt.gca()
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.get_xaxis().set_visible(False)

    # sns.heatmap(groups, mask=mask, square=True, cmap='viridis_r',
    #             xticklabels=False, yticklabels=False, cbar=False, ax=ax)
    #sns.heatmap(groups, square=True, cmap='viridis_r',
                # xticklabels=False, yticklabels=False, cbar=False, ax=ax)


def plot_input_image(img, ax):
    colors = ['white', 'black']
    cmap = mpl.colors.ListedColormap(colors)
    # sns.heatmap(img, square=True, xticklabels=False,
    #             yticklabels=False, cmap='Greys', cbar=False, ax=ax)
    sns.heatmap(img, square=True, xticklabels=False,
                yticklabels=False, cmap=cmap, cbar=False, ax=ax)
    fig = plt.gca()
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.get_xaxis().set_visible(False)

