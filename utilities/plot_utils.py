"""
This module includes helper functions for plotting.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_image(image, figsize=None, title=None, xlim=None, ylim=None, extent=None, colorbar=False, cmap=None):
    if cmap is None:
        cmap = 'nipy_spectral'
    if figsize is None:
        figsize = (9, 6)
    fig = plt.figure(figsize=figsize)
    if extent is None:
        plt.imshow(image, origin='lower', cmap=cmap)  # 'hot' 'nipy_spectral'
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(image, origin='lower', cmap=cmap, extent=extent)
    if title:
        plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if colorbar:
        plt.colorbar(orientation="horizontal")
    plt.show()


def plot_images(images, figsize=None, title_list=None, xlim=None, ylim=None, extend=None, colorbar=False, cmap=None):
    if cmap is None:
        cmap = 'nipy_spectral'
    n_images = len(images)
    if figsize is None:
        figwidth = np.min((n_images * 6, 24))
        figsize = (figwidth, 6)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    for count, i_image in enumerate(images):
        if extend is None:
            im = axes[count].imshow(i_image, origin='lower', cmap=cmap)
        else:
            im = axes[count].imshow(i_image, origin='lower', cmap=cmap, extend=extend)
        if title_list is not None:
            axes[count].set_title(title_list[count])
        if xlim is not None:
            axes[count].set_xlim(xlim)
        if ylim is not None:
            axes[count].set_ylim(ylim)
        if colorbar:
            fig.colorbar(im, ax=axes[count])
    plt.show()


def plot_graph(y, figsize=None, title=None, xlim=None, ylim=None):
    if figsize is None:
        figsize = (9, 6)
    plt.figure(figsize=figsize)
    plt.plot(y)
    if title:
        plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    else:
        plt.xlim([0, len(y)])
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()


def plot_graphs(graph_list, title_list=[], xlim=None, ylim=None):
    n_graphs = len(graph_list)
    fig_depth = 2 * n_graphs
    # fig_depth = np.max((4, fig_depth))
    fig_depth = np.min((24, fig_depth))
    plt.figure(figsize=(12, fig_depth))
    for i in range(n_graphs):
        plt.subplot(n_graphs, 1, i + 1)
        plt.plot(graph_list[i])
        if title_list:
            plt.title(title_list[i])
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
    plt.tight_layout()
    plt.show()