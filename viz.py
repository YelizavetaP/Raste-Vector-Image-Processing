"""Reusable plotting helpers for HW2 notebook and scripts."""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_two_images(img_a, img_b, title_a='', title_b='', suptitle=None, figsize=(12, 5)):
    """Side-by-side plotter. Handles 1-ch (gray) and 3-ch BGR."""
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    for a, im, t in zip(ax, (img_a, img_b), (title_a, title_b)):
        if im.ndim == 2:
            a.imshow(im, cmap='gray', vmin=0, vmax=255)
        else:
            a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        a.set_title(t)
        a.set_xticks([]); a.set_yticks([])
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    plt.show()


def _draw_image_panel(ax, im, title):
    if im.ndim == 2:
        ax.imshow(im, cmap='gray', vmin=0, vmax=255)
    else:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])


def _draw_hist_panel(ax, im, title):
    data = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).ravel() if im.ndim == 3 else im.ravel()
    ax.hist(data, bins=256, range=(0, 256), color='steelblue')
    ax.set_xlim(0, 255); ax.set_title(title); ax.set_xlabel('intensity')


def show_compare_with_hists(orig_a, orig_b, mod_a, mod_b,
                            label_a='bing', label_b='landsat', tag='', figsize=(18, 8)):
    """2×4 grid. Row 0 = originals, Row 1 = modified. Columns: img_a | hist_a | img_b | hist_b."""
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    for r, (row_tag, im_a, im_b) in enumerate([
        ('original', orig_a, orig_b),
        (tag,        mod_a,  mod_b),
    ]):
        _draw_image_panel(axes[r][0], im_a, f'{label_a} — {row_tag}' if row_tag else label_a)
        _draw_hist_panel( axes[r][1], im_a, f'{label_a} hist ({row_tag})' if row_tag else f'{label_a} hist')
        _draw_image_panel(axes[r][2], im_b, f'{label_b} — {row_tag}' if row_tag else label_b)
        _draw_hist_panel( axes[r][3], im_b, f'{label_b} hist ({row_tag})' if row_tag else f'{label_b} hist')
    fig.tight_layout()
    plt.show()
