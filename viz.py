"""Reusable plotting helpers for HW1/HW2 notebook and scripts.

All save-to-disk is opt-in: pass `save_path=...` to write the figure to that
path (default = show only, don't touch the filesystem).
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Small reusable plotters (used by the experimentation notebook).             #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Pipeline visualisations (originally in main.py).                            #
# Pass `save_path=Path(...)` to write the figure; default is show-only.       #
# --------------------------------------------------------------------------- #
def _maybe_save(fig, save_path):
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)


def show_originals(bing, landsat, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(cv2.cvtColor(bing, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Bing Maps - high-res reference")
    ax[1].imshow(cv2.cvtColor(landsat, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Landsat - low-res tuning target")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Originals: KPI main campus (input images)")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    plt.show()


def show_pipeline_grid(snaps_top, snaps_bot, count_top, count_bot,
                       label_top, label_bot, save_path=None):
    cols = max(len(snaps_top), len(snaps_bot))
    fig, ax = plt.subplots(2, cols, figsize=(3.0 * cols, 6.5))
    if cols == 1:
        ax = np.array([[ax[0]], [ax[1]]])
    for row, (snaps, label, count) in enumerate((
        (snaps_top, label_top, count_top),
        (snaps_bot, label_bot, count_bot),
    )):
        for j in range(cols):
            a = ax[row, j]
            if j < len(snaps):
                name, img, _ = snaps[j]
                a.imshow(img, cmap=None if img.ndim == 3 else "gray")
                a.set_title(name, fontsize=10)
            else:
                a.axis("off")
            a.set_xticks([]); a.set_yticks([])
        ax[row, 0].set_ylabel(f"{label}\ncount = {count}", fontsize=11)
    fig.suptitle("Pipeline manipulations - top: Bing reference, bottom: Landsat")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    plt.show()


def show_contours_side_by_side(ctx_b, ctx_l, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(cv2.cvtColor(ctx_b["rect_overlay"], cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"Bing - detected rectangles: {ctx_b['count']}")
    ax[1].imshow(cv2.cvtColor(ctx_l["rect_overlay"], cv2.COLOR_BGR2RGB))
    ax[1].set_title(f"Landsat - detected rectangles: {ctx_l['count']}")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Vectorized contours (approxPolyDP, 4 vertices) - building candidates")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    plt.show()
