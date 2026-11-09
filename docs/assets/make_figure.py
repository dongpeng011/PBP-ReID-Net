# encoding: utf-8
"""Generate the architecture figure used in the README.

    python docs/assets/make_figure.py

Layout: one row with the forward pass, one row with a zoom on each of the three
contributions (EASA, HPRA, GAL).
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'pdf.fonttype': 42,
})

BLUE, BLUE_E = '#dbe9f6', '#2f6db3'
ORANGE, ORANGE_E = '#fbe3cf', '#c96a1b'
GREEN, GREEN_E = '#dff0d8', '#4b8b3b'
GREY, GREY_E = '#f0f0f0', '#8a8a8a'


def box(ax, x, y, w, h, text, face, edge, fontsize=7.4, weight='normal', pad=0.004):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad={},rounding_size=0.02'.format(pad),
                                linewidth=0.9, facecolor=face, edgecolor=edge))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, weight=weight, linespacing=1.32)


def arrow(ax, p0, p1, lw=0.9, color='#3d3d3d'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=8,
                                 linewidth=lw, color=color, shrinkA=1.0, shrinkB=1.0))


def clean(ax, xlim=(0, 1), ylim=(0, 1)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis('off')


def main():
    fig = plt.figure(figsize=(10.4, 4.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[0.62, 1.5],
                            left=0.015, right=0.985, top=0.96, bottom=0.03,
                            hspace=0.05, wspace=0.09)

    # =============================================================== flow
    ax = fig.add_subplot(grid[0, :])
    clean(ax, ylim=(0, 1))
    ax.text(0.0, 0.92, 'Forward pass', fontsize=9.6, weight='bold', color='#1f1f1f')
    ax.text(0.135, 0.92, 'ViT-Base · 256×128 crop · 16×16 patches → 16×8 = 128 tokens · P = 3 bands',
            fontsize=7.2, color='#5a5a5a')

    stages = [
        (0.100, 'input crop\n256 × 128', GREY, GREY_E),
        (0.125, 'patch embedding\n16×16 / stride 16\n→ 128 tokens', GREY, GREY_E),
        (0.125, 'blocks 0–5\nfull-token\nself-attention', BLUE, BLUE_E),
        (0.115, 'blocks 6–11\nHPRA  §3.2', ORANGE, ORANGE_E),
        (0.155, 'GAL  §3.3\n$F=x_g+W_f\\,\\mathrm{vec}(X_f)$', GREEN, GREEN_E),
        (0.095, 'identity\ndescriptor $F$', GREY, GREY_E),
    ]
    gap = 0.030
    total = sum(s[0] for s in stages) + gap * (len(stages) - 1)
    x, y, h = (1.0 - total) / 2, 0.22, 0.42
    for width, text, face, edge in stages:
        box(ax, x, y, width, h, text, face, edge, fontsize=7.0)
        x += width
        if width != stages[-1][0]:
            arrow(ax, (x + 0.002, y + h / 2), (x + gap - 0.002, y + h / 2))
            x += gap
    ax.text(0.985, 0.10, 'one descriptor per crop', ha='right', fontsize=6.6, color='#6a6a6a')

    # ============================================================== EASA
    ax = fig.add_subplot(grid[1, 0])
    clean(ax)
    ax.text(0.0, 0.96, '(a) EASA — body-prior anchoring', fontsize=8.4, weight='bold', color='#1f1f1f')
    ax.text(0.0, 0.88, 'parameter-free · no keypoints · no masks',
            fontsize=6.8, color='#5a5a5a')

    gx, gy, cw, ch = 0.02, 0.24, 0.036, 0.031
    band_of_row = [0] * 5 + [1] * 5 + [2] * 6
    colours = {0: '#f8d3b4', 1: '#f0a969', 2: '#dd7b36'}
    for row in range(16):
        for col in range(8):
            ax.add_patch(Rectangle((gx + col * cw, gy + (15 - row) * ch), cw * 0.92, ch * 0.86,
                                   facecolor=colours[band_of_row[row]],
                                   edgecolor='white', linewidth=0.35))
    for boundary in (5, 10):
        ax.plot([gx - 0.006, gx + 8 * cw], [gy + (16 - boundary) * ch] * 2,
                color='#7a3b12', linewidth=1.1, solid_capstyle='butt')
    for label, y, n in (('head', gy + 13.4 * ch, 40), ('torso', gy + 8.4 * ch, 40),
                        ('lower body', gy + 3.2 * ch, 48)):
        ax.text(gx + 8 * cw + 0.035, y, '{}\n{} tokens'.format(label, n),
                fontsize=6.5, va='center', color='#7a3b12')
    ax.text(0.0, 0.11, r'$N_i=\lfloor N_y/P\rfloor N_x$,  $N_P=N-(P-1)N_p$',
            fontsize=6.9, color='#5a5a5a')
    ax.text(0.0, 0.03, '16 rows → 5 / 5 / 6 rows → 40 / 40 / 48 tokens',
            fontsize=6.6, color='#5a5a5a')

    # ============================================================== HPRA
    ax = fig.add_subplot(grid[1, 1])
    clean(ax)
    ax.text(0.0, 0.96, '(b) HPRA — part-relation attention', fontsize=8.4, weight='bold', color='#1f1f1f')
    ax.text(0.0, 0.88, 'level 1 inside the bands · level 2 across them',
            fontsize=6.8, color='#5a5a5a')

    ys = [0.54, 0.42, 0.30]
    for index, (yy, size) in enumerate(zip(ys, (40, 40, 48))):
        box(ax, 0.00, yy, 0.14, 0.085, 'band {}\n{}'.format(index + 1, size), ORANGE, ORANGE_E, fontsize=6.4)
        box(ax, 0.175, yy, 0.21, 0.085, 'MHSA\n(shared)', ORANGE, ORANGE_E, fontsize=6.3)
        arrow(ax, (0.142, yy + 0.042), (0.172, yy + 0.042))
        arrow(ax, (0.387, yy + 0.042), (0.445, 0.395))
    box(ax, 0.45, 0.345, 0.15, 0.10, 'GAP\n$\\rightarrow t_i$', ORANGE, ORANGE_E, fontsize=6.5)
    arrow(ax, (0.603, 0.395), (0.640, 0.395))
    box(ax, 0.645, 0.330, 0.22, 0.13, 'MHSA over\n3 tokens\n$\\rightarrow X_f$', ORANGE, ORANGE_E, fontsize=6.3)
    ax.text(0.0, 0.11, r'level 1: $\mathcal{O}(N^2C/P)$', fontsize=6.9, color='#5a5a5a')
    ax.text(0.0, 0.03, r'level 2: $\mathcal{O}(P^2C)$,  $P=3$', fontsize=6.9, color='#5a5a5a')

    # =============================================================== GAL
    ax = fig.add_subplot(grid[1, 2])
    clean(ax)
    ax.text(0.0, 0.96, '(c) GAL — additive fusion', fontsize=8.4, weight='bold', color='#1f1f1f')
    ax.text(0.0, 0.88, 'no channel expansion · no concat + FC',
            fontsize=6.8, color='#5a5a5a')

    box(ax, 0.00, 0.56, 0.40, 0.12, '$x_g=\\mathrm{GAP}(X)\\in\\mathbb{R}^{C}$', GREEN, GREEN_E, fontsize=6.9)
    box(ax, 0.00, 0.34, 0.40, 0.12, '$x_l=W_f\\,\\mathrm{vec}(X_f)$', GREEN, GREEN_E, fontsize=6.9)
    box(ax, 0.56, 0.45, 0.42, 0.13, '$F=x_g+x_l$', GREEN, GREEN_E, fontsize=7.8, weight='bold')
    arrow(ax, (0.41, 0.62), (0.55, 0.54))
    arrow(ax, (0.41, 0.40), (0.55, 0.49))
    ax.text(0.00, 0.22, r'$W_f\in\mathbb{R}^{PC\times C}$: 1.8 M parameters',
            fontsize=6.9, color='#5a5a5a')
    ax.text(0.00, 0.13, 'descriptor keeps the backbone', fontsize=6.6, color='#5a5a5a')
    ax.text(0.00, 0.05, 'channel dimension $C=768$', fontsize=6.6, color='#5a5a5a')

    fig.savefig(os.path.join(OUT, 'pipeline.png'), dpi=220, facecolor='white')
    fig.savefig(os.path.join(OUT, 'pipeline.pdf'), facecolor='white')
    print('saved', os.path.join(OUT, 'pipeline.png'))


if __name__ == '__main__':
    main()
