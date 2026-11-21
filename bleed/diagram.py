"""Block diagrams of the bleed-through pipeline in ECE 253 course conventions.

Figure 1 follows the Gonzalez & Woods degradation/restoration model used in
SlideDeck 7 (p.2) and SlideDeck 10 (p.6-7): plain light-blue rectangles, a "+"
summing node with noise eta(x,y) entering from below, wide hollow arrows flowing
left to right, italic serif signal names, DEGRADATION / RESTORATION brackets. The
restoration stages are the proposal's Fig. 4 stages inside a dashed grouping box
(as in SlideDeck 2 p.46, 9 p.5), with the modified block in yellow.
Figure 2 draws the bleed/front decomposition as an iterative loop in the style of
the Richardson-Lucy diagram (SlideDeck 8 p.3).

Run: uv run --with matplotlib bleed/diagram.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Circle, FancyArrow, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent/'results'/'diagrams'
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'STIXGeneral', 'font.size': 11})  # math in Computer Modern (LaTeX's font)
BLUE, YELLOW, NAVY, GREY = '#7fa7d9', '#fff2a8', '#1f3b73', '#eeeeee'
SHADOW = [patheffects.withSimplePatchShadow(offset=(2, -2), alpha=.25)]


def block(ax, x, y, w, h, title, body='', fill='white', edge=BLUE, num=None):
    ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=edge, lw=1.6, path_effects=SHADOW, zorder=2))
    head = f'{num}  {title}' if num else title
    two = '\n' in title
    ty = (y + h - (.4 if two else .22)) if body else y + h/2
    ax.text(x + w/2, ty, head, ha='center', va='center', fontsize=11.5, weight='bold',
            color=NAVY if num else 'black', zorder=3, linespacing=1.15)
    if body:
        top = .78 if two else .4
        ax.text(x + w/2, y + (h - top)/2, body, ha='center', va='center', fontsize=9.6, zorder=3, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, width=.09):
    dx, dy = x1 - x0, y1 - y0
    ax.add_patch(FancyArrow(x0, y0, dx, dy, width=width, head_width=width*2.4, head_length=.16,
                            length_includes_head=True, fc='white', ec='black', lw=1.1, zorder=4))


def thin(ax, x0, y0, x1, y1, style='-|>', color='black', ls='-', rad=0.):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=4,
                arrowprops=dict(arrowstyle=style, color=color, lw=1.3, ls=ls,
                                connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))


def summing(ax, x, y, r=.2):
    ax.add_patch(Circle((x, y), r, fc='white', ec='black', lw=1.3, zorder=4))
    ax.text(x, y - .01, '+', ha='center', va='center', fontsize=15, zorder=5)


def figure1():
    fig, ax = plt.subplots(figsize=(16.8, 5.6))
    ax.set_xlim(0, 16.8); ax.set_ylim(-1.35, 5.35); ax.axis('off')

    # --- degradation -------------------------------------------------------
    ax.text(.1, 2.25, '$f(x,y)$', fontsize=14, ha='left', va='center')
    ax.text(.1, 1.85, 'Front writing\n(pencil, ballpoint, gel pen)', fontsize=9, ha='left', va='top', color='#444')
    ax.text(.1, 4.25, '$b(x,y)$', fontsize=14, ha='left', va='center')
    ax.text(.1, 3.9, 'Reverse writing', fontsize=9, ha='left', va='top', color='#444')
    block(ax, 1.25, 3.7, 2.2, 1.1, r'Bleed-through $\mathcal{H}_b$', 'mirror, diffuse through paper,\nattenuate: $a(x,y)\\,m_b(x,y)$')
    arrow(ax, .95, 4.25, 1.25, 4.25)
    summing(ax, 4.1, 2.25)
    arrow(ax, .95, 2.25, 3.9, 2.25)
    arrow(ax, 3.45, 4.25, 4.1, 4.25); arrow(ax, 4.1, 4.3, 4.1, 2.45)        # down into the node
    arrow(ax, 4.1, 1.15, 4.1, 2.05)                                        # noise from below
    ax.text(4.1, .95, r'Noise $\eta(x,y)$', ha='center', va='top', fontsize=12)
    ax.text(4.1, .6, 'camera, JPEG, illumination', ha='center', va='top', fontsize=8.5, color='#444')
    arrow(ax, 4.3, 2.25, 5.35, 2.25)
    ax.text(4.83, 2.55, '$g(x,y)$', ha='center', fontsize=13)
    ax.text(4.83, 1.85, 'front photo', ha='center', fontsize=8.5, color='#444')
    # back photo branch: b(x,y) is also photographed directly -> g_b(x,y) -> block 1
    ax.plot([.75, .75, 6.675], [4.45, 5.1, 5.1], color='#444', lw=1.3, zorder=3)
    thin(ax, 6.675, 5.1, 6.675, 3.1, color='#444')
    ax.text(3.0, 5.2, r'back photo $g_b(x,y)$ (photographed directly)', ha='center', fontsize=10.5, color='#444')

    # --- restoration (dashed group) ------------------------------------------
    gx, gy, gw, gh = 5.4, .45, 9.95, 4.25
    ax.add_patch(FancyBboxPatch((gx, gy), gw, gh, boxstyle='round,pad=0,rounding_size=.08', fc='none',
                                ec=NAVY, lw=1.4, ls=(0, (5, 4)), zorder=1))
    ax.text(gx + gw - .12, gy + gh - .2, 'Restoration filter (Rowley-Brooke, modified)', fontsize=11, color=NAVY,
            style='italic', va='center', ha='right')
    W, H, y = 2.3, 1.95, 1.25
    xs = [5.6, 8.1, 10.6, 13.1]
    block(ax, xs[0], y, W, H, 'Mirror back & align', 'SIFT/RANSAC homography\n+ dense NCC shift field\nregistration check', num=1)
    block(ax, xs[1], y, W, H, 'Identify interference',
          'joint-histogram 4-class MRF\n+ connected components;\n$d_g = a\\,m_b + p,\\ p\\geq 0$', num=2)
    block(ax, xs[2], y, W, H, 'Protect foreground\nstrokes', 'keep front ink $p>0$,\ndivide bleed out:\n$\\hat f = g\\,e^{\\,k_c a m_b}$',
          fill=YELLOW, num=3)
    block(ax, xs[3], y, W, H, 'Restore paper\nbackground', 'nearby paper texture\n+ grid continuation;\nuntouched if $a\\,m_b\\leq 0.02$', num=4)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + W, y + H/2, b, y + H/2)
    arrow(ax, xs[3] + W, y + H/2, 15.75, y + H/2)
    ax.text(15.82, y + H/2, r'$\hat f(x,y)$', fontsize=14, va='center')
    ax.text(15.82, y + H/2 - .38, 'Estimate', fontsize=9, color='#444', va='center')

    # --- brackets ------------------------------------------------------------
    for x0, x1, label in [(.1, 5.3, 'DEGRADATION'), (5.4, 16.6, 'RESTORATION')]:
        ax.plot([x0, x0, x1, x1], [-.2, -.35, -.35, -.2], color='black', lw=1)
        ax.text((x0 + x1)/2, -.6, label, ha='center', fontsize=12, style='italic', fontvariant='small-caps')
    ax.text(8.4, -1.05, r'Model (optical density, $d = \log(\mathrm{paper}/I)$):  '
            r'$d_g(x,y) = d_f(x,y) + a(x,y)\,m_b(x,y) + \eta(x,y)$     '
            r'$\Leftrightarrow\ g = f\,t + \eta,\ \ t = e^{-a m_b}$  (Deck 10 haze model, $A=0$)',
            fontsize=11, ha='center', va='center', color='black')
    return fig


def figure_simple():
    """Slide version: inputs, four restoration blocks with one equation each, output."""
    fig, ax = plt.subplots(figsize=(15, 4.2))
    ax.set_xlim(0, 15); ax.set_ylim(-.2, 4.1); ax.axis('off')
    W, H, y = 2.45, 1.6, 1.25
    xs = [2.0, 4.85, 7.7, 10.55]
    ax.add_patch(FancyBboxPatch((1.8, .75), 11.45, 3.05, boxstyle='round,pad=0,rounding_size=.08', fc='none',
                                ec=NAVY, lw=1.4, ls=(0, (5, 4)), zorder=1))
    ax.text(13.1, 3.55, 'Restoration (Rowley-Brooke, modified)', fontsize=12, color=NAVY, style='italic',
            ha='right', va='center')
    ax.text(.05, y + H*.72, '$g(x,y)$', fontsize=16, va='center')
    ax.text(.05, y + H*.72 - .38, 'front photo', fontsize=10, color='#444', va='center')
    ax.text(.05, y + H*.22, '$g_b(x,y)$', fontsize=16, va='center')
    ax.text(.05, y + H*.22 - .38, 'back photo', fontsize=10, color='#444', va='center')
    arrow(ax, 1.15, y + H*.72, xs[0], y + H*.72); arrow(ax, 1.15, y + H*.22, xs[0], y + H*.22)
    steps = [('Mirror back\n& align', r'$\tilde{g}_b(\mathbf{x}) = g_b(\mathbf{H}^{-1}\mathbf{x})$', 'white'),
             ('Identify\ninterference', r'$d_g = a\,m_b + p$', 'white'),
             ('Protect foreground\nstrokes', r'$\hat f = g\,e^{\,a\,m_b}$', YELLOW),
             ('Restore paper\nbackground', r'$\hat f = \mathrm{paper}\cdot\mathrm{texture}$', 'white')]
    for i, (x, (t, eq, fill)) in enumerate(zip(xs, steps), 1):
        ax.add_patch(Rectangle((x, y), W, H, fc=fill, ec=BLUE, lw=1.6, path_effects=SHADOW, zorder=2))
        ax.text(x + W/2, y + H - .45, f'{i}  {t}', ha='center', va='center', fontsize=13, weight='bold',
                color=NAVY, linespacing=1.1, zorder=3)
        ax.text(x + W/2, y + .38, eq, ha='center', va='center', fontsize=13.5, zorder=3)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + W, y + H/2, b, y + H/2)
    arrow(ax, xs[3] + W, y + H/2, 13.75, y + H/2)
    ax.text(13.85, y + H/2 + .05, r'$\hat f(x,y)$', fontsize=17, va='center')
    ax.text(13.85, y + H/2 - .4, 'restored', fontsize=10, color='#444', va='center')
    ax.text(7.5, .2, r'Bleed model:  $g = f\,t + \eta,\quad t(x,y) = e^{-a(x,y)\,m_b(x,y)}$', ha='center',
            va='center', fontsize=13)
    return fig


def figure2():
    fig, ax = plt.subplots(figsize=(14.5, 4.6))
    ax.set_xlim(0, 14.5); ax.set_ylim(-.15, 4.6); ax.axis('off')
    W, H, y = 2.75, 1.45, 2.2
    xs = [.4, 3.55, 6.7, 9.85]
    steps = [('Predict', r'$b_k = a_k(x,y)\,m_b(x,y)$'),
             ('Compare', r'$r_k = d_g - b_k$'),
             ('Shrink', r'$p_{k+1} = \max(r_k - \lambda(x,y)/2,\ 0)$'),
             ('Refit', r'$a_{k+1}$: Gaussian-window' '\n' r'least squares on $d_g - p_{k+1}$')]
    for i, (x, (t, b)) in enumerate(zip(xs, steps), 1):
        block(ax, x, y, W, H, t, b, fill=GREY, edge='#9a9a9a', num=i)
    for a, b in zip(xs[:-1], xs[1:]):
        thin(ax, a + W, y + H/2, b, y + H/2)
    block(ax, 3.8, 4.0, 2.25, .5, r'Observed $d_g,\ m_b$ (fixed)', fill='white', edge='#9a9a9a')
    thin(ax, 4.92, 4.0, 4.92, y + H)
    thin(ax, xs[3] + W/2, y, xs[0] + W/2, y, color='#1f5fbf', rad=-.32)
    ax.text(7.0, .42, r'Use $a_{k+1},\ p_{k+1}$ as the next $a_k,\ p_k$ (6 iterations)', ha='center', fontsize=11,
            color='#1f5fbf')
    thin(ax, xs[3] + W, y + H/2, 13.2, y + H/2)
    ax.text(13.3, y + H/2, r'$p(x,y)$,' '\n' r'$a\,m_b(x,y)$', ha='left', va='center', fontsize=13)
    ax.text(.4, -.05, r'$\lambda(x,y) = \lambda_0 + \lambda_1 m_b(x,y)$ grows with reverse-ink density;  '
            r'then: drop isolated residue, fill gaps by evidence-guided minimal paths (Dijkstra).',
            fontsize=10, color='#333')
    return fig


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    # The slide pipeline diagram is typeset in LaTeX/TikZ: bleed/diagrams/pipeline.tex
    # (tectonic bleed/diagrams/pipeline.tex). figure_simple is kept only as a fallback.
    for name, make in [('pipeline-block-diagram-detailed', figure1), ('decomposition-loop', figure2)]:
        fig = make()
        fig.savefig(OUT/f'{name}.svg', bbox_inches='tight')
        fig.savefig(OUT/f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print('wrote', OUT/f'{name}.svg')
