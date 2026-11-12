"""Plot paired fitted-model loss against observed full charged wall time."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent


def plot(data):
    figure, axes = plt.subplots(1, 3, figsize=(11.6, 3.9), layout='constrained')
    for ax, root in zip(axes, data['roots'], strict=True):
        day, _, decision = root['case']
        rows = root['summary']
        full = sorted((int(a.rsplit('_', 1)[1]), v) for a, v in rows.items() if a.startswith('rollout_'))
        matched = sorted((int(a.rsplit('_', 1)[1]), v) for a, v in rows.items()
                         if a.startswith('direct_rollout_'))
        for points, color, symbol, label in (
            (full, '#18598a', 'o', 'Broad-menu rollout'),
            (matched, '#8251a5', '^', 'Direct-menu rollout'),
        ):
            x = [v['mean_wall'] for _, v in points]
            y = [v['loss'] for _, v in points]
            ax.plot(x, y, color=color, marker=symbol, linewidth=1.25, markersize=4, label=label)
            for j, ((budget, _), xx, yy) in enumerate(zip(points, x, y, strict=True)):
                if j not in (0, 1, len(points)-1):
                    continue
                ax.annotate(str(budget), (xx, yy), xytext=((0, 8) if j == 0 else
                            (-7, -12) if j == 1 else (8, 8)), textcoords='offset points',
                            fontsize=7, color=color, ha='center')
        r = rows['direct_zero']
        ax.scatter(r['mean_wall'], r['loss'], color='#13774d', marker='D', s=32,
                   label='V20 direct', zorder=4)
        base = rows['first_visible']
        ax.text(.02, .97, f"First visible: {base['loss']:.1f} s loss  /  {base['mean_wall']:.3f} s wall",
                fontsize=7, color='#8b4a30', transform=ax.transAxes, va='top')
        ax.set(title=f'{day[5:]}  |  decision {decision}', xlabel='Full charged wall (s)')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.margins(x=.12, y=.21)
        ax.grid(axis='y', color='#dfe4e9', linewidth=.75)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Mean truncated travel loss (s)')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncols=4, frameon=False, fontsize=8)
    figure.savefig(HERE/'budget_curve.png', dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    data = json.loads((HERE/'paired_results.json').read_text())
    assert data['status'] == 'complete'
    plot(data)
