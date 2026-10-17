"""Static scientific figures; no training, inference or selection."""
from pathlib import Path
import os
import json
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'results'
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'matplotlib-config'))
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def main():
    summary = read('summary.json')
    contrasts = read('paired_comparisons.json')
    groups = read('subgroups.json')
    fig, ax = plt.subplots(1, 3, figsize=(13.8, 4.3), constrained_layout=True)
    arches = ['I', 'Pool', 'S2']
    colors = ['#2878A0', '#D48132']
    for i, feature in enumerate(['P', 'C']):
        rows = [next(r for r in summary if r['variant'] == f'{a}_{feature}' and r['mode'] == 'filtered') for a in arches]
        ax[0].errorbar(np.arange(3), [r['mean_accuracy'] for r in rows], yerr=[r['sample_sd'] for r in rows],
                       fmt='o-', capsize=4, color=colors[i], label=f'{feature}: ' + ('parse only' if feature == 'P' else 'parse + consensus'))
        vals = [next(r['accuracy'] for r in groups if r['variant'] == f'{a}_{feature}' and r['mode'] == 'filtered'
                     and r['kind'] == 'single_mean' and r['subgroup'] == 'wrong_unique_plurality') for a in arches]
        ax[1].plot(np.arange(3), vals, 'o-', color=colors[i], label=feature)
    ax[0].set(title='Validation accuracy (mean +/- seed SD)', ylabel='Selection accuracy (%)', xticks=range(3), xticklabels=arches)
    ax[0].legend(fontsize=8)
    ax[1].set(title='Wrong consensus (60 questions)', ylabel='Diagnostic accuracy (%)', xticks=range(3), xticklabels=arches)
    selected = [next(r for r in contrasts if r['contrast'] == name and r['mode'] == 'filtered' and r['kind'] == 'single_mean' and r['subgroup'] == 'all')
                for name in ['interaction (S2-I): C-P', 'interaction (Pool-I): C-P', 'interaction (S2-Pool): C-P']]
    for i, row in enumerate(selected):
        point = row['difference_pp']; low, high = row['ci95_pp']
        ax[2].errorbar(point, i, xerr=[[point-low], [high-point]], fmt='o', color='#336B49', capsize=4)
    ax[2].axvline(0, color='#777777', linestyle='--', linewidth=1)
    ax[2].set(title='Interaction contrasts (descriptive 95% CI)', xlabel='Change in structure advantage (pp)',
              yticks=range(3), yticklabels=['S2 - I', 'Pool - I', 'S2 - Pool'])
    for a in ax:
        a.grid(alpha=.2)
        a.spines[['top', 'right']].set_visible(False)
    fig.savefig(OUT / 'factorial_results.png', dpi=180)
    fig.savefig(OUT / 'factorial_results.pdf')
    plt.close(fig)
    import csv
    with (OUT / 'learning_curves.csv').open(encoding='utf-8-sig') as f:
        history = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.7), constrained_layout=True)
    for i, arch in enumerate(arches):
        for color, feature in zip(colors, ['P', 'C']):
            variant = f'{arch}_{feature}'
            rows = [r for r in history if r['variant'] == variant]
            epochs = list(range(21))
            for split, ls in [('train', '--'), ('validation', '-')]:
                loss = [np.mean([float(r[f'{split}_bce']) for r in rows if int(r['epoch']) == e]) for e in epochs]
                axes[0, i].plot(epochs, loss, ls, color=color, label=f'{feature} {split}')
            acc = [np.mean([float(r['validation_accuracy_filtered']) for r in rows if int(r['epoch']) == e]) for e in epochs]
            axes[1, i].plot(epochs, acc, color=color, label=feature)
        axes[0, i].set(title=arch, ylabel='Candidate BCE')
        axes[1, i].set(xlabel='Epoch (fixed final = 20)', ylabel='Filtered validation accuracy (%)')
        axes[0, i].legend(fontsize=8)
        for a in axes[:, i]:
            a.grid(alpha=.2)
            a.spines[['top', 'right']].set_visible(False)
    fig.savefig(OUT / 'learning_curves.png', dpi=180)
    fig.savefig(OUT / 'learning_curves.pdf')
    plt.close(fig)
    print('PLOTS SAVED', flush=True)


if __name__ == '__main__':
    main()
