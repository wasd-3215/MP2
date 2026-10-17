"""Static scientific figure, five-seed means and sample standard deviations."""
import os
from pathlib import Path
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'results/matplotlib-config')
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=HERE/'results';rows=json.loads((out/'group_summary.json').read_text())
    subs=json.loads((out/'subgroup_summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    schemes=['U','K','H'];x=np.arange(3);colors={'P':'#3a78a4','C':'#ae884c'}
    for model,shift in [('P',-.18),('C',.18)]:
        selected=[next(r for r in rows if r['group']==f'{model}_{s}') for s in schemes]
        axes[0].bar(x+shift,[r['mean_accuracy'] for r in selected],.34,yerr=[r['sd_accuracy'] for r in selected],color=colors[model],capsize=3,label=model)
        h=[next(r for r in subs if r['group']==f'{model}_{s}' and r['split']=='validation' and r['subgroup']=='wrong_unique_plurality') for s in schemes]
        axes[1].bar(x+shift,[r['mean_accuracy_diagnostic'] for r in h],.34,yerr=[r['sd_accuracy'] for r in h],color=colors[model],capsize=3,label=model)
    lo=min(r['mean_accuracy']-r['sd_accuracy'] for r in rows);hi=max(r['mean_accuracy']+r['sd_accuracy'] for r in rows)
    axes[0].set_ylim(lo-1,hi+1);axes[1].set_ylim(0,100)
    for ax in axes[:2]:
        ax.set_xticks(x,['Uniform','Count-matched','Hard-targeted']);ax.tick_params(axis='x',rotation=20)
        ax.set_ylabel('Selection accuracy (%)');ax.legend(title='P: parse; C: consistency')
    axes[0].set_title('All 798 validation questions')
    axes[1].set_title('Wrong unique plurality: 60 questions')
    with (out/'learning_curves.csv').open(encoding='utf-8-sig',newline='') as f:curves=list(csv.DictReader(f))
    for group,color in [('C_U','#3a78a4'),('C_H','#ae884c')]:
        for field,style,label in [('clean_unweighted_train_bce','--','train'),('clean_unweighted_validation_bce','-','validation')]:
            values=[np.mean([float(r[field]) for r in curves if r['group']==group and int(r['epoch'])==epoch]) for epoch in range(1,21)]
            axes[2].plot(range(1,21),values,style,color=color,label=f'{group} {label}')
    axes[2].set_title('Clean unweighted BCE, consistency model')
    axes[2].set_xlabel('Epoch');axes[2].set_ylabel('BCE');axes[2].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Training-weight experiment: validation development results; error bars = seed SD')
    fig.savefig(out/'experiment_summary.png',dpi=180);fig.savefig(out/'experiment_summary.pdf')
    print('FIGURE WRITTEN')


if __name__=='__main__':main()
