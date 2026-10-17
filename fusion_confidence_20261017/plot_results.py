"""Standalone research figure using matplotlib; run after report generation."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['MPLCONFIGDIR']=str(__import__('pathlib').Path(__file__).resolve().parent/'results/matplotlib-config')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path(__file__).resolve().parent/'results'
    read=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
    rows=read('feature_summary.json');records=read('pipelines.json');routes=read('routing_fits.json')
    get=lambda name:next(r for r in records if r['name']==name)
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    selected=[r for r in rows if r['variant'] not in ['s0','pool']]
    labels=[r['variant'].replace('s2_','+') for r in selected]
    axes[0].bar(np.arange(len(selected)),[r['mean_accuracy'] for r in selected],yerr=[r['sd_accuracy'] for r in selected],capsize=3,color='#3a78a4')
    axes[0].set_xticks(np.arange(len(selected)),labels,rotation=35,ha='right')
    axes[0].set_ylim(80,96);axes[0].set_title('Features: five-seed mean and SD')
    axes[0].set_ylabel('Validation selection accuracy (%)')
    names=['s2_seed17','s2_5','heterogeneous_15','consistency_5']
    axes[1].bar(np.arange(4),[get(n)['accuracy'] for n in names],color=['#8f9ba3','#3a78a4','#658c54','#ae884c'])
    axes[1].set_xticks(np.arange(4),['S2 seed17','S2 x5','S2/S0/pool x15','Consistency x5'],rotation=35,ha='right')
    axes[1].set_ylim(88,96);axes[1].set_title('Frozen ensembles: official accuracy')
    names2=[];gains=[]
    for base in names:
        these=[get(r['name']) for r in routes if r['base']==base]
        names2.extend([base]*len(these));gains.extend([r['accuracy']-get(base)['accuracy'] for r in these])
    for i,base in enumerate(names):
        start=i*6;ys=gains[start:start+6]
        axes[2].scatter(i+np.linspace(-.18,.18,6),ys,s=35,color='#3a78a4')
    axes[2].axhline(0,color='black',linewidth=.8)
    axes[2].set_xticks(np.arange(4),['S2 seed17','S2 x5','Heterogeneous','Consistency x5'],rotation=35,ha='right')
    axes[2].set_title('OOF-selected fallback: all six rules/base')
    axes[2].set_ylabel('Change in accuracy (percentage points)')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Validation development results; no test evaluation',fontsize=14)
    fig.savefig(out/'experiment_summary.png',dpi=180)
    fig.savefig(out/'experiment_summary.pdf')
    print('FIGURE WRITTEN')


if __name__=='__main__':main()
