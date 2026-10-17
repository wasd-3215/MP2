"""Static factorial and learning-curve figures, independent of training code."""
from pathlib import Path
import os,json,csv
HERE=Path(__file__).resolve().parents[1];OUT=HERE/'results'
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib-config'));os.environ['OPENBLAS_NUM_THREADS']='1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def main():
    summary=read('summary.json');contrasts=read('paired_comparisons.json');groups=read('subgroups.json')
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
    colors=['#2878A0','#D48132']
    for col,scope in enumerate(['All','D']):
        for color,feature in zip(colors,['P','C']):
            rows=[next(r for r in summary if r['variant']==f'{scope}_{feature}_{pos}') for pos in ['Late','Early']]
            axes[col].errorbar([0,1],[r['D_mean_accuracy'] for r in rows],yerr=[r['D_sample_sd'] for r in rows],fmt='o-',capsize=4,color=color,label=feature)
        axes[col].set(title=f'{scope} training: disagreement accuracy',xticks=[0,1],xticklabels=['Late','Early'],ylabel='Accuracy on 355 questions (%)')
        axes[col].legend()
    names=['scope_main: D-All','early_at_C_main','early_feature_interaction','early_scope_interaction']
    for i,name in enumerate(names):
        r=next(r for r in contrasts if r['contrast']==name and r['kind']=='single_mean' and r['subgroup']=='all')
        v=r['difference_pp'];low,high=r['ci95_pp'];axes[2].errorbar(v,i,xerr=[[v-low],[high-v]],fmt='o',capsize=4,color='#336B49')
    axes[2].axvline(0,color='#777',ls='--',lw=1)
    axes[2].set(title='Main contrasts: descriptive 95% CI',xlabel='Complete-bank difference (pp)',
        yticks=range(4),yticklabels=['D - All','Early - Late at C','Early x feature','Early x scope'])
    for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'factorial_results.png',dpi=180);fig.savefig(OUT/'factorial_results.pdf');plt.close(fig)
    with (OUT/'learning_curves.csv').open(encoding='utf-8-sig') as f:history=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,4,figsize=(15,6.5),constrained_layout=True)
    for col,(scope,feature) in enumerate([(s,f) for s in ['All','D'] for f in ['P','C']]):
        for color,position in zip(colors,['Late','Early']):
            rows=[r for r in history if r['variant']==f'{scope}_{feature}_{position}'];epochs=list(range(21))
            for split,ls in [('train','--'),('validation','-')]:
                vals=[np.mean([float(r[f'{split}_bce_disagreement']) for r in rows if int(r['budget_epoch'])==e]) for e in epochs]
                axes[0,col].plot(epochs,vals,ls,color=color,label=f'{position} {split}')
            vals=[np.mean([float(r['validation_accuracy_disagreement']) for r in rows if int(r['budget_epoch'])==e]) for e in epochs]
            axes[1,col].plot(epochs,vals,color=color,label=position)
        axes[0,col].set(title=f'{scope} / {feature}',ylabel='Disagreement candidate BCE');axes[0,col].legend(fontsize=7)
        axes[1,col].set(xlabel='Budget epoch (19 updates each)',ylabel='Disagreement accuracy (%)')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'learning_curves.png',dpi=180);fig.savefig(OUT/'learning_curves.pdf');plt.close(fig)
    print('PLOTS SAVED',flush=True)
if __name__=='__main__':main()
