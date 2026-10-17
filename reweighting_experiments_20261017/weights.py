"""Training-only group labels and globally normalized question weights."""
import numpy as np
from explicit_features import answer_structure


def groups(bank,y):
    k=np.asarray(y).sum(1).astype(int)
    if len(bank)!=len(k) or not np.isin(k,[1,2,3,4]).all():
        raise ValueError('Aligned questions with at least one correct candidate required')
    answers,support,unique=answer_structure(bank)
    vote=support.argmax(1)
    voted_correct=np.asarray(y)[np.arange(len(k)),vote]>0
    mixed=k<4
    return {'all':np.ones(len(k),dtype=bool),'all_correct':k==4,'mixed':mixed,
      'wrong_unique_plurality':mixed&unique&~voted_correct,
      'correct_unique_plurality_mixed':mixed&unique&voted_correct,
      'plurality_tie':~unique,'one_correct':k==1,
      'two_correct':k==2,'three_correct':k==3,
      'multiple_valid_answer_groups':np.asarray([len({v for v in a if v is not None})>=2 for a in answers]),
      'any_invalid_parse':np.asarray([None in a for a in answers])}


def make_weights(bank,y,ratio=2.):
    g=groups(bank,y);k=np.asarray(y).sum(1).astype(int)
    n=len(k);mixed=g['mixed'];hard=g['wrong_unique_plurality'];rest=mixed&~hard
    if not hard.any() or not rest.any():raise ValueError('Both hard and other mixed training groups required')
    u=np.ones(n,dtype=np.float64)
    h=u.copy()
    normalizer=mixed.sum()/(ratio*hard.sum()+rest.sum())
    h[hard]=ratio*normalizer;h[rest]=normalizer
    matched=u.copy()
    for count in [1,2,3]:
        mask=k==count
        if mask.any():matched[mask]=h[mask].mean()
    weights={'U':u,'K':matched,'H':h}
    for name,w in weights.items():
        assert np.isfinite(w).all() and (w>0).all()
        assert np.allclose(w.mean(),1.,atol=1e-12)
        assert np.array_equal(w[g['all_correct']],np.ones(g['all_correct'].sum()))
        assert np.isclose(w[mixed].sum(),mixed.sum(),atol=1e-9)
    for count in range(1,5):
        mask=k==count
        assert np.isclose(h[mask].sum(),matched[mask].sum(),atol=1e-9)
    assert np.isclose((h[:,None]*y).sum(),(matched[:,None]*y).sum(),atol=1e-9)
    summary={'fit_split':'train only','ratio_hard_to_other_mixed':ratio,
      'normalizer':float(normalizer),'counts':{name:int(mask.sum()) for name,mask in g.items()},'schemes':{}}
    for name,w in weights.items():
        summary['schemes'][name]={'total_mass':float(w.sum()),'mixed_mass':float(w[mixed].sum()),
          'all_correct_mass':float(w[g['all_correct']].sum()),
          'positive_candidate_fraction':float((w[:,None]*y).sum()/(4*w.sum())),
          'per_correct_count_mass':{str(c):float(w[k==c].sum()) for c in range(1,5)},
          'hard_weight':float(w[hard][0]),'other_one_correct_weight':float(w[(k==1)&~hard][0]),
          'two_correct_weight':float(w[k==2][0]),'three_correct_weight':float(w[k==3][0])}
    return weights,g,summary
