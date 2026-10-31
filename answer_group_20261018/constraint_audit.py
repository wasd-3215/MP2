"""Train-only feasibility gate, not a solver and not a validation rule search."""
from support import OUT,write
from common import load_inputs
import re

def main():
    bank,_=load_inputs('train')
    screens={'explicit_xyz_equation':lambda q:'=' in q and bool(re.search(r'\b[xyz]\b',q,re.I)),
             'interest_original_or_principal':lambda q:'interest' in q.lower() and any(w in q.lower() for w in ('original','principal'))}
    results={name:[{'row':i,'id':r['id'],'question':r['question']} for i,r in enumerate(bank) if screen(r['question'])]
             for name,screen in screens.items()}
    write(OUT/'constraint_feasibility.json',{'split':'train','n':len(bank),'minimum_supported_questions':20,
          'screens':results,'counts':{k:len(v) for k,v in results.items()},'gate_passed':False,
          'manual_review':'The sole interest question asks original principal given one-year 12% interest and interest amount1500; principal=1500/0.12=12500. Wording is clear, but one question cannot support a general extraction family.',
          'reference_labels_used':False,'hard_constraint_extension_added':False,
          'interpretation':'Insufficient coverage for these two narrow screens. This does not establish that other question-side constraint families are unavailable.'})
    assert all(len(v)<20 for v in results.values())
    print({k:len(v) for k,v in results.items()},flush=True)

if __name__=='__main__':main()
