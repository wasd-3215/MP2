"""Prevalence audit only; probe never replaces official parser or makes predictions."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
from fractions import Fraction
import json
import re
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))
from common import check_data, read_jsonl, DATA, parse_answer

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def probe(text):
    boxes = list(re.finditer(r'\\boxed\{([^{}]+)\}', text))
    result = {'numeric_box_parse_conflict': False, 'symbolic_box_numeric_prefix': False, 'box_value': None, 'last_box_literal': None}
    if not boxes:
        return result
    box = boxes[-1]
    token = box.group(1).strip()
    result['last_box_literal'] = token
    parsed = parse_answer(text)
    if re.fullmatch(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?', token):
        try:
            value = str(Fraction(token.replace(',', '')))
        except (ValueError, ZeroDivisionError):
            return result
        result['box_value'] = value
        # Restrict to last box occurring after the last ####, if present.
        result['numeric_box_parse_conflict'] = value != parsed and box.start() > text.rfind('####')
    else:
        # A literal algebraic final box starts with a number, e.g.32-d; no nested-box or LaTeX fraction claim.
        result['symbolic_box_numeric_prefix'] = bool(parsed is not None and re.match(r'\s*[-+]?\d', token)
              and re.search(r'[A-Za-z]', token) and box.start() > text.rfind('####'))
    return result

def main():
    result = {'purpose': 'Training recurrence and coverage diagnosis, no method selection or score changes',
      'probe_limit': 'Only last non-nested boxed expression; full numeric literal or number-prefixed algebraic string. It is not a general mathematical parser.',
      'test_used': False, 'splits': {}}
    common_ids = {c['id'] for c in [json.loads(s) for s in (HERE / 'evidence_cards.jsonl').read_text(encoding='utf-8').splitlines()]}
    for split in ['train', 'validation']:
        check_data(DATA, [split])
        bank = read_jsonl(DATA / f'candidates/gsm8k_{split}.jsonl')
        refs = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
        assert [r['id'] for r in bank] == [r['id'] for r in refs]
        gold = [parse_answer(r['answer']) for r in refs]
        flags = {key: np.zeros((len(bank), 4), bool) for key in ['numeric_box_parse_conflict', 'symbolic_box_numeric_prefix']}
        details = []
        for q, row in enumerate(bank):
            for i, candidate in enumerate(row['candidates']):
                info = probe(candidate['text'])
                for key in flags:
                    flags[key][q, i] = info[key]
                if info['numeric_box_parse_conflict'] or info['symbolic_box_numeric_prefix']:
                    details.append({'row': q, 'candidate': i, 'id': row['id'], 'gold': gold[q],
                       'official_parse': parse_answer(candidate['text']), **info,
                       'in_pair_common_failures': row['id'] in common_ids})
        item = {'questions': len(bank), 'candidates': 4*len(bank),
             'flags': {key: {'candidates': int(v.sum()), 'questions': int(v.any(1).sum())} for key, v in flags.items()},
             'details': details}
        for key in flags:
            subset = [d for d in details if d[key]]
            item['flags'][key]['officially_correct_candidates'] = sum(d['official_parse'] == d['gold'] for d in subset)
            item['flags'][key]['box_matches_reference_but_official_parse_wrong'] = sum(d['box_value'] == d['gold'] and d['official_parse'] != d['gold'] for d in subset)
        if split == 'train':
            # Only base-model OOF decisions are audited; no new selector or penalty is fitted here.
            files = {'Pool_P': CODE / 'method_complementarity_20261017/results/oof-Pool_P.npy',
                     'S2_C': CODE / 'fusion_confidence_20261017/results/oof-s2_consistency.npy'}
            selected = {}
            valid = np.asarray([[parse_answer(c['text']) is not None for c in r['candidates']] for r in bank])
            for expert, path in files.items():
                z = np.load(path)
                p = 1/(1+np.exp(-np.clip(z.astype(np.float64), -40, 40)))
                scores = p.mean(0)
                choice = scores.argmax(1)
                change = valid.any(1) & ~valid[np.arange(len(bank)), choice]
                choice[change] = np.where(valid[change], scores[change], -np.inf).argmax(1)
                selected[expert] = {key: int(v[np.arange(len(bank)), choice].sum()) for key, v in flags.items()}
                selected[expert]['flagged_details'] = [d for d in details if int(choice[d['row']]) == d['candidate']]
            item['selected_flagged_candidates_by_train_OOF'] = selected
        result['splits'][split] = item
    (HERE / 'format_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({s: {k:v for k,v in item.items() if k!='details'} for s,item in result['splits'].items()}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
