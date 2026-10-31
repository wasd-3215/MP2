"""Read-only validation diagnosis from saved predictions, no training/inference/test."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))
from common import check_data, read_jsonl, parse_answer, DATA

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    check_data(DATA, ['validation'])
    bank = read_jsonl(DATA / 'candidates/gsm8k_validation.jsonl')
    refs = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    assert [r['id'] for r in bank] == [r['id'] for r in refs]
    answers = [[parse_answer(c['text']) for c in r['candidates']] for r in bank]
    gold = [parse_answer(r['answer']) for r in refs]
    y = np.asarray([[a is not None and a == g for a in row] for row, g in zip(answers, gold)], bool)
    valid = np.asarray([[a is not None for a in row] for row in answers], bool)
    paths = {}
    interaction = CODE / 'interaction_consensus_20261017/results/pipelines'
    for variant in ['I_P', 'I_C', 'Pool_P', 'Pool_C', 'S2_P', 'S2_C']:
        paths[variant] = interaction / f'{variant}_ensemble5_filtered/predictions.json'
        for seed in [17, 29, 43, 71, 101]:
            paths[f'{variant}_seed{seed}'] = interaction / f'{variant}_seed{seed}_filtered/predictions.json'
    paths['S2_route'] = CODE / 'validity_filter_20261017/results/pipelines/s2_ensemble5_route_filtered/predictions.json'
    choices, oks = {}, {}
    for name, path in paths.items():
        pred = read(path)
        assert pred['ids'] == [r['id'] for r in bank]
        choices[name] = np.asarray(pred['selected'])
        oks[name] = y[np.arange(len(bank)), choices[name]]
    common = ~oks['Pool_P'] & ~oks['S2_C']
    indices = np.flatnonzero(common)
    assert len(indices) == 42
    probability_path = CODE / 'method_complementarity_20261017/results/validation-probabilities.npz'
    with np.load(probability_path) as a:
        probabilities = {e: a[e].copy() for e in ['Pool_P', 'S2_C']}
    rows, cards = [], []
    ensemble_names = ['I_P', 'I_C', 'Pool_P', 'Pool_C', 'S2_P', 'S2_C']
    for number, q in enumerate(indices, 1):
        a = answers[q]
        counts = Counter(v for v in a if v is not None)
        top = max(counts.values())
        leaders = [v for v, c in counts.items() if c == top]
        ensemble_rescues = [n for n in ensemble_names if oks[n][q]]
        seed_rescues = [n for n in paths if '_seed' in n and oks[n][q]]
        row = {'case': number, 'row': int(q), 'source_line': int(q+1), 'id': bank[q]['id'],
               'answers': '|'.join(str(v) for v in a), 'gold': gold[q], 'gold_support': counts[gold[q]],
               'correct_candidates': int(y[q].sum()), 'valid_candidates': int(valid[q].sum()),
               'answer_groups': len(counts), 'group_sizes': ','.join(map(str, sorted(counts.values(), reverse=True))),
               'all_valid_groups_singleton': all(c == 1 for c in counts.values()),
               'wrong_unique_plurality': len(leaders) == 1 and leaders[0] != gold[q],
               'plurality_tied': len(leaders) > 1,
               'representatives_same_wrong_answer': a[int(choices['Pool_P'][q])] == a[int(choices['S2_C'][q])],
               'route_correct': bool(oks['S2_route'][q]), 'other_ensemble_rescues': '|'.join(ensemble_rescues),
               'seed_rescues_n': len(seed_rescues), 'any_seed_rescue': bool(seed_rescues)}
        confidences = {}
        for e, p in probabilities.items():
            mean = p[:, q].mean(0)
            choice = int(choices[e][q])
            masked = np.where(valid[q], mean, -np.inf)
            order = np.argsort(-masked, kind='stable')
            rank = min(int(np.flatnonzero(order == c)[0])+1 for c in np.flatnonzero(y[q]))
            seeds = np.where(valid[q], p[:, q], -np.inf).argmax(1)
            agreement = np.mean([a[int(i)] == a[choice] for i in seeds])
            best_correct = max(mean[y[q]])
            confidences[e] = {'selected': choice, 'mean_probabilities': mean.tolist(),
                'correct_candidate_best_rank': rank, 'wrong_minus_best_correct_probability': float(mean[choice] - best_correct),
                'seed_agreement_selected_answer': float(agreement), 'correct_seed_count': int(y[q, seeds].sum())}
            row[e+'_correct_rank'] = rank
            row[e+'_seed_agreement'] = float(agreement)
            row[e+'_correct_seeds'] = int(y[q, seeds].sum())
            row[e+'_wrong_correct_gap'] = float(mean[choice] - best_correct)
        rows.append(row)
        evidence = []
        for i, c in enumerate(bank[q]['candidates']):
            evidence.append({'index': i, 'parsed_answer': a[i], 'officially_correct': bool(y[q, i]),
                             'text': c['text'], 'picked_by': [e for e in ['Pool_P', 'S2_C', 'S2_route'] if choices[e][q] == i]})
        cards.append({'case': number, 'row': int(q), 'id': bank[q]['id'], 'question': bank[q]['question'],
           'gold': gold[q], 'reference_work': refs[q]['answer'], 'parsed_answers': a,
           'method_choices': {e: int(choices[e][q]) for e in ensemble_names + ['S2_route']},
           'correct_indices': np.flatnonzero(y[q]).tolist(), 'candidates': evidence, 'statistics': row,
           'confidence': confidences, 'human_review_status': 'UNREVIEWED'})
    with (HERE / 'case_index.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with (HERE / 'evidence_cards.jsonl').open('w', encoding='utf-8') as f:
        for card in cards:
            f.write(json.dumps(card, ensure_ascii=False) + '\n')
    summary = {'common_failures': len(indices), 'one_correct_candidate': sum(r['correct_candidates']==1 for r in rows),
       'group_sizes': dict(Counter(r['group_sizes'] for r in rows)),
       'all_valid_groups_singleton': sum(r['all_valid_groups_singleton'] for r in rows),
       'wrong_unique_plurality': sum(r['wrong_unique_plurality'] for r in rows),
       'plurality_tied': sum(r['plurality_tied'] for r in rows),
       'some_invalid_candidate': sum(r['valid_candidates']<4 for r in rows),
       'representatives_same_wrong_answer': sum(r['representatives_same_wrong_answer'] for r in rows),
       'other_ensemble_rescues': {e: int(oks[e][indices].sum()) for e in ensemble_names + ['S2_route']},
       'any_of_six_ensemble_correct': sum(bool(r['other_ensemble_rescues']) for r in rows),
       'any_of_six_plus_route_correct': sum(bool(r['other_ensemble_rescues']) or r['route_correct'] for r in rows),
       'any_of_thirty_seeds_correct': sum(r['any_seed_rescue'] for r in rows),
       'confidence': {e: {'all_five_seeds_wrong': sum(r[e+'_correct_seeds']==0 for r in rows),
         'all_five_same_wrong_answer': sum(r[e+'_seed_agreement']==1 for r in rows),
         'correct_best_rank': dict(Counter(r[e+'_correct_rank'] for r in rows)),
         'wrong_correct_gap_median': float(np.median([r[e+'_wrong_correct_gap'] for r in rows]))} for e in probabilities},
       'full_four_candidate_characters': sum(len(c['question'])+sum(len(v['text']) for v in c['candidates']) for c in cards),
       'test_used': False, 'fresh_training_or_inference': False, 'causal_reviews_completed': 0}
    write(HERE / 'automatic_summary.json', summary)
    write(HERE / 'sources.json', {'predictions': {n: {'path': str(p), 'sha256': sha(p)} for n, p in paths.items()},
        'probabilities': {'path': str(probability_path), 'sha256': sha(probability_path)},
        'script_sha256': sha(Path(__file__)), 'candidate_data': sha(DATA / 'candidates/gsm8k_validation.jsonl'),
        'reference_data': sha(DATA / 'gsm8k_validation.jsonl'), 'parser': sha(CODE / 'common.py')})
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
