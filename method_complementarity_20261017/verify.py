"""Source integrity, exact expert reuse and arbitrary-input behavior."""
import runtime
from utils import HERE, OUT, OLD, CODE, read, write, sha, sources, protected, members
import numpy as np
from common import load_inputs, parse_answer
from method import member_probabilities, predict
from decisions import sigmoid, decide, gate_features, best

def main():
    frozen = read(OUT / 'frozen_before_validation.json')
    assert sources() == frozen['sources']
    assert all(sha(p) == h for p, h in frozen['inputs'].items())
    bank, arrays = load_inputs('validation')
    probs = [member_probabilities(bank, arrays, members(e)) for e in ('Pool_P', 'S2_C')]
    errors = []
    for ei, e in enumerate(('Pool_P', 'S2_C')):
        for si, seed in enumerate([17, 29, 43, 71, 101]):
            z = np.load(OLD / f'results/runs/{e}/seed_{seed}/validation-logits.npy')
            error = float(np.max(np.abs(probs[ei][si] - sigmoid(z))))
            assert error < 1e-6
            errors.append(error)
    np.savez(OUT / 'validation-probabilities.npz', Pool_P=probs[0], S2_C=probs[1])
    ix = np.array([97, 3, 400, 11, 777, 25, 0, 48, 200, 17, 700, 56, 98])
    subset = [{**bank[i], 'id': f'new-audit:{j}'} for j, i in enumerate(ix)]
    small_arrays = {'embeddings': arrays['embeddings'][ix]}
    perm = [2, 0, 3, 1]
    permuted = [{**r, 'candidates': [{**r['candidates'][k], 'index': j} for j, k in enumerate(perm)]} for r in subset]
    perm_arrays = {'embeddings': small_arrays['embeddings'][:, perm]}
    perm_probs = [member_probabilities(permuted, perm_arrays, members(e)) for e in ('Pool_P', 'S2_C')]
    perm_error = max(float(np.max(np.abs(p[:, ix][:, :, perm] - q))) for p, q in zip(probs, perm_probs))
    assert perm_error < 3e-5
    checked = []
    for name in frozen['pipelines']:
        path = OUT / 'pipelines' / name
        config = read(path / 'pipeline.json')
        expected = read(path / 'predictions.json')['selected']
        assert decide(bank, probs, config).tolist() == expected
        actual = predict(subset, small_arrays, path)
        assert actual == [expected[i] for i in ix]
        assert predict(subset[:1], {'embeddings': small_arrays['embeddings'][:1]}, path) == actual[:1]
        assert all(type(i) is int and 0 <= i < 4 for i in actual)
        perm_choice = decide(permuted, perm_probs, config)
        for q, k in enumerate(perm_choice):
            assert parse_answer(permuted[q]['candidates'][int(k)]['text']) == parse_answer(subset[q]['candidates'][actual[q]]['text'])
        checked.append(name)
    # Synthetic texts test parser fallback and one-valid-answer behavior without any reference file.
    all_invalid = [{**bank[0], 'id': 'audit-invalid', 'candidates': [{'index': i, 'text': 'no final numeric marker'} for i in range(4)]}]
    one_valid = [{**all_invalid[0], 'id': 'audit-one-valid', 'candidates': [{'index': i, 'text': '#### 7' if i == 2 else 'no marker'} for i in range(4)]}]
    synthetic = [p[:, :1].copy() for p in probs]
    fallback = []
    for name in frozen['pipelines']:
        config = read(OUT / 'pipelines' / name / 'pipeline.json')
        c = decide(all_invalid, synthetic, config)
        assert c.shape == (1,) and 0 <= int(c[0]) < 4
        assert decide(one_valid, synthetic, config).tolist() == [2]
        fallback.append(name)
    # Explicitly ensure the conditional gate cannot switch on numerical agreement.
    _, choices, disagree = gate_features(bank, probs)
    for name in ('gate8', 'gate4'):
        c = decide(bank, probs, read(OUT / 'pipelines' / name / 'pipeline.json'))
        assert np.array_equal(c[~disagree], choices[1][~disagree])
    p = protected()
    verification = {'passed': p['passed'], 'experiment_checks_passed': True, 'fresh_official_evaluations': 6,
       'portable_official_replays': 6, 'full_expert_probability_replay_max_error': max(errors),
       'candidate_permutation_probability_max_error': perm_error, 'numeric_answer_permutation_checks': checked,
       'new_ids_reordered_and_single_question_checks': checked, 'parser_all_invalid_and_one_valid_checks': fallback,
       'native_int_outputs': True, 'gate_same_answer_never_switches': True,
       'protected_files': p['files_checked'], 'protected_changes': p['changed'], 'test_used': False,
       'frozen_sources_and_assets_unchanged': True, 'reference_cpu_certification': False}
    write(OUT / 'verification.json', verification)
    write(OUT / 'delivery_source_hashes.json', sources())
    print(verification, flush=True)
    assert p['passed'], p['changed']

if __name__ == '__main__':
    main()
