"""Boundary tests for a parser-based decision constraint, without references."""
from itertools import product
import numpy as np
from method import validity_filter
from common import parse_answer
from run_suite import OUT, write


def main():
    assert parse_answer('work only 42') is None
    assert parse_answer('#### 1/0') is None
    assert parse_answer('#### 1/2') == parse_answer('#### 0.5')
    assert parse_answer(r'\boxed{42}') == '42'
    checks = 0
    rng = np.random.default_rng(731)
    for bits in product([False, True], repeat=4):
        bank = [{'candidates': [{'index': i, 'text': '#### 42' if bits[i] else 'unparseable'} for i in range(4)]}]
        for original in range(4):
            for scores in [np.array([[.9, .8, .7, .6]]), np.ones((1, 4)), rng.random((1, 4))]:
                before = scores.copy()
                actual, diag = validity_filter(bank, scores, np.array([original]))
                eligible = [i for i, bit in enumerate(bits) if bit]
                expected = original if bits[original] or not eligible else max(eligible, key=lambda i: (scores[0, i], -i))
                assert int(actual[0]) == expected
                assert np.array_equal(scores, before)
                assert diag['replaceable'] == int(bool(eligible) and not bits[original])
                # Any original correct choice must be valid, hence is preserved.
                if bits[original]:
                    assert actual[0] == original
                checks += 1
    # Batched and single-question execution agree, including all-invalid fallback.
    bank = [{'candidates': [{'text': t} for t in texts]} for texts in [
        ['bad', '#### 1', '#### 2', 'bad'], ['bad'] * 4, ['#### 1'] * 4]]
    scores = np.array([[.9, .4, .7, .8], [.9, .1, .2, .3], [.8, .9, .1, .2]])
    choices = np.array([0, 2, 2])
    actual = validity_filter(bank, scores, choices)[0]
    assert actual.tolist() == [2, 2, 2]
    for q in range(3):
        assert validity_filter(bank[q:q+1], scores[q:q+1], choices[q:q+1])[0][0] == actual[q]
    OUT.mkdir(exist_ok=True)
    write(OUT / 'implementation_checks.json', {'passed': True, 'exhaustive_mask_choice_score_cases': checks,
          'official_parser_boundaries': True, 'ties_earliest': True, 'scores_unchanged': True,
          'valid_routed_choice_preserved_even_if_not_highest_score': True,
          'all_invalid_fallback': True, 'batch_single_agreement': True})
    print('IMPLEMENTATION CHECKS PASSED', checks)


if __name__ == '__main__':
    main()
