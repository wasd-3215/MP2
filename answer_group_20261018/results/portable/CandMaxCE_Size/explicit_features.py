"""Pure input features. Numeric grouping always uses the untouched official parser."""
from collections import Counter
import re
import numpy as np
from common import parse_answer

NAMES = ['valid_parse', 'own_answer_support', 'distinct_valid_answers',
         'in_largest_group', 'largest_group_tied', 'support_gap',
         'log_characters', 'log_words', 'log_lines', 'log_operators',
         'log_math_markers', 'log_final_markers', 'log_confident_words']
FEATURES = {'none': [], 'parse': [0], 'consistency': list(range(6)),
            'style': [0] + list(range(6,13)), 'all': list(range(13))}


def answer_structure(bank):
    answers, support, unique = [], [], []
    for row in bank:
        a = [parse_answer(c['text']) for c in row['candidates']]
        counts = Counter(v for v in a if v is not None)
        f = [counts[v] if v is not None else 0 for v in a]
        sizes = sorted(counts.values(), reverse=True)
        answers.append(a)
        support.append(f)
        unique.append(bool(sizes) and (len(sizes)==1 or sizes[0]>sizes[1]))
    return answers, np.asarray(support, dtype=np.int64).reshape(-1,4), np.asarray(unique)


def extract(bank):
    out = np.zeros((len(bank),4,len(NAMES)),dtype=np.float32)
    answers, support, unique = answer_structure(bank)
    for q,row in enumerate(bank):
        sizes = sorted(Counter(a for a in answers[q] if a is not None).values(),reverse=True)
        maximum = sizes[0] if sizes else 0
        gap = maximum - (sizes[1] if len(sizes)>1 else 0)
        for i,c in enumerate(row['candidates']):
            t = c['text']
            out[q,i,:6] = [answers[q][i] is not None, support[q,i]/4,
                           len(sizes)/4, support[q,i]>0 and support[q,i]==maximum,
                           bool(sizes) and not unique[q], gap/4]
            counts = [len(t),len(t.split()),len(t.splitlines()),
                      len(re.findall(r'[+*/=−×÷-]',t)),
                      len(re.findall(r'\$|\\(?:frac|times|boxed|begin)',t)),
                      len(re.findall(r'####|\\boxed|final answer',t,re.I)),
                      len(re.findall(r'\b(?:clearly|obviously|certainly|definitely)\b',t,re.I))]
            out[q,i,6:] = np.log1p(counts)
    return out


def fit_standardization(x):
    flat=x.reshape(-1,x.shape[-1])
    return flat.mean(0,dtype=np.float64).astype(np.float32), np.maximum(flat.std(0,dtype=np.float64),1e-4).astype(np.float32)
