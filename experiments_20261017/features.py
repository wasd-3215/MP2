"""Train-only standardization and a small, train-only question TF-IDF vocabulary."""
import re
from collections import Counter

import numpy as np


def question_terms(text):
    words = re.findall(r"[a-z]+|\d+(?:\.\d+)?|[-+*/%=]", text.lower())
    return words + [a + " " + b for a, b in zip(words, words[1:])]


def fit_question_features(bank, max_features=1024):
    document_counts = Counter()
    for row in bank:
        document_counts.update(set(question_terms(row["question"])))
    ranked = sorted(document_counts, key=lambda term: (-document_counts[term], term))
    vocabulary = [term for term in ranked if document_counts[term] >= 2][:max_features]
    idf = np.asarray([np.log((1 + len(bank)) / (1 + document_counts[t])) + 1
                      for t in vocabulary], dtype=np.float32)
    return vocabulary, idf


def transform_questions(bank, vocabulary, idf):
    positions = {term: i for i, term in enumerate(vocabulary)}
    result = np.zeros((len(bank), len(vocabulary)), dtype=np.float32)
    for row_index, row in enumerate(bank):
        counts = Counter(question_terms(row["question"]))
        for term, count in counts.items():
            index = positions.get(term)
            if index is not None:
                result[row_index, index] = (1 + np.log(count)) * idf[index]
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / np.maximum(norms, 1e-12)


def fit_normalization(embeddings):
    flat = embeddings.reshape(-1, embeddings.shape[-1])
    mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-4)

