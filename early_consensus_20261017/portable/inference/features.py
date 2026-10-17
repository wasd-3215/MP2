"""Identical six training-standardized features for every condition."""
from explicit_features import extract as original_extract, fit_standardization


def extract(bank):
    return original_extract(bank)[..., :6].copy()
