"""Offline predict(bank, arrays, checkpoint). Reads only inference assets."""
import runtime
runtime.configure_torch()

import json
from pathlib import Path

import numpy as np
import torch

from features import transform_questions
from models import CandidateScorer


def load_bundle(checkpoint):
    path = Path(checkpoint)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model = CandidateScorer(**config["model"])
    with np.load(path / "weights.npz", allow_pickle=False) as archive:
        state = {key: torch.from_numpy(archive[key].copy()) for key in archive.files}
    model.load_state_dict(state, strict=True)
    model.eval()
    with np.load(path / "preprocessing.npz", allow_pickle=False) as archive:
        preprocessing = {key: archive[key].copy() for key in archive.files}
    return model, config, preprocessing


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("Supply --checkpoint pointing to a generated bundle directory")
    model, config, prep = load_bundle(checkpoint)
    x = np.asarray(arrays["embeddings"], dtype=np.float32)
    if x.shape != (len(bank), 4, 1024) or not np.isfinite(x).all():
        raise ValueError("Expected finite, aligned (N, 4, 1024) candidate embeddings")
    x = torch.from_numpy((x - prep["mean"]) / prep["std"])
    question = None
    if model.has_question:
        question = torch.from_numpy(transform_questions(bank, config["vocabulary"], prep["idf"]))
    choices = []
    with torch.inference_mode():
        for start in range(0, len(bank), 256):
            q = None if question is None else question[start:start + 256]
            choices.extend(model(x[start:start + 256], q).argmax(dim=1).tolist())
    return choices

