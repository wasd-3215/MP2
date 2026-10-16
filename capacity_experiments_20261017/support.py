"""Reproducible CPU training and validation, with untouched official evaluators."""
import runtime
runtime.configure_torch()

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))
from common import DATA, PROTOCOL, check_data, load_inputs, parse_answer, read_jsonl, score_choices
from features import fit_normalization, fit_question_features, transform_questions
from models import CandidateScorer


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_command(command, log):
    tick = time.perf_counter()
    completed = subprocess.run(command, cwd=CODE, env=os.environ.copy(), capture_output=True, text=True)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Command failed; see {log}: {completed.stderr}")
    return time.perf_counter() - tick


def labels_for(split, bank):
    # Only train/validation may enter this training pipeline.
    if split not in ("train", "validation"):
        raise ValueError("Training pipeline cannot access test/audit references")
    check_data(DATA, (split,))
    references = read_jsonl(DATA / f"gsm8k_{split}.jsonl")
    if len(references) != len(bank):
        raise ValueError("Reference count mismatch")
    labels = []
    for row, reference in zip(bank, references):
        if row["id"] != reference["id"] or row["question"] != reference["question"]:
            raise ValueError("Reference alignment mismatch")
        gold = parse_answer(reference["answer"])
        if gold is None:
            raise ValueError("Unparseable reference")
        parsed = [parse_answer(c["text"]) for c in row["candidates"]]
        labels.append([int(a is not None and a == gold) for a in parsed])
    result = np.asarray(labels, dtype=np.float32)
    if not (result.sum(axis=1) >= 1).all():
        raise ValueError("A group lacks a correct candidate")
    return result


def asset_bytes(bundle):
    total = 0
    for path in bundle.iterdir():
        if path.suffix == ".npz":
            with zipfile.ZipFile(path) as archive:
                total += sum(info.file_size for info in archive.infolist())
        else:
            total += path.stat().st_size
    return total


def official_evaluate(run_dir):
    prediction_path = run_dir / "validation-predictions.json"
    prediction_seconds = run_command([
        sys.executable, "-B", str(CODE / "predict.py"), "--method", str(HERE / "method.py"),
        "--checkpoint", str(run_dir / "bundle"), "--split", "validation",
        "--output", str(prediction_path)], run_dir / "predict.log")
    scoring_seconds = run_command([
        sys.executable, "-B", str(CODE / "score_predictions.py"), "--split", "validation",
        "--predictions", str(prediction_path), "--output", str(run_dir / "validation-score.json")],
        run_dir / "score.log")
    score = json.loads((run_dir / "validation-score.json").read_text())
    return score, prediction_seconds, scoring_seconds


def semantic_checks(model, x, q):
    model.eval()
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.inference_mode():
        original = model(x[:13], q[:13])
        permuted = model(x[:13, permutation], q[:13])
        error = (permuted - original[:, permutation]).abs().max().item()
        alone = model(x[:1], q[:1])
        batch_error = (alone[0] - original[0]).abs().max().item()
        perturbed = x[:13].clone()
        perturbed[:, 1:] = torch.roll(perturbed[:, 1:], shifts=1, dims=0)
        context_effect = (model(perturbed, q[:13])[:, 0] - original[:, 0]).abs().max().item()
    if not np.isfinite(error) or error > 2e-5 or batch_error > 2e-5:
        raise AssertionError("Candidate permutation or batch independence check failed")
    independent = model.name in ("linear", "independent", "independent_large",
                                  "attention_self", "independent_question")
    if independent and context_effect > 2e-5:
        raise AssertionError("Independent/self-only scorer depends on other candidates")
    return {"permutation_max_abs_error": error, "batch_size_max_abs_error": batch_error,
            "other_candidate_perturbation_max_abs_effect": context_effect,
            "non_fixed_batch_sizes_checked": [1, 13], "passed": True}


def verify_protected(output):
    before = json.loads((HERE / "protected_before.json").read_text(encoding="utf-8-sig"))
    changes = [name for name, digest in before.items()
               if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    write_json(output / "protected_verification.json", {"files_checked": len(before), "changed": changes,
                                                        "passed": not changes})
    if changes:
        raise AssertionError(f"Protected files changed: {changes}")


