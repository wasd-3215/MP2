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


def train_one(name, seed, lr, protocol, tensors, bank, preprocessing, output):
    lr_tag = str(lr).replace(".", "p")
    run_dir = output / "runs" / name / f"lr_{lr_tag}" / f"seed_{seed}"
    config = {"model": {"name": name, "hidden": protocol["hidden"],
                        "dropout": protocol["dropout"],
                        "question_dim": len(preprocessing["vocabulary"]) if name.endswith("_question") else 0},
              "seed": seed, "learning_rate": lr, "protocol_sha256": sha(HERE / "protocol.json")}
    completed_path = run_dir / "result.json"
    if completed_path.exists():
        cached = json.loads(completed_path.read_text(encoding="utf-8"))
        if cached["configuration"] != config:
            raise ValueError("Existing run has different configuration; use a new output directory")
        print(f"RESUME {name} seed={seed} lr={lr} accuracy={cached['accuracy']:.3f}", flush=True)
        return cached
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = CandidateScorer(**config["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=protocol["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    train_x, train_q, train_y, val_x, val_q, val_y = tensors
    ids = [row["id"] for row in bank]
    generator = torch.Generator().manual_seed(seed + 10000)
    best_key = (-1, float("-inf"))
    best_state, best_epoch = None, 0
    history = []
    tick = time.perf_counter()
    print(f"START {name} seed={seed} lr={lr}", flush=True)
    for epoch in range(1, protocol["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        accumulated = 0.0
        for start in range(0, len(order), protocol["batch_size"]):
            index = order[start:start + protocol["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            scores = model(train_x[index], train_q[index])
            loss = criterion(scores, train_y[index])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip"])
            optimizer.step()
            accumulated += loss.item() * len(index)
        model.eval()
        with torch.inference_mode():
            val_scores = torch.cat([model(val_x[start:start + 256], val_q[start:start + 256])
                                    for start in range(0, len(val_x), 256)])
            val_loss = criterion(val_scores, val_y).item()
            choices = val_scores.argmax(dim=1).tolist()
        # Same fixed official function used by score_predictions.py, without modification.
        score = score_choices("validation", ids, choices)
        history.append({"epoch": epoch, "train_bce": accumulated / len(train_x),
                        "validation_bce": val_loss, "official_correct": score["correct"],
                        "official_accuracy": score["accuracy"],
                        "elapsed_seconds": time.perf_counter() - tick})
        key = (score["correct"], -val_loss)
        if key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"EPOCH {name} seed={seed} {epoch}: val={score['accuracy']:.3f} best={100*best_key[0]/len(bank):.3f}", flush=True)
        if epoch - best_epoch >= protocol["patience"]:
            break
    training_seconds = time.perf_counter() - tick
    model.load_state_dict(best_state)
    model.eval()
    checks = semantic_checks(model, val_x, val_q)
    with torch.inference_mode():
        logits = torch.cat([model(val_x[start:start + 256], val_q[start:start + 256])
                            for start in range(0, len(val_x), 256)]).numpy()
    bundle = run_dir / "bundle"
    bundle.mkdir(exist_ok=True)
    bundle_config = {"model": config["model"]}
    if model.has_question:
        bundle_config["vocabulary"] = preprocessing["vocabulary"]
    write_json(bundle / "config.json", bundle_config)
    np.savez(bundle / "weights.npz", **{k: v.numpy() for k, v in best_state.items()})
    prep = {"mean": preprocessing["mean"], "std": preprocessing["std"]}
    if model.has_question:
        prep["idf"] = preprocessing["idf"]
    np.savez(bundle / "preprocessing.npz", **prep)
    np.save(run_dir / "validation-logits.npy", logits, allow_pickle=False)
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "checks.json", checks)
    size = asset_bytes(bundle)
    if size > 16 * 1024**2:
        raise ValueError("Bundle exceeds uncompressed asset budget")
    official, prediction_seconds, scoring_seconds = official_evaluate(run_dir)
    frozen_choices = json.loads((run_dir / "validation-predictions.json").read_text())["selected"]
    if frozen_choices != logits.argmax(axis=1).tolist() or official["correct"] != best_key[0]:
        raise AssertionError("Fresh inference differs from selected training checkpoint")
    result = {"configuration": config, "model": name, "seed": seed, "learning_rate": lr,
              "accuracy": official["accuracy"], "correct": official["correct"], "n": official["n"],
              "best_epoch": best_epoch, "epochs_run": len(history), "best_validation_bce": -best_key[1],
              "parameters": sum(p.numel() for p in model.parameters()),
              "training_seconds": training_seconds,
              "prediction_wall_seconds": prediction_seconds, "scoring_wall_seconds": scoring_seconds,
              "validation_total_wall_seconds": prediction_seconds + scoring_seconds,
              "uncompressed_asset_bytes": size,
              "bundle_sha256": {p.name: sha(p) for p in bundle.iterdir()},
              "run_dir": str(run_dir.relative_to(HERE)), "checks": checks}
    write_json(completed_path, result)
    print(f"DONE {name} seed={seed} lr={lr} val={official['accuracy']:.3f} epoch={best_epoch} train={training_seconds:.1f}s", flush=True)
    return result


def evaluate_baselines(bank, labels, output):
    ids = [row["id"] for row in bank]
    majority = []
    from collections import Counter
    for row in bank:
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        counts = Counter(answer for answer in answers if answer is not None)
        frequencies = [counts.get(answer, 0) if answer is not None else -1 for answer in answers]
        majority.append(int(np.argmax(frequencies)))
    baselines = {}
    for name, choices in [("always_first", [0] * len(bank)), ("answer_majority", majority),
                          ("random_seed17", np.random.default_rng(17).integers(0, 4, len(bank)).tolist())]:
        folder = output / "baselines" / name
        write_json(folder / "predictions.json", {"protocol": PROTOCOL, "split": "validation", "ids": ids,
                                                 "selected": choices})
        run_command([sys.executable, "-B", str(CODE / "score_predictions.py"), "--split", "validation",
                     "--predictions", str(folder / "predictions.json"), "--output", str(folder / "score.json")],
                    folder / "score.log")
        baselines[name] = json.loads((folder / "score.json").read_text())
    baselines["uniform_random_expectation_percent"] = float(100 * labels.mean())
    write_json(output / "baselines.json", baselines)


def verify_protected(output):
    before = json.loads((HERE / "protected_before.json").read_text(encoding="utf-8-sig"))
    changes = [name for name, digest in before.items()
               if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    write_json(output / "protected_verification.json", {"files_checked": len(before), "changed": changes,
                                                        "passed": not changes})
    if changes:
        raise AssertionError(f"Protected files changed: {changes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((HERE / "protocol.json").read_text())
    write_json(output / "source_hashes.json", {p.name: sha(p) for p in HERE.iterdir()
                                              if p.is_file() and p.suffix in (".py", ".json", ".txt")})
    write_json(output / "environment.json", {"python": sys.version, "executable": sys.executable,
               "platform": platform.platform(), "processor": "AMD Ryzen 7 4800U with Radeon Graphics",
               "processor_platform": platform.processor(), "numpy": np.__version__, "torch": torch.__version__,
               "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
               "device": "cpu", "date_local": "2026-10-17", "timezone": "Asia/Shanghai"})
    verify_protected(output)
    run_command([sys.executable, "-B", str(CODE / "check_data.py")], output / "data_check.log")
    tick = time.perf_counter()
    train_bank, train_arrays = load_inputs("train")
    val_bank, val_arrays = load_inputs("validation")
    train_labels = labels_for("train", train_bank)
    val_labels = labels_for("validation", val_bank)
    mean, std = fit_normalization(train_arrays["embeddings"])
    vocabulary, idf = fit_question_features(train_bank)
    train_q = transform_questions(train_bank, vocabulary, idf)
    val_q = transform_questions(val_bank, vocabulary, idf)
    tensors = (torch.from_numpy((train_arrays["embeddings"] - mean) / std),
               torch.from_numpy(train_q), torch.from_numpy(train_labels),
               torch.from_numpy((val_arrays["embeddings"] - mean) / std),
               torch.from_numpy(val_q), torch.from_numpy(val_labels))
    prep = {"mean": mean, "std": std, "vocabulary": vocabulary, "idf": idf}
    np.save(output / "train-labels.npy", train_labels.astype(np.int64), allow_pickle=False)
    np.save(output / "validation-labels.npy", val_labels.astype(np.int64), allow_pickle=False)
    write_json(output / "data_summary.json", {"train_questions": len(train_bank), "validation_questions": len(val_bank),
               "train_correct_candidate_histogram": np.bincount(train_labels.sum(axis=1).astype(int), minlength=5).tolist(),
               "validation_correct_candidate_histogram": np.bincount(val_labels.sum(axis=1).astype(int), minlength=5).tolist(),
               "question_vocabulary_size": len(vocabulary), "preprocessing_seconds": time.perf_counter() - tick,
               "test_used_for_training_selection_or_scoring": False})
    evaluate_baselines(val_bank, val_labels, output)
    all_results, chosen_results, selections = [], [], {}
    for name in protocol["models"]:
        search = [train_one(name, protocol["search_seed"], lr, protocol, tensors, val_bank, prep, output)
                  for lr in protocol["learning_rates"]]
        all_results.extend(search)
        selected = max(search, key=lambda r: (r["correct"], -r["best_validation_bce"]))
        selections[name] = selected["learning_rate"]
        chosen_results.append(selected)
        write_json(output / "selected_learning_rates.json", selections)
        for seed in protocol["seeds"]:
            if seed == protocol["search_seed"]:
                continue
            result = train_one(name, seed, selected["learning_rate"], protocol, tensors, val_bank, prep, output)
            all_results.append(result)
            chosen_results.append(result)
        write_json(output / "all_runs.json", all_results)
        write_json(output / "selected_runs.json", chosen_results)
    verify_protected(output)
    # Report generator has no training logic and never accesses test references.
    run_command([sys.executable, "-B", str(HERE / "summarize.py"), "--output", str(output)], output / "summarize.log")
    print(f"COMPLETE: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
