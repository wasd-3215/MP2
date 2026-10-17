"""Check the selected copied bundle and original files without reading test labels."""
import runtime

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
from common import load_inputs


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results")
    output = parser.parse_args().output.resolve()
    folder = output / "delivery_check"
    folder.mkdir(exist_ok=True)
    selection = read(output / "selection.json")
    bundle = output / "selected_bundle"
    assert {p.name: sha(p) for p in bundle.iterdir()} == selection["bundle_sha256"]
    commands = [
        [sys.executable, "-B", str(HERE.parent / "predict.py"), "--method", str(HERE / "method.py"),
         "--checkpoint", str(bundle), "--split", "validation", "--output", str(folder / "predictions.json")],
        [sys.executable, "-B", str(HERE.parent / "score_predictions.py"), "--split", "validation",
         "--predictions", str(folder / "predictions.json"), "--output", str(folder / "score.json")]]
    durations = []
    for i, command in enumerate(commands):
        start = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy(), cwd=HERE.parent)
        durations.append(time.perf_counter() - start)
        (folder / f"command_{i}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        assert result.returncode == 0, result.stderr
    score = read(folder / "score.json")
    assert score["accuracy"] == selection["validation_accuracy"]
    expected = read(folder / "predictions.json")["selected"]
    from method import predict
    bank, arrays = load_inputs("validation")
    index = np.array([97, 3, 400, 11, 777, 25, 0, 48, 200, 17, 700, 56, 98])
    subset = [{**bank[i], "id": f"new-bank:{j}"} for j, i in enumerate(index)]
    subarrays = {"embeddings": arrays["embeddings"][index]}
    actual = predict(subset, subarrays, bundle)
    assert actual == [expected[i] for i in index]
    assert predict(subset[:1], {"embeddings": subarrays["embeddings"][:1]}, bundle) == actual[:1]
    assert all(type(i) is int and 0 <= i <= 3 for i in actual)
    checked_models = []
    for run in read(output / 'selected_runs.json'):
        if run['seed'] != 17:
            continue
        run_dir = HERE / run['run_dir']
        reference = read(run_dir / 'validation-predictions.json')['selected']
        replay = predict(subset, subarrays, run_dir / 'bundle')
        assert replay == [reference[i] for i in index]
        assert predict(subset[:1], {'embeddings': subarrays['embeddings'][:1]}, run_dir / 'bundle') == replay[:1]
        assert all(type(i) is int and 0 <= i <= 3 for i in replay)
        checked_models.append(run['group'])
    before = read(HERE / "protected_before.json")
    changed = [name for name, digest in before.items() if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    assert not changed, changed
    training_sources = read(output / "source_hashes.json")
    source_changes = [name for name, digest in training_sources.items() if sha(HERE / name) != digest]
    assert not source_changes, source_changes
    result = {"passed": True, "model": selection["model"], "validation_accuracy": score["accuracy"],
              "official_prediction_and_scoring_wall_seconds": sum(durations),
              "bundle_hashes_match": True, "reordered_new_ids_13_questions": True, "single_question": True,
              "native_integer_output": True, "protected_files_checked": len(before), "protected_changes": changed,
              "training_source_changes": source_changes,
              "all_seed17_models_new_ids_reordered_and_single_question": checked_models}
    (folder / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sources = {p.name: sha(p) for p in HERE.iterdir() if p.is_file()}
    (output / "delivery_source_hashes.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
