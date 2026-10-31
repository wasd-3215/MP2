"""Train only the 25 missing Pool-P fold models; never loads validation/test."""
import runtime
runtime.configure_torch()
from utils import HERE, OUT, OLD, FUSION, SEEDS, read, write, sha, sources
import hashlib
import math
import platform
import random
import sys
import time
import numpy as np
import torch
from common import load_inputs, read_jsonl, parse_answer, DATA, check_data
from models import Scorer
from features import extract, fit_standardization

def infer(model, x, f):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(x[i:i+256], f[i:i+256]).numpy() for i in range(0, len(x), 256)])

def main():
    OUT.mkdir(exist_ok=True)
    frozen = sources()
    write(OUT / 'training_sources.json', frozen)
    protocol = read(HERE / 'protocol.json')['base_training']
    bank, arrays = load_inputs('train')
    check_data(DATA, ['train'])
    refs = read_jsonl(DATA / 'gsm8k_train.jsonl')
    assert [r['id'] for r in refs] == [r['id'] for r in bank]
    y = np.asarray([[parse_answer(c['text']) == parse_answer(ref['answer']) for c in row['candidates']]
                    for row, ref in zip(bank, refs)], np.float32)
    assert (y.sum(1) > 0).all()
    assert np.array_equal(y, np.load(FUSION / 'results/train-labels.npy'))
    fold_doc = read(FUSION / 'results/folds.json')
    assert fold_doc['ids'] == [r['id'] for r in bank]
    folds = np.load(FUSION / 'results/question-folds.npy')
    assert np.array_equal(folds, fold_doc['folds'])
    check = np.empty(len(bank), np.int64)
    for fold, ix in enumerate(np.array_split(np.random.default_rng(protocol['fold_seed']).permutation(len(bank)), 5)):
        check[ix] = fold
    assert np.array_equal(folds, check)
    raw_x, raw_f = arrays['embeddings'], extract(bank)
    reused = FUSION / 'results/oof-s2_consistency.npy'
    reused_oof = np.load(reused)
    assert reused_oof.shape == (5, len(bank), 4) and np.isfinite(reused_oof).all()
    write(OUT / 'environment.json', {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
          'cpu': platform.processor(), 'platform': platform.platform(), 'torch_threads': torch.get_num_threads(),
          'interop_threads': torch.get_num_interop_threads(), 'openblas_threads': 1, 'serial_jobs': True, 'device': 'cpu'})
    write(OUT / 'reused_inputs.json', {str(p): sha(p) for p in [reused, FUSION / 'results/folds.json', FUSION / 'results/question-folds.npy']})
    np.save(OUT / 'train-labels.npy', y, allow_pickle=False)
    np.save(OUT / 'question-folds.npy', folds, allow_pickle=False)
    all_oof, records = [], []
    for si, seed in enumerate(SEEDS):
        oof = np.full((len(bank), 4), np.nan, np.float32)
        for fold in range(5):
            begin_time = time.perf_counter()
            folder = OUT / f'oof_runs/seed_{seed}/fold_{fold}'
            folder.mkdir(parents=True, exist_ok=True)
            held, train = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
            assert not np.intersect1d(train, held).size
            mean, std = fit_standardization(raw_x[train])
            fm, fs = fit_standardization(raw_f[train])
            # Verify reused C standardization and OOF alignment; its six head features are identical.
            old_run = FUSION / f'results/runs/s2_consistency/seed_{seed}/fold_{fold}'
            assert np.array_equal(np.load(old_run / 'held_indices.npy'), held)
            old_z = np.load(old_run / 'evaluation-logits.npy')
            assert np.array_equal(old_z, reused_oof[si, held])
            with np.load(old_run / 'bundle/preprocessing.npz') as a:
                assert np.array_equal(mean, a['mean']) and np.array_equal(std, a['std'])
                assert np.array_equal(fm, a['feature_mean'][:6]) and np.array_equal(fs, a['feature_std'][:6])
            # Re-run the old C weights with this exact architecture to validate safe reuse.
            c_model = Scorer('S2', 'C')
            with np.load(old_run / 'bundle/weights.npz') as a:
                c_model.load_state_dict({k: torch.from_numpy(a[k].copy()) for k in a.files}, strict=True)
            cx = torch.from_numpy((raw_x[held] - mean) / std)
            cf = torch.from_numpy((raw_f[held] - fm) / fs)
            c_error = float(np.max(np.abs(infer(c_model, cx, cf) - old_z)))
            assert c_error < 3e-5, c_error
            if (folder / 'result.json').exists():
                record = read(folder / 'result.json')
                assert record['sources'] == frozen
                assert sha(folder / 'logits.npy') == record['logits_sha256']
                oof[held] = np.load(folder / 'logits.npy')
                records.append(record)
                continue
            x = torch.from_numpy((raw_x[train] - mean) / std)
            f = torch.from_numpy((raw_f[train] - fm) / fs)
            target = torch.from_numpy(y[train])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = Scorer('Pool', 'P')
            initial = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            torch.manual_seed(seed + 20000)
            optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
            generator = torch.Generator().manual_seed(seed + 10000)
            history, order_hash = [], hashlib.sha256()
            tick = time.perf_counter()
            for epoch in range(1, protocol['epochs'] + 1):
                model.train()
                order = torch.randperm(len(x), generator=generator)
                order_hash.update(order.numpy().tobytes())
                total = 0.
                for start in range(0, len(x), protocol['batch_size']):
                    ix = order[start:start+protocol['batch_size']]
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[ix], f[ix]), target[ix])
                    assert torch.isfinite(loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['gradient_clip'])
                    optimizer.step()
                    total += loss.item() * len(ix)
                history.append({'epoch': epoch, 'train_bce': total / len(x)})
            fit_seconds = time.perf_counter() - tick
            z = infer(model, cx, cf)
            assert np.isfinite(z).all() and torch.count_nonzero(model.base.head[0].weight[:, -5:]).item() == 0
            perm = [2, 0, 3, 1]
            a = infer(model, cx[:13], cf[:13])
            error = float(np.max(np.abs(a[:, perm] - infer(model, cx[:13, perm], cf[:13, perm]))))
            assert error < 3e-5
            bundle = folder / 'bundle'
            bundle.mkdir(exist_ok=True)
            write(bundle / 'config.json', {'model': {'architecture': 'Pool', 'feature_mode': 'P'}})
            np.savez(bundle / 'weights.npz', **{k: v.detach().numpy() for k, v in model.state_dict().items()})
            np.savez(bundle / 'preprocessing.npz', mean=mean, std=std, feature_mean=fm, feature_std=fs)
            np.save(folder / 'held_indices.npy', held, allow_pickle=False)
            np.save(folder / 'logits.npy', z, allow_pickle=False)
            write(folder / 'history.json', history)
            record = {'seed': seed, 'fold': fold, 'train_questions': len(train), 'held_questions': len(held),
                  'epochs': 20, 'steps': 20 * math.ceil(len(train) / 256), 'parameters': sum(p.numel() for p in model.parameters()),
                  'fit_seconds': fit_seconds, 'total_seconds': time.perf_counter() - begin_time,
                  'initial_state_sha256': initial, 'order_sha256': order_hash.hexdigest(), 'sources': frozen,
                  'logits_sha256': sha(folder / 'logits.npy'), 'candidate_permutation_max_error': error,
                  'reused_C_logit_max_error': c_error, 'standardization_fit_heldout': False,
                  'bundle_hashes': {p.name: sha(p) for p in bundle.iterdir()}}
            write(folder / 'result.json', record)
            oof[held] = z
            records.append(record)
            write(OUT / 'training_runs.json', records)
            print(f'{len(records)}/25 Pool-P seed={seed} fold={fold}: {fit_seconds:.1f}s fit, reused C error {c_error:.2g}', flush=True)
        assert np.isfinite(oof).all()
        all_oof.append(oof)
    np.save(OUT / 'oof-Pool_P.npy', np.stack(all_oof), allow_pickle=False)
    assert sources() == frozen
    write(OUT / 'training_verification.json', {'passed': True, 'new_fits': 25, 'reused_C_folds_checked': 25,
          'reused_C_max_logit_error': max(r['reused_C_logit_max_error'] for r in records),
          'fit_seconds': sum(r['fit_seconds'] for r in records), 'total_run_seconds': sum(r['total_seconds'] for r in records),
          'validation_loaded': False, 'test_used': False, 'same_folds_and_seeds_as_reused_C': True})
    print('ALL 25 OOF FITS COMPLETE', flush=True)

if __name__ == '__main__':
    main()
