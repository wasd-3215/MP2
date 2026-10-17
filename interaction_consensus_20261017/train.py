"""Fixed last-epoch training of six cells; validation is diagnostic only."""
import runtime
from support import HERE, CODE, OUT, SEEDS, VARIANTS, read, write, sha, source_hashes, protected
from method import infer, sigmoid
from models import Scorer
from features import extract, fit_standardization
from common import load_inputs, read_jsonl, parse_answer, DATA, check_data
import hashlib
import platform
import random
import sys
import time
import numpy as np
import torch


def labels(split, bank):
    assert split in ('train', 'validation')
    check_data(DATA, [split])
    refs = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    assert len(refs) == len(bank)
    result = []
    for row, ref in zip(bank, refs):
        assert row['id'] == ref['id'] and row['question'] == ref['question']
        answer = parse_answer(ref['answer'])
        assert answer is not None
        result.append([parse_answer(c['text']) == answer for c in row['candidates']])
    y = np.asarray(result, dtype=np.float32)
    assert (y.sum(axis=1) > 0).all()
    return y


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def metrics(z, y, valid):
    scores = sigmoid(z)
    original = scores.argmax(axis=1)
    filtered = original.copy()
    change = valid.any(axis=1) & ~valid[np.arange(len(y)), original]
    filtered[change] = np.where(valid[change], scores[change], -np.inf).argmax(axis=1)
    return {'bce': float(np.mean(np.logaddexp(0, z.astype(np.float64)) - y * z)),
            'accuracy_raw': float(100 * y[np.arange(len(y)), original].mean()),
            'accuracy_filtered': float(100 * y[np.arange(len(y)), filtered].mean())}


def main():
    OUT.mkdir(exist_ok=True)
    assert read(OUT / 'implementation_checks.json')['passed']
    protocol = read(HERE / 'protocol.json')
    sources = source_hashes()
    if (OUT / 'training_sources.json').exists():
        assert read(OUT / 'training_sources.json') == sources
    write(OUT / 'training_sources.json', sources)
    protected()
    cpu = platform.processor()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0]
    except (ImportError, OSError):
        pass
    write(OUT / 'environment.json', {'cpu': cpu, 'python': sys.version, 'executable': sys.executable,
       'platform': platform.platform(), 'numpy': np.__version__, 'torch': torch.__version__,
       'torch_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
       'openblas_threads': 1, 'device': 'cpu', 'serial_jobs': True})
    train, ta = load_inputs('train')
    validation, va = load_inputs('validation')
    y, vy = labels('train', train), labels('validation', validation)
    raw_f, val_f = extract(train), extract(validation)
    mean, std = fit_standardization(ta['embeddings'])
    fm, fs = fit_standardization(raw_f)
    prep = {'mean': mean, 'std': std, 'feature_mean': fm, 'feature_std': fs}
    x = torch.from_numpy((ta['embeddings'] - mean) / std)
    f = torch.from_numpy((raw_f - fm) / fs)
    vx = torch.from_numpy((va['embeddings'] - mean) / std)
    vf = torch.from_numpy((val_f - fm) / fs)
    target = torch.from_numpy(y)
    valid, vvalid = raw_f[..., 0].astype(bool), val_f[..., 0].astype(bool)
    np.save(OUT / 'train-labels.npy', y, allow_pickle=False)
    np.save(OUT / 'validation-labels.npy', vy, allow_pickle=False)
    write(OUT / 'data_summary.json', {'train': len(train), 'validation': len(validation),
       'train_correct_histogram': np.bincount(y.sum(axis=1).astype(int), minlength=5).tolist(),
       'validation_correct_histogram': np.bincount(vy.sum(axis=1).astype(int), minlength=5).tolist(),
       'test_used': False, 'standardization_fit_split': 'train'})
    results = []
    for variant in VARIANTS:
        architecture, feature_mode = variant.split('_')
        for seed in SEEDS:
            folder = OUT / 'runs' / variant / f'seed_{seed}'
            identity = {'variant': variant, 'seed': seed, 'sources': sources}
            if (folder / 'result.json').exists():
                result = read(folder / 'result.json')
                assert result['identity'] == identity
                assert all(sha(folder / 'bundle' / k) == v for k, v in result['bundle_hashes'].items())
                results.append(result)
                print('RESUME', variant, seed, flush=True)
                continue
            folder.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = Scorer(architecture, feature_mode)
            initial_digest = state_digest(model)
            torch.manual_seed(seed + 20000)
            optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
            generator = torch.Generator().manual_seed(seed + 10000)
            order_digest = hashlib.sha256()
            history = []
            initial = None
            fit_seconds, diagnostic_seconds = 0., 0.
            for epoch in range(0, protocol['epochs'] + 1):
                stochastic = None
                if epoch:
                    tick = time.perf_counter()
                    model.train()
                    order = torch.randperm(len(x), generator=generator)
                    order_digest.update(order.numpy().tobytes())
                    total_loss = 0.
                    for begin in range(0, len(x), protocol['batch_size']):
                        ix = order[begin:begin+protocol['batch_size']]
                        optimizer.zero_grad(set_to_none=True)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[ix], f[ix]), target[ix])
                        assert torch.isfinite(loss)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['gradient_clip'])
                        optimizer.step()
                        total_loss += loss.item() * len(ix)
                    stochastic = total_loss / len(x)
                    fit_seconds += time.perf_counter() - tick
                tick = time.perf_counter()
                rng_before = torch.get_rng_state().clone()
                tz, vz = infer(model, x, f), infer(model, vx, vf)
                assert torch.equal(rng_before, torch.get_rng_state())
                row = {'epoch': epoch, 'optimizer_steps': epoch * 19, 'stochastic_train_bce': stochastic,
                       'train': metrics(tz, y, valid), 'validation': metrics(vz, vy, vvalid)}
                history.append(row)
                if epoch == 0:
                    initial = vz.copy()
                diagnostic_seconds += time.perf_counter() - tick
            assert np.isfinite(vz).all() and len(history) == 21
            if feature_mode == 'P':
                assert torch.count_nonzero(model.base.head[0].weight[:, -5:]).item() == 0
            bundle = folder / 'bundle'
            bundle.mkdir(exist_ok=True)
            write(bundle / 'config.json', {'model': {'architecture': architecture, 'feature_mode': feature_mode}})
            np.savez(bundle / 'weights.npz', **{k: v.detach().numpy() for k, v in model.state_dict().items()})
            np.savez(bundle / 'preprocessing.npz', **prep)
            np.save(folder / 'validation-logits.npy', vz, allow_pickle=False)
            np.save(folder / 'train-logits.npy', tz, allow_pickle=False)
            np.save(folder / 'initial-validation-logits.npy', initial, allow_pickle=False)
            write(folder / 'history.json', history)
            permutation = [2, 0, 3, 1]
            a = infer(model, vx[:13], vf[:13])
            b = infer(model, vx[:13, permutation], vf[:13, permutation])
            error = float(np.max(np.abs(a[:, permutation] - b)))
            batch_error = float(np.max(np.abs(a[:1] - infer(model, vx[:1], vf[:1]))))
            assert error < 3e-5 and batch_error < 3e-5
            result = {'identity': identity, 'variant': variant, 'seed': seed, 'epochs': 20,
                      'optimizer_steps': 380, 'parameters': sum(p.numel() for p in model.parameters()),
                      'initial_state_sha256': initial_digest, 'sample_order_sha256': order_digest.hexdigest(),
                      'fit_seconds': fit_seconds, 'diagnostic_seconds': diagnostic_seconds,
                      'total_training_and_diagnostics_seconds': time.perf_counter() - start,
                      'final_metrics': history[-1], 'run_dir': str(folder.relative_to(HERE)),
                      'permutation_max_error': error, 'single_batch_max_error': batch_error,
                      'bundle_hashes': {p.name: sha(p) for p in bundle.iterdir()}}
            write(folder / 'result.json', result)
            results.append(result)
            write(OUT / 'training_runs.json', results)
            print(f'{len(results)}/30 {variant} seed{seed} final raw={row["validation"]["accuracy_raw"]:.3f} filtered={row["validation"]["accuracy_filtered"]:.3f}, {result["total_training_and_diagnostics_seconds"]:.1f}s', flush=True)
    assert len(results) == 30
    for architecture in ('I', 'Pool', 'S2'):
        for seed in SEEDS:
            p = next(r for r in results if r['variant'] == architecture + '_P' and r['seed'] == seed)
            c = next(r for r in results if r['variant'] == architecture + '_C' and r['seed'] == seed)
            assert p['initial_state_sha256'] == c['initial_state_sha256']
            assert p['sample_order_sha256'] == c['sample_order_sha256']
            pi = np.load(HERE / p['run_dir'] / 'initial-validation-logits.npy')
            ci = np.load(HERE / c['run_dir'] / 'initial-validation-logits.npy')
            assert np.array_equal(pi, ci)
    assert source_hashes() == sources
    write(OUT / 'training_runs.json', results)
    write(OUT / 'training_verification.json', {'passed': True, 'runs': 30,
           'PC_identical_initial_weights_and_functions': 15, 'PC_identical_training_orders': 15,
           'final_epoch_only': True, 'epochs': 20, 'optimizer_steps_per_run': 380,
           'eval_diagnostics_preserve_rng': True, 'test_used': False})
    print('ALL 30 FIXED-EPOCH RUNS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
