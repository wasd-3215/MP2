"""Frozen independent judgment plus residual candidate correction."""
from support import (HERE, CODE, ROOT, write_json, sha, run_command, labels_for,
                     asset_bytes, official_evaluate, semantic_checks, verify_protected)
import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from common import load_inputs, score_choices, PROTOCOL
from features import fit_normalization
from models import CandidateScorer, load_anchor
from measurements import measure


def train_one(name, seed, lr, protocol, tensors, bank, train_bank, prep, output, sources, anchor=None):
    group = name
    folder = output/'runs'/group/f"lr_{str(lr).replace('.', 'p')}"/f'seed_{seed}'
    config = {'model': {'name': name, 'hidden': protocol['hidden'], 'dropout': protocol['dropout'],
                        'question_dim': 0}, 'weight_decay': protocol['weight_decay'], 'seed': seed, 'learning_rate': lr,
              'protocol_sha256': sha(HERE/'protocol.json'), 'source_hashes': sources,
              'anchor':None if anchor is None else {'run_dir':anchor['run_dir'],'bundle_sha256':anchor['bundle_sha256']}}
    if (folder/'result.json').exists():
        cached = json.loads((folder/'result.json').read_text())
        if cached['configuration'] != config:
            raise ValueError('Source/configuration mismatch on resume; use a new output directory')
        for path, digest in cached['bundle_sha256'].items():
            if sha(folder/'bundle'/path) != digest:
                raise ValueError('Cached bundle changed')
        print(f'RESUME {group} seed={seed} lr={lr}',flush=True)
        return cached
    folder.mkdir(parents=True,exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = CandidateScorer(**config['model'])
    anchor_state = None
    if anchor is not None:
        with np.load(HERE/anchor['run_dir']/'bundle/weights.npz',allow_pickle=False) as saved:
            anchor_state = {k:torch.from_numpy(saved[k].copy()) for k in saved.files}
        load_anchor(model,anchor_state)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=protocol['weight_decay'])
    train_x, train_y, val_x, val_y = tensors
    generator = torch.Generator().manual_seed(seed+10000)
    ids = [row['id'] for row in bank]
    best_key, best_epoch, best_state = (-1,-float('inf')),0,None
    history = []
    tick = time.perf_counter()
    model.eval()
    _, initial_train = measure(model,train_x,train_y)
    initial_logits, initial_val = measure(model,val_x,val_y)
    write_json(folder/'initial_metrics.json',{'train_eval':initial_train,'validation_eval':initial_val})
    if anchor is not None:
        initial_official = score_choices('validation',ids,initial_logits.argmax(1).tolist())
        assert initial_official['correct']==anchor['correct']
        anchor_logits = np.load(HERE/anchor['run_dir']/'validation-logits.npy',allow_pickle=False)
        np.testing.assert_array_equal(initial_logits.numpy(),anchor_logits)
        best_key = (initial_official['correct'],-initial_val['bce'])
        best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
    print(f'START {group} seed={seed} lr={lr}',flush=True)
    for epoch in range(1,protocol['max_epochs']+1):
        model.train()
        order = torch.randperm(len(train_x),generator=generator)
        accumulated = 0.
        for start in range(0,len(order),protocol['batch_size']):
            index = order[start:start+protocol['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            scores = model(train_x[index])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores,train_y[index])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'])
            optimizer.step()
            accumulated += loss.item()*len(index)
        model.eval()
        with torch.inference_mode():
            _, train_metrics = measure(model,train_x,train_y)
            logits, val_metrics = measure(model,val_x,val_y)
            val_loss = val_metrics['bce']
            score = score_choices('validation',ids,logits.argmax(1).tolist())
        history.append({'budget_epoch':epoch,'optimizer_steps':epoch*math.ceil(len(train_x)/protocol['batch_size']),
                        'training_draws':epoch*len(train_x),'eligible_pool_passes':epoch,
                        'stochastic_train_bce':accumulated/len(order),'validation_bce':val_loss,
                        'train_eval':train_metrics,'validation_eval':val_metrics,
                        'eval_bce_gap':val_metrics['bce']-train_metrics['bce'],
                        'eval_mixed_bce_gap':val_metrics['mixed_bce']-train_metrics['mixed_bce'],
                        'official_correct':score['correct'],'official_accuracy':score['accuracy'],
                        'elapsed_seconds':time.perf_counter()-tick})
        key = (score['correct'],-val_loss)
        if key > best_key:
            best_key,best_epoch = key,epoch
            best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f'EPOCH {group} seed={seed} {epoch}: val={score["accuracy"]:.3f} best={100*best_key[0]/len(bank):.3f}',flush=True)
        if epoch-best_epoch >= protocol['patience']:
            break
    training_seconds = time.perf_counter()-tick
    model.load_state_dict(best_state)
    model.eval()
    checks = semantic_checks(model,val_x,torch.zeros((len(val_x),0)))
    if model.frozen_base:
        checks['base_parameters_unchanged'] = all(torch.equal(value,anchor_state['base.'+key]) for key,value in model.base.state_dict().items())
        assert checks['base_parameters_unchanged']
    if name in ('finetune','residual_self'):
        assert checks['other_candidate_perturbation_max_abs_effect']<2e-5
    with torch.inference_mode():
        train_logits, train_metrics = measure(model,train_x,train_y)
        val_logits, val_metrics = measure(model,val_x,val_y)
        logits = val_logits.numpy()
    bundle = folder/'bundle'
    bundle.mkdir(exist_ok=True)
    write_json(bundle/'config.json',{'model':config['model']})
    np.savez(bundle/'weights.npz',**{k:v.numpy() for k,v in best_state.items()})
    np.savez(bundle/'preprocessing.npz',**prep)
    np.save(folder/'validation-logits.npy',logits,allow_pickle=False)
    write_json(folder/'history.json',history)
    write_json(folder/'checks.json',checks)
    correction_diagnostics = {}
    base_score_seconds = 0.0
    if anchor is not None:
        with torch.inference_mode():
            chunks = [model.components(chunk) for chunk in val_x.split(256)]
            parts = {k:torch.cat([p[k] for p in chunks]).numpy() for k in chunks[0]}
        np.savez(folder/'validation-components.npz',**parts)
        np.testing.assert_array_equal(parts['final'],logits)
        if model.frozen_base:
            np.testing.assert_array_equal(parts['base'],anchor_logits)
        write_json(folder/'base-only-predictions.json',{'protocol':PROTOCOL,'split':'validation',
            'ids':ids,'selected':parts['base'].argmax(1).tolist()})
        base_score_seconds = run_command([sys.executable,'-B',str(CODE/'score_predictions.py'),'--split','validation',
            '--predictions',str(folder/'base-only-predictions.json'),'--output',str(folder/'base-only-score.json')],folder/'base-only-score.log')
        base_official = json.loads((folder/'base-only-score.json').read_text())
        correction_diagnostics = {'base_only_accuracy':base_official['accuracy'],
            'anchor_accuracy':anchor['accuracy'],'anchor_run':anchor['run_dir'],
            'base_max_abs_logit_drift':float(np.max(np.abs(parts['base']-anchor_logits))),
            'mean_abs_applied_delta':float(np.abs(parts['applied_delta']).mean()),
            'mean_gate':float(parts['gate'].mean())}
        write_json(folder/'correction_diagnostics.json',correction_diagnostics)
    write_json(folder/'train-predictions.json',{'protocol':PROTOCOL,'split':'train',
        'ids':[r['id'] for r in train_bank],'selected':train_logits.argmax(1).tolist()})
    train_score_seconds = run_command([sys.executable,'-B',str(CODE/'score_predictions.py'),
        '--split','train','--predictions',str(folder/'train-predictions.json'),
        '--output',str(folder/'train-score.json')],folder/'train-score.log')
    train_official = json.loads((folder/'train-score.json').read_text())
    assert abs(train_official['accuracy']-train_metrics['selection_accuracy_diagnostic'])<1e-10
    size = asset_bytes(bundle)
    assert size <= 16*1024**2
    official,predict_seconds,score_seconds = official_evaluate(folder)
    choices = json.loads((folder/'validation-predictions.json').read_text())['selected']
    assert choices == logits.argmax(1).tolist() and official['correct'] == best_key[0]
    result = {'configuration':config,'group':group,'model':name,'seed':seed,
              'correction_diagnostics':correction_diagnostics,
              'base_only_scoring_wall_seconds':base_score_seconds,
              'anchor_training_seconds':0.0 if anchor is None else anchor['training_seconds'],
              'anchor_best_optimizer_steps':0 if anchor is None else anchor['best_optimizer_steps'],
              'hidden':protocol['hidden'],'dropout':protocol['dropout'],'weight_decay':protocol['weight_decay'],
              'train_accuracy':train_official['accuracy'],'accuracy_gap_pp':train_official['accuracy']-official['accuracy'],
              'train_eval':train_metrics,'validation_eval':val_metrics,
              'eval_bce_gap':val_metrics['bce']-train_metrics['bce'],
              'eval_mixed_bce_gap':val_metrics['mixed_bce']-train_metrics['mixed_bce'],
              'epoch10_metrics':history[9],'train_scoring_wall_seconds':train_score_seconds,
              'learning_rate':lr,'accuracy':official['accuracy'],'correct':official['correct'],'n':official['n'],
              'best_epoch':best_epoch,'epochs_run':len(history),'best_validation_loss':-best_key[1],
              'optimizer_steps':history[-1]['optimizer_steps'],'best_optimizer_steps':best_epoch*math.ceil(len(train_x)/protocol['batch_size']),
              'eligible_training_questions':len(train_x),'training_draws':history[-1]['training_draws'],
              'parameters':sum(p.numel() for p in model.parameters()),
              'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'training_seconds':training_seconds,
              'prediction_wall_seconds':predict_seconds,'scoring_wall_seconds':score_seconds,
              'validation_total_wall_seconds':predict_seconds+score_seconds,'uncompressed_asset_bytes':size,
              'bundle_sha256':{p.name:sha(p) for p in bundle.iterdir()},
              'run_dir':str(folder.relative_to(HERE)),'checks':checks}
    write_json(folder/'result.json',result)
    print(f'DONE {group} seed={seed} lr={lr} val={official["accuracy"]:.3f} epoch={best_epoch} train={training_seconds:.1f}s',flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=HERE/'results')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    protocol = json.loads((HERE/'protocol.json').read_text())
    sources = {p.name:sha(p) for p in HERE.iterdir() if p.suffix=='.py' and p.name not in ('summarize.py','verify_delivery.py')}
    prior_source_path = output/'source_hashes.json'
    if prior_source_path.exists() and json.loads(prior_source_path.read_text()) != sources:
        raise ValueError('Training sources changed; refuse to mix runs')
    write_json(prior_source_path,sources)
    processor = platform.processor()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            processor = winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    except (ImportError,OSError):
        pass
    write_json(output/'environment.json',{'python':sys.version,'executable':sys.executable,
        'platform':platform.platform(),'processor':processor,'numpy':np.__version__,'torch':torch.__version__,
        'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
        'device':'cpu','started_local':time.strftime('%Y-%m-%d %H:%M:%S')})
    verify_protected(output)
    run_command([sys.executable,'-B',str(HERE/'check_models.py')],output/'model_checks.log')
    run_command([sys.executable,'-B',str(CODE/'check_data.py')],output/'data_check.log')
    train_bank,train_arrays = load_inputs('train')
    val_bank,val_arrays = load_inputs('validation')
    train_y,val_y = labels_for('train',train_bank),labels_for('validation',val_bank)
    mean,std = fit_normalization(train_arrays['embeddings'])
    tensors = (torch.from_numpy((train_arrays['embeddings']-mean)/std),torch.from_numpy(train_y),
               torch.from_numpy((val_arrays['embeddings']-mean)/std),torch.from_numpy(val_y))
    np.save(output/'train-labels.npy',train_y.astype(np.int64),allow_pickle=False)
    np.save(output/'validation-labels.npy',val_y.astype(np.int64),allow_pickle=False)
    write_json(output/'data_summary.json',{'train_questions':len(train_bank),'validation_questions':len(val_bank),
        'train_correct_candidate_histogram':np.bincount(train_y.sum(1).astype(int),minlength=5).tolist(),
        'validation_correct_candidate_histogram':np.bincount(val_y.sum(1).astype(int),minlength=5).tolist(),
        'train_mixed':int((train_y.sum(1)<4).sum()),'validation_mixed':int((val_y.sum(1)<4).sum()),
        'draws_per_budget_epoch':len(train_bank),'updates_per_budget_epoch':math.ceil(len(train_bank)/protocol['batch_size']),
        'test_used_for_training_selection_or_scoring':False})
    all_runs,selected_runs,selections = [],[],{}
    anchors = {}
    for name in protocol['models']:
        staged = name not in ('independent','attention')
        search = [train_one(name,protocol['search_seed'],lr,protocol,tensors,val_bank,train_bank,
                            {'mean':mean,'std':std},output,sources,
                            anchors[protocol['search_seed']] if staged else None) for lr in protocol['learning_rates']]
        all_runs.extend(search)
        selected = max(search,key=lambda r:(r['correct'],-r['best_validation_loss']))
        selections[name] = selected['learning_rate']
        selected_runs.append(selected)
        if name=='independent':
            anchors[selected['seed']] = selected
        write_json(output/'selected_learning_rates.json',selections)
        for seed in protocol['seeds']:
            if seed==protocol['search_seed']:
                continue
            result = train_one(name,seed,selected['learning_rate'],protocol,tensors,val_bank,train_bank,
                               {'mean':mean,'std':std},output,sources,anchors[seed] if staged else None)
            all_runs.append(result)
            selected_runs.append(result)
            if name=='independent':
                anchors[seed] = result
        write_json(output/'all_runs.json',all_runs)
        write_json(output/'selected_runs.json',selected_runs)
    verify_protected(output)
    print(f'TRAINING COMPLETE: {len(all_runs)} runs, {len(selected_runs)} selected',flush=True)


if __name__ == '__main__':
    main()
