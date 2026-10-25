"""Fixed A/B/C auxiliary experiment. Persist fold1 gates before reading fold3.

Run with the bundled Python and the existing .venv-literature site-packages.
This driver never edits frozen data or original training/model modules.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
import tomllib

P = Path(__file__).resolve().parents[1]
E = P.parent
# Existing dependencies only; the executable stays the bundled Python.
sys.path.insert(0, str(E / '.venv-literature/lib/python3.12/site-packages'))
sys.path.insert(0, str(P))
import numpy as np
import pandas as pd
import torch
from torch import nn
from notice_pipeline.alignment import make_alignment_head, combined_loss, clip_separately
from notice_pipeline.supervision import load_auxiliary_vectors, alignment_batch
from notice_pipeline.storage import file_digest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def gate(records, candidate, reference, years, seeds):
    lookup = {(r['year'], r['seed'], r['arm']): r['selection_loss'] for r in records}
    cells = [{'year': y, 'seed': seed, 'difference': lookup[y, seed, candidate] - lookup[y, seed, reference]}
             for y in years for seed in seeds]
    yearly = {str(y): float(np.mean([c['difference'] for c in cells if c['year'] == y])) for y in years}
    mean = float(np.mean([c['difference'] for c in cells]))
    return {'candidate': candidate, 'reference': reference, 'cells': cells,
            'year_mean_differences': yearly, 'four_cell_mean_difference': mean,
            'passed': all(v <= 0 for v in yearly.values()) and mean <= -0.001,
            'rule': 'each year mean <=0 and all-four mean <=-0.001', 'fold': 1,
            'test_used': False}


def load_loader():
    spec = importlib.util.spec_from_file_location('semantic_corrected_loader', E / '财务年度配对修正/pipeline.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    return loader


def load_pack(year, folds):
    folder = E / '财务年度配对修正/cache'
    meta = json.loads((folder / f'{year}.json').read_text())
    with np.load(folder / f'{year}.npz', allow_pickle=False) as cache:
        splits = []
        for f in folds:
            split = {'xs': [cache[f'{f}_x{i}'].copy() for i in range(3)],
                     **{k: cache[f'{f}_{k}'].copy() for k in ('industry', 'q', 'state', 'y', 'types')}}
            split['keys'] = pd.DataFrame({'Symbol': cache[f'{f}_symbol'].astype(str),
                                          'economic_year': cache[f'{f}_year'].copy()})
            splits.append(split)
    return dict(meta, splits=splits)


def fit_calibration(formal, raw, pack, weights):
    s = pack['splits'][2]
    cal = formal.prev.calibrate(raw['2_a_logit'], raw['2_b_logit'], s, weights)
    positive = s['y'] == 1
    cal['types_shrink'] = formal.exp.prob.fit_shrink(
        (raw['2_b_logit'] - np.log(weights[1]))[positive], s['types'][positive], cal['types_shared'])
    return cal


def digest_state(module):
    h = hashlib.sha256()
    for key, value in module.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_one(formal, pack, aux, year, seed, arm, path, cfg, source_hashes):
    path.mkdir(parents=True, exist_ok=False)
    prev = formal.prev
    model = formal.RiskModel(pack, 32, 'Plain', seed)
    init_hash = digest_state(model)
    rng_before = torch.random.get_rng_state().clone()
    head = make_alignment_head(seed + cfg['model']['head_seed_offset'], aux['dimension'])
    assert torch.equal(rng_before, torch.random.get_rng_state())
    head_hash = digest_state(head)
    weight = 0.0 if arm == 'A' else cfg['model']['auxiliary_weight']
    train_cfg = cfg['training']
    tensors = [prev.tensor_split(s, 'cpu') for s in pack['splits'][:2]]
    weights = prev.cost_weights(pack['splits'][0])
    aw, tw = torch.tensor(weights[0]), torch.tensor(weights[1])
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                                lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    # The original factory finishes by setting this same RNG state. Explicitly
    # resetting here also makes each arm independent of the preceding arm.
    prev.seed_all(seed)
    dropout_rng_hash = hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest()
    order_hash = hashlib.sha256()
    history = []
    best, stale = float('inf'), 0
    up = down = valup = 0
    started = time.perf_counter()
    for epoch in range(1, train_cfg['max_epochs'] + 1):
        model.train()
        head.train()
        order = torch.randperm(len(pack['splits'][0]['y']), generator=generator)
        order_hash.update(order.numpy().tobytes())
        total, aux_total, supervised_total = 0.0, 0.0, 0
        for start in range(0, len(order), train_cfg['batch_size']):
            ix = order[start:start + train_cfg['batch_size']]
            optimizer.zero_grad()
            a, b = model(*prev.batch(tensors[0], ix))
            y = tensors[0]['y'][ix]
            typ = nn.functional.binary_cross_entropy_with_logits(
                b, tensors[0]['types'][ix], pos_weight=tw, reduction='none').mean(1)
            task = nn.functional.binary_cross_entropy_with_logits(a, y, pos_weight=aw) + (typ * y).sum() / y.sum().clamp_min(1)
            main = task + model.penalty()
            vectors, mask = alignment_batch(aux, ix.numpy(), 'B' if arm == 'B' else 'C')
            targets, mask_tensor = torch.as_tensor(vectors), torch.as_tensor(mask)
            loss = combined_loss(main, model.messages, targets, mask_tensor, head, weight)
            assert torch.isfinite(loss)
            for message in model.messages:
                message.retain_grad()
            loss.backward()
            up += sum(m.numel() * 4 for m in model.messages)
            down += sum(m.grad[:, :d].numel() * 4 for m, d in zip(model.messages, model.content_dims))
            clip_separately(model, head, train_cfg['clip_backbone_norm'])
            optimizer.step()
            total += float(task.detach()) * len(ix)
            n = int(mask.sum())
            supervised_total += n
            if weight and n:
                aux_total += float(((loss - main) / weight).detach()) * n
        a, b = formal.logits(model, pack['splits'][1])
        score = prev.selection_loss(a, b, pack['splits'][1], weights)
        valup += len(a) * model.payload_bytes
        history.append({'epoch': epoch, 'selection_loss': score, 'train_task_loss': total / len(order),
                        'auxiliary_loss_per_available_sample': aux_total / supervised_total if supervised_total and weight else 0.,
                        'auxiliary_samples': supervised_total, 'seconds': time.perf_counter() - started,
                        'batch_order_cumulative_sha256': order_hash.hexdigest()})
        if score < best - train_cfg['min_improvement']:
            best, stale, best_epoch = score, 0, epoch
            torch.save({'state_dict': model.state_dict(), 'best_epoch': epoch}, path / 'model.pt')
            torch.save({'state_dict': head.state_dict(), 'best_epoch': epoch}, path / 'head.pt')
        else:
            stale += 1
        if stale >= train_cfg['patience']:
            break
    train_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(path / 'model.pt', weights_only=True)['state_dict'])
    head.load_state_dict(torch.load(path / 'head.pt', weights_only=True)['state_dict'])
    pd.DataFrame(history).to_csv(path / 'history.csv', index=False)
    # Only folds 0,1,2 are loaded until the pilot gate is durably written.
    raw = {}
    for f, split in enumerate(pack['splits']):
        raw[f'{f}_a_logit'], raw[f'{f}_b_logit'] = formal.logits(model, split)
    calibration_started = time.perf_counter()
    cal = fit_calibration(formal, raw, pack, weights)
    second_cal = fit_calibration(formal, raw, pack, weights)
    assert second_cal == cal
    calibration_seconds = time.perf_counter() - calibration_started
    fresh = formal.RiskModel(pack, 32, 'Plain', seed)
    fresh.load_state_dict(torch.load(path / 'model.pt', weights_only=True)['state_dict'])
    for f, split in enumerate(pack['splits']):
        a, b = formal.logits(fresh, split)
        np.testing.assert_array_equal(a, raw[f'{f}_a_logit'])
        np.testing.assert_array_equal(b, raw[f'{f}_b_logit'])
    np.savez_compressed(path / 'fit_predictions.npz', **raw)
    write_json(path / 'calibration.json', cal)
    rec = {'year': year, 'seed': seed, 'arm': arm, 'lambda': weight, 'time_protocol': cfg['study']['time_protocol'],
           'initial_hash': init_hash, 'initial_parameter_hash': model.initial_parameter_hash,
           'head_initial_hash': head_hash, 'head_seed': seed + cfg['model']['head_seed_offset'],
           'dropout_initial_rng_sha256': dropout_rng_hash, 'model_final_state_hash': digest_state(model),
           'model_file_sha256': file_digest(path / 'model.pt'), 'head_final_state_hash': digest_state(head),
           'epochs': len(history), 'best_epoch': best_epoch, 'selection_loss': best,
           'train_seconds': train_seconds, 'calibration_seconds_including_repeat_check': calibration_seconds,
           'fit_seconds_before_save': time.perf_counter() - started, 'selection_fold': 1, 'calibration_fold': 2,
           'test_fold_loaded': False, 'fit_folds': [0, 1, 2], 'fit_reload_exact': True,
           'calibration_refit_exact': True, 'auxiliary_mask_rows': int(aux['aux_mask'].sum()),
           'auxiliary_dimension': aux['dimension'], 'auxiliary_target_sha256': source_hashes['targets'],
           'auxiliary_embedding_sha256': source_hashes['embeddings'], 'source_hashes': source_hashes,
           'parameters': sum(p.numel() for p in model.parameters()),
           'head_parameters': sum(p.numel() for p in head.parameters()), 'prediction_bytes': model.payload_bytes,
           'train_forward_bytes': up, 'train_backward_bytes': down, 'epoch_validation_bytes': valup}
    write_json(path / 'run.json', rec)
    return rec


def assert_original_parity(formal, pack, rec, path):
    original = E / f"财务年度配对修正/results/{rec['year']}/S32_s{rec['seed']}"
    oldrec = json.loads((original / 'run.json').read_text())
    assert rec['initial_hash'] == oldrec['initial_hash']
    original_state = torch.load(original / 'model.pt', weights_only=True)['state_dict']
    current_state = torch.load(path / 'model.pt', weights_only=True)['state_dict']
    assert original_state.keys() == current_state.keys()
    assert all(torch.equal(original_state[k], current_state[k]) for k in original_state)
    oldhist, newhist = pd.read_csv(original / 'history.csv'), pd.read_csv(path / 'history.csv')
    for field in ('epoch', 'selection_loss', 'train_task_loss'):
        np.testing.assert_array_equal(newhist[field].to_numpy(), oldhist[field].to_numpy())
    assert rec['best_epoch'] == oldrec['best_epoch'] and rec['epochs'] == oldrec['epochs']
    assert rec['selection_loss'] == oldrec['selection_loss']
    with np.load(original / 'predictions.npz', allow_pickle=False) as old, np.load(path / 'fit_predictions.npz', allow_pickle=False) as new:
        for f in range(3):
            for name in ('a_logit', 'b_logit'):
                np.testing.assert_array_equal(new[f'{f}_{name}'], old[f'{f}_{name}'])
    assert json.loads((path / 'calibration.json').read_text()) == json.loads((original / 'calibration.json').read_text())
    check = {'passed': True, 'original_run': str(original), 'state_dict_exact': True,
             'epochs_and_selection_history_exact': True, 'train_task_loss_history_exact': True,
             'fold_0_1_2_logits_exact': True, 'fold2_calibration_exact': True,
             'fold3_not_read_for_parity': True}
    write_json(path / 'original_parity.json', check)
    return check


def verify_arm_randomness(root, year, seed, arms):
    runs = [json.loads((root / str(year) / f'{arm}_s{seed}/run.json').read_text()) for arm in arms]
    for field in ('initial_hash', 'head_initial_hash', 'dropout_initial_rng_sha256'):
        assert len({r[field] for r in runs}) == 1, field
    histories = [pd.read_csv(root / str(year) / f'{arm}_s{seed}/history.csv') for arm in arms]
    length = min(len(h) for h in histories)
    for h in histories[1:]:
        np.testing.assert_array_equal(histories[0]['batch_order_cumulative_sha256'][:length].to_numpy(),
                                      h['batch_order_cumulative_sha256'][:length].to_numpy())


def evaluate_after_gate(formal, root, records, gate_path):
    assert gate_path.exists()
    gate_hash = file_digest(gate_path)
    packs, rows = {}, []
    for rec in records:
        year = rec['year']
        if year not in packs:
            packs[year] = load_pack(year, range(4))
        pack = packs[year]
        path = root / str(year) / f"{rec['arm']}_s{rec['seed']}"
        model = formal.RiskModel(pack, 32, 'Plain', rec['seed'])
        model.load_state_dict(torch.load(path / 'model.pt', weights_only=True)['state_dict'])
        a, b = formal.logits(model, pack['splits'][3])
        weights = formal.prev.cost_weights(pack['splits'][0])
        cal = json.loads((path / 'calibration.json').read_text())
        az = formal.exp.prob.apply_cal(a - np.log(weights[0]), cal['overall'])
        bz = formal.exp.prob.apply_cal(b - np.log(weights[1]), cal['types_shrink'])
        raw = {'3_a_logit': a, '3_b_logit': b, 'a_logit': az, 'b_logit': bz}
        row = formal.exp.old.summary_row(year, rec['arm'], rec['seed'], pack['splits'][3], raw, pack['splits'][0])
        row.update(arm=rec['arm'], pilot_gate_sha256=gate_hash)
        np.savez_compressed(path / 'predictions.npz', **raw)
        write_json(path / 'test_metrics.json', row)
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / 'test_metrics.csv', index=False)
    # Metrics are reported after the frozen gate and cannot modify its decision.
    assert file_digest(gate_path) == gate_hash
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=P / 'configs/semantic_20261016.toml')
    args = parser.parse_args()
    cfg = tomllib.loads(args.config.read_text())
    root = P / cfg['study']['output']
    root.mkdir(parents=True, exist_ok=True)
    # Intentional explicit failure prevents mixing a rerun with old official outputs.
    assert not (root / 'pilot_gate.json').exists(), 'Gate already exists; use the saved experiment.'
    loader = load_loader()
    formal = loader.formal
    torch.set_num_threads(cfg['training']['threads'])
    directory = P / 'artifacts/followup/supervision_targets'
    paths = [args.config, Path(__file__), P / 'notice_pipeline/alignment.py', P / 'notice_pipeline/supervision.py',
             directory / 'targets.jsonl', directory / 'embeddings/encoder_manifest.json', directory / 'embeddings/embeddings.npz']
    paths += [E / f'财务年度配对修正/cache/{y}{ext}' for y in cfg['study']['evaluation_years'] for ext in ('.npz', '.json')]
    paths += [E / f'财务年度配对修正/results/{y}/S32_s{s}/{name}'
              for y in cfg['study']['evaluation_years'] for s in cfg['study']['pilot_seeds'] + cfg['study']['expansion_seeds']
              for name in ('model.pt', 'run.json', 'calibration.json', 'predictions.npz')]
    paths += [E / f'财务年度配对修正/{n}' for n in ('train.py', 'pipeline.py')]
    paths += [E / '节点表格网络与分段训练/economic_risk_residual.py']
    before = {str(path): file_digest(path) for path in paths}
    write_json(root / 'source_hashes.json', before)
    write_json(root / 'protocol.json', cfg)
    packs = {year: load_pack(year, range(3)) for year in cfg['study']['evaluation_years']}
    auxes = {year: load_auxiliary_vectors(directory, year, cfg['study']['time_protocol'], 0,
                pack['splits'][0]['keys'].Symbol.to_numpy(), pack['splits'][0]['keys'].economic_year.to_numpy())
             for year, pack in packs.items()}
    records = []
    def run_seeds(seeds):
        for year, pack in packs.items():
            hashes = {'cache': before[str(E / f'财务年度配对修正/cache/{year}.npz')],
                      'cache_metadata': before[str(E / f'财务年度配对修正/cache/{year}.json')],
                      'targets': before[str(directory / 'targets.jsonl')],
                      'embeddings': before[str(directory / 'embeddings/embeddings.npz')]}
            for seed in seeds:
                for arm in cfg['study']['arms']:
                    path = root / str(year) / f'{arm}_s{seed}'
                    rec = train_one(formal, pack, auxes[year], year, seed, arm, path, cfg, hashes)
                    if arm == 'A':
                        assert_original_parity(formal, pack, rec, path)
                        rec['original_parity_exact'] = True
                        write_json(path / 'run.json', rec)
                    records.append(rec)
                    write_json(root / 'runs.json', records)
                    print(json.dumps({'year': year, 'seed': seed, 'arm': arm, 'selection_loss': rec['selection_loss'],
                                      'epochs': rec['epochs'], 'A_parity': rec.get('original_parity_exact')}), flush=True)
                verify_arm_randomness(root, year, seed, cfg['study']['arms'])
    run_seeds(cfg['study']['pilot_seeds'])
    decisions = [gate(records, arm, 'A', cfg['study']['evaluation_years'], cfg['study']['pilot_seeds']) for arm in ('B', 'C')]
    local = gate(records, 'C', 'B', cfg['study']['evaluation_years'], cfg['study']['pilot_seeds'])
    expand = any(d['passed'] for d in decisions)
    result = {'stage': 'initial_12_runs', 'decisions': decisions, 'expand_all_abc': expand,
              'expansion_seeds': cfg['study']['expansion_seeds'] if expand else [],
              'C_vs_B_local_diagnostic_gate': local,
              'local_diagnostic_execution': 'not implemented in this A/B/C driver; report eligibility only',
              'test_metrics_opened': False, 'lambda_search_performed': False}
    # Durable gate artifact precedes any fold3 loading/prediction/metrics.
    write_json(root / 'pilot_gate.json', result)
    print('PILOT_GATE ' + json.dumps(result), flush=True)
    if expand:
        run_seeds(cfg['study']['expansion_seeds'])
    write_json(root / 'training_complete.json', {'runs': len(records), 'pilot_gate_sha256': file_digest(root / 'pilot_gate.json'),
               'all_A_parity_exact': all(r.get('original_parity_exact') for r in records if r['arm'] == 'A'),
               'test_opened': False})
    evaluate_after_gate(formal, root, records, root / 'pilot_gate.json')
    assert before == {str(path): file_digest(Path(path)) for path in before}
    write_json(root / 'verification.json', {'source_inputs_and_frozen_models_unchanged': True,
               'all_A_parity_exact': True, 'same_backbone_head_rng_and_batch_order_verified': True,
               'test_metrics_only_after_gate': True, 'formal_runs': len(records),
               'expanded': expand, 'gate_sha256': file_digest(root / 'pilot_gate.json'), 'lambda_search_performed': False})
    print('COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
