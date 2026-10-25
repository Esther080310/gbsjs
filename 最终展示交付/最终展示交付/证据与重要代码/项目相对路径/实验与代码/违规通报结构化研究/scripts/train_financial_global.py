"""Matched F0/FM/FV GlobalNN experiment; gate on FV-FM before fold3."""
from pathlib import Path
import argparse
import contextlib
import copy
import json
import sys
import time
import tomllib

P = Path(__file__).resolve().parents[1]
E = P.parent
sys.path.insert(0, str(E / '.venv-literature/lib/python3.12/site-packages'))
sys.path.insert(0, str(P))
import numpy as np
import pandas as pd
import torch
import train_auxiliary_semantic as shared
from notice_pipeline.financial_features import load_financial_features
from notice_pipeline.storage import file_digest

write_json = shared.write_json
FEATURES = P / 'artifacts/experiments/20261016-evidence-v1/financial_features'


def make_pack(year, variant, folds):
    pack = shared.load_pack(year, folds)
    with np.load(FEATURES / f'financial_features_{year}.npz', allow_pickle=False) as feats:
        feature_names = feats['feature_names'].astype(str).tolist()
        for f, split in zip(folds, pack['splits']):
            for key, values in [('symbol', split['keys'].Symbol.to_numpy()), ('year', split['keys'].economic_year.to_numpy()),
                                ('y', split['y']), ('types', split['types'])]:
                np.testing.assert_array_equal(feats[f'{f}_{key}'], values)
            extra = load_financial_features(FEATURES, year, f, variant)
            assert extra.shape == (len(split['y']), 16) and extra.dtype == np.float32
            old = split['xs'][1]
            split['xs'][1] = np.concatenate([old, extra], axis=1)
            np.testing.assert_array_equal(split['xs'][1][:, :old.shape[1]], old)
    pack['names'][1] = pack['names'][1] + feature_names
    # Names remain real source names. No S32 grouping or synthetic economic prefix.
    pack['meta'][1]['selected_raw'] = pack['meta'][1]['selected_raw'] + feature_names
    return pack


def verify_matched_packs(formal, year, packs, seed):
    models = {arm: formal.RiskModel(pack, 0, 'Plain', seed) for arm, pack in packs.items()}
    states = [shared.digest_state(model) for model in models.values()]
    assert len(set(states)) == 1
    baseline = shared.load_pack(year, range(3))
    for arm, pack in packs.items():
        for original, split in zip(baseline['splits'], pack['splits']):
            for field in ('q', 'state', 'industry', 'y', 'types'):
                np.testing.assert_array_equal(split[field], original[field])
            for node in (0, 2):
                np.testing.assert_array_equal(split['xs'][node], original['xs'][node])
            np.testing.assert_array_equal(split['keys'].to_numpy(), original['keys'].to_numpy())
        assert models[arm].messages is None
    return {'same_parameter_initial_values': True, 'initial_state_hash': states[0],
            'old_x1_x2_x3_q_keys_labels_unchanged': True, 'added_shape': 16,
            'old_global_reused': False, 'F0_retrained_for_matching_capacity': True}


def run_one(formal, pack, year, seed, arm, path, hashes):
    path.mkdir(parents=True, exist_ok=False)
    model = formal.RiskModel(pack, 0, 'Plain', seed)
    init_hash = shared.digest_state(model)
    started = time.perf_counter()
    with (path / 'training.log').open('w') as log, contextlib.redirect_stdout(log):
        runtime = formal.training(model, pack, path, seed)
    raw = {}
    for f, split in enumerate(pack['splits']):
        raw[f'{f}_a_logit'], raw[f'{f}_b_logit'] = formal.logits(model, split)
    weights = formal.prev.cost_weights(pack['splits'][0])
    cal = shared.fit_calibration(formal, raw, pack, weights)
    assert cal == shared.fit_calibration(formal, raw, pack, weights)
    fresh = formal.RiskModel(pack, 0, 'Plain', seed)
    fresh.load_state_dict(torch.load(path / 'model.pt', weights_only=True)['state_dict'])
    for f, split in enumerate(pack['splits']):
        a, b = formal.logits(fresh, split)
        np.testing.assert_array_equal(a, raw[f'{f}_a_logit'])
        np.testing.assert_array_equal(b, raw[f'{f}_b_logit'])
    np.savez_compressed(path / 'fit_predictions.npz', **raw)
    write_json(path / 'calibration.json', cal)
    rec = {**runtime, 'year': year, 'seed': seed, 'arm': arm,
           'initial_hash': init_hash, 'model_final_state_hash': shared.digest_state(model),
           'model_file_sha256': file_digest(path / 'model.pt'), 'source_hashes': hashes,
           'parameters': sum(v.numel() for v in model.parameters()), 'added_input_dimensions': 16,
           'old_q_unchanged': True, 'selection_fold': 1, 'calibration_fold': 2,
           'calibration_refit_exact': True, 'fit_reload_exact': True, 'fit_folds': [0, 1, 2],
           'fold3_loaded': False, 'fit_seconds_including_checks': time.perf_counter() - started,
           'runtime_context': 'possible overlap with separately authorized local semantic experiment',
           'old_frozen_GlobalNN_reused': False}
    write_json(path / 'run.json', rec)
    return rec


def test_after_gate(formal, root, records):
    gate_hash = file_digest(root / 'pilot_gate.json')
    packs, metrics = {}, []
    for rec in records:
        year, arm, seed = rec['year'], rec['arm'], rec['seed']
        key = (year, arm)
        if key not in packs:
            packs[key] = make_pack(year, arm, range(4))
        pack = packs[key]
        path = root / str(year) / f'{arm}_s{seed}'
        model = formal.RiskModel(pack, 0, 'Plain', seed)
        model.load_state_dict(torch.load(path / 'model.pt', weights_only=True)['state_dict'])
        a, b = formal.logits(model, pack['splits'][3])
        weights = formal.prev.cost_weights(pack['splits'][0])
        cal = json.loads((path / 'calibration.json').read_text())
        raw = {'3_a_logit': a, '3_b_logit': b,
               'a_logit': formal.exp.prob.apply_cal(a - np.log(weights[0]), cal['overall']),
               'b_logit': formal.exp.prob.apply_cal(b - np.log(weights[1]), cal['types_shrink'])}
        row = formal.exp.old.summary_row(year, arm, seed, pack['splits'][3], raw, pack['splits'][0])
        row.update(arm=arm, pilot_gate_sha256=gate_hash)
        np.savez_compressed(path / 'predictions.npz', **raw)
        write_json(path / 'test_metrics.json', row)
        metrics.append(row)
    assert file_digest(root / 'pilot_gate.json') == gate_hash
    pd.DataFrame(metrics).to_csv(root / 'test_metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=P / 'configs/financial_20261016.toml')
    args = parser.parse_args()
    cfg = tomllib.loads(args.config.read_text())
    root = P / cfg['study']['output']
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'pilot_gate.json').exists()
    formal = shared.load_loader().formal
    torch.set_num_threads(cfg['training']['threads'])
    paths = [Path(__file__), Path(shared.__file__), args.config, P / 'notice_pipeline/financial_features.py',
             P / 'docs/20261016_数据后评估与实验协议.md', E / '节点表格网络与分段训练/economic_risk_residual.py']
    paths += [FEATURES / f'financial_features_{year}.npz' for year in cfg['study']['evaluation_years']]
    paths += [E / f'财务年度配对修正/cache/{year}{ext}' for year in cfg['study']['evaluation_years'] for ext in ('.npz', '.json')]
    before = {str(path): file_digest(path) for path in paths}
    write_json(root / 'source_hashes.json', before)
    write_json(root / 'protocol.json', cfg)
    packs = {year: {arm: make_pack(year, arm, range(3)) for arm in cfg['study']['arms']}
             for year in cfg['study']['evaluation_years']}
    checks, records = [], []
    def run_seeds(seeds):
        for year, variants in packs.items():
            for seed in seeds:
                check = verify_matched_packs(formal, year, variants, seed)
                checks.append(dict(check, year=year, seed=seed))
                for arm, pack in variants.items():
                    rec = run_one(formal, pack, year, seed, arm, root / str(year) / f'{arm}_s{seed}', before)
                    assert rec['initial_hash'] == check['initial_state_hash']
                    records.append(rec)
                    write_json(root / 'runs.json', records)
                    print(json.dumps({'year': year, 'seed': seed, 'arm': arm,
                                      'selection_loss': rec['selection_loss'], 'epochs': rec['epochs']}), flush=True)
    run_seeds(cfg['study']['pilot_seeds'])
    decision = shared.gate(records, 'FV', 'FM', cfg['study']['evaluation_years'], cfg['study']['pilot_seeds'])
    expanded = decision['passed']
    result = {'stage': 'initial_12_runs', 'decision': decision, 'expand_all_F0_FM_FV': expanded,
              'expansion_seeds': cfg['study']['expansion_seeds'] if expanded else [],
              'test_metrics_opened': False, 'FV_minus_F0_used_to_expand': False}
    write_json(root / 'pilot_gate.json', result)
    print('FINANCIAL_GATE ' + json.dumps(result), flush=True)
    if expanded:
        run_seeds(cfg['study']['expansion_seeds'])
    write_json(root / 'training_complete.json', {'runs': len(records), 'test_opened': False,
               'gate_sha256': file_digest(root / 'pilot_gate.json')})
    test_after_gate(formal, root, records)
    assert before == {path: file_digest(Path(path)) for path in before}
    write_json(root / 'verification.json', {'source_inputs_unchanged': True,
               'matched_shape_and_initial_parameters': checks, 'test_only_after_gate': True,
               'runs': len(records), 'expanded': expanded, 'old_q_unchanged': True})
    print('COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
