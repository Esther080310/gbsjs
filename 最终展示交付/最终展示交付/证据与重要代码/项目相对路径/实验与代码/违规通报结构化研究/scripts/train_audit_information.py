"""U0/UM/UA matched GlobalNN source information test, appended to old X3."""
from pathlib import Path
import argparse
import copy
import json
import shutil
import sys
import tomllib
import contextlib

P = Path(__file__).resolve().parents[1]
E = P.parent
sys.path.insert(0, str(E / '.venv-literature/lib/python3.12/site-packages'))
sys.path.insert(0, str(P))
import numpy as np
import torch
import pandas as pd
import train_auxiliary_semantic as shared
import train_financial_global as financial
from notice_pipeline.audit_information import load_audit_features
from notice_pipeline.storage import file_digest

FEATURES = P / 'artifacts/experiments/20261016-evidence-v1/audit_information/features'
write_json = shared.write_json


def make_pack(year, variant, folds=range(3), append_node=2):
    pack = shared.load_pack(year, folds)
    with np.load(FEATURES / f'audit_features_{year}.npz', allow_pickle=False) as feature:
        for fold, split in zip(folds, pack['splits']):
            for key, values in [('symbol', split['keys'].Symbol.to_numpy()), ('year', split['keys'].economic_year.to_numpy()),
                                ('y', split['y']), ('types', split['types'])]:
                np.testing.assert_array_equal(feature[f'{fold}_{key}'], values)
            extra = load_audit_features(FEATURES, year, fold, variant)
            assert extra.shape == (len(split['y']), 7) and extra.dtype == np.float32
            old = split['xs'][append_node]
            split['xs'][append_node] = np.concatenate([old, extra], axis=1)
            np.testing.assert_array_equal(split['xs'][append_node][:, :old.shape[1]], old)
        names = feature['feature_names'].astype(str).tolist()
    pack['names'][append_node] = pack['names'][append_node] + names
    pack['meta'][append_node]['selected_raw'] = pack['meta'][append_node]['selected_raw'] + names
    # pack.events deliberately remains the old exec indices. GlobalNN reads X3
    # directly; a future local encoder requires a separate approved factory.
    return pack


def matched_check(formal, year, variants, seed):
    old = shared.load_pack(year, range(3))
    hashes = []
    for arm, pack in variants.items():
        model = formal.RiskModel(pack, 0, 'Plain', seed)
        hashes.append(shared.digest_state(model))
        assert pack['events'] == old['events']
        for base, split in zip(old['splits'], pack['splits']):
            for key in ('q', 'industry', 'state', 'y', 'types'):
                np.testing.assert_array_equal(split[key], base[key])
            for node in (0, 1):
                np.testing.assert_array_equal(split['xs'][node], base['xs'][node])
            np.testing.assert_array_equal(split['xs'][2][:, :base['xs'][2].shape[1]], base['xs'][2])
            np.testing.assert_array_equal(split['keys'].to_numpy(), base['keys'].to_numpy())
    assert len(set(hashes)) == 1
    return {'year': year, 'seed': seed, 'same_initial_parameters': True,
            'initial_state_hash': hashes[0], 'old_all_X_q_labels_events_keys_exact': True,
            'X3_old_dimension': old['splits'][0]['xs'][2].shape[1], 'X3_added_dimensions': 7}


def evaluate_after_gate(formal, root, records):
    gate_hash = file_digest(root / 'pilot_gate.json')
    packs, metrics, checks = {}, [], []
    for rec in records:
        year, seed, arm = rec['year'], rec['seed'], rec['arm']
        if (year, arm) not in packs:
            packs[year, arm] = make_pack(year, arm, range(4))
        pack = packs[year, arm]
        path = root / str(year) / f'{arm}_s{seed}'
        factory = lambda: formal.RiskModel(pack, 0, 'Plain', seed)
        model = factory()
        model.load_state_dict(torch.load(path / 'model.pt', weights_only=True)['state_dict'])
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        checkdir = path / 'original_persist_check'
        checkdir.mkdir(exist_ok=True)
        shutil.copyfile(path / 'model.pt', checkdir / 'model.pt')
        with (checkdir / 'verification.log').open('w') as log, contextlib.redirect_stdout(log):
            _, row = formal.persist(model, factory, pack, checkdir, arm, year, seed, rec)
        with np.load(checkdir / 'predictions.npz', allow_pickle=False) as validated, np.load(path / 'fit_predictions.npz', allow_pickle=False) as fit:
            for key in fit.files:
                np.testing.assert_array_equal(fit[key], validated[key])
            raw = {key: validated[key].copy() for key in validated.files}
        assert json.loads((checkdir / 'calibration.json').read_text()) == json.loads((path / 'calibration.json').read_text())
        np.savez_compressed(path / 'predictions.npz', **raw)
        row.update(arm=arm, pilot_gate_sha256=gate_hash)
        write_json(path / 'test_metrics.json', row)
        metrics.append(row)
        checks.append({'year': year, 'seed': seed, 'arm': arm, 'four_fold_original_persist_exact': True,
                       'calibration_exact': True, 'saved_model_parameters_finite': True})
    assert file_digest(root / 'pilot_gate.json') == gate_hash
    pd.DataFrame(metrics).to_csv(root / 'test_metrics.csv', index=False)
    write_json(root / 'original_persist_verification.json', {'runs': len(records), 'checks': checks,
               'additional_training_runs': 0, 'gate_sha256': gate_hash})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=P / 'configs/audit_information_20261016.toml')
    args = parser.parse_args()
    cfg = tomllib.loads(args.config.read_text())
    assert cfg['study']['append_node'] == 2
    root = P / cfg['study']['output']
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'pilot_gate.json').exists()
    formal = shared.load_loader().formal
    torch.set_num_threads(cfg['training']['threads'])
    paths = [Path(__file__), args.config, Path(shared.__file__), Path(financial.__file__),
             P / 'notice_pipeline/audit_information.py', FEATURES / 'manifest.json',
             E / '节点表格网络与分段训练/economic_risk_residual.py']
    paths += [FEATURES / f'audit_features_{year}.npz' for year in cfg['study']['evaluation_years']]
    paths += [E / f'财务年度配对修正/cache/{year}{ext}' for year in cfg['study']['evaluation_years'] for ext in ('.npz', '.json')]
    protocol = P / 'docs/20261016_第二轮评估与审计信息实验协议.md'
    if protocol.exists():
        paths.append(protocol)
    before = {str(path): file_digest(path) for path in paths}
    write_json(root / 'source_hashes.json', before)
    write_json(root / 'protocol.json', cfg)
    packs = {year: {arm: make_pack(year, arm) for arm in cfg['study']['arms']} for year in cfg['study']['evaluation_years']}
    records, checks = [], []
    def run_seeds(seeds):
        for year, variants in packs.items():
            for seed in seeds:
                check = matched_check(formal, year, variants, seed)
                checks.append(check)
                for arm, pack in variants.items():
                    rec = financial.run_one(formal, pack, year, seed, arm, root / str(year) / f'{arm}_s{seed}', before)
                    rec.update(added_input_dimensions=7, append_node='X3', uses_semantic_auxiliary=False,
                               uses_new_amounts=False, same_initial_parameters_verified=True,
                               r_is_current_snapshot_match_not_date_gate=True,
                               global_gate_does_not_constrain_distributed_experiments=True)
                    assert rec['initial_hash'] == check['initial_state_hash']
                    write_json(root / str(year) / f'{arm}_s{seed}/run.json', rec)
                    records.append(rec)
                    write_json(root / 'runs.json', records)
                    print(json.dumps({'year': year, 'seed': seed, 'arm': arm,
                                      'selection_loss': rec['selection_loss'], 'epochs': rec['epochs']}), flush=True)
    run_seeds(cfg['study']['pilot_seeds'])
    decision = shared.gate(records, 'UA', 'UM', cfg['study']['evaluation_years'], cfg['study']['pilot_seeds'])
    expand = decision['passed']
    result = {'stage': 'initial_12_global_runs', 'decision': decision, 'expand_global_all_three_arms': expand,
              'expansion_seeds': cfg['study']['expansion_seeds'] if expand else [], 'test_metrics_opened': False,
              'gate_scope': 'GlobalNN seed expansion only; not a distributed research prerequisite',
              'UA_minus_U0_used_to_expand': False}
    write_json(root / 'pilot_gate.json', result)
    print('AUDIT_GLOBAL_GATE ' + json.dumps(result), flush=True)
    if expand:
        run_seeds(cfg['study']['expansion_seeds'])
    write_json(root / 'training_complete.json', {'runs': len(records), 'test_opened': False,
               'gate_sha256': file_digest(root / 'pilot_gate.json')})
    evaluate_after_gate(formal, root, records)
    assert before == {path: file_digest(Path(path)) for path in before}
    # Source features in turn protect the original alignment, source categories,
    # frozen sample keys and labels. Dates/indices are not model input columns.
    source_manifest = json.loads((FEATURES / 'manifest.json').read_text())
    assert source_manifest['source_sha256'] == {path: file_digest(Path(path)) for path in source_manifest['source_sha256']}
    write_json(root / 'verification.json', {'source_inputs_unchanged': True, 'original_persist_four_fold_exact': True,
               'matched_shape_and_initial_parameters': checks, 'test_only_after_gate': True, 'runs': len(records),
               'expanded': expand, 'old_q_unchanged': True, 'append_node': 'X3', 'additional_dimensions': 7,
               'unknown_matched_configurations': len(source_manifest['unknown_rows']),
               'global_gate_does_not_constrain_distributed_experiments': True})
    print('COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
