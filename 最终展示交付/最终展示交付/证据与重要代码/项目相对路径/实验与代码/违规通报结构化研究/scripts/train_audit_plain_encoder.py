"""One ordinary full-X3 MLP reference at the already selected capacity."""
from pathlib import Path
import argparse
import contextlib
from datetime import datetime, timezone
import json
import shutil
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
import train_audit_information as audit_global
from notice_pipeline.audit_distributed import AuditDistributed, CONTENT
from notice_pipeline.audit_plain_encoder import PlainAuditEncoder
from notice_pipeline.storage import file_digest

write_json = shared.write_json
BRANCH = P / 'artifacts/experiments/20261016-evidence-v1/audit_information/distributed'


def preflight(formal, packs, budget):
    checks = []
    for year, variants in packs.items():
        for seed in (42, 100):
            plain = [PlainAuditEncoder(formal, pack, budget, seed) for pack in variants.values()]
            assert len({shared.digest_state(m) for m in plain}) == 1
            branch = AuditDistributed(formal, variants['UA'], budget, seed)
            common = branch.state_dict()
            for name, value in plain[-1].state_dict().items():
                if not name.startswith('base.enc.2.'):
                    assert torch.equal(value, common[name]), name
            model = plain[-1]
            split = variants['UA']['splits'][0]
            data = formal.prev.tensor_split(split, 'cpu')
            model.eval()
            with torch.no_grad():
                model(*formal.prev.batch(data, slice(0, 16)))
                assert sum(m.shape[1] for m in model.messages) == budget
            checks.append({'year': year, 'seed': seed, 'input_dimensions': 23, 'output_dimensions': CONTENT[budget][2],
                'hidden': model.plain_hidden, 'branch_X3_parameters': model.branch_parameter_reference,
                'plain_X3_parameters': model.plain_node_parameters, 'parameter_difference': model.node_parameter_difference,
                'parameter_relative_difference': model.node_parameter_difference / model.branch_parameter_reference,
                'other_component_initial_parameters_equal_to_branch': True,
                'three_arm_initial_weights_exact': True, 'message_floats': budget, 'message_bytes': budget * 4,
                'seed_private_encoder': model.encoder_seed})
    return checks


def train_one(formal, pack, budget, year, seed, arm, path, hashes):
    path.mkdir(parents=True, exist_ok=False)
    model = PlainAuditEncoder(formal, pack, budget, seed)
    initial = shared.digest_state(model)
    start = time.perf_counter()
    with (path / 'training.log').open('w') as log, contextlib.redirect_stdout(log):
        runtime = formal.training(model, pack, path, seed)
    raw = {}
    for fold, split in enumerate(pack['splits']):
        raw[f'{fold}_a_logit'], raw[f'{fold}_b_logit'] = formal.logits(model, split)
    weights = formal.prev.cost_weights(pack['splits'][0])
    calibration = shared.fit_calibration(formal, raw, pack, weights)
    assert calibration == shared.fit_calibration(formal, raw, pack, weights)
    fresh = PlainAuditEncoder(formal, pack, budget, seed)
    fresh.load_state_dict(torch.load(path / 'model.pt', weights_only=True)['state_dict'])
    for fold, split in enumerate(pack['splits']):
        a, b = formal.logits(fresh, split)
        np.testing.assert_array_equal(a, raw[f'{fold}_a_logit'])
        np.testing.assert_array_equal(b, raw[f'{fold}_b_logit'])
    np.savez_compressed(path / 'fit_predictions.npz', **raw)
    write_json(path / 'calibration.json', calibration)
    rec = {**runtime, 'year': year, 'seed': seed, 'arm': arm, 'budget': budget,
           'encoder': 'one_plain_full_X3_mlp', 'hidden': model.plain_hidden,
           'branch_X3_parameters': model.branch_parameter_reference, 'plain_X3_parameters': model.plain_node_parameters,
           'node_parameter_difference': model.node_parameter_difference, 'encoder_seed': model.encoder_seed,
           'initial_hash': initial, 'initial_parameter_hash': model.initial_parameter_hash,
           'model_final_state_hash': shared.digest_state(model),
           'parameters': sum(p.numel() for p in model.parameters()), 'prediction_bytes': budget * 4,
           'content_dimensions': CONTENT[budget], 'source_hashes': hashes,
           'fit_folds': [0, 1, 2], 'selection_fold': 1, 'calibration_fold': 2, 'fold3_loaded': False,
           'fit_reload_exact': True, 'calibration_refit_exact': True,
           'fit_seconds_including_checks': time.perf_counter() - start,
           'runtime_context': 'sequential after branch training; postprocessing may overlap'}
    write_json(path / 'run.json', rec)
    return rec


def evaluate(formal, root, records):
    gate_hash = file_digest(root / 'pilot_gate.json')
    packs, metrics, checks = {}, [], []
    for rec in records:
        year, arm, budget, seed = rec['year'], rec['arm'], rec['budget'], rec['seed']
        if (year, arm) not in packs:
            packs[year, arm] = audit_global.make_pack(year, arm, range(4))
        pack = packs[year, arm]
        path = root / str(year) / f'{arm}_s{seed}'
        factory = lambda: PlainAuditEncoder(formal, pack, budget, seed)
        model = factory()
        model.load_state_dict(torch.load(path / 'model.pt', weights_only=True)['state_dict'])
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        checkdir = path / 'original_persist_check'
        checkdir.mkdir(exist_ok=True)
        shutil.copyfile(path / 'model.pt', checkdir / 'model.pt')
        with (checkdir / 'verification.log').open('w') as log, contextlib.redirect_stdout(log):
            _, row = formal.persist(model, factory, pack, checkdir, arm, year, seed, rec)
        with np.load(checkdir / 'predictions.npz', allow_pickle=False) as saved, np.load(path / 'fit_predictions.npz', allow_pickle=False) as fit:
            for key in fit.files:
                np.testing.assert_array_equal(fit[key], saved[key])
            raw = {key: saved[key].copy() for key in saved.files}
        assert json.loads((path / 'calibration.json').read_text()) == json.loads((checkdir / 'calibration.json').read_text())
        np.savez_compressed(path / 'predictions.npz', **raw)
        row.update(arm=arm, budget=budget, encoder='plain_full_X3', pilot_gate_sha256=gate_hash)
        write_json(path / 'test_metrics.json', row)
        metrics.append(row)
        checks.append({'year': year, 'seed': seed, 'arm': arm, 'four_fold_original_persist_exact': True,
                       'calibration_exact': True, 'saved_model_finite': True})
    pd.DataFrame(metrics).to_csv(root / 'test_metrics.csv', index=False)
    write_json(root / 'original_persist_verification.json', {'runs': len(records), 'checks': checks,
               'additional_training_runs': 0})
    assert gate_hash == file_digest(root / 'pilot_gate.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=P / 'configs/audit_plain_encoder_20261016.toml')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    cfg = tomllib.loads(args.config.read_text())
    budget = json.loads((BRANCH / 'capacity_selection.json').read_text())['selected_budget']
    root = P / cfg['study']['output']
    root.mkdir(parents=True, exist_ok=True)
    formal = shared.load_loader().formal
    torch.set_num_threads(cfg['training']['threads'])
    packs = {year: {arm: audit_global.make_pack(year, arm) for arm in cfg['study']['arms']} for year in cfg['study']['evaluation_years']}
    checks = preflight(formal, packs, budget)
    write_json(root / 'engineering_checks.json', checks)
    if args.check_only:
        print(json.dumps({'engineering_checks': 'passed', 'budget': budget, 'first': checks[0]}), flush=True)
        return
    assert (BRANCH / 'verification.json').exists(), 'Finish branch training before the added plain baseline.'
    assert not (root / 'pilot_gate.json').exists()
    paths = [Path(__file__), args.config, P / 'notice_pipeline/audit_plain_encoder.py', P / 'notice_pipeline/audit_distributed.py',
             Path(shared.__file__), Path(audit_global.__file__), BRANCH / 'capacity_selection.json', BRANCH / 'pilot_gate.json',
             BRANCH / 'runs.json', BRANCH / 'verification.json']
    paths += [audit_global.FEATURES / f'audit_features_{year}.npz' for year in cfg['study']['evaluation_years']]
    paths += [E / f'财务年度配对修正/cache/{year}{ext}' for year in cfg['study']['evaluation_years'] for ext in ('.npz', '.json')]
    predeclared = root / 'predeclared_protocol_v2.json'
    assert predeclared.exists(), 'Supplement must be frozen before branch fold3.'
    paths.append(predeclared)
    before = {str(path): file_digest(path) for path in paths}
    write_json(root / 'source_hashes.json', before)
    write_json(root / 'protocol.json', {'configuration': cfg, 'selected_budget': budget,
        'frozen_before_plain_training_utc': datetime.now(timezone.utc).isoformat(),
        'hypothesis': 'An ordinary matched-parameter full-X3 MLP may preserve audit information without the structured executive/audit branch; no structure sweep.',
        'original_capacity_and_gate_unchanged': True, 'parameter_checks': checks})
    branch_records = json.loads((BRANCH / 'runs.json').read_text())
    records = []
    def run_seeds(seeds):
        for year, variants in packs.items():
            for seed in seeds:
                hashes = []
                for arm, pack in variants.items():
                    rec = train_one(formal, pack, budget, year, seed, arm, root / str(year) / f'{arm}_s{seed}', before)
                    hashes.append(rec['initial_hash'])
                    records.append(rec)
                    write_json(root / 'runs.json', records)
                    print(json.dumps({'year': year, 'seed': seed, 'arm': arm, 'budget': budget,
                                      'hidden': rec['hidden'], 'selection_loss': rec['selection_loss'], 'epochs': rec['epochs']}), flush=True)
                assert len(set(hashes)) == 1
    run_seeds(cfg['study']['pilot_seeds'])
    compare = [dict(r, arm='plain_UA') for r in records if r['arm'] == 'UA']
    compare += [dict(r, arm='branch_UA') for r in branch_records if r['arm'] == 'UA' and r['budget'] == budget]
    decision = shared.gate(compare, 'plain_UA', 'branch_UA', cfg['study']['evaluation_years'], cfg['study']['pilot_seeds'])
    expand = decision['passed']
    gate = {'selected_capacity_fixed': budget, 'comparison': decision, 'expand_all_plain_U0_UM_UA': expand,
            'expansion_seeds': cfg['study']['expansion_seeds'] if expand else [],
            'test_metrics_opened': False, 'original_capacity_selection_changed': False,
            'hidden_width_search': False}
    write_json(root / 'pilot_gate.json', gate)
    print('PLAIN_ENCODER_GATE ' + json.dumps(gate), flush=True)
    if expand:
        run_seeds(cfg['study']['expansion_seeds'])
    write_json(root / 'training_complete.json', {'runs': len(records), 'test_opened': False, 'gate_sha256': file_digest(root / 'pilot_gate.json')})
    evaluate(formal, root, records)
    assert before == {path: file_digest(Path(path)) for path in before}
    write_json(root / 'verification.json', {'runs': len(records), 'source_inputs_unchanged': True,
        'initial_weights_same_within_plain_arms': True, 'other_components_initial_weights_equal_to_branch': True,
        'four_fold_original_persist_exact': True, 'test_only_after_plain_gate': True,
        'budget': budget, 'expanded': expand, 'hidden_width_search': False, 'original_capacity_selection_unchanged': True})
    print('COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
