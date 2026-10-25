"""Independent-year replication on 2022 of the frozen audit-information candidate.

Frozen candidate and protocol: artifacts/experiments/20261016-audit-year2022-v1/训练前约定.md

Two stages, gate strictly before any fold3 metric is read:
  stage 1 (pilot)  : D32_U0, D32_UA, D128_U0, D128_UM, D128_UA x seeds 42/100
  gate             : fold1 proper BCE, D128_UA vs D32_U0, written to pilot_gate.json
  stage 2 (expand) : the same five arms x seeds 17/223/483/907, only if the gate passes
  evaluation       : four-fold persist + calibration recomputation for every trained run

Nothing frozen is modified: the 2020/2021 caches, the frozen evidence-v1 features and
the existing training scripts are only read. Cache and feature paths are redirected by
patching module attributes in this process, never by editing those modules.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import shutil
import sys
import time
from pathlib import Path

P = Path(__file__).resolve().parents[1]
E = P.parent
sys.path.insert(0, str(E / '.venv-literature/lib/python3.12/site-packages'))
sys.path.insert(0, str(P))

import numpy as np
import pandas as pd
import torch

import train_auxiliary_semantic as shared
import train_audit_information as audit_global
import train_audit_distributed as dist
from notice_pipeline.audit_distributed import AuditDistributed, CONTENT
from notice_pipeline.storage import file_digest

OUT = P / 'artifacts/experiments/20261016-audit-year2022-v1'
CACHE = OUT / 'cache'
FEATURES = OUT / 'features'
ROOT = OUT / 'distributed'
YEAR = 2022
SPECS = ((32, 'U0'), (32, 'UA'), (128, 'U0'), (128, 'UM'), (128, 'UA'))
PILOT_SEEDS = (42, 100)
EXPANSION_SEEDS = (17, 223, 483, 907)
THREADS = 4
write_json = shared.write_json


def install_redirected_loaders():
    """Point the shared pack loader at the 2022 cache without touching frozen code."""
    def load_pack(year, folds):
        meta = json.loads((CACHE / f'{year}.json').read_text())
        with np.load(CACHE / f'{year}.npz', allow_pickle=False) as cache:
            splits = []
            for f in folds:
                split = {'xs': [cache[f'{f}_x{i}'].copy() for i in range(3)],
                         **{k: cache[f'{f}_{k}'].copy() for k in ('industry', 'q', 'state', 'y', 'types')}}
                split['keys'] = pd.DataFrame({'Symbol': cache[f'{f}_symbol'].astype(str),
                                              'economic_year': cache[f'{f}_year'].copy()})
                splits.append(split)
        return dict(meta, splits=splits)
    shared.load_pack = load_pack
    audit_global.FEATURES = FEATURES


def pack_for(arm, folds=range(3)):
    return audit_global.make_pack(YEAR, arm, folds)


def train_one(formal, pack, budget, seed, arm, path, sources):
    path.mkdir(parents=True, exist_ok=False)
    model = AuditDistributed(formal, pack, budget, seed)
    initial_hash = shared.digest_state(model)
    start = time.perf_counter()
    with (path / 'training.log').open('w') as log, contextlib.redirect_stdout(log):
        runtime = formal.training(model, pack, path, seed)
    raw = {}
    for fold, split in enumerate(pack['splits']):
        raw[f'{fold}_a_logit'], raw[f'{fold}_b_logit'] = formal.logits(model, split)
    weights = formal.prev.cost_weights(pack['splits'][0])
    cal = shared.fit_calibration(formal, raw, pack, weights)
    assert cal == shared.fit_calibration(formal, raw, pack, weights)
    fresh = AuditDistributed(formal, pack, budget, seed)
    fresh.load_state_dict(torch.load(path / 'model.pt', weights_only=True)['state_dict'])
    for fold, split in enumerate(pack['splits']):
        a, b = formal.logits(fresh, split)
        np.testing.assert_array_equal(a, raw[f'{fold}_a_logit'])
        np.testing.assert_array_equal(b, raw[f'{fold}_b_logit'])
    np.savez_compressed(path / 'fit_predictions.npz', **raw)
    write_json(path / 'calibration.json', cal)
    record = {**runtime, 'year': YEAR, 'seed': seed, 'arm': arm, 'budget': budget,
              'tag': f'D{budget}_{arm}', 'initial_hash': initial_hash,
              'old_components_initial_hash': model.old_components_initial_hash,
              'audit_initial_hash': model.audit_initial_hash,
              'model_final_state_hash': shared.digest_state(model),
              'initial_parameter_hash': model.initial_parameter_hash,
              'content_dimensions': CONTENT[budget], 'total_message_floats': budget,
              'prediction_bytes': budget * 4, 'audit_seed': seed + 2000,
              'parameters': sum(v.numel() for v in model.parameters()),
              'model_file_sha256': file_digest(path / 'model.pt'), 'source_hashes': sources,
              'fit_seconds_including_checks': time.perf_counter() - start,
              'fit_folds': [0, 1, 2], 'selection_fold': 1, 'calibration_fold': 2,
              'fold3_loaded': False, 'fit_reload_exact': True, 'calibration_refit_exact': True}
    write_json(path / 'run.json', record)
    return record


def plain_parity_check(formal, seed, sources):
    """Full-training parity of D32_U0 against an unwired RiskModel on the same 2022 pack.

    2020/2021 had frozen formal S32 models to compare with; 2022 has none, so the
    unwired baseline is trained here on the identical augmented pack. The extra seven
    X3 columns are never read by the old executive indices, so equality is expected.
    """
    pack = pack_for('U0')
    root = ROOT / 'engineering' / f'parity_s{seed}'
    root.mkdir(parents=True, exist_ok=True)
    audit_path, plain_path = root / 'D32_U0', root / 'plain_S32'
    for path in (audit_path, plain_path):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    wired = AuditDistributed(formal, pack, 32, seed)
    plain = formal.RiskModel(pack, 32, 'Plain', seed)
    assert wired.original_s32_state_dict().keys() == plain.state_dict().keys()
    assert all(torch.equal(v, plain.state_dict()[k]) for k, v in wired.original_s32_state_dict().items())
    # formal.training seeds only its shuffle generator and takes the dropout stream
    # from the ambient global RNG state, which model construction leaves at seed_all(seed).
    # Re-seed before each call so the two runs are independent, exactly as separate
    # training invocations would be.
    formal.prev.seed_all(seed)
    with (audit_path / 'training.log').open('w') as log, contextlib.redirect_stdout(log):
        a_run = formal.training(wired, pack, audit_path, seed)
    formal.prev.seed_all(seed)
    with (plain_path / 'training.log').open('w') as log, contextlib.redirect_stdout(log):
        p_run = formal.training(plain, pack, plain_path, seed)
    assert a_run['selection_loss'] == p_run['selection_loss']
    assert a_run['best_epoch'] == p_run['best_epoch']
    a_state = torch.load(audit_path / 'model.pt', weights_only=True)['state_dict']
    p_state = torch.load(plain_path / 'model.pt', weights_only=True)['state_dict']
    audit_keys = {k for k in a_state if k.startswith('audit.')}
    assert set(a_state) == set(p_state) | audit_keys, sorted(set(a_state) ^ (set(p_state) | audit_keys))
    for key, value in p_state.items():
        assert torch.equal(value, a_state[key]), key
    assert all(torch.isfinite(v).all() for v in a_state.values())
    old_history = pd.read_csv(plain_path / 'history.csv')
    new_history = pd.read_csv(audit_path / 'history.csv')
    for field in ('epoch', 'selection_loss', 'train_task_loss'):
        np.testing.assert_array_equal(old_history[field], new_history[field])
    for fold, split in enumerate(pack['splits']):
        fa, fb = formal.logits(wired, split)
        pa, pb = formal.logits(plain, split)
        np.testing.assert_array_equal(fa, pa)
        np.testing.assert_array_equal(fb, pb)
    result = {'year': YEAR, 'seed': seed, 'frozen_model_available': False,
              'unwired_baseline_trained_on_same_pack': True,
              'initial_old_components_exact': True, 'final_state_dict_exact': True,
              'full_training_history_exact': True, 'fold0_1_2_raw_logits_exact': True,
              'selection_loss_exact': a_run['selection_loss'],
              'new_training_runs_in_this_check': 2, 'fold3_read': False}
    write_json(root / 'parity.json', result)
    return result


def pilot_gate(records):
    """Pre-registered expansion rule: fold1 proper BCE only, never fold3."""
    def mean_loss(budget, arm):
        rows = [r for r in records if r['budget'] == budget and r['arm'] == arm and r['seed'] in PILOT_SEEDS]
        assert len(rows) == len(PILOT_SEEDS), (budget, arm, len(rows))
        return float(np.mean([r['selection_loss'] for r in rows]))
    candidate, reference = mean_loss(128, 'UA'), mean_loss(32, 'U0')
    difference = candidate - reference
    passed = bool(difference < -1e-4)
    return {'fold': 1, 'fold_definition': '2022 split fold1 = economic_year 2020',
            'metric': 'proper BCE selection loss on fold1 (Pilot seeds 42/100)',
            'candidate': 'D128_UA', 'reference': 'D32_U0',
            'candidate_mean_selection_loss': candidate, 'reference_mean_selection_loss': reference,
            'difference': difference, 'threshold': -1e-4,
            'rule': 'expand only if mean(fold1 proper BCE, D128_UA) - mean(..., D32_U0) < -1e-4',
            'passed': passed, 'expand': passed, 'test_used': False}


def evaluate_after_gate(formal, records, gate_hash):
    packs, rows, checks = {}, [], []
    for rec in records:
        key = rec['arm']
        if key not in packs:
            packs[key] = pack_for(key, range(4))
        pack = packs[key]
        path = ROOT / f"D{rec['budget']}" / f"{rec['arm']}_s{rec['seed']}"
        factory = (lambda p=pack, b=rec['budget'], s=rec['seed']: AuditDistributed(formal, p, b, s))
        model = factory()
        model.load_state_dict(torch.load(path / 'model.pt', weights_only=True)['state_dict'])
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        checkdir = path / 'original_persist_check'
        checkdir.mkdir(exist_ok=True)
        shutil.copyfile(path / 'model.pt', checkdir / 'model.pt')
        with (checkdir / 'verification.log').open('w') as log, contextlib.redirect_stdout(log):
            _, row = formal.persist(model, factory, pack, checkdir, rec['tag'], YEAR, rec['seed'], rec)
        with np.load(checkdir / 'predictions.npz', allow_pickle=False) as validated, \
                np.load(path / 'fit_predictions.npz', allow_pickle=False) as fit:
            for k in fit.files:
                np.testing.assert_array_equal(fit[k], validated[k])
            raw = {k: validated[k].copy() for k in validated.files}
        assert json.loads((checkdir / 'calibration.json').read_text()) == json.loads((path / 'calibration.json').read_text())
        np.savez_compressed(path / 'predictions.npz', **raw)
        row.update(arm=rec['arm'], budget=rec['budget'], total_message_floats=rec['budget'],
                   prediction_bytes=rec['budget'] * 4, pilot_gate_sha256=gate_hash)
        write_json(path / 'test_metrics.json', row)
        rows.append(row)
        checks.append({'budget': rec['budget'], 'seed': rec['seed'], 'arm': rec['arm'],
                       'persist_four_fold_exact': True, 'calibration_exact': True, 'saved_model_finite': True})
    pd.DataFrame(rows).to_csv(ROOT / 'test_metrics.csv', index=False)
    write_json(ROOT / 'original_persist_verification.json',
               {'runs': len(records), 'checks': checks, 'additional_training_runs': 0,
                'test_only_after_fold1_gate': True})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--no-expand', action='store_true', help='pilot only, never train expansion seeds')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    install_redirected_loaders()
    formal = shared.load_loader().formal
    torch.set_num_threads(THREADS)

    sources = {str(p): file_digest(p) for p in (
        Path(__file__), OUT / '训练前约定.md', CACHE / f'{YEAR}.npz', CACHE / f'{YEAR}.json',
        FEATURES / f'audit_features_{YEAR}.npz', FEATURES / 'manifest.json',
        P / 'notice_pipeline/audit_distributed.py', P / 'notice_pipeline/audit_information.py',
        Path(shared.__file__), Path(audit_global.__file__), Path(dist.__file__),
        E / '财务年度配对修正/cache/2020.npz', E / '财务年度配对修正/cache/2021.npz')}
    write_json(ROOT / 'source_hashes.json', sources)

    parity = plain_parity_check(formal, PILOT_SEEDS[0], sources)
    print('PARITY ' + json.dumps(parity), flush=True)
    if args.check_only:
        print('ENGINEERING_CHECKS_PASS', flush=True)
        return

    packs = {arm: pack_for(arm) for _, arm in SPECS}
    records = []

    def run(seeds):
        for budget, arm in SPECS:
            for seed in seeds:
                path = ROOT / f'D{budget}' / f'{arm}_s{seed}'
                if (path / 'run.json').exists():
                    print('SKIP existing', path, flush=True)
                    continue
                rec = train_one(formal, packs[arm], budget, seed, arm, path, sources)
                records.append(rec)
                write_json(ROOT / 'runs.json', records)
                print(json.dumps({'budget': budget, 'seed': seed, 'arm': arm,
                                  'selection_loss': rec['selection_loss'], 'epochs': rec['epochs']}), flush=True)
        for budget in sorted({b for b, _ in SPECS}):
            for seed in seeds:
                hashes = {r['initial_hash'] for r in records if r['budget'] == budget and r['seed'] == seed}
                assert len(hashes) == 1, (budget, seed, hashes)

    run(PILOT_SEEDS)
    gate = pilot_gate(records)
    write_json(ROOT / 'pilot_gate.json', gate)
    print('PILOT_GATE ' + json.dumps(gate), flush=True)
    if gate['expand'] and not args.no_expand:
        run(EXPANSION_SEEDS)
    write_json(ROOT / 'training_complete.json', {'runs': len(records),
               'gate_sha256': file_digest(ROOT / 'pilot_gate.json'), 'test_opened': False,
               'expanded': bool(gate['expand'] and not args.no_expand)})
    rows = evaluate_after_gate(formal, records, file_digest(ROOT / 'pilot_gate.json'))
    after = {p: file_digest(Path(p)) for p in sources}
    assert after == sources, 'a protected frozen input changed'
    write_json(ROOT / 'verification.json', {
        'year': YEAR, 'runs': len(records), 'expanded': bool(gate['expand'] and not args.no_expand),
        'source_inputs_unchanged': True, 'four_fold_persist_and_calibration_exact': True,
        'three_arms_initial_weights_exact_within_capacity': True,
        'fold1_gate_written_before_fold3_read': True,
        'd32_u0_unwired_parity_seed': PILOT_SEEDS[0], 'payload_bytes_by_budget': {b: b * 4 for b in CONTENT},
        'rows': len(rows)})
    print('COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
