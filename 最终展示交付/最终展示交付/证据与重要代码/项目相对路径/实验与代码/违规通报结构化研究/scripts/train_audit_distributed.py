"""Fixed audit information/capacity pilot, selection entirely on fold1."""
from pathlib import Path
import argparse
import contextlib
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
from torch import nn
import train_auxiliary_semantic as shared
import train_audit_information as audit_global
from notice_pipeline.audit_distributed import AuditDistributed, CONTENT, FINANCE, EXECUTIVE, make_audit_branch
from notice_pipeline.storage import file_digest

write_json = shared.write_json
FEATURES = audit_global.FEATURES


def select_capacity(records, budgets=(32, 64, 128), years=(2020, 2021), seeds=(42, 100), tolerance=1e-4):
    scores = {}
    for budget in budgets:
        rows = [r for r in records if r['budget'] == budget and r['arm'] == 'UA' and r['year'] in years and r['seed'] in seeds]
        assert len(rows) == len(years) * len(seeds)
        scores[budget] = float(np.mean([r['selection_loss'] for r in rows]))
    minimum = min(scores.values())
    selected = min(b for b, value in scores.items() if value <= minimum + tolerance)
    return {'scores_UA_four_cell_mean': scores, 'selected_budget': selected, 'tie_tolerance': tolerance,
            'tie_rule': 'smallest budget within tolerance of minimum', 'fold': 1, 'test_used': False}


def engineering_checks(formal, root):
    oldpack = shared.load_pack(2020, range(3))
    pack = audit_global.make_pack(2020, 'U0')
    old = formal.RiskModel(oldpack, 32, 'Plain', 42)
    new = AuditDistributed(formal, pack, 32, 42)
    assert all(torch.equal(v, new.original_s32_state_dict()[k]) for k, v in old.state_dict().items())
    prev = formal.prev
    tensors = prev.tensor_split(pack['splits'][0], 'cpu')
    old_tensors = prev.tensor_split(oldpack['splits'][0], 'cpu')
    indices = torch.arange(64)
    aw, tw = map(torch.as_tensor, prev.cost_weights(pack['splits'][0]))
    def step(model, data):
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        prev.seed_all(123)
        a, b = model(*prev.batch(data, indices))
        y = data['y'][indices]
        typ = nn.functional.binary_cross_entropy_with_logits(b, data['types'][indices], pos_weight=tw, reduction='none').mean(1)
        loss = nn.functional.binary_cross_entropy_with_logits(a, y, pos_weight=aw) + (typ * y).sum() / y.sum().clamp_min(1) + model.penalty()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        return a.detach(), b.detach()
    oa, ob = step(old, old_tensors)
    na, nb = step(new, tensors)
    assert torch.equal(oa, na) and torch.equal(ob, nb)
    assert all(torch.equal(v, new.original_s32_state_dict()[k]) for k, v in old.state_dict().items())
    assert all(parameter.grad is None for parameter in new.audit.parameters())
    per_budget = []
    for budget in CONTENT:
        models = [AuditDistributed(formal, audit_global.make_pack(2020, arm), budget, 42) for arm in ('U0', 'UM', 'UA')]
        assert len({shared.digest_state(model) for model in models}) == 1
        model = models[-1]
        data = prev.tensor_split(audit_global.make_pack(2020, 'UA')['splits'][0], 'cpu')
        xs, industry, q, state = prev.batch(data, indices)
        xs = [x.clone() for x in xs]
        xs[2][:, -7:] = 0
        model.eval()
        with torch.no_grad():
            model(xs, industry, q, torch.zeros_like(state))
            zero = model.messages[2].clone()
            xs[2][:, -7] = 1
            xs[2][:, -6] = 1
            model(xs, industry, q, torch.zeros_like(state))
            matched = model.messages[2].clone()
            assert not torch.equal(zero[:, :-1], matched[:, :-1])
            assert sum(message.shape[1] for message in model.messages) == budget
            assert sum(message.numel() * message.element_size() for message in model.messages) // len(indices) == budget * 4
            xs[2][:, -7] = 0
            model(xs, industry, q, torch.zeros_like(state))
            assert torch.equal(model.messages[2], zero)
        rng = torch.random.get_rng_state().clone()
        make_audit_branch(42, CONTENT[budget][2])
        assert torch.equal(rng, torch.random.get_rng_state())
        per_budget.append({'budget': budget, 'content_dimensions': CONTENT[budget], 'finance_group_outputs': FINANCE[budget],
                           'executive_group_outputs': EXECUTIVE[budget], 'payload_bytes': budget * 4,
                           'arm_initial_states_exact': True, 'state0_audit_still_active': True,
                           'r0_branch_exact_zero': True, 'audit_init_preserves_rng': True})
    result = {'D32_U0_initial_old_components_exact': True, 'D32_U0_first_step_logits_exact': True,
              'D32_U0_first_update_old_components_exact': True, 'r0_audit_gradients_None': True,
              'budgets': per_budget, 'engineering_check_optimizer_updates': 2,
              'formal_effect_training_runs_in_this_check': 0}
    write_json(root / 'engineering_checks.json', result)
    return result


def train_one(formal, pack, budget, year, seed, arm, path, sources):
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
    record = {**runtime, 'year': year, 'seed': seed, 'arm': arm, 'budget': budget,
              'initial_hash': initial_hash, 'old_components_initial_hash': model.old_components_initial_hash,
              'audit_initial_hash': model.audit_initial_hash, 'model_final_state_hash': shared.digest_state(model),
              'initial_parameter_hash': model.initial_parameter_hash,
              'content_dimensions': CONTENT[budget], 'total_message_floats': budget,
              'prediction_bytes': budget * 4, 'audit_seed': seed + 2000,
              'parameters': sum(v.numel() for v in model.parameters()),
              'model_file_sha256': file_digest(path / 'model.pt'), 'source_hashes': sources,
              'fit_seconds_including_checks': time.perf_counter() - start,
              'fit_folds': [0, 1, 2], 'selection_fold': 1, 'calibration_fold': 2,
              'fold3_loaded': False, 'fit_reload_exact': True, 'calibration_refit_exact': True,
              'runtime_context': 'possible authorized parallel experiments; no speed comparison'}
    if budget == 32 and arm == 'U0':
        original = E / f'财务年度配对修正/results/{year}/S32_s{seed}'
        state = torch.load(original / 'model.pt', weights_only=True)['state_dict']
        assert state.keys() == model.original_s32_state_dict().keys()
        assert all(torch.equal(v, model.original_s32_state_dict()[k]) for k, v in state.items())
        old_history, new_history = pd.read_csv(original / 'history.csv'), pd.read_csv(path / 'history.csv')
        for field in ('epoch', 'selection_loss', 'train_task_loss'):
            np.testing.assert_array_equal(old_history[field], new_history[field])
        old_record = json.loads((original / 'run.json').read_text())
        assert runtime['selection_loss'] == old_record['selection_loss']
        assert runtime['best_epoch'] == old_record['best_epoch']
        with np.load(original / 'predictions.npz', allow_pickle=False) as old:
            for key, values in raw.items():
                np.testing.assert_array_equal(values, old[key])
        assert cal == json.loads((original / 'calibration.json').read_text())
        record['original_S32_training_and_fit_parity_exact'] = True
        write_json(path / 'original_parity.json', {'initial_old_components_exact': True,
            'final_old_components_exact': True, 'full_training_history_exact': True,
            'fold0_1_2_raw_logits_exact': True, 'fold2_calibration_exact': True,
            'fold3_not_read_yet': True})
    write_json(path / 'run.json', record)
    return record


def evaluate_after_gate(formal, root, records):
    gate_hash, selection_hash = file_digest(root / 'pilot_gate.json'), file_digest(root / 'capacity_selection.json')
    packs, rows, checks = {}, [], []
    for rec in records:
        year, budget, arm, seed = rec['year'], rec['budget'], rec['arm'], rec['seed']
        if (year, arm) not in packs:
            packs[year, arm] = audit_global.make_pack(year, arm, range(4))
        pack = packs[year, arm]
        path = root / f'D{budget}' / str(year) / f'{arm}_s{seed}'
        factory = lambda: AuditDistributed(formal, pack, budget, seed)
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
        if budget == 32 and arm == 'U0':
            with np.load(E / f'财务年度配对修正/results/{year}/S32_s{seed}/predictions.npz', allow_pickle=False) as old:
                for key in old.files:
                    np.testing.assert_array_equal(raw[key], old[key])
            parity = json.loads((path / 'original_parity.json').read_text())
            parity.update(four_fold_raw_logits_and_calibrated_outputs_exact=True, checked_after_gate=True)
            write_json(path / 'original_parity.json', parity)
        np.savez_compressed(path / 'predictions.npz', **raw)
        row.update(arm=arm, budget=budget, total_message_floats=budget, prediction_bytes=budget * 4,
                   pilot_gate_sha256=gate_hash, capacity_selection_sha256=selection_hash)
        write_json(path / 'test_metrics.json', row)
        rows.append(row)
        checks.append({'budget': budget, 'year': year, 'seed': seed, 'arm': arm,
                       'original_persist_four_fold_exact': True, 'calibration_exact': True,
                       'saved_model_finite': True})
    pd.DataFrame(rows).to_csv(root / 'test_metrics.csv', index=False)
    assert (file_digest(root / 'pilot_gate.json'), file_digest(root / 'capacity_selection.json')) == (gate_hash, selection_hash)
    write_json(root / 'original_persist_verification.json', {'runs': len(records), 'checks': checks,
               'additional_training_runs': 0, 'test_only_after_capacity_selection_and_gate': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=P / 'configs/audit_distributed_20261016.toml')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    cfg = tomllib.loads(args.config.read_text())
    root = P / cfg['study']['output']
    root.mkdir(parents=True, exist_ok=True)
    formal = shared.load_loader().formal
    torch.set_num_threads(cfg['training']['threads'])
    engineering_checks(formal, root)
    if args.check_only:
        print('ENGINEERING_CHECKS_PASS', flush=True)
        return
    assert not (root / 'pilot_gate.json').exists()
    paths = [Path(__file__), args.config, P / 'notice_pipeline/audit_distributed.py',
             P / 'notice_pipeline/audit_information.py', Path(shared.__file__), Path(audit_global.__file__), FEATURES / 'manifest.json',
             E / '节点表格网络与分段训练/economic_risk_residual.py']
    paths += [FEATURES / f'audit_features_{year}.npz' for year in cfg['study']['evaluation_years']]
    paths += [E / f'财务年度配对修正/cache/{year}{ext}' for year in cfg['study']['evaluation_years'] for ext in ('.npz', '.json')]
    paths += [E / f'财务年度配对修正/results/{year}/S32_s{seed}/{name}'
              for year in cfg['study']['evaluation_years'] for seed in cfg['study']['pilot_seeds'] + cfg['study']['expansion_seeds']
              for name in ('model.pt', 'run.json', 'calibration.json', 'predictions.npz')]
    before = {str(path): file_digest(path) for path in paths}
    write_json(root / 'source_hashes.json', before)
    write_json(root / 'protocol.json', cfg)
    packs = {year: {arm: audit_global.make_pack(year, arm) for arm in cfg['study']['arms']} for year in cfg['study']['evaluation_years']}
    records = []
    def run_seeds(budgets, seeds):
        for budget in budgets:
            for year, variants in packs.items():
                for seed in seeds:
                    init_hashes = []
                    for arm, pack in variants.items():
                        path = root / f'D{budget}' / str(year) / f'{arm}_s{seed}'
                        rec = train_one(formal, pack, budget, year, seed, arm, path, before)
                        init_hashes.append(rec['initial_hash'])
                        records.append(rec)
                        write_json(root / 'runs.json', records)
                        print(json.dumps({'budget': budget, 'year': year, 'seed': seed, 'arm': arm,
                            'selection_loss': rec['selection_loss'], 'epochs': rec['epochs'],
                            'original_S32_parity': rec.get('original_S32_training_and_fit_parity_exact')}), flush=True)
                    assert len(set(init_hashes)) == 1
    run_seeds(cfg['study']['budgets'], cfg['study']['pilot_seeds'])
    selected = select_capacity(records, cfg['study']['budgets'], cfg['study']['evaluation_years'], cfg['study']['pilot_seeds'], cfg['selection']['tie_tolerance'])
    chosen = selected['selected_budget']
    write_json(root / 'capacity_selection.json', selected)
    remapped = []
    for rec in records:
        if rec['budget'] == chosen and rec['arm'] == 'UA':
            remapped.append(dict(rec, arm='selected_UA'))
        if rec['budget'] == 32 and rec['arm'] == 'U0':
            remapped.append(dict(rec, arm='D32_U0'))
    decision = shared.gate(remapped, 'selected_UA', 'D32_U0', cfg['study']['evaluation_years'], cfg['study']['pilot_seeds'])
    expand = decision['passed']
    expansion_budgets = sorted({32, chosen}) if expand else []
    gate_result = {'capacity_selection': selected, 'decision': decision, 'expand': expand,
        'expansion_budgets': expansion_budgets, 'expansion_arms': cfg['study']['arms'] if expand else [],
        'expansion_seeds': cfg['study']['expansion_seeds'] if expand else [], 'test_metrics_opened': False,
        'global_improvement_used_as_prerequisite': False}
    write_json(root / 'pilot_gate.json', gate_result)
    print('DISTRIBUTED_GATE ' + json.dumps(gate_result), flush=True)
    if expand:
        run_seeds(expansion_budgets, cfg['study']['expansion_seeds'])
    write_json(root / 'training_complete.json', {'runs': len(records), 'test_opened': False,
        'gate_sha256': file_digest(root / 'pilot_gate.json'), 'selection_sha256': file_digest(root / 'capacity_selection.json')})
    evaluate_after_gate(formal, root, records)
    assert before == {path: file_digest(Path(path)) for path in before}
    write_json(root / 'verification.json', {'runs': len(records), 'source_inputs_and_frozen_models_unchanged': True,
        'original_persist_all_four_folds_exact': True, 'D32_U0_all_full_training_parity_exact': True,
        'three_arms_initial_weights_exact_within_capacity': True, 'test_after_capacity_selection_and_gate': True,
        'selected_budget': chosen, 'expanded_budgets': expansion_budgets, 'quality_dimensions_unchanged': 3,
        'payload_bytes_by_budget': {b: b * 4 for b in CONTENT}, 'global_gate_not_used': True})
    print('COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
