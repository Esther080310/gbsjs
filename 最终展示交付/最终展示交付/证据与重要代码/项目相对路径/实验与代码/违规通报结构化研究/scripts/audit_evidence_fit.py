"""Post-gate original-persist verification; never retrains a model."""
from pathlib import Path
import argparse
import contextlib
import json
import shutil
import sys

P = Path(__file__).resolve().parents[1]
E = P.parent
sys.path.insert(0, str(E / '.venv-literature/lib/python3.12/site-packages'))
sys.path.insert(0, str(P))
import numpy as np
import pandas as pd
import torch
import train_auxiliary_semantic as shared
import train_financial_global as financial
from notice_pipeline.supervision import load_auxiliary_vectors
from notice_pipeline.storage import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('route', choices=('semantic', 'financial'))
    args = parser.parse_args()
    root = P / 'artifacts/experiments/20261016-evidence-v1' / args.route
    assert (root / 'pilot_gate.json').exists()
    records = json.loads((root / 'runs.json').read_text())
    formal = shared.load_loader().formal
    torch.set_num_threads(4)
    packs, auxes, checks = {}, {}, []
    gate_hash = file_digest(root / 'pilot_gate.json')
    protocol = P / 'docs/20261016_数据后评估与实验协议.md'
    for rec in records:
        year, seed, arm = rec['year'], rec['seed'], rec['arm']
        key = (year, arm) if args.route == 'financial' else (year,)
        if key not in packs:
            packs[key] = financial.make_pack(year, arm, range(4)) if args.route == 'financial' else shared.load_pack(year, range(4))
        pack = packs[key]
        path = root / str(year) / f'{arm}_s{seed}'
        checkdir = path / 'original_persist_check'
        checkdir.mkdir(exist_ok=True)
        shutil.copyfile(path / 'model.pt', checkdir / 'model.pt')
        budget = 0 if args.route == 'financial' else 32
        factory = lambda: formal.RiskModel(pack, budget, 'Plain', seed)
        model = factory()
        model.load_state_dict(torch.load(path / 'model.pt', weights_only=True)['state_dict'])
        assert all(torch.isfinite(tensor).all() for tensor in model.state_dict().values())
        with (checkdir / 'verification.log').open('w') as log, contextlib.redirect_stdout(log):
            formal.persist(model, factory, pack, checkdir, arm, year, seed, rec)
        with np.load(checkdir / 'predictions.npz', allow_pickle=False) as verified, np.load(path / 'fit_predictions.npz', allow_pickle=False) as fit, np.load(path / 'predictions.npz', allow_pickle=False) as test:
            for name in fit.files:
                np.testing.assert_array_equal(verified[name], fit[name])
            for name in test.files:
                np.testing.assert_array_equal(verified[name], test[name])
            if args.route == 'semantic' and arm == 'A':
                original = E / f'财务年度配对修正/results/{year}/S32_s{seed}/predictions.npz'
                with np.load(original, allow_pickle=False) as old:
                    for name in old.files:
                        np.testing.assert_array_equal(verified[name], old[name])
                parity = json.loads((path / 'original_parity.json').read_text())
                parity.update(four_fold_raw_logits_exact=True, calibrated_fold3_logits_exact=True,
                              four_fold_parity_checked_after_gate=True)
                shared.write_json(path / 'original_parity.json', parity)
        assert json.loads((checkdir / 'calibration.json').read_text()) == json.loads((path / 'calibration.json').read_text())
        check = {'year': year, 'seed': seed, 'arm': arm, 'original_persist_four_fold_exact': True,
                 'original_persist_calibration_exact': True, 'saved_backbone_finite': True,
                 'post_gate_check': True, 'no_additional_training': True}
        if args.route == 'semantic':
            head_state = torch.load(path / 'head.pt', weights_only=True)['state_dict']
            assert all(torch.isfinite(v).all() for v in head_state.values())
            check.update(saved_head_finite=True, actual_message_dimensions=[8, 16, 8],
                         per_batch_auxiliary_loss_saved=False,
                         per_batch_gradient_finiteness_saved=False,
                         note='Driver predated full logging spec. Epoch losses/counts are saved; no batch loss or gradient audit is reconstructed.')
            if year not in auxes:
                s = pack['splits'][0]
                auxes[year] = load_auxiliary_vectors(P / 'artifacts/followup/supervision_targets', year,
                    'retrospective_snapshot2023', 0, s['keys'].Symbol.to_numpy(), s['keys'].economic_year.to_numpy())
            generator = torch.Generator().manual_seed(seed)
            rows = []
            for epoch in range(1, rec['epochs'] + 1):
                order = torch.randperm(len(pack['splits'][0]['y']), generator=generator).numpy()
                for batch_index, start in enumerate(range(0, len(order), 1024)):
                    rows.append({'epoch': epoch, 'batch_index': batch_index,
                                 'auxiliary_eligible_rows': int(auxes[year]['aux_mask'][order[start:start + 1024]].sum())})
            pd.DataFrame(rows).to_csv(path / 'deterministic_batch_mask_counts.csv', index=False)
            shared.write_json(path / 'logging_scope.json', check)
        checks.append(check)
    assert file_digest(root / 'pilot_gate.json') == gate_hash
    shared.write_json(root / 'original_persist_verification.json', {'runs': len(records), 'checks': checks,
        'original_persist_all_four_folds_exact': True, 'original_calibration_all_exact': True,
        'A_original_four_fold_parity_all_exact': args.route == 'semantic',
        'protocol_sha256_at_post_gate_audit': file_digest(protocol), 'protocol_landed_after_semantic_start': args.route == 'semantic',
        'pilot_gate_sha256': gate_hash, 'additional_training_runs': 0,
        'runtime_context': 'training or audits may overlap local semantic/financial runs; no speed claims'})
    print(json.dumps({'route': args.route, 'runs': len(records), 'original_persist_exact': True, 'additional_training': 0}))


if __name__ == '__main__':
    main()
