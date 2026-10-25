"""Phase A: parameter-matched CMA-L on 2020/2021.  Phase B: CMA and CMA-L on 2022 (third year)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cma_lib as C  # noqa: E402


def run(year: int, arm: str, seeds=C.SEEDS) -> list:
    recs = []
    for s in seeds:
        t = time.perf_counter()
        r = C.fit(year, s, arm)
        recs.append(r)
        print(json.dumps({'year': year, 'arm': arm, 'seed': s, 'epochs': r['epochs'],
                          'params': r['parameters'], 'seconds': round(time.perf_counter() - t, 2)},
                         ensure_ascii=False), flush=True)
    return recs


def main() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else 'all'
    recs = []
    if phase in ('a', 'all'):
        recs += run(2020, 'CMA-L') + run(2021, 'CMA-L')
    if phase in ('b', 'all'):
        P = C.P
        sys.path.insert(0, str(P / 'scripts'))
        import train_audit_year2022 as fix22
        fix22.install_redirected_loaders()
        # 核对：2022 pack 的行数与冻结运行一致
        pack4 = C.pack_for(2022, range(4))
        n = len(pack4['splits'][3]['y'])
        frozen = json.loads((P / 'artifacts/experiments/20261016-audit-year2022-v1/distributed/D128/UA_s42/run.json').read_text())
        print(json.dumps({'2022_fold3_rows': n, 'x3_dim': int(pack4['splits'][0]['xs'][2].shape[1]),
                          'positives': int(pack4['splits'][3]['y'].sum())}, ensure_ascii=False), flush=True)
        assert n == 5199, n
        recs += run(2022, 'CMA') + run(2022, 'CMA-L')
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'runs_extend.json').write_text(
        json.dumps(recs, ensure_ascii=False, indent=2), encoding='utf8')
    print('DONE', len(recs))


if __name__ == '__main__':
    main()
