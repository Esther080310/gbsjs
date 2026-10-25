import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cma_lib as C, ablation_lib as A
sys.path.insert(0, str(C.P / 'scripts'))
import train_audit_year2022 as fix22
fix22.install_redirected_loaders()
recs = []
for s in (42, 100, 17, 223, 483, 907):
    t = time.perf_counter()
    r = A.fit(2022, 'CMA-G', s)
    recs.append(r)
    print(json.dumps({'year': 2022, 'arm': 'CMA-G', 'seed': s, 'epochs': r['epochs'],
                      'params': r['parameters'], 'budget': r['budget_floats'],
                      'seconds': round(time.perf_counter() - t, 2)}, ensure_ascii=False), flush=True)
(HERE / 'results' / 'runs_cmag_2022.json').write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding='utf8')
print('DONE', len(recs))
