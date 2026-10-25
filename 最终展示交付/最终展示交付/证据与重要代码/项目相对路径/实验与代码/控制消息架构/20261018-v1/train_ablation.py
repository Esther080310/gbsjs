"""Train CMA-G (2020/2021) and FiLM-L (2020/2021/2022)."""
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cma_lib as C, ablation_lib as A

recs = []
def run(year, arm):
    for s in (42, 100, 17, 223, 483, 907):
        t = time.perf_counter()
        r = A.fit(year, arm, s)
        recs.append(r)
        print(json.dumps({'year': year, 'arm': arm, 'seed': s, 'epochs': r['epochs'],
                          'params': r['parameters'], 'budget': r['budget_floats'],
                          'seconds': round(time.perf_counter() - t, 2)}, ensure_ascii=False), flush=True)

for y in (2020, 2021):
    run(y, 'CMA-G'); run(y, 'FiLM-L')
sys.path.insert(0, str(C.P / 'scripts'))
import train_audit_year2022 as fix22
fix22.install_redirected_loaders()
run(2022, 'FiLM-L')
(HERE / 'results').mkdir(exist_ok=True)
(HERE / 'results' / 'runs_ablation.json').write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding='utf8')
print('DONE', len(recs))
