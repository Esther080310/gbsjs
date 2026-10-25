"""Train CMA: 6 seeds x 2 years."""
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cma_lib as C

recs = []
for year in C.YEARS:
    for seed in C.SEEDS:
        t = time.perf_counter()
        r = C.fit(year, seed)
        recs.append(r)
        print(json.dumps({'year': year, 'seed': seed, 'epochs': r['epochs'],
                          'params': r['parameters'], 'budget': r['budget_floats'],
                          'seconds': round(time.perf_counter() - t, 2)}, ensure_ascii=False), flush=True)
(HERE / 'results').mkdir(exist_ok=True)
(HERE / 'results' / 'runs.json').write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding='utf8')
print('TRAIN_DONE', len(recs))
