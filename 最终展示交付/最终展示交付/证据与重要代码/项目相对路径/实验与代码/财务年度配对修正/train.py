"""Matched retraining after annual key repair; original six-seed models frozen."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
import pipeline
formal=pipeline.formal;prev=formal.prev;OUT=HERE/'results'
YEARS=(2020,2021);SEEDS=(42,100,17,223,483,907)
MODELS={'GlobalNN':(0,'GlobalNN_Matched'),'S32':(32,'R32_Plain')}

def source(year,seed,model):
    folder='results_economic_risk' if seed in (42,100) else 'results_economic_risk_confirmation'
    return ROOT/'节点表格网络与分段训练'/folder/str(year)/f'{MODELS[model][1]}_s{seed}'

def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    verification=json.loads((HERE/'data/cache_verification.json').read_text());assert verification['old_cache_reproduced_exactly']
    sources=[HERE/f for f in ('train.py','pipeline.py','执行约定.md')]
    sources += [HERE/'data/x2_annual.csv.gz',HERE/'data/verification.json',HERE/'data/cache_verification.json',Path(formal.__file__)]
    sources += [HERE/f'cache/{year}{ext}' for year in YEARS for ext in ('.npz','.json')]
    for year in YEARS:
        for seed in SEEDS:
            for model in MODELS:
                sources += [source(year,seed,model)/name for name in ('model.pt','run.json','predictions.npz','calibration.json')]
    before={str(p):prev.sha(p) for p in sources};prev.save_json(OUT/'source_hashes.json',before)
    records=[];rows=[]
    for year in YEARS:
        old=pipeline.verify_old_cache(year);new=pipeline.prepare(year,True)
        for seed in SEEDS:
            for model,(budget,tag) in MODELS.items():
                oldpath=source(year,seed,model);oldrec=json.loads((oldpath/'run.json').read_text())
                factory=lambda:formal.RiskModel(new,budget,'Plain',seed)
                initial=factory();assert initial.initial_hash==oldrec['initial_hash']
                # Check the original frozen outputs using the unchanged old data.
                frozen=formal.RiskModel(old,budget,'Plain',seed)
                frozen.load_state_dict(torch.load(oldpath/'model.pt',weights_only=True)['state_dict'])
                raw_old=dict(np.load(oldpath/'predictions.npz'))
                a,b=formal.logits(frozen,old['splits'][3])
                np.testing.assert_array_equal(a,raw_old['3_a_logit']);np.testing.assert_array_equal(b,raw_old['3_b_logit'])
                oldrow=formal.exp.old.summary_row(year,model,seed,old['splits'][3],raw_old,old['splits'][0]);oldrow.update(data_version='Old',model=model);rows.append(oldrow)
                # Same checkpoint on corrected data, retaining its original calibration.
                a,b=formal.logits(frozen,new['splits'][3]);w=prev.cost_weights(old['splits'][0]);cal=json.loads((oldpath/'calibration.json').read_text())
                az=formal.exp.prob.apply_cal(a-np.log(w[0]),cal['overall']);bz=formal.exp.prob.apply_cal(b-np.log(w[1]),cal['types_shrink'])
                fixed={'3_a_logit':a,'3_b_logit':b,'a_logit':az,'b_logit':bz}
                row=formal.exp.old.summary_row(year,model,seed,new['splits'][3],fixed,new['splits'][0]);row.update(data_version='FrozenCorrected',model=model);rows.append(row)
                folder=OUT/str(year)/f'{model}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/'frozen_corrected_predictions.npz',**fixed)
                if not (folder/'completed.json').exists():
                    trained=factory();runtime=formal.training(trained,new,folder,seed)
                    rec,row=formal.persist(trained,factory,new,folder,model,year,seed,runtime)
                    prev.save_json(folder/'completed.json',{'record':rec,'metrics':row})
                else:
                    done=json.loads((folder/'completed.json').read_text());rec,row=done['record'],done['metrics']
                row.update(data_version='Corrected',model=model);rows.append(row);records.append(rec)
                pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);prev.save_json(OUT/'runs.json',records)
        print('Year completed',year,flush=True)
    assert len(records)==24 and len(rows)==72
    assert before=={str(p):prev.sha(p) for p in sources}
    prev.save_json(OUT/'verification.json',{'new_trainings':24,'original_checkpoints_unchanged':True,'all_sources_unchanged':True,'original_initializations_matched':True,'old_model_eval_logits_reproduced_exactly':True,'four_fold_reload_and_calibration_checked':True,'model_and_loss_and_selector_unchanged':True})
    print('MATCHED RETRAINING COMPLETE',flush=True)
if __name__=='__main__':main()
