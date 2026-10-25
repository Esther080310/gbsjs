"""Freeze corrected data/models and replay existing decision-preserving codecs."""
from pathlib import Path
import runpy,importlib.util,json,time,random
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;ROOT=H.parent
R=runpy.run_path(str(ROOT/'通信机制收束验证/run.py'),run_name='final_codec_functions');S=R['S'];P=R['P'];torch=R['torch'];torch.set_num_threads(2)
sp=importlib.util.spec_from_file_location('final_finance_pipeline',ROOT/'财务年度配对修正/pipeline.py');pipe=importlib.util.module_from_spec(sp);sp.loader.exec_module(pipe)
formal=pipe.formal

def main():
    out=H/'results';out.mkdir(exist_ok=True);metrics=[];predict=[];manifest={};models=[]
    sources=[Path(__file__),ROOT/'财务年度配对修正/pipeline.py',pipe.FINANCE,ROOT/'通信机制收束验证/run.py',ROOT/'通信机制收束验证/crown.py',ROOT/'区间松弛诊断/stable_bounds.py',ROOT/'名单按需通信验证/experiment.py']
    for year in [2020,2021]:
        sources.extend([ROOT/f'财务年度配对修正/cache/{year}{e}' for e in ['.npz','.json']])
        for seed in [42,100,17,223,483,907]:
            for model in ['GlobalNN','S32']:
                src=ROOT/f'财务年度配对修正/results/{year}/{model}_s{seed}';sources.extend(src/n for n in ['model.pt','predictions.npz','calibration.json','run.json'])
    manifest={str(p):P.sha(p) for p in sources};P.save_json(H/'final_manifest.json',{'version':'FINAL-2026-10-14-Corrected-Snapshot2023','data':'财务年度配对修正/cache','label_cutoff':'2023-12-31','sample_unit':'company-occurrence-year','years':[2020,2021],'seeds':[42,100,17,223,483,907],'model':'corrected retrained S32 Plain','content_dims':[7,15,7],'quality_dims':[1,1,1],'model_budget':32,'target':'top10% list; selected Top2 sets; all selected type probabilities abs error<=.001','sources':manifest})
    for year in [2020,2021]:
        p=pipe.prepare(year,True);ss=p['splits'];w=P.cost_weights(ss[0]);s=ss[3]
        cache=np.load(ROOT/f'财务年度配对修正/cache/{year}.npz')
        for f,split in enumerate(ss):
            for j,x in enumerate(split['xs']):np.testing.assert_array_equal(x,cache[f'{f}_x{j}'])
            for k in ['y','types','q','industry']:np.testing.assert_array_equal(split[k],cache[f'{f}_{k}'])
        for seed in [42,100,17,223,483,907]:
            for kind,budget in [('GlobalNN',0),('S32',32)]:
                path=ROOT/f'财务年度配对修正/results/{year}/{kind}_s{seed}';model=formal.RiskModel(p,budget,'Plain',seed);model.load_state_dict(torch.load(path/'model.pt',weights_only=True)['state_dict']);model.eval();a,b=formal.logits(model,s);old=np.load(path/'predictions.npz');np.testing.assert_array_equal(a,old['3_a_logit']);np.testing.assert_array_equal(b,old['3_b_logit']);row=formal.exp.old.summary_row(year,kind,seed,s,dict(old),ss[0]);predict.append({'year':year,'seed':seed,'model':kind,**row})
                if kind=='GlobalNN':continue
                cal=json.loads((path/'calibration.json').read_text());train,_,_=S['extract'](model,ss[0]);ms,aa,bb=S['extract'](model,s);np.testing.assert_array_equal(aa,a);np.testing.assert_array_equal(bb,b);refa,refb=S['calibrate'](a,b,w,cal);np.testing.assert_array_equal(refa,old['a_logit']);np.testing.assert_array_equal(refb,old['b_logit']);steps=[np.maximum(abs(m).max(0)/127,np.finfo('f4').tiny).astype('f4') for m in train]
                for step,m in zip(steps,train):step[abs(m).max(0)==0]=1;step[-1]=np.float32(1/127)
                tags=['StableINT8','StableFP16','CrownINT8','FullFP16','FullFP32'];times={t:[] for t in tags};saved={}
                for rep in range(3):
                    order=tags.copy();random.Random(year+seed+rep*819).shuffle(order)
                    for tag in order:
                        if tag.startswith('Full'):result=R['full'](model,ms,s['industry'],steps,w,cal,refa,refb,tag[4:])
                        else:start,bound=R['METHODS'][tag];result=R['protocol'](model,ms,s['industry'],steps,w,cal,refa,refb,start,bound,.001)
                        record,frames,phases,pred,*_=result;times[tag].append(record['processing_seconds'])
                        if rep==0:
                            dest=out/f'{year}/s{seed}/{tag}';dest.mkdir(parents=True,exist_ok=True)
                            for name,wire in frames.items():(dest/name).write_bytes(wire)
                            np.savez_compressed(dest/'predictions.npz',**pred);P.save_json(dest/'phases.json',phases);saved[tag]=record
                        else:
                            assert saved[tag]['wire_bytes']==record['wire_bytes'];stored=np.load(out/f'{year}/s{seed}/{tag}/predictions.npz');np.testing.assert_array_equal(stored['a'],pred['a']);np.testing.assert_array_equal(stored['b'],pred['b'])
                for tag in tags:metrics.append({'year':year,'seed':seed,'method':tag,**saved[tag],'processing_seconds':float(np.median(times[tag]))})
                models.append({'year':year,'seed':seed,'source_model':str(path),'feature_cache_reproduced':True,'model_probabilities_reproduced':True});pd.DataFrame(metrics).to_csv(out/'communication_metrics.csv',index=False);pd.DataFrame(predict).to_csv(out/'prediction_metrics.csv',index=False);print('FINAL_REPLAY',year,seed,flush=True)
    assert all(P.sha(Path(p))==sha for p,sha in manifest.items());df=pd.DataFrame(metrics);df.groupby('method').mean(numeric_only=True).to_csv(out/'communication_summary.csv');pd.DataFrame(predict).groupby('model').mean(numeric_only=True).to_csv(out/'prediction_summary.csv');P.save_json(out/'verification.json',{'new_trainings':0,'corrected_model_versions':12,'global_versions':12,'protocol_configurations':len(df),'timing_replays':len(df)*3,'source_hashes_unchanged':True,'feature_cache_exact':True,'saved_logits_and_calibrated_probabilities_exact':True,'codec_actual_frames_roundtrip':True,'repeated_protocol_predictions_exact':True,'main_target':'epsilon=.001, top10% list and selected Top2 set','timing_scope':'in-memory encode/compress/decode/fuse/bounds/request; excludes local encoder, disk and actual network','model_inventory':models})
if __name__=='__main__':main()
