from pathlib import Path
import sys,json,hashlib,math
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'实验与代码/MID稳定性复检/20261022-v1'))
import check_mid as T
P,E,ST,K=T.P,T.E,T.ST,T.K
FA=K.FA
ARMS=('Main','S0','MID');GROUPS=('X1市场','X2财务','X3治理事件','审计调制分支');B=4

def write(p,x):K.write(p,x)

def take(s,ids):
    return {'xs':[v[ids].copy() for v in s['xs']],**{k:s[k][ids].copy() for k in ['industry','q','state']},'keys':s['keys'].iloc[ids].reset_index(drop=True),'noise':[v[ids].copy() for v in s['noise']]}

class Engine:
    def __init__(self,formal,pack):
        self.main=P.models(formal,pack);self.shared={};self.sources=[]
        for arm in ['S0','MID']:
            self.shared[arm]=[]
            for seed in T.SEEDS:
                if arm=='S0':m=ST.construct(formal,pack,'S0',seed);path=ST.OUT/f'S0_s{seed}/model.pt'
                else:m=T.construct(formal,pack,seed);path=T.OUT/f'models/MID_s{seed}/model.pt'
                m.load_state_dict(torch.load(path,weights_only=True));m.eval();self.shared[arm].append(m)
                self.sources.append(dict(arm=arm,seed=seed,path=str(path),sha256=T.M.sha(path)))
        for seed in self.main:
            path=P.D.CONTROL/f'artifacts/2020/FiLM-L_s{seed}/model.pt';self.sources.append(dict(arm='Main',seed=seed,path=str(path),sha256=T.M.sha(path)))

    @torch.no_grad()
    def predict(self,s,off=False):
        def run(m,mode,entry=None):
            hooks=[]
            if off:
                modules=[m.film] if mode=='main' else [h.film for h in m.heads]
                hooks=[module.register_forward_hook(lambda module,args,value:torch.zeros_like(value)) for module in modules]
            out=[]
            try:
                bs=256 if mode=='main' else 4096
                for start in range(0,len(s['industry']),bs):
                    sl=slice(start,start+bs)
                    args=([torch.tensor(v[sl]) for v in s['xs']],torch.tensor(s['industry'][sl]),torch.tensor(s['q'][sl]),torch.tensor(s['state'][sl]))
                    a,b=m(*args,noise=[torch.tensor(v[sl]) for v in s['noise']]) if mode=='mid' else m(*args)
                    out.append(np.column_stack([a.numpy(),b.numpy()]).astype('f8'))
            finally:
                for h in hooks:h.remove()
            out=np.concatenate(out)
            if entry is not None:
                w,c=entry['weights'],entry['calibration'];out[:,0]=c['overall']['slope']*(out[:,0]-np.log(w['overall']))+c['overall']['intercept']
                cb=c['types_shrink'];out[:,1:]=cb['slope']*(out[:,1:]-np.log(np.asarray(w['types'],dtype='f4')))+cb['intercept']+np.asarray(cb.get('delta',0))
            return out
        main=np.mean([run(m,'main',entry) for m,entry in self.main.values()],axis=0)[None,:,:]
        return {'Main':main,'S0':np.stack([run(m,'none') for m in self.shared['S0']]),'MID':np.stack([run(m,'mid') for m in self.shared['MID']])}

def percentile(values,reference):
    return np.stack([100*np.searchsorted(np.sort(ref),v,side='right')/len(ref) for ref,v in zip(reference,values)])

def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    formal=P.D.C.shared.load_loader().formal;pack=P.D.C.pack_for(2020,(0,1));tr,va=pack['splits']
    va['noise']=E.noise_for(va);tr['noise']=E.noise_for(tr)
    engine=Engine(formal,pack);base=engine.predict(va);off=engine.predict(va,True)
    # Reproduce the actual frozen primary predictor, including its calibration/ensemble convention.
    checkidx=np.arange(48);check=FA.take(va,checkidx);reference,_=FA.predict(engine.main,check)
    np.testing.assert_allclose(FA.sigmoid(base['Main'][0,checkidx,:]),reference[:,:11],atol=1e-6,rtol=1e-6)
    rng=np.random.default_rng(20261023)
    randomids=np.concatenate([np.sort(rng.choice(np.flatnonzero(va['industry']==i),64,replace=False)) for i in [2,8]])
    with np.load(ROOT/'实验与代码/归因与关键字段隐私/20261022-v1/results-with-decomposition/sample_predictions.npz') as z:positiveids=z['validation_indices']
    main_a=base['Main'][0,:,0];selected=np.zeros(len(main_a),bool);selected[np.argsort(-main_a,kind='stable')[:int(np.ceil(.1*len(main_a)))]]=True
    cases=[]
    for ind in [2,8]:
        for hit in [True,False]:
            pool=np.flatnonzero((va['industry']==ind)&(va['y']>0)&(selected==hit));order=pool[np.argsort(main_a[pool],kind='stable')];idx=int(order[len(order)//2])
            cases.append(dict(validation_index=idx,industry=chr(65+ind),case_kind='主模型名单内已记录正例' if hit else '主模型名单外已记录正例'))
    ids=np.array(sorted(set(randomids.tolist()+positiveids.tolist()+[c['validation_index'] for c in cases])));s=take(va,ids);positions={int(v):i for i,v in enumerate(ids)}
    sample=s['keys'].copy();sample['validation_index']=ids;sample['industry']=[chr(65+i) if i<19 else 'UNK' for i in s['industry']];sample['random_sample']=np.isin(ids,randomids);sample['positive_case_pool']=np.isin(ids,positiveids);sample.to_csv(OUT/'sample.csv',index=False)
    target=engine.predict(s);targetoff=engine.predict(s,True)
    donors=[];trsy=tr['keys'].Symbol.astype(str).to_numpy()
    for i in range(len(ids)):
        pool=np.flatnonzero((tr['industry']==s['industry'][i])&(trsy!=str(s['keys'].iloc[i].Symbol)))
        donors.append(rng.choice(pool,B,replace=False))
    donors=np.stack(donors);np.savez_compressed(OUT/'backgrounds.npz',validation_indices=ids,donor_train_indices=donors)
    values=[]
    for mask in range(16):
        hybrid=take(s,np.repeat(np.arange(len(ids)),B));d=donors.ravel()
        for node in range(3):
            if mask&(1<<node):continue
            if node<2:hybrid['xs'][node]=tr['xs'][node][d].copy()
            else:hybrid['xs'][2][:,:16]=tr['xs'][2][d,:16];hybrid['state']=tr['state'][d].copy()
            hybrid['q'][:,node]=tr['q'][d,node]
        pred=engine.predict(hybrid,off=not bool(mask&8));values.append({a:v.reshape(v.shape[0],len(ids),B,11).mean(2) for a,v in pred.items()})
    phis={};errors={};noderows=[]
    for arm in ARMS:
        phi=np.zeros((*values[0][arm].shape[:2],4,11))
        for j in range(4):
            for mask in range(16):
                if mask&(1<<j):continue
                k=mask.bit_count();weight=math.factorial(k)*math.factorial(3-k)/24
                phi[:,:,j,:]+=weight*(values[mask|(1<<j)][arm]-values[mask][arm])
        err=float(np.max(np.abs(phi.sum(2)-(values[15][arm]-values[0][arm]))));assert err<1e-10
        np.testing.assert_allclose(values[15][arm],target[arm],atol=2e-5,rtol=1e-5)
        phis[arm]=phi;errors[arm]=err
        np.savez_compressed(OUT/f'{arm}_node_attribution.npz',phi=phi,baseline=values[0][arm],prediction=target[arm],prediction_no_film=targetoff[arm])
        ix=np.array([positions[int(v)] for v in randomids]);raw=np.abs(phi[:,:,:,0][:,ix,:]).mean(1);shares=raw/raw.sum(1,keepdims=True)*100
        for rep in range(len(shares)):
            for j,label in enumerate(GROUPS):noderows.append(dict(arm=arm,replicate=rep,group=label,share=float(shares[rep,j]),mean_abs_logit_attribution=float(raw[rep,j])))
    pd.DataFrame(noderows).to_csv(OUT/'node_shares.csv',index=False)
    # Full-population audit-branch diagnostic, expressed against fixed original-score ECDF.
    audit=va['xs'][2][:,16:23];names=pack['names'][2][16:]
    write(OUT/'audit_input_dictionary.json',dict(feature_names=names,patterns=pd.DataFrame(audit,columns=names).value_counts().reset_index(name='rows').to_dict('records')))
    # Retain explicit one-hot names rather than silently guessing what numeric codes mean.
    auditgroups=[]
    for row in audit:
        if row[0]<.5:auditgroups.append('未匹配审计记录')
        else:
            active=[names[j] for j in range(1,7) if row[j]>.5];auditgroups.append(' / '.join(active) if active else '已匹配：其他模式')
    auditrows=[]
    for arm in ARMS:
        delta=percentile(base[arm][:,:,0],base[arm][:,:,0])-percentile(off[arm][:,:,0],base[arm][:,:,0])
        np.savez_compressed(OUT/f'{arm}_validation_scores.npz',full=base[arm],film_off=off[arm],audit_rank_shift=delta)
        for i in range(len(va['y'])):
            auditrows.append(dict(arm=arm,validation_index=i,audit_group=auditgroups[i],matched=bool(audit[i,0]>.5),rank_shift_median=float(np.median(delta[:,i])),rank_shift_mean=float(delta[:,i].mean()),known_violation=int(va['y'][i])))
    pd.DataFrame(auditrows).to_csv(OUT/'audit_rank_shift.csv',index=False)
    # Fixed field set: 13 finance fields, 2 public age fields, every numeric X3 event field.
    previous=json.loads((ROOT/'实验与代码/关键字段隐私检验/20261022-v1/results/field_sets.json').read_text())['sets']['key_financial']
    fields=[(1,n.split('::',1)[1]) for n in previous]+[(0,n) for n in ['mkt__company_age','mkt__listing_age']]+[(2,n) for n in pack['names'][2][:16] if 'missing::' not in n]
    fieldrows=[];casefields=[]
    for node,name in fields:
        j=pack['names'][node].index(name);ok=FA.observed(s,tr,pack['names'],node,j);okt=FA.observed(tr,s,pack['names'],node,j)
        rng=np.random.default_rng(int(hashlib.sha256(('field-'+name).encode()).hexdigest()[:8],16));hy=take(s,np.repeat(np.arange(len(ids)),B))
        for i in range(len(ids)):
            if not ok[i]:continue
            pool=np.flatnonzero(okt&(tr['industry']==s['industry'][i])&(trsy!=str(s['keys'].iloc[i].Symbol)))
            if not len(pool):ok[i]=False;continue
            d=rng.choice(pool,B,replace=len(pool)<B);hy['xs'][node][i*B:(i+1)*B,j]=tr['xs'][node][d,j]
        full=engine.predict(hy);noaudit=engine.predict(hy,True)
        for arm in ARMS:
            delta=target[arm]-full[arm].reshape(full[arm].shape[0],len(ids),B,11).mean(2)
            deltaoff=targetoff[arm]-noaudit[arm].reshape(noaudit[arm].shape[0],len(ids),B,11).mean(2)
            np.savez_compressed(OUT/f'{arm}_{name}_field.npz',delta=delta,delta_off=deltaoff,observed=ok)
            for group,mask,col in [('random',np.isin(ids,randomids),0),*[(ind+'_'+code,np.isin(ids,positiveids)&(s['industry']==ord(ind)-65)&(va['types'][ids,FA.CODES.index(code)]>0),1+FA.CODES.index(code)) for ind in ['C','I'] for code in FA.LABELS]]:
                mask=mask&ok
                if not mask.any():continue
                for rep in range(delta.shape[0]):fieldrows.append(dict(arm=arm,replicate=rep,group=group,node=f'X{node+1}',field=name,n_observed=int(mask.sum()),mean_abs_delta=float(np.abs(delta[rep,mask,col]).mean()),mean_signed_delta=float(delta[rep,mask,col].mean()),mean_abs_delta_off=float(np.abs(deltaoff[rep,mask,col]).mean())))
            for case in cases:
                i=positions[case['validation_index']]
                if ok[i]:casefields.append(dict(arm=arm,validation_index=case['validation_index'],field=name,node=f'X{node+1}',mean_signed_risk_delta=float(delta[:,i,0].mean()),mean_abs_risk_delta=float(np.abs(delta[:,i,0]).mean())))
        print('FIELD',name,flush=True)
    pd.DataFrame(fieldrows).to_csv(OUT/'field_sensitivity.csv',index=False);pd.DataFrame(casefields).to_csv(OUT/'case_fields.csv',index=False)
    caserows=[]
    for case in cases:
        ix=case['validation_index'];pos=positions[ix]
        for arm in ARMS:
            fullrank=percentile(base[arm][:,ix:ix+1,0],base[arm][:,:,0]).ravel();offrank=percentile(off[arm][:,ix:ix+1,0],base[arm][:,:,0]).ravel()
            top=np.argsort(-base[arm][:,ix,1:],axis=1,kind='stable')[:,:2];votes={code:int((top==j).any(1).sum()) for j,code in enumerate(FA.CODES)}
            caserows.append(dict(**case,arm=arm,symbol=str(va['keys'].iloc[ix].Symbol),economic_year=int(va['keys'].iloc[ix].economic_year),known_type_codes=';'.join(code for j,code in enumerate(FA.CODES) if va['types'][ix,j]>0),risk_percentile_median=float(np.median(fullrank)),risk_percentile_min=float(fullrank.min()),risk_percentile_max=float(fullrank.max()),film_off_percentile_median=float(np.median(offrank)),top2_votes=json.dumps(votes),n_models=len(fullrank)))
    pd.DataFrame(caserows).to_csv(OUT/'cases.csv',index=False)
    write(OUT/'manifest.json',dict(status='complete',random_sample=128,positive_case_pool=len(positiveids),distinct_explained_rows=len(ids),case_count=len(cases),backgrounds_per_row=B,fields=len(fields),folds_loaded=[0,1],economic_years=[2018],main_prediction_replay=True,additivity_max_errors=errors,model_sources=engine.sources,protocol_sha256=T.M.sha(HERE/'分析与图表约定.md'),source_sha256=T.M.sha(__file__),new_training=False))
    print('COMPLETE',errors,flush=True)

if __name__=='__main__':main()
