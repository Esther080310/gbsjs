from pathlib import Path
import json,sys,hashlib,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'实验与代码/隐私多维评价/20261022-v1'))
import evaluate_privacy_dimensions as M
K,P,E,ST=M.K,M.P,M.E,M.ST
PREVIOUS=M.OUT;SEEDS=(100,17,223,483,907)

def write(p,x):K.write(p,x)

def construct(formal,pack,seed):
    model=E.Shared(formal,pack,'mid',seed=seed)
    model.heads=nn.ModuleList([E.Fusion(E.A.FilmKeepX3(formal,pack,(s+seed-42)%(2**32))) for s in E.C.SEEDS])
    torch.manual_seed(seed)
    return model

def fit_model(formal,pack,seed):
    dest=OUT/'models'/f'MID_s{seed}';dest.mkdir(parents=True,exist_ok=True)
    model=construct(formal,pack,seed)
    if (dest/'run.json').exists():
        model.load_state_dict(torch.load(dest/'model.pt',weights_only=True));return model,json.loads((dest/'run.json').read_text())
    baseline=ST.construct(formal,pack,'S0',seed);midstate=model.state_dict()
    for name,value in baseline.state_dict().items():assert torch.equal(value,midstate[name]),name
    old=json.loads((ST.OUT/f'S0_s{seed}/run.json').read_text())
    assert ST.digest(baseline)==old['initial_hash']
    torch.manual_seed(seed)
    prev=formal.prev;ts=[prev.tensor_split(s,'cpu') for s in pack['splits']];weights=prev.cost_weights(pack['splits'][0]);aw,tw=map(torch.tensor,weights)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    batches=torch.Generator().manual_seed(seed);noisegen=torch.Generator().manual_seed(2026102200+seed)
    best=float('inf');stale=0;history=[];hashes=[];started=time.perf_counter()
    for epoch in range(1,31):
        model.train();order=torch.randperm(len(pack['splits'][0]['y']),generator=batches);hashes.append(hashlib.sha256(order.numpy().tobytes()).hexdigest());total=regtotal=0.
        for start in range(0,len(order),1024):
            ids=order[start:start+1024];batch=prev.batch(ts[0],ids);y=ts[0]['y'][ids]
            noise=[torch.randn((len(ids),d),generator=noisegen) for d in model.private_dimensions]
            opt.zero_grad(set_to_none=True);model(*batch,noise=noise)
            al=nn.functional.binary_cross_entropy_with_logits(model.all_any,y[None,:].expand_as(model.all_any),pos_weight=aw,reduction='none').mean()
            ty=ts[0]['types'][ids];tl=nn.functional.binary_cross_entropy_with_logits(model.all_type,ty[None,:,:].expand_as(model.all_type),pos_weight=tw,reduction='none').mean(2)
            task=al+(tl*y[None,:]).sum()/(y.sum().clamp_min(1)*6);reg=model.regularizer(batch[0]);loss=task+.1*reg
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=float(task.detach())*len(ids);regtotal+=float(reg.detach())*len(ids)
        val=E.extract(model,pack['splits'][1]);score=prev.selection_loss(val['a'],val['b'],pack['splits'][1],weights)
        history.append(dict(epoch=epoch,task=total/len(order),regularizer=regtotal/len(order),selection_loss=score))
        if score<best-1e-4:best=score;stale=0;bestepoch=epoch;torch.save(model.state_dict(),dest/'model.pt')
        else:
            stale+=1
            if stale>=6:break
        print('TRAIN MID',seed,epoch,round(score,4),flush=True)
    n=min(len(hashes),len(old['epoch_batch_hashes']));assert hashes[:n]==old['epoch_batch_hashes'][:n]
    model.load_state_dict(torch.load(dest/'model.pt',weights_only=True));pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
    record=dict(seed=seed,strength=.1,initial_common_parameters_exact=True,baseline_initial_hash=old['initial_hash'],common_batch_prefix_exact=True,epoch_batch_hashes=hashes,epochs=len(history),best_epoch=bestepoch,parameters=sum(p.numel() for p in model.parameters()),baseline_parameters=old['parameters'],noise_generator_separate=True,seconds=time.perf_counter()-started,model_sha256=M.sha(dest/'model.pt'))
    write(dest/'run.json',record);return model,record

def salt_extract(model,split,seed,salt):
    original=E.noise_for
    def noise(s):
        rows=[]
        for r in s['keys'].itertuples():
            key=f'{salt}|{seed}|{r.Symbol}|{r.economic_year}'
            n=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little');rows.append(np.random.default_rng(n).standard_normal(125).astype('f4'))
        return np.split(np.stack(rows),[16,104],1)
    try:E.noise_for=noise;return E.extract(model,split)
    finally:E.noise_for=original

def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);M.OUT=OUT
    # Redirect only new auxiliary attacks; previous result directories remain untouched.
    K.OUT=OUT/'auxiliary_recovery'
    formal=P.D.C.shared.load_loader().formal;pack=P.D.C.pack_for(2020,(0,1));tr,va=pack['splits']
    y0all,names=P.numeric_targets(tr,pack);y1all,_=P.numeric_targets(va,pack);nf,masks=K.numeric_masks(pack);assert names==nf
    prior=json.loads((PREVIOUS/'manifest.json').read_text());keys=prior['key_fields'];ix=np.array([names.index(n) for n in keys]);m0all,m1all=masks
    sy=tr['keys'].Symbol.astype(str).to_numpy();inner=np.array([int(hashlib.sha256(('attack-inner-20261022|'+v).encode()).hexdigest()[:8],16)%5==0 for v in sy]);fit=~inner
    meanall,scaleall=K.scale_targets(y0all[fit],m0all[fit]);mean=meanall[ix];scale=scaleall[ix]
    y0,y1=y0all[:,ix],y1all[:,ix];m0,m1=m0all[:,ix],m1all[:,ix]
    cuts=np.empty((20,2,13),dtype='f4')
    for r in json.loads((PREVIOUS/'attribute_definitions.json').read_text()):cuts[r['industry'],:,keys.index(r['field'])]=[r['low'],r['high']]
    a0=np.column_stack([y0<cuts[tr['industry'],0],y0>cuts[tr['industry'],1]]).astype('f4');a1=np.column_stack([y1<cuts[va['industry'],0],y1>cuts[va['industry'],1]]).astype('f4')
    with np.load(PREVIOUS/'context.npz') as z:
        for a,b in [(y1,z['y']),(m1,z['observed']),(scale,z['scale']),(a1,z['attribute_labels'])]:np.testing.assert_array_equal(a,b)
    with np.load(PREVIOUS/'attacks/public/selected.npz') as z:public={k:z[k] for k in z.files}
    candidate=pd.read_csv(ROOT/'实验与代码/归因与关键字段隐私/20261022-v1/results-with-decomposition/privacy_candidate_union.csv').drop_duplicates('field')
    auxidx=np.array([names.index(f'{r.node.lower()}::{r.field}') for r in candidate.itertuples()]);K.PUBLIC_AGE_INDICES=[names.index('x1::mkt__company_age'),names.index('x1::mkt__listing_age')]
    pub0=K.public_view(tr,pack);pub1=K.public_view(va,pack)
    write(OUT/'manifest.json',dict(seeds=list(SEEDS),strength=.1,folds_loaded=[0,1],validation_economic_years=[2018],target_parity_exact=True,previous_manifest_sha256=M.sha(PREVIOUS/'manifest.json'),protocol_sha256=M.sha(HERE/'复检前约定.md'),source_sha256=M.sha(__file__),key_fields=keys,new_predictor_training=True,new_predictor_count=5))
    np.savez_compressed(OUT/'context.npz',y=y1,observed=m1,scale=scale,attribute_labels=a1)
    fields=[];attrs=[];tasks=[];noisechecks=[];runs=[];started=time.perf_counter()
    for seed in SEEDS:
        tag=f'MID_s{seed}';model,run=fit_model(formal,pack,seed);runs.append(run);ex=[E.extract(model,s) for s in [tr,va]]
        task=P.EV.task_metrics(tr,va,ex[1]['a'],ex[1]['b']);tasks.append(dict(tag=tag,seed=seed,recall=task['recall_at_10pct'],ap=task['overall_average_precision'],top2_hit=task['conditional_top2_hit']))
        for salt in [73101,73102,73103]:
            v=salt_extract(model,va,seed,salt);t=P.EV.task_metrics(tr,va,v['a'],v['b']);noisechecks.append(dict(seed=seed,salt=salt,recall=t['recall_at_10pct'],ap=t['overall_average_precision']))
        views=[M.output_views(v,p) for v,p in zip(ex,[pub0,pub1])];solutions={'public':public}
        for view in ['top2','outputs','messages']:
            legacy=None
            if view=='messages':
                old0=np.column_stack([ex[0]['full_deployment'],pub0]);old1=np.column_stack([ex[1]['full_deployment'],pub1])
                K.attack_library(tag,old0,old1,y0all,y1all,m0all,m1all,fit,inner,auxidx,meanall,scaleall)
                legacy=M.legacy_candidates(tag,old0,old1,fit,inner,meanall,scaleall,ix)
            solutions[view]=M.library(tag+'_'+view,views[0][view],views[1][view],y0,y1,m0,m1,a0,a1,fit,inner,mean,scale,list(solutions.items()),legacy)
        score=np.load(ST.OUT/f'S0_s{seed}/validation_predictions.npz')['a_logit'];high=np.zeros(len(y1),bool);high[np.argsort(-score,kind='stable')[:int(np.ceil(.1*len(score)))]]=True
        groups={'all':np.ones(len(y1),bool),'C':va['industry']==2,'I':va['industry']==8,'S0_highrisk':high,'S0_other':~high}
        for ind in ['C','I']:
            for code in ['P2503','P2504','P2505']:groups[ind+'_'+code]=(va['industry']==ord(ind)-65)&(va['types'][:,K.FA.CODES.index(code)]>0)
        for view,s in solutions.items():
            for group,rows in groups.items():
                mm=m1&rows[:,None];n=mm.sum(0);pr=s['numeric_validation'];rm=M.numbers(y1,pr[0],mm,scale)[0];hits=[M.numbers(y1,pr[j+1],mm,scale)[j+1] for j in range(3)]
                am=np.column_stack([mm,mm]);av=M.aucs(a1,s['auc_validation'],am)
                for j,field in enumerate(keys):fields.append(dict(seed=seed,view=view,group=group,field=field,n_observed=int(n[j]),nrmse=float(rm[j]) if n[j] else None,**{f'success_{tol:g}':float(hits[k][j]) if n[j] else None for k,tol in enumerate(M.TOLS)}))
                for j in range(26):
                    ok=am[:,j];yy=a1[ok,j];support=ok.sum()>=30 and len(np.unique(yy))==2
                    ba=balanced_accuracy_score(yy,s['ba_validation'][ok,j]>=s['thresholds'][j]) if support else None
                    attrs.append(dict(seed=seed,view=view,group=group,field=keys[j%13],attribute='low' if j<13 else 'high',n_observed=int(ok.sum()),prevalence=float(yy.mean()) if ok.any() else None,auc=float(av[j]) if support and np.isfinite(av[j]) else None,balanced_accuracy=float(ba) if ba is not None else None,supported=bool(support)))
        for fn,rows in [('fields.csv',fields),('attributes.csv',attrs),('tasks.csv',tasks),('noise_task_repeats.csv',noisechecks)]:pd.DataFrame(rows).to_csv(OUT/fn,index=False)
        write(OUT/'pair_checks.json',runs);print('DONE MID',seed,round(task['recall_at_10pct'],4),round(task['overall_average_precision'],4),flush=True)
    write(OUT/'completion.json',dict(status='complete',seconds=time.perf_counter()-started,seeds=list(SEEDS),new_predictor_count=5,protocol_sha256=M.sha(HERE/'复检前约定.md'),source_sha256=M.sha(__file__)))

if __name__=='__main__':main()
