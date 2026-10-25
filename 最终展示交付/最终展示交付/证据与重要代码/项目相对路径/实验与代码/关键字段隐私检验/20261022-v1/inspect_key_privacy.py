"""Field-level recovery audit; frozen predictors, internally selected attackers."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'实验与代码/dCor稳定性复检/20261021-v1'))
import stability as ST
P,E=ST.P,ST.E
sys.path.insert(0,str(ROOT/'实验与代码/归因与关键字段隐私/20261022-v1'))
import field_attribution as FA
SEEDS=(100,17,223,483,907)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))

def masked_error(y,pred,mask,scale):
    ss=(((y.astype('f8')-pred.astype('f8'))/scale)**2*mask).sum(0)
    return np.sqrt(ss/np.maximum(mask.sum(0),1))

def scale_targets(y,mask):
    n=np.maximum(mask.sum(0),1)
    mean=(y.astype('f8')*mask).sum(0)/n
    var=(((y-mean)**2)*mask).sum(0)/n
    scale=np.sqrt(var);scale[scale<1e-7]=1
    return mean.astype('f4'),scale.astype('f4')

def numeric_masks(pack):
    fields=[];parts=[[],[]]
    tr,va=pack['splits']
    for node,names in enumerate(pack['names']):
        for j,name in enumerate(names[:16] if node==2 else names):
            if 'missing::' in name or '=' in name:continue
            fields.append(f'x{node+1}::{name}')
            parts[0].append(FA.observed(tr,va,pack['names'],node,j))
            parts[1].append(FA.observed(va,tr,pack['names'],node,j))
    return fields,[np.column_stack(p) for p in parts]

def public_view(s,pack):
    age=[pack['names'][0].index(n) for n in ['mkt__company_age','mkt__listing_age']]
    return np.concatenate([np.eye(20,dtype='f4')[s['industry']],s['xs'][2][:,16:23],s['q'],s['state'],s['xs'][0][:,age]],1).astype('f4')

def attack_library(tag,x0,x1,y0,y1,m0,m1,fit,inner,key_idx,mean,scale):
    dest=OUT/'attackers'/tag;dest.mkdir(parents=True,exist_ok=True)
    selected_path=dest/'selected_predictions.npz'
    if selected_path.exists():
        with np.load(selected_path) as z:return z['pred'],json.loads((dest/'selection.json').read_text())
    allidx=np.arange(y0.shape[1]);candidates=[]
    for alpha in (.1,1.,10.,100.):
        ridge=P.A.RidgeAttack(alpha).fit(x0[fit],y0[fit])
        score=masked_error(y0[inner],ridge.predict(x0[inner]),m0[inner],scale)
        candidates.append((f'ridge_{alpha:g}',allidx,score,ridge.predict(x1)))
    xs=P.A.Standardizer.fit(x0[fit]);xf=torch.from_numpy(xs.apply(x0[fit]));xi=torch.from_numpy(xs.apply(x0[inner]));xv=torch.from_numpy(xs.apply(x1))
    for label,idx in [('deep_all',allidx),('deep_key',key_idx)]:
        path=dest/f'{label}.npz';checkpoint=dest/f'{label}.pt'
        if path.exists():
            with np.load(path) as z:candidates.append((label,z['idx'],z['inner_errors'],z['pred']))
            continue
        P.A.seed_all(20261022)
        net=nn.Sequential(nn.Linear(x0.shape[1],512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,len(idx)))
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        yf=torch.from_numpy(((y0[fit][:,idx]-mean[idx])/scale[idx]).astype('f4'))
        mf=torch.from_numpy(m0[fit][:,idx].astype('f4'))
        loader=DataLoader(TensorDataset(xf,yf,mf),batch_size=1024,shuffle=True,generator=torch.Generator().manual_seed(20261022))
        history=[];best=np.inf;stale=0;best_epoch=0
        for epoch in range(1,81):
            net.train()
            for xb,yb,mb in loader:
                opt.zero_grad(set_to_none=True)
                loss=((net(xb)-yb).square()*mb).sum()/mb.sum().clamp_min(1)
                assert torch.isfinite(loss)
                loss.backward();opt.step()
            net.eval()
            with torch.no_grad():pr=torch.cat([net(v) for v in xi.split(4096)]).numpy()*scale[idx]+mean[idx]
            errors=masked_error(y0[inner][:,idx],pr,m0[inner][:,idx],scale[idx]);score=float(errors.mean())
            history.append(dict(epoch=epoch,inner_nrmse=score))
            if score<best-1e-5:
                best=score;stale=0;best_epoch=epoch;torch.save(net.state_dict(),checkpoint)
            else:
                stale+=1
                if stale>=8:break
        net.load_state_dict(torch.load(checkpoint,weights_only=True));net.eval()
        with torch.no_grad():
            ip=torch.cat([net(v) for v in xi.split(4096)]).numpy()*scale[idx]+mean[idx]
            vp=torch.cat([net(v) for v in xv.split(4096)]).numpy()*scale[idx]+mean[idx]
        errors=masked_error(y0[inner][:,idx],ip,m0[inner][:,idx],scale[idx])
        assert abs(float(errors.mean())-best)<1e-6
        np.savez_compressed(path,idx=idx,inner_errors=errors,pred=vp)
        write(dest/f'{label}_training.json',dict(best_epoch=best_epoch,epochs=len(history),history=history,checkpoint_sha256=digest(checkpoint)))
        candidates.append((label,idx,errors,vp));print('ATTACK',tag,label,best_epoch,round(best,4),flush=True)
    # A constant attack is also available; weak extra inputs cannot force a worse choice.
    const=np.broadcast_to(mean,y1.shape).copy()
    candidates.append(('constant',allidx,masked_error(y0[inner],np.broadcast_to(mean,y0[inner].shape),m0[inner],scale),const))
    # Ages are explicitly known; copy them rather than attributing their recovery to a message.
    for j in key_idx:
        if j not in PUBLIC_AGE_INDICES:continue
        candidates.append(('known_public_value',np.array([j]),np.array([0.]),y1[:,j:j+1].copy()))
    best=np.full(y0.shape[1],np.inf);pred=np.empty_like(y1);chosen=['']*y0.shape[1]
    for label,idx,errors,pr in candidates:
        for col,j in enumerate(idx):
            if errors[col]<best[j]:best[j]=errors[col];pred[:,j]=pr[:,col];chosen[j]=label
    assert np.isfinite(pred).all() and all(chosen)
    selection=dict(tag=tag,field_attacker=chosen,inner_field_nrmse=best.tolist(),fit_rows=int(fit.sum()),inner_rows=int(inner.sum()),final_validation_used_for_selection=False)
    np.savez_compressed(selected_path,pred=pred)
    write(dest/'selection.json',selection)
    return pred,selection

def attribution_recheck(model,split,tr,pack,keys):
    with np.load(ROOT/'实验与代码/归因与关键字段隐私/20261022-v1/results-with-decomposition/sample_predictions.npz') as z:ids=z['validation_indices']
    s=FA.take(split,ids)
    def forward(v):
        out=[]
        with torch.no_grad():
            for start in range(0,len(v['industry']),256):
                sl=slice(start,start+256)
                a,b=model([torch.from_numpy(x[sl]) for x in v['xs']],torch.from_numpy(v['industry'][sl]),torch.from_numpy(v['q'][sl]),torch.from_numpy(v['state'][sl]))
                out.append(np.column_stack([a.numpy(),b.numpy()]))
        return np.concatenate(out)
    base=forward(s);records=[];trsy=tr['keys'].Symbol.astype(str).to_numpy();sy=split['keys'].Symbol.astype(str).to_numpy()[ids]
    for _,kr in keys.drop_duplicates('field').iterrows():
        field=kr['field'];node=int(kr.node[1:])-1;j=pack['names'][node].index(field)
        ok=FA.observed(s,tr,pack['names'],node,j);okt=FA.observed(tr,s,pack['names'],node,j)
        rng=np.random.default_rng(int(hashlib.sha256(field.encode()).hexdigest()[:8],16))
        rep=FA.take(s,np.repeat(np.arange(len(ids)),8))
        for i in range(len(ids)):
            if not ok[i]:continue
            pool=np.flatnonzero(okt&(tr['industry']==s['industry'][i])&(trsy!=sy[i]))
            assert len(pool)
            donors=rng.choice(pool,8,replace=len(pool)<8)
            rep['xs'][node][i*8:i*8+8,j]=tr['xs'][node][donors,j]
        delta=base-forward(rep).reshape(len(ids),8,11).mean(1)
        for _,cell in keys[keys.field==field].iterrows():
            ind=ord(cell.industry)-65;t=FA.CODES.index(cell['type'])
            eligible=ok&(s['industry']==ind)&(split['types'][ids,t]>0)
            if not eligible.any():continue
            records.append(dict(industry=cell.industry,type=cell['type'],field=field,n_observed=int(eligible.sum()),abs_overall_logit_delta=float(np.abs(delta[eligible,0]).mean()),abs_type_logit_delta=float(np.abs(delta[eligible,t+1]).mean())))
    return records

def main():
    global PUBLIC_AGE_INDICES
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    started=time.perf_counter();formal=P.D.C.shared.load_loader().formal;pack=P.D.C.pack_for(2020,(0,1));tr,va=pack['splits']
    y0,names=P.numeric_targets(tr,pack);y1,n1=P.numeric_targets(va,pack);nf,masks=numeric_masks(pack);m0,m1=masks
    assert names==n1==nf
    keysource=ROOT/'实验与代码/归因与关键字段隐私/20261022-v1/results-with-decomposition/privacy_candidate_union.csv'
    keys=pd.read_csv(keysource);unique=keys.drop_duplicates('field').copy()
    key_idx=np.array([names.index(f'{r.node.lower()}::{r.field}') for r in unique.itertuples()])
    financial=np.array([j for j in key_idx if names[j].startswith('x2::')]);PUBLIC_AGE_INDICES=[names.index('x1::mkt__company_age'),names.index('x1::mkt__listing_age')]
    assert len(key_idx)==15 and len(financial)==13
    # A deterministic corporate split is shared by all paired models and attackers.
    symbols=tr['keys'].Symbol.astype(str).to_numpy()
    inner=np.array([int(hashlib.sha256(('attack-inner-20261022|'+v).encode()).hexdigest()[:8],16)%5==0 for v in symbols]);fit=~inner
    assert not set(symbols[fit])&set(symbols[inner])
    mean,scale=scale_targets(y0[fit],m0[fit]);pub0=public_view(tr,pack);pub1=public_view(va,pack)
    rng=np.random.default_rng(20261022);controls=[]
    for j in financial:
        group=names[j].split('::')[1].split('__')[0]
        pool=np.array([k for k,n in enumerate(names) if n.startswith('x2::'+group+'__') and k not in key_idx and k not in controls])
        distance=np.abs(m0[:,pool].mean(0)-m0[:,j].mean());pool=pool[np.isclose(distance,distance.min())]
        controls.append(int(rng.choice(pool)))
    sets={'key_financial':financial.tolist(),'matched_controls':controls,'all_numeric':list(range(len(names))),'known_public_ages':PUBLIC_AGE_INDICES}
    write(OUT/'field_sets.json',{'sets':{k:[names[j] for j in v] for k,v in sets.items()},'key_source_sha256':digest(keysource)})
    splitmanifest=dict(fit_rows=int(fit.sum()),inner_rows=int(inner.sum()),validation_rows=len(y1),validation_company_overlap_with_attack_fit=len(set(va['keys'].Symbol.astype(str))&set(symbols[fit])),folds_loaded=[0,1],validation_economic_years=sorted(va['keys'].economic_year.unique().tolist()),public_dimensions=pub0.shape[1],fields=len(names),source_sha256=digest(__file__),protocol_sha256=digest(HERE/'检测前约定.md'))
    write(OUT/'manifest.json',splitmanifest)
    pubpred,_=attack_library('public_only',pub0,pub1,y0,y1,m0,m1,fit,inner,key_idx,mean,scale)
    np.savez_compressed(OUT/'evaluation_context.npz',targets=y1,observed=m1,scale=scale,mean=mean,industry=va['industry'],types=va['types'],public_prediction=pubpred)
    rowmetrics=[];fieldmetrics=[];cellmetrics=[];attributions=[]
    for seed in SEEDS:
        for arm in ('S0','D1'):
            tag=f'{arm}_s{seed}';model=ST.construct(formal,pack,arm,seed)
            checkpoint=ST.OUT/tag/'model.pt';model.load_state_dict(torch.load(checkpoint,weights_only=True));model.eval()
            ex=[E.extract(model,s) for s in (tr,va)]
            x0=np.column_stack([ex[0]['full_deployment'],pub0]);x1=np.column_stack([ex[1]['full_deployment'],pub1])
            pred,selection=attack_library(tag,x0,x1,y0,y1,m0,m1,fit,inner,key_idx,mean,scale)
            task=P.EV.task_metrics(tr,va,ex[1]['a'],ex[1]['b'])
            masks_group={'all':np.ones(len(y1),bool),'C':va['industry']==2,'I':va['industry']==8}
            # Compare the same enterprises for both arms, as ranked by frozen S0.
            s0=np.load(ST.OUT/f'S0_s{seed}'/'validation_predictions.npz')['a_logit']
            high=np.zeros(len(y1),bool);high[np.argsort(-s0,kind='stable')[:int(np.ceil(.1*len(s0)))]]=True
            masks_group.update(S0_highrisk=high,S0_other=~high)
            for group,rows in masks_group.items():
                mm=m1&rows[:,None]
                er=masked_error(y1,pred,mm,scale);pe=masked_error(y1,pubpred,mm,scale)
                ce=masked_error(y1,np.broadcast_to(mean,y1.shape),mm,scale)
                allerr=masked_error(y1,pred,np.broadcast_to(rows[:,None],y1.shape),scale)
                counts=mm.sum(0)
                for j in range(len(names)):
                    fieldmetrics.append(dict(seed=seed,arm=arm,group=group,field=names[j],n_observed=int(counts[j]),n_rows=int(rows.sum()),nrmse_observed=float(er[j]),public_nrmse=float(pe[j]),constant_nrmse=float(ce[j]),gain_over_public=float(pe[j]-er[j]),nrmse_including_imputation=float(allerr[j]),attacker=selection['field_attacker'][j]))
                for label,cols in sets.items():
                    jj=np.array(cols);jj=jj[counts[jj]>=30]
                    if not len(jj):continue
                    rowmetrics.append(dict(seed=seed,arm=arm,group=group,field_set=label,n_fields=len(jj),nrmse=float(er[jj].mean()),public_nrmse=float(pe[jj].mean()),constant_nrmse=float(ce[jj].mean()),gain_over_public=float((pe[jj]-er[jj]).mean()),recall=float(task['recall_at_10pct']),ap=float(task['overall_average_precision'])))
            for (ind,code),cell in keys.groupby(['industry','type']):
                rows=(va['industry']==ord(ind)-65)&(va['types'][:,FA.CODES.index(code)]>0)
                mm=m1&rows[:,None];err=masked_error(y1,pred,mm,scale);pe=masked_error(y1,pubpred,mm,scale);counts=mm.sum(0)
                for kr in cell.itertuples():
                    j=names.index(f'{kr.node.lower()}::{kr.field}')
                    cellmetrics.append(dict(seed=seed,arm=arm,industry=ind,type=code,field=names[j],n_observed=int(counts[j]),nrmse=float(err[j]),public_nrmse=float(pe[j]),gain_over_public=float(pe[j]-err[j]),descriptive_only=bool(counts[j]<30)))
            attrpath=OUT/'attackers'/tag/'attribution.csv'
            if attrpath.exists():ar=pd.read_csv(attrpath).to_dict('records')
            else:
                ar=attribution_recheck(model,va,tr,pack,keys);pd.DataFrame(ar).to_csv(attrpath,index=False)
            attributions.extend([dict(seed=seed,arm=arm,**r) for r in ar])
            write(OUT/'attackers'/tag/'model_reference.json',dict(path=str(checkpoint),sha256=digest(checkpoint),task=task,new_predictor_training=False))
            for fn,rows in [('aggregate.csv',rowmetrics),('fields.csv',fieldmetrics),('industry_type.csv',cellmetrics),('attribution_recheck.csv',attributions)]:pd.DataFrame(rows).to_csv(OUT/fn,index=False)
            print('DONE',tag,'key observed NRMSE',rowmetrics[-4:] if False else round(np.mean([r['nrmse'] for r in rowmetrics if r['seed']==seed and r['arm']==arm and r['group']=='all' and r['field_set']=='key_financial']),4),flush=True)
    write(OUT/'completion.json',dict(status='complete',seeds=list(SEEDS),new_predictor_training=False,seconds=time.perf_counter()-started,protocol_sha256=digest(HERE/'检测前约定.md'),source_sha256=digest(__file__)))

if __name__=='__main__':main()
