"""Observed-value recovery, attribute inference, and authorized-output baselines."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import roc_auc_score,roc_curve,balanced_accuracy_score

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'实验与代码/关键字段隐私检验/20261022-v1'))
import inspect_key_privacy as K
P,E,ST=K.P,K.E,K.ST
TOLS=(.1,.319,.5);SEEDS=K.SEEDS

def write(p,x):K.write(p,x)
def sha(p):return K.digest(p)
def aucs(y,s,mask):
    result=[]
    for j in range(y.shape[1]):
        ok=mask[:,j]
        result.append(roc_auc_score(y[ok,j],s[ok,j]) if ok.sum()>=10 and len(np.unique(y[ok,j]))==2 else np.nan)
    return np.array(result)

def thresholds(y,s,mask):
    cut=[];ba=[]
    for j in range(y.shape[1]):
        ok=mask[:,j]
        if ok.sum()<10 or len(np.unique(y[ok,j]))<2:cut.append(0.);ba.append(np.nan);continue
        fpr,tpr,t=roc_curve(y[ok,j],s[ok,j]);i=int(np.argmax(tpr-fpr));v=t[i]
        cut.append(float(v) if np.isfinite(v) else float(np.nextafter(s[ok,j].max(),np.inf)))
        ba.append(float(.5*(1+tpr[i]-fpr[i])))
    return np.array(cut),np.array(ba)

def numbers(y,p,m,scale):
    n=np.maximum(m.sum(0),1);z=np.abs((y-p)/scale)
    rm=np.sqrt((z*z*m).sum(0)/n)
    return [rm,*[((z<=t)*m).sum(0)/n for t in TOLS]]

def deep_train(dest,label,x0,xi,xv,y0,yi,m0,mi,mean,scale,classification=False):
    path=dest/f'{label}.npz'
    if path.exists():
        with np.load(path) as z:return z['inner'],z['validation']
    xs=P.A.Standardizer.fit(x0);xf=torch.tensor(xs.apply(x0));xt=torch.tensor(xs.apply(xi));xv=torch.tensor(xs.apply(xv))
    P.A.seed_all(20261022)
    net=nn.Sequential(nn.Linear(x0.shape[1],512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,y0.shape[1]))
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    yy=y0 if classification else (y0-mean)/scale
    loader=DataLoader(TensorDataset(xf,torch.tensor(yy.astype('f4')),torch.tensor(m0.astype('f4'))),batch_size=1024,shuffle=True,generator=torch.Generator().manual_seed(20261022))
    pos=(y0*m0).sum(0);neg=((1-y0)*m0).sum(0)
    pw=torch.tensor(np.clip(neg/np.maximum(pos,1),.1,50).astype('f4')) if classification else None
    best=-np.inf if classification else np.inf;stale=0;history=[];checkpoint=dest/f'{label}.pt'
    for epoch in range(1,81):
        net.train()
        for x,y,m in loader:
            opt.zero_grad(set_to_none=True);pr=net(x)
            loss=(nn.functional.binary_cross_entropy_with_logits(pr,y,pos_weight=pw,reduction='none') if classification else (pr-y).square())
            loss=(loss*m).sum()/m.sum().clamp_min(1);assert torch.isfinite(loss)
            loss.backward();opt.step()
        net.eval()
        with torch.no_grad():p=torch.cat([net(v) for v in xt.split(4096)]).numpy()
        score=float(np.nanmean(aucs(yi,p,mi))) if classification else float(np.mean(numbers(yi,p*scale+mean,mi,scale)[0]))
        history.append(dict(epoch=epoch,selection_score=score))
        better=score>best+1e-5 if classification else score<best-1e-5
        if better:best=score;stale=0;bestepoch=epoch;torch.save(net.state_dict(),checkpoint)
        else:
            stale+=1
            if stale>=8:break
    net.load_state_dict(torch.load(checkpoint,weights_only=True));net.eval()
    with torch.no_grad():pi=torch.cat([net(v) for v in xt.split(4096)]).numpy();pv=torch.cat([net(v) for v in xv.split(4096)]).numpy()
    if not classification:pi=pi*scale+mean;pv=pv*scale+mean
    np.savez_compressed(path,inner=pi,validation=pv)
    write(dest/f'{label}.json',dict(classification=classification,best_epoch=bestepoch,best_score=best,history=history,checkpoint_sha256=sha(checkpoint)))
    print('FIT',dest.name,label,bestepoch,round(best,4),flush=True)
    return pi,pv

def library(tag,x0,x1,y0,y1,m0,m1,labels0,labels1,fit,inner,mean,scale,parents,legacy=None):
    dest=OUT/'attacks'/tag;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'done.json').exists():
        with np.load(dest/'selected.npz') as z:return {k:z[k] for k in z.files}
    numeric=[];attrs=[];mi=m0[inner];yi=y0[inner];ai=labels0[inner];ami=np.column_stack([mi,mi]);am0=np.column_stack([m0[fit],m0[fit]])
    for alpha in (.1,1.,10.,100.):
        ridge=P.A.RidgeAttack(alpha).fit(x0[fit],y0[fit]);numeric.append((f'ridge{alpha:g}',ridge.predict(x0[inner]),ridge.predict(x1)))
        clf=P.A.RidgeAttack(alpha).fit(x0[fit],labels0[fit]);attrs.append((f'linear{alpha:g}',clf.predict(x0[inner]),clf.predict(x1)))
    pi,pv=deep_train(dest,'regressor',x0[fit],x0[inner],x1,y0[fit],yi,m0[fit],mi,mean,scale)
    numeric.append(('regressor',pi,pv))
    pi,pv=deep_train(dest,'attribute',x0[fit],x0[inner],x1,labels0[fit],ai,am0,ami,None,None,True)
    attrs.append(('attribute',pi,pv))
    for label,const in [('mean',mean),('median',np.array([np.median(y0[fit,j][m0[fit,j]]) for j in range(y0.shape[1])]))]:
        numeric.append((label,np.broadcast_to(const,yi.shape),np.broadcast_to(const,y1.shape)))
    for name,parent in parents:
        for j in range(4):numeric.append((name+f'_numeric{j}',parent['numeric_inner'][j],parent['numeric_validation'][j]))
        attrs.extend([(name+'_attribute_auc',parent['auc_inner'],parent['auc_validation']),(name+'_attribute_ba',parent['ba_inner'],parent['ba_validation'])])
    if legacy is not None:numeric.extend(legacy)
    for name,pi,pv in numeric:attrs.append(('from_'+name,np.column_stack([-pi,pi]),np.column_stack([-pv,pv])))
    p=yi.shape[1];best=np.vstack([np.full(p,np.inf),np.full((3,p),-np.inf)])
    selected_i=np.zeros((4,*yi.shape));selected_v=np.zeros((4,*y1.shape));names=[['']*p for _ in range(4)]
    for name,pi,pv in numeric:
        scores=numbers(yi,pi,mi,scale)
        for metric,score in enumerate(scores):
            use=score<best[metric] if metric==0 else score>best[metric]
            best[metric,use]=score[use];selected_i[metric][:,use]=pi[:,use];selected_v[metric][:,use]=pv[:,use]
            for j in np.flatnonzero(use):names[metric][j]=name
    q=ai.shape[1];abest=np.full(q,-np.inf);bbest=np.full(q,-np.inf);a_i=np.zeros_like(ai);a_v=np.zeros_like(labels1);b_i=a_i.copy();b_v=a_v.copy();cuts=np.zeros(q);anames=['']*q;bnames=['']*q
    for name,pi,pv in attrs:
        av=aucs(ai,pi,ami);ct,bv=thresholds(ai,pi,ami)
        use=np.isfinite(av)&(av>abest);abest[use]=av[use];a_i[:,use]=pi[:,use];a_v[:,use]=pv[:,use]
        for j in np.flatnonzero(use):anames[j]=name
        use=np.isfinite(bv)&(bv>bbest);bbest[use]=bv[use];b_i[:,use]=pi[:,use];b_v[:,use]=pv[:,use];cuts[use]=ct[use]
        for j in np.flatnonzero(use):bnames[j]=name
    result=dict(numeric_inner=selected_i,numeric_validation=selected_v,auc_inner=a_i,auc_validation=a_v,ba_inner=b_i,ba_validation=b_v,thresholds=cuts)
    np.savez_compressed(dest/'selected.npz',**result)
    write(dest/'done.json',dict(numeric_attacker=names,attribute_auc_attacker=anames,attribute_ba_attacker=bnames,final_validation_used_for_selection=False))
    return result

def legacy_candidates(tag,x0,x1,fit,inner,allmean,allscale,keyindices):
    """Replay the prior deeper all-field and key-field attackers, without refitting."""
    dest=K.OUT/'attackers'/tag
    if not dest.exists():return []
    xs=P.A.Standardizer.fit(x0[fit]);out=[]
    for name in ['deep_all','deep_key']:
        with np.load(dest/f'{name}.npz') as z:idx=z['idx'];oldpred=z['pred']
        net=nn.Sequential(nn.Linear(x0.shape[1],512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,len(idx)))
        net.load_state_dict(torch.load(dest/f'{name}.pt',weights_only=True));net.eval()
        with torch.no_grad():
            pi=torch.cat([net(v) for v in torch.tensor(xs.apply(x0[inner])).split(4096)]).numpy()*allscale[idx]+allmean[idx]
            pv=torch.cat([net(v) for v in torch.tensor(xs.apply(x1)).split(4096)]).numpy()*allscale[idx]+allmean[idx]
        np.testing.assert_allclose(pv,oldpred,atol=1e-6,rtol=1e-6)
        cols=[int(np.flatnonzero(idx==k)[0]) for k in keyindices]
        out.append(('legacy_'+name,pi[:,cols],pv[:,cols]))
    return out

def output_views(ex,pub):
    a=P.EV.sigmoid(ex['a']).astype('f4');b=P.EV.sigmoid(ex['b']).astype('f4')
    top=np.argsort(-b,axis=1,kind='stable')[:,:2];mask=np.zeros_like(b);mask[np.arange(len(b))[:,None],top]=1
    return {'public':pub,'top2':np.column_stack([pub,a,b*mask,mask]),'outputs':np.column_stack([pub,a,b]),'messages':np.column_stack([ex['full_deployment'],pub,a,b])}

def gaussian_views(model,splits,ex):
    result=[];scales=[ex[0][h].std(0) for h in ['h1','h2','h3']]
    for s,original in zip(splits,ex):
        noise=E.noise_for(s);v={k:x.copy() for k,x in original.items()}
        for i,h in enumerate(['h1','h2','h3']):v[h]+=noise[i]*scales[i]
        aa=[];bb=[]
        with torch.no_grad():
            for lo in range(0,len(s['y']),4096):
                sl=slice(lo,lo+4096);hs=[torch.tensor(v[h][sl]) for h in ['h1','h2','h3']]
                ps=[head(hs,torch.tensor(v['c'][sl]),torch.tensor(v['r'][sl]),torch.tensor(v['q'][sl]),torch.tensor(s['industry'][sl])) for head in model.heads]
                aa.append(torch.stack([p[0] for p in ps]).mean(0).numpy());bb.append(torch.stack([p[1] for p in ps]).mean(0).numpy())
        v['a']=np.concatenate(aa);v['b']=np.concatenate(bb);v['full_deployment']=np.column_stack([v[k] for k in ['h1','h2','h3','q','c','r','industry']]);result.append(v)
    return result

def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);started=time.perf_counter()
    formal=P.D.C.shared.load_loader().formal;pack=P.D.C.pack_for(2020,(0,1));tr,va=pack['splits']
    all_y0,names=P.numeric_targets(tr,pack);all_y1,_=P.numeric_targets(va,pack);nf,masks=K.numeric_masks(pack);assert nf==names
    keys=json.loads((K.OUT/'field_sets.json').read_text())['sets']['key_financial'];ix=np.array([names.index(n) for n in keys]);all_m0,all_m1=masks
    sy=tr['keys'].Symbol.astype(str).to_numpy();inner=np.array([int(hashlib.sha256(('attack-inner-20261022|'+v).encode()).hexdigest()[:8],16)%5==0 for v in sy]);fit=~inner
    assert not set(sy[fit])&set(sy[inner])
    allmean,allscale=K.scale_targets(all_y0[fit],all_m0[fit]);mean=allmean[ix];scale=allscale[ix]
    y0,y1=all_y0[:,ix],all_y1[:,ix];m0,m1=all_m0[:,ix],all_m1[:,ix];pub0=K.public_view(tr,pack);pub1=K.public_view(va,pack)
    # Industry-conditioned attribute cutpoints, fixed strictly in the attack fit set.
    cut=np.empty((20,2,len(ix)),dtype='f4');attrmeta=[]
    for ind in range(20):
        for j in range(len(ix)):
            ok=fit&m0[:,j]&(tr['industry']==ind);fallback=ok.sum()<100
            if fallback:ok=fit&m0[:,j]
            cut[ind,:,j]=np.quantile(y0[ok,j],[.25,.75])
            attrmeta.append(dict(industry=ind,field=keys[j],low=float(cut[ind,0,j]),high=float(cut[ind,1,j]),fallback_global=bool(fallback),fit_observed=int(ok.sum())))
    labels=[]
    for s,y in [(tr,y0),(va,y1)]:labels.append(np.column_stack([y<cut[s['industry'],0,:],y>cut[s['industry'],1,:]]).astype('f4'))
    a0,a1=labels;am1=np.column_stack([m1,m1])
    write(OUT/'attribute_definitions.json',attrmeta)
    manifest=dict(folds_loaded=[0,1],evaluation_economic_years=sorted(va['keys'].economic_year.unique().tolist()),fit_rows=int(fit.sum()),inner_rows=int(inner.sum()),validation_rows=len(y1),key_fields=keys,field_source_sha256=sha(K.OUT/'field_sets.json'),protocol_sha256=sha(HERE/'评价前约定.md'),source_sha256=sha(__file__),new_predictor_training=False,tolerances=list(TOLS))
    write(OUT/'manifest.json',manifest)
    np.savez_compressed(OUT/'context.npz',y=y1,observed=m1,scale=scale,attribute_labels=a1,industry=va['industry'],types=va['types'])
    public=library('public',pub0,pub1,y0,y1,m0,m1,a0,a1,fit,inner,mean,scale,[])
    fields=[];attrs=[];tasks=[];modelsources=[]
    jobs=[(seed,arm) for seed in SEEDS for arm in ['S0','D1']]+[(42,arm) for arm in ['S0','D1','M01','G1']]
    for seed,arm in jobs:
        tag=f'{arm}_s{seed}'
        if seed!=42:
            model=ST.construct(formal,pack,arm,seed);checkpoint=ST.OUT/tag/'model.pt';model.load_state_dict(torch.load(checkpoint,weights_only=True))
        else:
            loadarm='S0' if arm=='G1' else arm;model=E.load(formal,pack,loadarm);checkpoint=E.OUT/loadarm/'model.pt'
        model.eval();ex=[E.extract(model,s) for s in [tr,va]]
        if arm=='G1':ex=gaussian_views(model,[tr,va],ex)
        views=[output_views(v,p) for v,p in zip(ex,[pub0,pub1])]
        solutions={'public':public}
        task=P.EV.task_metrics(tr,va,ex[1]['a'],ex[1]['b']);tasks.append(dict(tag=tag,arm=arm,seed=seed,recall=task['recall_at_10pct'],ap=task['overall_average_precision'],top2_hit=task['conditional_top2_hit'],exploratory=seed==42))
        modelsources.append(dict(tag=tag,path=str(checkpoint),sha256=sha(checkpoint),simulated_sigma=1. if arm=='G1' else None))
        for view in ['top2','outputs','messages']:
            legacy=None
            if view=='messages' and seed!=42:
                old0=np.column_stack([ex[0]['full_deployment'],pub0]);old1=np.column_stack([ex[1]['full_deployment'],pub1])
                legacy=legacy_candidates(tag,old0,old1,fit,inner,allmean,allscale,ix)
            parents=[(v,s) for v,s in solutions.items()]
            solutions[view]=library(tag+'_'+view,views[0][view],views[1][view],y0,y1,m0,m1,a0,a1,fit,inner,mean,scale,parents,legacy)
        # Shared reference enterprises for high-risk comparisons, not arm-specific selected sets.
        if seed!=42:score=np.load(ST.OUT/f'S0_s{seed}/validation_predictions.npz')['a_logit']
        else:score=E.extract(E.load(formal,pack,'S0'),va)['a']
        high=np.zeros(len(y1),bool);high[np.argsort(-score,kind='stable')[:int(np.ceil(.1*len(score)))]]=True
        groups={'all':np.ones(len(y1),bool),'C':va['industry']==2,'I':va['industry']==8,'S0_highrisk':high,'S0_other':~high}
        for ind in ['C','I']:
            for code in ['P2503','P2504','P2505']:groups[ind+'_'+code]=(va['industry']==ord(ind)-65)&(va['types'][:,K.FA.CODES.index(code)]>0)
        for view,s in solutions.items():
            for group,rows in groups.items():
                mm=m1&rows[:,None];n=mm.sum(0);pred=s['numeric_validation'];er=numbers(y1,pred[0],mm,scale)[0]
                hits=[numbers(y1,pred[k+1],mm,scale)[k+1] for k in range(3)]
                am=np.column_stack([mm,mm]);av=aucs(a1,s['auc_validation'],am)
                for j,field in enumerate(keys):fields.append(dict(tag=tag,arm=arm,seed=seed,view=view,group=group,field=field,n_observed=int(n[j]),nrmse=float(er[j]) if n[j] else None,**{f'success_{t:g}':float(hits[k][j]) if n[j] else None for k,t in enumerate(TOLS)}))
                for j in range(26):
                    ok=am[:,j];yy=a1[ok,j];supported=ok.sum()>=30 and len(np.unique(yy))==2
                    ba=balanced_accuracy_score(yy,s['ba_validation'][ok,j]>=s['thresholds'][j]) if supported else None
                    attrs.append(dict(tag=tag,arm=arm,seed=seed,view=view,group=group,field=keys[j%13],attribute='low' if j<13 else 'high',n_observed=int(ok.sum()),prevalence=float(yy.mean()) if ok.any() else None,auc=float(av[j]) if supported and np.isfinite(av[j]) else None,balanced_accuracy=float(ba) if ba is not None else None,supported=bool(supported)))
        for fn,rows in [('fields.csv',fields),('attributes.csv',attrs),('tasks.csv',tasks)]:pd.DataFrame(rows).to_csv(OUT/fn,index=False)
        write(OUT/'model_sources.json',modelsources)
        print('DONE',tag,'recall',round(task['recall_at_10pct'],4),flush=True)
    write(OUT/'completion.json',dict(status='complete',seconds=time.perf_counter()-started,models=len(jobs),new_predictor_training=False,protocol_sha256=sha(HERE/'评价前约定.md'),source_sha256=sha(__file__)))

if __name__=='__main__':main()
