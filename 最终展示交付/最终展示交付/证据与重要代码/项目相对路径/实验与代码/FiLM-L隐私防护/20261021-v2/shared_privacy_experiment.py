"""Shared local encoders, six fusion heads, and independently audited privacy constraints."""
from __future__ import annotations
import argparse, copy, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=HERE.parent/'20261020-v1'
sys.path.insert(0,str(OLD))
import defense_lib as D
import evaluate_selected as EV
import select_lambda as SELECT
AUDIT=ROOT/'实验与代码/FiLM-L隐私审计/20261020-v1'
sys.path.insert(0,str(AUDIT))
import run_formal_audit as audit
C,A=D.C,D.A
OUT=HERE/'results'
ARMS={'S0':('none',0.), 'D001':('dcor',.01),'D01':('dcor',.1),'D1':('dcor',1.),
      'M0001':('mid',.001),'M001':('mid',.01),'M01':('mid',.1)}


def write(p,x): D.write_json(p,x)

class Fusion(nn.Module):
    def __init__(self, original):
        super().__init__()
        for name in ['context','projections','score','party','trunk','any_head','type_head']:
            setattr(self,name,copy.deepcopy(getattr(original.base,name)))
        self.film=copy.deepcopy(original.film)
    def forward(self, hs,c,r,q,industry):
        context=self.context(industry)
        gamma,beta=self.film(c).chunk(2,1)
        h1,h2,h3=hs
        post=h2+r*(gamma*h2+beta)
        messages=[torch.cat([h,q[:,i:i+1]],1) for i,h in enumerate([h1,post,h3])]
        score=[self.score(torch.cat([proj(m[:,:d]),context,self.party.weight[i].expand(len(m),-1)],1))
               for i,(m,proj,d) in enumerate(zip(messages,self.projections,(16,88,21)))]
        alpha=torch.softmax(torch.cat(score,1),1)
        z=self.trunk(torch.cat([*[m*alpha[:,i:i+1] for i,m in enumerate(messages)],context],1))
        return self.any_head(z).squeeze(-1),self.type_head(z)

class Shared(nn.Module):
    def __init__(self,formal,pack,mode='none',seed=42):
        super().__init__()
        self.mode=mode
        original=A.FilmKeepX3(formal,pack,seed)
        self.enc=copy.deepcopy(original.base.enc)
        self.control=copy.deepcopy(original.control)
        self.heads=nn.ModuleList([Fusion(A.FilmKeepX3(formal,pack,s)) for s in C.SEEDS])
        self.private_dimensions=(16,88,21)
        if mode=='mid':
            self.logvars=nn.ModuleList([nn.Linear(d,d) for d in self.private_dimensions])
            for layer in self.logvars:
                nn.init.zeros_(layer.weight); nn.init.constant_(layer.bias,-4.)
        for i,v in enumerate(D.local_targets(pack['splits'][0])):
            std=v.std(0); std[std<1e-7]=1
            self.register_buffer(f'mean{i}',torch.tensor(v.mean(0)))
            self.register_buffer(f'std{i}',torch.tensor(std))
        torch.manual_seed(seed)
    def encode(self,xs,state,noise=None):
        hs=[self.enc[0](xs[0]),self.enc[1](xs[1]),self.enc[2](xs[2],state)]
        self.kl=torch.zeros(())
        if self.mode=='mid':
            zs=[]; kls=[]
            for i,(h,layer) in enumerate(zip(hs,self.logvars)):
                lv=layer(h).clamp(-6,2)
                eps=torch.randn_like(h) if noise is None else noise[i]
                zs.append(h+torch.exp(.5*lv)*eps)
                kls.append(.5*(h.square()+lv.exp()-1-lv).mean())
            hs=zs; self.kl=torch.stack(kls).mean()
        c=self.control(xs[2][:,16:23]); r=xs[2][:,16:17]
        self.hs=hs; self.c=c; self.r=r
        return hs,c,r
    def forward(self,xs,industry,q,state,noise=None):
        hs,c,r=self.encode(xs,state,noise)
        predictions=[head(hs,c,r,q,industry) for head in self.heads]
        self.all_any=torch.stack([p[0] for p in predictions])
        self.all_type=torch.stack([p[1] for p in predictions])
        return self.all_any.mean(0),self.all_type.mean(0)
    def regularizer(self,xs):
        if self.mode=='mid':return self.kl
        if self.mode=='none':return torch.zeros(())
        targets=(xs[0],xs[1],xs[2][:,:16]); scores=[]
        for i,(x,h) in enumerate(zip(targets,self.hs)):
            z=((x-getattr(self,f'mean{i}'))/getattr(self,f'std{i}'))[:128]
            scores.append(dcor2(z,h[:128]))
        return torch.stack(scores).mean()

def dcor2(x,y):
    def center(z):
        d=torch.cdist(z,z,p=2)
        return d-d.mean(0,keepdim=True)-d.mean(1,keepdim=True)+d.mean()
    a,b=center(x),center(y)
    return (a*b).mean()/torch.sqrt(a.square().mean()*b.square().mean()+1e-12)

def noise_for(split):
    """Private deployment randomness is simulated reproducibly; keys never enter features."""
    seeds=[]
    for row in split['keys'].itertuples():
        key=f'20261021|{row.Symbol}|{row.economic_year}'
        seeds.append(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
    joined=np.stack([np.random.default_rng(s).standard_normal(125).astype('f4') for s in seeds])
    return np.split(joined,[16,104],axis=1)

@torch.no_grad()
def extract(model,split):
    model.eval(); ns=noise_for(split) if model.mode=='mid' else None
    out={k:[] for k in ['a','b','h1','h2','h3','c','r','q','industry']}
    for start in range(0,len(split['y']),4096):
        end=start+4096
        xs=[torch.tensor(x[start:end]) for x in split['xs']]
        q=torch.tensor(split['q'][start:end]); state=torch.tensor(split['state'][start:end])
        ind=torch.tensor(split['industry'][start:end])
        noise=None if ns is None else [torch.tensor(v[start:end]) for v in ns]
        a,b=model(xs,ind,q,state,noise)
        for k,v in [('a',a),('b',b),*zip(['h1','h2','h3'],model.hs),('c',model.c),('r',model.r),('q',q),('industry',ind[:,None].float())]:
            out[k].append(v.numpy())
    out={k:np.concatenate(v) for k,v in out.items()}
    out['full_deployment']=np.concatenate([out[k] for k in ['h1','h2','h3','q','c','r','industry']],1)
    out['public_side_only']=np.concatenate([out[k] for k in ['q','c','r','industry']],1)
    return out

def train(formal,pack,arm):
    dest=OUT/arm
    if (dest/'run.json').exists():return json.loads((dest/'run.json').read_text())
    dest.mkdir(parents=True,exist_ok=False)
    mode,strength=ARMS[arm]
    model=Shared(formal,pack,mode)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    prev=formal.prev; tensors=[prev.tensor_split(s,'cpu') for s in pack['splits']]
    weights=prev.cost_weights(pack['splits'][0]); aw,tw=map(torch.tensor,weights)
    gen=torch.Generator().manual_seed(42); history=[]; best=float('inf'); stale=0; started=time.perf_counter()
    for epoch in range(1,31):
        model.train(); order=torch.randperm(len(pack['splits'][0]['y']),generator=gen)
        task_sum=reg_sum=0.
        for start in range(0,len(order),1024):
            idx=order[start:start+1024]; batch=prev.batch(tensors[0],idx); y=tensors[0]['y'][idx]
            opt.zero_grad(); model(*batch)
            any_loss=nn.functional.binary_cross_entropy_with_logits(model.all_any,y[None,:].expand_as(model.all_any),pos_weight=aw,reduction='none').mean()
            types=tensors[0]['types'][idx]
            tl=nn.functional.binary_cross_entropy_with_logits(model.all_type,types[None,:,:].expand_as(model.all_type),pos_weight=tw,reduction='none').mean(2)
            task=any_loss+(tl*y[None,:]).sum()/(y.sum().clamp_min(1)*6)
            reg=model.regularizer(batch[0]); loss=task+strength*reg
            assert torch.isfinite(loss)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            task_sum+=float(task.detach())*len(idx); reg_sum+=float(reg.detach())*len(idx)
        val=extract(model,pack['splits'][1]); score=prev.selection_loss(val['a'],val['b'],pack['splits'][1],weights)
        history.append(dict(epoch=epoch,selection_loss=score,task=task_sum/len(order),regularizer=reg_sum/len(order)))
        print(arm,epoch,round(score,5),round(reg_sum/len(order),5),flush=True)
        if score<best-1e-4:
            best=score; stale=0; best_epoch=epoch
            torch.save(model.state_dict(),dest/'model.pt')
        else:
            stale+=1
            if stale>=6:break
    pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
    record=dict(arm=arm,mode=mode,strength=strength,best_epoch=best_epoch,epochs=len(history),seconds=time.perf_counter()-started,selection_loss=best,parameters=sum(p.numel() for p in model.parameters()),folds_loaded=[0,1],protocol_sha=D.file_digest(HERE/'训练前约定.md'),code_sha=D.file_digest(Path(__file__)))
    write(dest/'run.json',record); return record

def load(formal,pack,arm):
    m=Shared(formal,pack,ARMS[arm][0]); m.load_state_dict(torch.load(OUT/arm/'model.pt',weights_only=True));m.eval();return m

def baseline(formal,pack):
    frozen_cal=json.loads((ROOT/'实验与代码/FiLM-L通信优化/20261020-v1/results/calibration_replay_preflight.json').read_text())
    cal_lookup={r['seed']:r for r in frozen_cal['checks'] if r['year']==2020}
    per_fold={i:{} for i in range(len(pack['splits']))}
    logits={i:[] for i in per_fold}
    for seed in C.SEEDS:
        m=A.FilmKeepX3(formal,pack,seed)
        m.load_state_dict(torch.load(D.CONTROL/f'artifacts/2020/FiLM-L_s{seed}/model.pt',weights_only=True)['state_dict'])
        for f,s in enumerate(pack['splits']):
            per_fold[f][seed]=audit.extract(m,s)
            ra,rt=formal.logits(m,s)
            entry=cal_lookup[seed]; ca=entry['calibration']; w=entry['weights']
            logits[f].append((formal.exp.prob.apply_cal(ra-np.log(w['overall']),ca['overall']),
                              formal.exp.prob.apply_cal(rt-np.log(np.asarray(w['types'],dtype='f4')),ca['types_shrink'])))
    result={}
    for f in per_fold:
        result[f]=audit.build_views(per_fold[f])['joint6']
        result[f]['a']=np.mean([p[0] for p in logits[f]],0)
        result[f]['b']=np.mean([p[1] for p in logits[f]],0)
    return result

def select(formal,pack):
    path=OUT/'selection.json'
    if path.exists():return json.loads(path.read_text())
    base=baseline(formal,pack); s=pack['splits']; reference=EV.task_metrics(s[0],s[1],base[1]['a'],base[1]['b'])
    rows=[]
    for arm in ARMS:
        m=load(formal,pack,arm); views=[extract(m,sp) for sp in s]
        metrics=EV.task_metrics(s[0],s[1],views[1]['a'],views[1]['b'])
        rec=SELECT.attack_fold1('full_deployment',views[0]['full_deployment'],views[1]['full_deployment'],s[0],s[1],pack['names'])
        row=dict(arm=arm,**metrics,**{'reconstruction_gain':rec['gain_vs_constant'],'attack':rec})
        row['task_eligible']=metrics['recall_at_10pct']>=reference['recall_at_10pct']-.01 and metrics['overall_average_precision']>=reference['overall_average_precision']-.01
        rows.append(row);write(OUT/arm/'fold1_evaluation.json',row)
        print('SELECT',arm,row['recall_at_10pct'],row['overall_average_precision'],row['reconstruction_gain'],row['task_eligible'],flush=True)
    s0=rows[0]['reconstruction_gain']; chosen=['S0']; choices={}
    for family,prefix in [('dcor','D'),('mid','M')]:
        candidates=[r for r in rows if r['arm'].startswith(prefix) and r['task_eligible'] and s0-r['reconstruction_gain']>=.10]
        best=min(candidates,key=lambda r:(r['reconstruction_gain'],ARMS[r['arm']][1])) if candidates else None
        choices[family]=None if best is None else best['arm']
        if best:chosen.append(best['arm'])
    result=dict(status='selected_before_test',year=2020,reference_fold1=reference,rows=rows,selected=chosen,family_choices=choices,folds_loaded=[0,1],fold2_loaded=False,fold3_loaded=False)
    write(path,result);return result

def final(formal,decision):
    dest=OUT/'final'
    if (dest/'summary.json').exists():return
    dest.mkdir(exist_ok=False)
    pack=C.pack_for(2020,range(4)); folds={i:s for i,s in enumerate(pack['splits'])}; weights=formal.prev.cost_weights(folds[0])
    symbols=folds[3]['keys'].Symbol.astype(str).to_numpy(); companies=sorted(set(symbols)); mapping={s:i for i,s in enumerate(companies)}
    boot=np.random.default_rng(audit.ATTACK_SEED).poisson(1,size=(audit.BOOT_RECON,len(companies))).astype('i2');boot[boot.sum(1)==0,0]=1
    summaries=[]
    for arm in decision['selected']:
        model=load(formal,pack,arm); ex={f:extract(model,folds[f]) for f in (0,1,2,3)}
        cal=C.shared.fit_calibration(formal,{'2_a_logit':ex[2]['a'],'2_b_logit':ex[2]['b']},pack,weights)
        a=formal.exp.prob.apply_cal(ex[3]['a']-np.log(weights[0]),cal['overall'])
        b=formal.exp.prob.apply_cal(ex[3]['b']-np.log(weights[1]),cal['types_shrink'])
        metrics=EV.task_metrics(folds[0],folds[3],a,b)
        out=dest/arm;out.mkdir();write(out/'calibration.json',cal);np.savez_compressed(out/'predictions.npz',a_logit=a,b_logit=b)
        views={f:{'shared6':{k:ex[f][k] for k in ['h1','h2','h3','full_deployment','public_side_only']}} for f in (0,1,3)}
        for name in ('records','fields','bootstrap'):(out/name).mkdir()
        records=[]
        for view in ['full_deployment','public_side_only','h1','h2','h3']:
            print('FINAL ATTACK',arm,view,flush=True)
            r=audit.attack_one(2020,'shared6',view,{f:folds[f] for f in (0,1,3)},views,pack['names'],pack['meta'],mapping,boot,out,primary=True)
            records.append(dict(view=view,gain=r['selected_gain_vs_constant']['point_estimate'],auc=(r['attribute_macro_auc'] or {}).get('point_estimate'),practical_leakage=r['practical_leakage_criterion_met']))
        summaries.append(dict(arm=arm,task=metrics,privacy=records,payload_fp32_bytes=545,shared_message_dimensions=137,public_industry_transmitted_separately=True))
        write(out/'summary.json',summaries[-1])
    write(dest/'summary.json',dict(status='complete',selection_sha=D.file_digest(OUT/'selection.json'),results=summaries,folds_used=[0,1,2,3],year=2020))

def preflight(formal,pack):
    m=Shared(formal,pack); old=A.FilmKeepX3(formal,pack,42);m.eval();old.eval()
    s=pack['splits'][0]; b=[torch.tensor(v[:32]) for v in s['xs']];ind=torch.tensor(s['industry'][:32]);q=torch.tensor(s['q'][:32]);state=torch.tensor(s['state'][:32])
    with torch.no_grad():
        hs,c,r=m.encode(b,state); a,t=m.heads[0](hs,c,r,q,ind); ao,to=old(b,ind,q,state)
        torch.testing.assert_close(a,ao,rtol=0,atol=0);torch.testing.assert_close(t,to,rtol=0,atol=0)
    x=torch.randn(24,5,requires_grad=True);dc=dcor2(x,x);assert abs(float(dc.detach())-1)<1e-5;dc.backward();assert torch.isfinite(x.grad).all()
    mm=Shared(formal,pack,'mid'); sm={k:(v[:32] if isinstance(v,np.ndarray) else v) for k,v in s.items()};sm['xs']=[v[:32] for v in s['xs']];sm['keys']=s['keys'].iloc[:32]
    ex=extract(mm,sm); ex2=extract(mm,sm);np.testing.assert_array_equal(ex['full_deployment'],ex2['full_deployment'])
    assert len({id(v) for v in m.parameters()})==len(list(m.parameters()))
    write(OUT/'preflight.json',dict(head42_forward_exact=True,dcor_identity_and_finite_gradient=True,mid_repeat_transcript_exact=True,folds_loaded=[0,1],protocol_sha=D.file_digest(HERE/'训练前约定.md')))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['train','select','final','all'],default='all');args=parser.parse_args()
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    formal=C.shared.load_loader().formal
    if args.phase in ['train','select','all']:
        pack=C.pack_for(2020,(0,1))
        if args.phase in ['train','all']:
            preflight(formal,pack)
            for arm in ARMS:train(formal,pack,arm)
        if args.phase in ['select','all']:select(formal,pack)
    if args.phase in ['final','all']:
        decision=json.loads((OUT/'selection.json').read_text());final(formal,decision)

if __name__=='__main__':main()
