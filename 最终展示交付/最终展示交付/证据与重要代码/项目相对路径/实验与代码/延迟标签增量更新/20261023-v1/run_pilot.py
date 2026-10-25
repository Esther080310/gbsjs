from pathlib import Path
import sys,json,hashlib,time,copy,importlib.util
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
H=Path(__file__).resolve().parent;ROOT=H.parents[2];O=H/'results';O.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'实验与代码/控制消息架构/20261018-v1'))
import cma_lib as C,ablation_lib as A
spec=importlib.util.spec_from_file_location('incremental_events',ROOT/'实验与代码/延迟监督与近期队列验证/data_builder.py');DB=importlib.util.module_from_spec(spec);spec.loader.exec_module(DB)
SEEDS=[42,100,17];torch.set_num_threads(4)
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def digest(m):return C.shared.digest_state(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def select(s,ix):return {'xs':[x[ix].copy() for x in s['xs']],**{k:s[k][ix].copy() for k in ['industry','q','state','y','types']},'keys':s['keys'].iloc[ix].reset_index(drop=True)}
def join(ss):return {'xs':[np.concatenate([s['xs'][i] for s in ss]) for i in range(3)],**{k:np.concatenate([s[k] for s in ss]) for k in ['industry','q','state','y','types']},'keys':pd.concat([s['keys'] for s in ss],ignore_index=True)}
def tensors(s):return {'xs':[torch.tensor(v) for v in s['xs']],**{k:torch.tensor(s[k]) for k in ['q','state','y','types']},'industry':torch.tensor(s['industry'],dtype=torch.long)}
def forward(m,t,ix):return m([v[ix] for v in t['xs']],t['industry'][ix],t['q'][ix],t['state'][ix])
def task(a,b,y,typ,aw,tw):
 terms=F.binary_cross_entropy_with_logits(b,typ,pos_weight=tw,reduction='none').mean(1)
 return F.binary_cross_entropy_with_logits(a,y,pos_weight=aw)+(terms*y).sum()/y.sum().clamp_min(1)
def enc_hash(m):return hashlib.sha256(b''.join(v.detach().numpy().tobytes() for k,v in m.state_dict().items() if k.startswith('base.enc.') or k.startswith('control.'))).hexdigest()
def fullfit(m,t,ids,weights,seed,epochs=30):
 start=time.perf_counter();aw,tw=map(torch.tensor,weights);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);gen=torch.Generator().manual_seed(seed);m.train();hist=[];steps=0
 for ep in range(epochs):
  order=torch.tensor(ids)[torch.randperm(len(ids),generator=gen)];vals=[]
  for st in range(0,len(order),1024):
   ix=order[st:st+1024];torch.manual_seed(seed*100003+ep*1000+st);opt.zero_grad();a,b=forward(m,t,ix);loss=task(a,b,t['y'][ix],t['types'][ix],aw,tw);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();steps+=1;vals.append(float(loss.detach()))
  hist.append(float(np.mean(vals)))
 return dict(seconds=time.perf_counter()-start,steps=steps,exposures=epochs*len(ids),epochs=epochs,history=hist)
@torch.no_grad()
def encode(m,t):
 m.eval();cache={k:[] for k in ['h1','h2','h3','c','r']};start=time.perf_counter()
 for st in range(0,len(t['y']),2048):
  sl=slice(st,st+2048);a=t['xs'][2][sl,m.old_x3_dimension:m.old_x3_dimension+7];vals=[m.base.enc[0](t['xs'][0][sl]),m.base.enc[1](t['xs'][1][sl]),m.base.enc[2](t['xs'][2][sl],t['state'][sl]),m.control(a),a[:,:1]]
  for k,v in zip(cache,vals):cache[k].append(v)
 cache={k:torch.cat(v) for k,v in cache.items()};cache.update(q=t['q'],industry=t['industry']);return cache,time.perf_counter()-start

def cached_forward(m,z,ix):
 h1,h2,h3,c,r=[z[k][ix] for k in ['h1','h2','h3','c','r']];gamma,beta=m.film(c).chunk(2,dim=1);h2=h2+r*(gamma*h2+beta);context=m.base.context(z['industry'][ix]);q=z['q'][ix]
 msg=[torch.cat([v,q[:,i:i+1]],1) for i,v in enumerate([h1,h2,h3])]
 scores=[m.base.score(torch.cat([proj(v[:,:d]),context,m.base.party.weight[i].expand(len(v),-1)],1)) for i,(v,proj,d) in enumerate(zip(msg,m.base.projections,m.content_dims))];alpha=torch.softmax(torch.cat(scores,1),1);f=m.base.trunk(torch.cat([*[v*alpha[:,i:i+1] for i,v in enumerate(msg)],context],1));return m.base.any_head(f).squeeze(-1),m.base.type_head(f)
@torch.no_grad()
def scores(m,t):
 m.eval();out=[]
 for st in range(0,len(t['y']),2048):a,b=forward(m,t,slice(st,st+2048));out.append(np.column_stack([a.numpy(),b.numpy()]))
 return np.concatenate(out)
def update(m,z,t,delta,replay,weights,seed):
 for k,v in m.named_parameters():v.requires_grad_(not(k.startswith('base.enc.') or k.startswith('control.')))
 params=[v for v in m.parameters() if v.requires_grad];opt=torch.optim.AdamW(params,lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed+9931);aw,tw=map(torch.tensor,weights);start=time.perf_counter();hist=[];m.train();before=enc_hash(m)
 for st in range(60):
  if replay is None:ix=rng.choice(delta,256,replace=True)
  else:ix=np.r_[rng.choice(delta,128,replace=True),rng.choice(replay,128,replace=True)]
  ix=torch.tensor(ix);torch.manual_seed(seed*100003+st);opt.zero_grad();a,b=cached_forward(m,z,ix);loss=task(a,b,t['y'][ix],t['types'][ix],aw,tw);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(params,1);opt.step();hist.append(float(loss.detach()))
 assert enc_hash(m)==before
 return dict(seconds=time.perf_counter()-start,steps=60,exposures=60*256,history=hist,trainable_parameters=sum(v.numel() for v in params),encoder_unchanged=True)
def metric(pred,s,tw):
 y=s['y'];n=len(y);k=int(np.ceil(.1*n));order=np.argsort(-pred[:,0],kind='stable');top=order[:k];hit=top[y[top]>.5];b=pred[:,1:]-np.log(np.maximum(tw,1e-8));top2=np.argsort(-b,axis=1,kind='stable')[:,:2]
 return dict(n=n,positives=int(y.sum()),recall10=float(y[top].sum()/max(y.sum(),1)),precision10=float(y[top].mean()),ap=float(average_precision_score(y,pred[:,0])),conditional_top2=float(np.any(np.take_along_axis(s['types'][hit],top2[hit],axis=1)>0,axis=1).mean()) if len(hit) else 0.)
def bootstrap_diff(y,a,b,seed=193):
 rng=np.random.default_rng(seed);d=[]
 for i in range(1000):
  ix=rng.integers(0,len(y),len(y));k=int(np.ceil(.1*len(ix)));den=max(y[ix].sum(),1);d.append((y[ix[np.argsort(-a[ix],kind='stable')[:k]]].sum()-y[ix[np.argsort(-b[ix],kind='stable')[:k]]].sum())/den)
 return np.percentile(d,[2.5,97.5]).tolist()

def real():
 formal=C.shared.load_loader().formal;pack=C.pack_for(2020,(0,1,2));s=join(pack['splits']);n0=len(pack['splits'][0]['y']);n1=len(pack['splits'][1]['y']);n=len(s['y']);keys=s['keys'];assert not keys.duplicated().any()
 ev,issues=DB.build_events();dates=['2019-04-30','2020-04-30','2023-12-31'];lab={}
 for cut in dates:
  first,typed=DB.first_records(ev,pd.Timestamp(cut));lab[cut]=DB.aligned(keys,first,typed);assert first.known_date.le(cut).all()
 y0,b0,_=lab[dates[0]];y1,b1,_=lab[dates[1]];yf,bf,_=lab[dates[2]];assert np.all(y1>=y0) and np.all(b1>=b0)
 pub=pd.read_csv(DB.PUBLICATION,dtype={'Symbol':str});pub.Symbol=pub.Symbol.str.zfill(6);assert not pub.duplicated(['Symbol','economic_year']).any();pubdate=pd.to_datetime(keys.merge(pub[['Symbol','economic_year','publication_date_min']],on=['Symbol','economic_year'],how='left',validate='one_to_one').publication_date_min)
 hold=keys.Symbol.map(lambda v:int(hashlib.sha256(str(v).encode()).hexdigest()[:8],16)%5==0).to_numpy();years=keys.economic_year.to_numpy();base=np.flatnonzero((years<=2017)&~hold&pubdate.le(dates[0]).to_numpy());alltrain=np.flatnonzero((years<=2018)&~hold&pubdate.le(dates[1]).to_numpy());oldret=np.flatnonzero((years==2017)&hold);target=np.flatnonzero(years==2019)
 oldmask=np.isin(alltrain,base);changed=(y1!=y0)|np.any(b1!=b0,axis=1);delta=alltrain[(~oldmask)|changed[alltrain]];unchanged=base[~changed[base]];replay=np.random.default_rng(712).choice(unchanged,min(2048,len(unchanged)),replace=False);assert not np.intersect1d(alltrain,np.r_[oldret,target]).size
 # Save source-keyed label revisions and training indices before any optimization.
 snapshot=keys.copy();snapshot['heldout_firm']=hold;snapshot['publication_date']=pubdate.dt.strftime('%Y-%m-%d');snapshot['y_initial']=y0.astype(int);snapshot['y_update']=y1.astype(int);snapshot['y_evaluation']=yf.astype(int);snapshot['changed_label']=changed;snapshot.to_csv(O/'rows_and_labels.csv',index=False)
 np.savez_compressed(O/'label_ledger.npz',symbol=keys.Symbol.to_numpy(str),year=years,initial_y=y0,initial_types=b0,update_y=y1,update_types=b1,evaluation_y=yf,evaluation_types=bf,base_index=base,update_index=alltrain,delta_index=delta,replay_index=replay,old_eval_index=oldret,target_index=target)
 frozen_y=s['y'].copy();frozen_b=s['types'].copy();audit=dict(initial_cutoff=dates[0],update_cutoff=dates[1],evaluation_cutoff=dates[2],initial_rows=len(base),update_rows=len(alltrain),new_rows=int(sum(~oldmask)),revised_old_rows=int(sum(changed[base])),newly_positive_old_rows=int(sum((y1[base]>y0[base]))),type_only_revisions=int(sum((y1[base]==y0[base])&np.any(b1[base]!=b0[base],axis=1))),delta_rows=len(delta),replay_rows=len(replay),old_eval_rows=len(oldret),target_rows=len(target),type_codes=list(DB.CODES),label_parser_source=str(DB.PARSER),raw_sha256=sha(DB.RAW),features='frozen 2020 pack; not historical input-version reconstruction',all_keys_unique=True,target_training_disjoint=True,preprocessing_note='existing fold0-fitted transforms; old heldout diagnostic shares unsupervised preprocessing',snapshot2023_agreement=float(np.mean(yf==frozen_y)),snapshot2023_types_agreement=float(np.mean(np.all(bf==frozen_b,axis=1))))
 write(O/'data_audit.json',audit);print('DATA',json.dumps(audit,ensure_ascii=False),flush=True)
 s0=copy.copy(s);s0['y'],s0['types']=y0,b0;s1=copy.copy(s);s1['y'],s1['types']=y1,b1;t0,t1=tensors(s0),tensors(s1);weights=formal.prev.cost_weights(select(s0,base));weights=(float(weights[0]),np.maximum(weights[1],1e-6).astype('f4'));write(O/'weights.json',dict(overall=weights[0],types=weights[1].tolist()))
 # Physical FP32 interface: 16+88+21 h coordinates, three q, eight audit code, byte r.
 payload=545;records=[];preds={};training_complete=False
 for seed in SEEDS:
  folder=O/f's{seed}';folder.mkdir(exist_ok=True);m=A.FilmKeepX3(formal,pack,seed);init=digest(m);initial=fullfit(m,t0,base,weights,seed);torch.save(m.state_dict(),folder/'initial.pt');print('INITIAL',seed,round(initial['seconds'],2),flush=True)
  cache,cachetime=encode(m,t1);m.eval()
  with torch.no_grad():
   ref=forward(m,t1,slice(0,2048));cached=cached_forward(m,cache,slice(0,2048));replayerr=max(float(torch.max(torch.abs(a-b))) for a,b in zip(ref,cached));assert replayerr==0
  torch.save({k:v for k,v in cache.items() if k in ['h1','h2','h3','c','r','q','industry']},folder/'cached_messages.pt')
  # Cache construction time here includes evaluation rows, conservatively charged in reported update time.
  models={'NO_UPDATE':copy.deepcopy(m)};runs={'NO_UPDATE':dict(seconds=0.,steps=0,exposures=0,trainable_parameters=0)}
  for arm,rp in [('FINETUNE',None),('REPLAY',replay)]:
   md=copy.deepcopy(m);runs[arm]=update(md,cache,t1,delta,rp,weights,seed);models[arm]=md;print(arm,seed,round(runs[arm]['seconds'],2),flush=True)
  full=A.FilmKeepX3(formal,pack,seed);assert digest(full)==init;runs['RETRAIN']=fullfit(full,t1,alltrain,weights,seed);runs['RETRAIN']['trainable_parameters']=sum(v.numel() for v in full.parameters());models['RETRAIN']=full;print('RETRAIN',seed,round(runs['RETRAIN']['seconds'],2),flush=True)
  # Evaluation labels are not passed to any optimizer or cache-forward selection.
  for arm,md in models.items():
   pdct=scores(md,t1);preds[(seed,arm)]=pdct;np.savez_compressed(folder/f'{arm}_scores.npz',logits=pdct);torch.save(md.state_dict(),folder/f'{arm}.pt')
   r=runs[arm];r.update(seed=seed,arm=arm,initial_sha256=init,initial_training_seconds=initial['seconds'],cache_build_seconds=cachetime,cache_replay_max_error=replayerr,parameters=sum(v.numel() for v in md.parameters()),model_sha256=digest(md),update_seconds_including_cache=r['seconds']+(cachetime if arm in ['FINETUNE','REPLAY'] else 0.),stored_initial_cache_payload_bytes=len(base)*payload,new_rows_upload_payload_bytes=(len(alltrain)-len(base))*payload if arm in ['FINETUNE','REPLAY'] else None)
   # Monolithic PyTorch retraining has no real network; count logical protocol tensor traffic only.
   if arm=='RETRAIN':r['logical_train_forward_payload_bytes']=r['exposures']*payload;r['logical_train_backward_payload_bytes']=r['exposures']*133*4
   else:r['logical_train_forward_payload_bytes']=r['new_rows_upload_payload_bytes'] or 0;r['logical_train_backward_payload_bytes']=0
   for subset,ids in [('future',target),('old',oldret)]:
    ss=dict(y=yf[ids],types=bf[ids]);met=metric(pdct[ids],ss,weights[1]);records.append(dict(seed=seed,arm=arm,subset=subset,**met,**{k:v for k,v in r.items() if k not in ['history','seed','arm']}))
   write(folder/f'{arm}_run.json',r)
  pd.DataFrame(records).to_csv(O/'real_metrics.csv',index=False)
 # Equal-weight seed point differences with paired bootstrap of companies in single future year.
 comparisons=[]
 for a,b in [('FINETUNE','NO_UPDATE'),('REPLAY','NO_UPDATE'),('REPLAY','FINETUNE'),('REPLAY','RETRAIN')]:
  da=pd.DataFrame(records);aa=da[(da.arm==a)&(da.subset=='future')].set_index('seed');bb=da[(da.arm==b)&(da.subset=='future')].set_index('seed');comparisons.append(dict(a=a,b=b,mean_recall_difference_pp=float((aa.recall10-bb.recall10).mean()*100),positive_seeds=int(sum(aa.recall10>bb.recall10))))
 write(O/'real_comparisons.json',comparisons)
 # Recompute scalar scores from saved files in a separate function; no ensemble inference claimed.
 checks=[]
 for row in records:
  pred=np.load(O/f's{row["seed"]}'/f'{row["arm"]}_scores.npz')['logits'];ids=target if row['subset']=='future' else oldret;met=metric(pred[ids],dict(y=yf[ids],types=bf[ids]),weights[1]);assert abs(met['recall10']-row['recall10'])<1e-12;checks.append(True)
 write(O/'real_verification.json',dict(saved_score_recomputations=len(checks),all_exact=all(checks),no_target_label_used_for_training=True,protocol_sha256=sha(H/'训练前约定.md'),source_sha256=sha(Path(__file__)),frozen_models_modified=False))
 return s,base,alltrain,(y0,y1,yf),audit

class Tiny(nn.Module):
 def __init__(self):
  super().__init__();self.enc=nn.ModuleList(nn.Sequential(nn.Linear(8,2),nn.Tanh()) for i in range(3));self.head=nn.Sequential(nn.Linear(6,16),nn.ReLU(),nn.Linear(16,1))
 def features(self,x):return torch.cat([m(x[:,i*8:(i+1)*8]) for i,m in enumerate(self.enc)],1)
 def forward(self,x):return self.head(self.features(x)).squeeze(1)
def tinyfit(m,x,y,steps,seed,source=None,replay=None,frozen=False):
 if frozen:
  for v in m.enc.parameters():v.requires_grad_(False)
  with torch.no_grad():z=m.features(x)
 else:z=x
 par=[v for v in m.parameters() if v.requires_grad];opt=torch.optim.AdamW(par,lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed);start=time.perf_counter()
 for st in range(steps):
  if replay is not None:ix=np.r_[rng.choice(source,128),rng.choice(replay,128)]
  else:ix=rng.choice(source if source is not None else len(y),256)
  opt.zero_grad();logit=m.head(z[ix]).squeeze(1) if frozen else m(x[ix]);loss=F.binary_cross_entropy_with_logits(logit,y[ix]);loss.backward();opt.step()
 return time.perf_counter()-start
@torch.no_grad()
def tinymetric(m,x,y):
 z=m(x).numpy();k=int(np.ceil(len(y)*.1));return dict(recall10=float(y[np.argsort(-z,kind='stable')[:k]].sum()/max(y.sum(),1)),ap=float(average_precision_score(y,z)))
def simulate(s,base,alltrain,labels,audit):
 y0,y1,yf=labels;keys=s['keys'];years=keys.economic_year.to_numpy();old=base;new=alltrain[~np.isin(alltrain,base)];p=float(yf[base].mean());c0=float(y0[base].sum()/max(yf[base].sum(),1));c1=float(y1[base].sum()/max(yf[base].sum(),1));cn=float(y1[new].sum()/max(yf[new].sum(),1));columns=[];xcols=[]
 for ni in range(3):
  idx=[j for j,n in enumerate(C.pack_for(2020,(0,))['names'][ni]) if 'missing::' not in n and not n.startswith('audit::')][:8];columns.append(idx);xcols.append(s['xs'][ni][base][:,idx])
 pool=np.concatenate(xcols,1);mu=pool.mean(0);sd=np.maximum(pool.std(0),.1);pool=np.clip((pool-mu)/sd,-8,8).astype('f4');rows=[]
 write(O/'simulation_parameters.json',dict(feature_column_indices=columns,n_initial=3000,n_new=1000,n_test=2000,finite_snapshot_positive_rate=p,old_initial_recording_probability=c0,old_updated_recording_probability=c1,new_recording_probability=cn,mechanisms=['stable','head_shift','new_coordinates'],signal_scale=1.5,note='empirical feature resampling plus imposed response/delay; not a fitted real-world mechanism'))
 for mech in ['stable','head_shift','new_coordinates']:
  for seed in SEEDS:
   rng=np.random.default_rng(seed+512);x=pool[rng.integers(0,len(pool),7000)].copy();x=torch.tensor(x);v=x.numpy();w=np.zeros(24);w[[0,8,16]]=[.9,-.7,.5];w2=w.copy()
   if mech=='head_shift':w2[[0,8,16]]=[-.6,.9,.6]
   if mech=='new_coordinates':w2[:]=0;w2[[6,14,22]]=[.9,-.7,.5]
   linear0=v@w;linear1=v@w2;std0=max(float(np.std(linear0[:3000])),.1);std1=max(float(np.std(linear1[:3000])),.1)
   # Match base prevalence using a fixed numerical intercept solve, not evaluation labels.
   def probabilities(z,std):
    raw=1.5*z/std;lo,hi=-15.,15.
    for _ in range(60):mid=(lo+hi)/2;val=np.mean(1/(1+np.exp(-(raw[:3000]+mid))));lo,hi=(mid,hi) if val<p else (lo,mid)
    return 1/(1+np.exp(-(raw+(lo+hi)/2)))
   pr0=probabilities(linear0,std0);pr1=probabilities(linear1,std1);prob=np.r_[pr0[:3000],pr1[3000:]];truth=(rng.random(7000)<prob).astype('f4');u=rng.random(7000);initial=truth[:3000]*(u[:3000]<c0);updated=truth[:4000]*np.r_[u[:3000]<c1,u[3000:4000]<cn];assert np.all(updated[:3000]>=initial)
   # New-year feature/response scenario retains initial labels on old rows, updating observation only.
   target=np.arange(4000,6000);oldtest=np.arange(6000,7000);oldtruth=(rng.random(1000)<pr0[oldtest]).astype('f4');delta=np.r_[np.flatnonzero(updated[:3000]!=initial),np.arange(3000,4000)];unch=np.flatnonzero(updated[:3000]==initial);rep=rng.choice(unch,min(512,len(unch)),False)
   torch.manual_seed(seed);m=Tiny();initial_state=copy.deepcopy(m.state_dict());initialsteps=int(np.ceil(3000/256))*30;tinyfit(m,x[:3000],torch.tensor(initial),initialsteps,seed);models={'NO_UPDATE':copy.deepcopy(m)};secs={'NO_UPDATE':0}
   for arm,rp in [('FINETUNE',None),('REPLAY',rep)]:
    mm=copy.deepcopy(m);secs[arm]=tinyfit(mm,x[:4000],torch.tensor(updated),60,seed,delta,rp,True);models[arm]=mm
   torch.manual_seed(seed);full=Tiny();full.load_state_dict(initial_state);secs['RETRAIN']=tinyfit(full,x[:4000],torch.tensor(updated),int(np.ceil(4000/256))*30,seed);models['RETRAIN']=full
   for arm,mm in models.items():
    for label,ids,ty in [('future',target,truth[target]),('old',oldtest,oldtruth)]:rows.append(dict(mechanism=mech,seed=seed,arm=arm,subset=label,seconds=secs[arm],**tinymetric(mm,x[ids],ty)))
   print('SIM',mech,seed,flush=True);pd.DataFrame(rows).to_csv(O/'simulation_metrics.csv',index=False)
 write(O/'simulation_complete.json',dict(settings=9,trained_models=36,evaluations=len(rows),source_sha256=sha(Path(__file__))))
if __name__=='__main__':
 s,base,alltrain,lab,audit=real();simulate(s,base,alltrain,lab,audit);write(O/'completion.json',dict(status='complete',real_initial_models=3,real_retrains=3,real_head_updates=6,simulation_settings=9,protocol_sha256=sha(H/'训练前约定.md')))
