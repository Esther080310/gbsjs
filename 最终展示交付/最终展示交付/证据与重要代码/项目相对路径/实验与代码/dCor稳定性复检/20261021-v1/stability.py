from pathlib import Path
import copy,json,sys,time,hashlib
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'实验与代码/FiLM-L随机扰动副模型/20261021-v1'))
import noise_pilot as P
sys.path.insert(0,str(ROOT/'实验与代码/FiLM-L隐私防护/20261021-v2'))
import shared_privacy_experiment as E


def digest(model):return E.D.state_digest(model)

def strong_attack(x0,y0,x1,y1):
 xs,ys=P.A.Standardizer.fit(x0),P.A.Standardizer.fit(y0)
 P.A.seed_all(20261021)
 model=nn.Sequential(nn.Linear(x0.shape[1],512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,y0.shape[1]))
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 loader=DataLoader(TensorDataset(torch.tensor(xs.apply(x0)),torch.tensor(ys.apply(y0))),batch_size=1024,shuffle=True,generator=torch.Generator().manual_seed(20261021))
 valid=torch.tensor(xs.apply(x1));best=float('inf');stale=0;history=[];beststate=None
 for epoch in range(1,81):
  model.train()
  for xb,yb in loader:
   opt.zero_grad();loss=(model(xb)-yb).square().mean();assert torch.isfinite(loss);loss.backward();opt.step()
  model.eval()
  with torch.no_grad():pred=ys.undo(torch.cat([model(v) for v in valid.split(4096)]).numpy())
  score=P.A.macro_nrmse(y1,pred,ys.scale);history.append(dict(epoch=epoch,nrmse=score))
  if score<best-1e-5:best=score;stale=0;best_epoch=epoch;beststate={k:v.detach().clone() for k,v in model.state_dict().items()}
  else:
   stale+=1
   if stale>=8:break
 ridge,rr=P.A.fit_ridge(x0,y0,x1,y1)
 ridge_score=P.A.macro_nrmse(y1,ridge.predict(x1),ys.scale)
 shallow=P.numeric_attack(x0,x1,y0,y1)
 scores=dict(deep_mlp=best,ridge=ridge_score,shallow_mlp=shallow['attacker_nrmse']['mlp'])
 return dict(nrmse=min(scores.values()),selected_attacker=min(scores,key=scores.get),attacker_nrmse=scores,deep_history=history,deep_best_epoch=best_epoch,ridge_history=rr)


def construct(formal,pack,arm,seed):
 model=E.Shared(formal,pack,'none' if arm=='S0' else 'dcor',seed=seed)
 if seed!=42:
  model.heads=nn.ModuleList([E.Fusion(E.A.FilmKeepX3(formal,pack,(s+seed-42) % (2**32))) for s in E.C.SEEDS])
 torch.manual_seed(seed)
 return model


def fit(formal,pack,arm,seed):
 dest=OUT/f'{arm}_s{seed}'
 if (dest/'run.json').exists():
  m=construct(formal,pack,arm,seed);m.load_state_dict(torch.load(dest/'model.pt',weights_only=True));return m,json.loads((dest/'run.json').read_text())
 if dest.exists():dest.rmdir()
 dest.mkdir(exist_ok=False)
 model=construct(formal,pack,arm,seed);initial=digest(model)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 prev=formal.prev;tensors=[prev.tensor_split(s,'cpu') for s in pack['splits']];weights=prev.cost_weights(pack['splits'][0]);aw,tw=map(torch.tensor,weights)
 gen=torch.Generator().manual_seed(seed);history=[];batchhash=[];best=float('inf');stale=0;started=time.perf_counter()
 for epoch in range(1,31):
  model.train();order=torch.randperm(len(pack['splits'][0]['y']),generator=gen)
  batchhash.append(hashlib.sha256(order.numpy().tobytes()).hexdigest());ts=rs=0.
  for start in range(0,len(order),1024):
   idx=order[start:start+1024];batch=prev.batch(tensors[0],idx);y=tensors[0]['y'][idx]
   opt.zero_grad();model(*batch)
   al=nn.functional.binary_cross_entropy_with_logits(model.all_any,y[None,:].expand_as(model.all_any),pos_weight=aw,reduction='none').mean()
   types=tensors[0]['types'][idx];tl=nn.functional.binary_cross_entropy_with_logits(model.all_type,types[None,:,:].expand_as(model.all_type),pos_weight=tw,reduction='none').mean(2)
   task=al+(tl*y[None,:]).sum()/(y.sum().clamp_min(1)*6);reg=model.regularizer(batch[0]);loss=task+reg
   assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
   ts+=float(task.detach())*len(idx);rs+=float(reg.detach())*len(idx)
  val=E.extract(model,pack['splits'][1]);score=prev.selection_loss(val['a'],val['b'],pack['splits'][1],weights)
  history.append(dict(epoch=epoch,selection_loss=score,task=ts/len(order),regularizer=rs/len(order)))
  if score<best-1e-4:best=score;stale=0;best_epoch=epoch;torch.save(model.state_dict(),dest/'model.pt')
  else:
   stale+=1
   if stale>=6:break
  print('TRAIN',arm,seed,epoch,round(score,5),flush=True)
 model.load_state_dict(torch.load(dest/'model.pt',weights_only=True));pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
 rec=dict(arm=arm,seed=seed,initial_hash=initial,final_hash=digest(model),epoch_batch_hashes=batchhash,epochs=len(history),best_epoch=best_epoch,seconds=time.perf_counter()-started,parameters=sum(v.numel() for v in model.parameters()))
 P.write(dest/'run.json',rec);return model,rec


def evaluate(formal,pack,model,arm,seed):
 dest=OUT/f'{arm}_s{seed}';dest.mkdir(exist_ok=True)
 if (dest/'evaluation.json').exists():return json.loads((dest/'evaluation.json').read_text())
 ex=[E.extract(model,s) for s in pack['splits']];y0,names=P.numeric_targets(pack['splits'][0],pack);y1,_=P.numeric_targets(pack['splits'][1],pack)
 task=P.EV.task_metrics(pack['splits'][0],pack['splits'][1],ex[1]['a'],ex[1]['b'])
 attack=strong_attack(ex[0]['full_deployment'],y0,ex[1]['full_deployment'],y1)
 np.savez_compressed(dest/'validation_predictions.npz',a_logit=ex[1]['a'],b_logit=ex[1]['b'])
 r=dict(arm=arm,seed=seed,recall=task['recall_at_10pct'],ap=task['overall_average_precision'],numeric_nrmse=attack['nrmse'],task=task,attack=attack,target_fields=len(names))
 P.write(dest/'evaluation.json',r);print('EVAL',arm,seed,r['recall'],r['ap'],r['numeric_nrmse'],attack['selected_attacker'],flush=True);return r


def main():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
 formal=P.D.C.shared.load_loader().formal;pack=P.D.C.pack_for(2020,(0,1));rows=[]
 if not (OUT/'manifest.json').exists():P.write(OUT/'manifest.json',dict(year=2020,folds_loaded=[0,1],protocol_sha=P.D.file_digest(HERE/'复检前约定.md'),source_sha=P.D.file_digest(Path(__file__)),old_exploratory_seed=42,new_seeds=list(E.C.SEEDS[1:])))
 for arm in ('S0','D1'):
  model=E.load(formal,pack,arm);rows.append(evaluate(formal,pack,model,arm,42))
  assert digest(construct(formal,pack,'S0',42))==digest(construct(formal,pack,'D1',42))
 relative=rows[1]['numeric_nrmse']/rows[0]['numeric_nrmse']-1
 gate=dict(stageA_relative_error_increase=relative,passed=relative>=.1,new_risk_networks_trained=False)
 P.write(OUT/'stageA_gate.json',gate)
 if not gate['passed']:
  P.write(OUT/'summary.json',dict(status='stop_attack_strength_gate',rows=rows,gate=gate));return
 pairchecks=[]
 for seed in E.C.SEEDS[1:]:
  records=[]
  for arm in ('S0','D1'):
   model,run=fit(formal,pack,arm,seed);records.append(run);rows.append(evaluate(formal,pack,model,arm,seed))
  left,right=records;assert left['initial_hash']==right['initial_hash'];n=min(left['epochs'],right['epochs']);assert left['epoch_batch_hashes'][:n]==right['epoch_batch_hashes'][:n];assert left['parameters']==right['parameters']
  pairchecks.append(dict(seed=seed,initial_exact=True,common_batch_prefix_exact=True,parameters_exact=True))
  P.write(OUT/'pair_checks.json',pairchecks)
 pairs=[]
 for seed in E.C.SEEDS:
  a=next(r for r in rows if r['seed']==seed and r['arm']=='S0');d=next(r for r in rows if r['seed']==seed and r['arm']=='D1')
  pairs.append(dict(seed=seed,recall_S0=a['recall'],recall_D1=d['recall'],recall_difference=d['recall']-a['recall'],ap_difference=d['ap']-a['ap'],nrmse_S0=a['numeric_nrmse'],nrmse_D1=d['numeric_nrmse'],relative_error_increase=d['numeric_nrmse']/a['numeric_nrmse']-1))
 new=[r for r in pairs if r['seed']!=42];privacy=sum(r['relative_error_increase']>=.1 for r in new)>=4 and np.mean([r['relative_error_increase'] for r in new])>=.1
 utility=np.mean([r['recall_difference'] for r in new])>=-.005 and sum(r['recall_difference']<-.01 for r in new)<=1 and np.mean([r['ap_difference'] for r in new])>=-.01
 pd.DataFrame(pairs).to_csv(OUT/'paired_results.csv',index=False)
 P.write(OUT/'summary.json',dict(status='complete',rows=rows,pairs=pairs,new_seed_privacy_stable=bool(privacy),new_seed_utility_preserved=bool(utility),eligible_for_future_cross_year=bool(privacy and utility),pairchecks=pairchecks,gate=gate,folds_loaded=[0,1],fold2_loaded=False,fold3_loaded=False))

if __name__=='__main__':main()
