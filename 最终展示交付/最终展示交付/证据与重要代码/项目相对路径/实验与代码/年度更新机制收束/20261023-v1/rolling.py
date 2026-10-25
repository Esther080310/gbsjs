from pathlib import Path
import sys,json,copy,time,hashlib
import numpy as np,pandas as pd,torch
H=Path(__file__).resolve().parent;O=H/'results';O.mkdir(exist_ok=True)
P=H.parents[1]/'延迟标签增量更新/20261023-v1';sys.path.insert(0,str(P));import run_pilot as R
from mechanism import CONFIG,audit_candidate,policy_state,consume_policy,cache_key

def w(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def model(seed,pack,formal,state):
 m=R.A.FilmKeepX3(formal,pack,seed);m.load_state_dict(state);return m

def main():
 torch.set_num_threads(4);pack=R.C.pack_for(2020,(0,1,2,3));s=R.join(pack['splits']);keys=s['keys'];years=keys.economic_year.to_numpy();old=R.C.pack_for(2020,(0,1,2));prior=R.join(old['splits']);n=len(prior['y'])
 for i in range(3):np.testing.assert_array_equal(s['xs'][i][:n],prior['xs'][i])
 np.testing.assert_array_equal(keys.iloc[:n].to_numpy(),prior['keys'].to_numpy());assert not keys.duplicated().any()
 ev,_=R.DB.build_events();labels={}
 for cut in ['2019-04-30','2020-04-30','2021-04-30','2023-12-31']:
  first,typed=R.DB.first_records(ev,pd.Timestamp(cut));labels[cut]=R.DB.aligned(keys,first,typed)[:2]
 yf,bf=labels['2023-12-31'];np.testing.assert_array_equal(yf,s['y']);np.testing.assert_array_equal(bf,s['types'])
 hold=keys.Symbol.map(lambda x:int(hashlib.sha256(str(x).encode()).hexdigest()[:8],16)%5==0).to_numpy();pub=pd.read_csv(R.DB.PUBLICATION,dtype={'Symbol':str});pub.Symbol=pub.Symbol.str.zfill(6);date=pd.to_datetime(keys.merge(pub[['Symbol','economic_year','publication_date_min']],on=['Symbol','economic_year'],how='left',validate='one_to_one').publication_date_min)
 base=np.flatnonzero((years<=2017)&~hold&date.le('2019-04-30').to_numpy());ledger=np.load(P/'results/label_ledger.npz');np.testing.assert_array_equal(base,ledger['base_index']);wg=json.loads((P/'results/weights.json').read_text());weights=(wg['overall'],np.asarray(wg['types'],dtype='f4'));formal=R.C.shared.load_loader().formal
 # Same-age recorded rates and input marginal shifts: descriptive, never a route condition.
 diagnostics=[];fieldrows=[]
 for year in [2017,2018,2019,2020]:
  ix=np.flatnonzero(years==year);cut=f'{year+2}-04-30';first,typed=R.DB.first_records(ev,pd.Timestamp(cut));yy,tt,_=R.DB.aligned(keys.iloc[ix],first,typed)
  diagnostics.append(dict(year=year,rows=len(ix),recording_cutoff=cut,recorded_rate_16months=float(yy.mean()),type_share_among_recorded={c:float(tt[yy==1,j].mean()) for j,c in enumerate(R.DB.CODES)}))
  for node in range(3):
   for j,name in enumerate(pack['names'][node]):
    if 'missing' in name or name.startswith('audit::'):continue
    ref=s['xs'][node][base,j];v=s['xs'][node][ix,j];fieldrows.append(dict(year=year,node=node+1,field=name,standardized_mean_difference=float(abs(v.mean()-ref.mean())/max(float(ref.std()),.1))))
 w(O/'annual_label_description.json',diagnostics);pd.DataFrame(fieldrows).to_csv(O/'annual_feature_description.csv',index=False)
 records=[];decisions=[];versions=[];monitorrows=[]
 for seed in [42,100,17]:
  init=torch.load(P/f'results/s{seed}/initial.pt',weights_only=True);active={a:model(seed,pack,formal,init) for a in ['STATIC','INCREMENT','ADAPTIVE']};state={a:dict(indices=base.copy(),cutoff='2019-04-30',version=f's{seed}-initial',policy_done={}) for a in active}
  folder=O/f's{seed}';folder.mkdir(exist_ok=True);torch.save(init,folder/'initial.pt')
  for step,(cut,trainyear,testyear) in enumerate([('2020-04-30',2018,2019),('2021-04-30',2019,2020)]):
   yy,bb=labels[cut];sv=dict(s,y=yy,types=bb);t=R.tensors(sv);tr=np.flatnonzero((years<=trainyear)&~hold&date.le(cut).to_numpy());target=np.flatnonzero(years==testyear)
   mons={g:np.flatnonzero((years==year)&hold&date.le(cut).to_numpy()) for g,year in [('anchor',2017),('latest',trainyear)]};assert all(not np.intersect1d(tr,ix).size for ix in [target,*mons.values()])
   def mon(pred):return {g:R.metric(pred[ix],dict(y=yy[ix],types=bb[ix]),weights[1]) for g,ix in mons.items()}
   # All routes access current visible labels only. Final evaluation is below route selection.
   static_pred=R.scores(active['STATIC'],t);eval_models={'STATIC':active['STATIC']};costs={'STATIC':dict(seconds=0.,exposures=0,forward_payload=0,backward_payload=0)}
   if step==0:
    fullstate=torch.load(P/f'results/s{seed}/RETRAIN.pt',weights_only=True);full=model(seed,pack,formal,fullstate);fr=json.loads((P/f'results/s{seed}/RETRAIN_run.json').read_text());reused=True
    np.testing.assert_array_equal(tr,ledger['update_index'])
   else:
    full=R.A.FilmKeepX3(formal,pack,seed);fr=R.fullfit(full,t,tr,weights,seed);reused=False
   full_pred=R.scores(full,t);eval_models['FULL']=full;fullcost=dict(seconds=float(fr['seconds']),exposures=int(fr['exposures']),forward_payload=int(fr['exposures']*545),backward_payload=int(fr['exposures']*532));costs['FULL']=fullcost
   fullpath=folder/f'{cut}-FULL.pt';torch.save(full.state_dict(),fullpath)
   # Joint snapshots are keyed by entity-year, not by presumed contiguous row numbers.
   for arm in ['INCREMENT','ADAPTIVE']:
    parent=active[arm];info=state[arm];py,pb=labels[info['cutoff']];revised=(yy!=py)|np.any(bb!=pb,axis=1);oldmask=np.isin(tr,info['indices']);delta=tr[(~oldmask)|revised[tr]];unchanged=info['indices'][~revised[info['indices']]];replay=np.random.default_rng(712+step).choice(unchanged,min(2048,len(unchanged)),False)
    parent_pred=R.scores(parent,t);before=mon(parent_pred);cache,ct=R.encode(parent,t);candidate=copy.deepcopy(parent);ur=R.update(candidate,cache,t,delta,replay,weights,seed);candidate_pred=R.scores(candidate,t);candidate_monitor=mon(candidate_pred);audit=audit_candidate(candidate_monitor,before)
    support={p['id']:int(np.sum((years[tr]>=p['first_full_cohort'])&(yy[tr]>.5))) for p in CONFIG['policies']};policy=policy_state(cut,support,info['policy_done']);route='INCREMENT';selected=candidate;candidatecost=dict(seconds=float(ct+ur['seconds']),exposures=int(ur['exposures']),forward_payload=int(sum(~oldmask)*545),backward_payload=0);cost=candidatecost.copy()
    if arm=='ADAPTIVE' and (policy['prefer_full'] or not audit['accept']):
     full_audit=audit_candidate(mon(full_pred),before);cost={k:candidatecost[k]+fullcost[k] for k in candidatecost}
     if full_audit['accept']:selected=full;route='FULL'
     else:selected=parent;route='KEEP'
    else:full_audit=None
    if arm=='INCREMENT':audit_applied=False
    else:audit_applied=True
    vname=f's{seed}-{cut}-{arm}-{route}';path=folder/f'{cut}-{arm}.pt';torch.save(selected.state_dict(),path);parentencoder=R.enc_hash(parent);selectedencoder=R.enc_hash(selected)
    checks=dict(candidate_encoder_unchanged=R.enc_hash(candidate)==parentencoder,cache_invalidated=(selectedencoder!=parentencoder),new_cache_key=cache_key(selectedencoder,'2020-frozen-preprocessing','2020-frozen-features'))
    assert checks['candidate_encoder_unchanged']
    versions.append(dict(version=vname,parent=info['version'],cutoff=cut,arm=arm,route=route,model_path=str(path),model_sha256=R.sha(path),encoder_hash=selectedencoder,**checks,publication_status='research_only',production_checks={k:False for k in CONFIG['release_checks_required']}))
    for group,met in candidate_monitor.items():monitorrows.append(dict(seed=seed,cutoff=cut,arm=arm,group=group,role='candidate',**met))
    for group,met in before.items():monitorrows.append(dict(seed=seed,cutoff=cut,arm=arm,group=group,role='parent',**met))
    decisions.append(dict(seed=seed,cutoff=cut,arm=arm,route=route,increment_gate=audit,gate_applied=audit_applied,full_gate=full_audit,policy=policy,policy_support=support,delta_rows=len(delta),revised_old_rows=int(sum(revised[info['indices']])),new_rows=int(sum(~oldmask)),replay_rows=len(replay),parent_version=info['version'],candidate_version=vname,cache_invalidated=checks['cache_invalidated'],cost=cost))
    active[arm]=selected;eval_models[arm]=selected;costs[arm]=cost
    if route!='KEEP':state[arm]=dict(indices=tr.copy(),cutoff=cut,version=vname,policy_done=consume_policy(policy,info['policy_done'],route=='FULL'))
    print('ROUTE',seed,cut,arm,route,audit['reasons'],flush=True)
   for arm,m in eval_models.items():
    pred=R.scores(m,t);np.savez_compressed(folder/f'{cut}-{arm}-evaluation.npz',future_logits=pred[target],future_symbol=keys.Symbol.iloc[target].to_numpy(str),future_y=yf[target],future_types=bf[target],old_logits=pred[mons['anchor']],old_y=yf[mons['anchor']],old_types=bf[mons['anchor']])
    for subset,ix in [('future',target),('old',mons['anchor'])]:records.append(dict(seed=seed,cutoff=cut,evaluation_year=testyear,arm=arm,subset=subset,**R.metric(pred[ix],dict(y=yf[ix],types=bf[ix]),weights[1]),**costs[arm]))
   pd.DataFrame(records).to_csv(O/'rolling_metrics.csv',index=False);pd.DataFrame(monitorrows).to_csv(O/'monitor_metrics.csv',index=False);w(O/'decisions.json',decisions);w(O/'versions.json',versions)
 w(O/'completion.json',dict(status='complete',seeds=[42,100,17],updates=2,initial_checkpoints_reused=3,first_full_checkpoints_reused=3,new_full_trainings=3,new_head_trainings=12,policy_effectiveness_tested=False,policy_pending_branches_exercised=True,final_year_labels_used_for_routing=False,source_sha256=R.sha(Path(__file__)),protocol_sha256=R.sha(H/'回放前约定.md'),config_sha256=R.sha(H/'config.json')))
 print('ROLLING COMPLETE',flush=True)
if __name__=='__main__':main()
