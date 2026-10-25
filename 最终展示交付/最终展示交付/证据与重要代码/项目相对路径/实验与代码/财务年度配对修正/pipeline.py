"""Explicit annual-finance loader and the unchanged clipped-asinh preprocessing."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'节点表格网络与分段训练'))
import economic_risk_residual as formal
prev=formal.prev
FINANCE=HERE/'data/x2_annual.csv.gz'

def load_data(max_year,corrected=True):
    d=prev.load_data(max_year)
    if not corrected:return d
    finance=pd.read_csv(FINANCE,dtype={'Symbol':'string'})
    f=d['keys'].merge(finance,on=['Symbol','economic_year'],how='left',validate='one_to_one')
    raw=f[d['names'][1]].to_numpy('f4');raw[~np.isfinite(raw)]=np.nan
    d['raw'][1]=raw;d['q']=d['q'].copy();d['q'][:,1]=np.isfinite(raw).mean(1).astype('f4')
    return d

def prepare(year,corrected=True):
    d=load_data(year,corrected);years=d['keys'].economic_year.to_numpy();train=years<=year-3
    transformed=[prev.transform_node(v,n,train) for v,n in zip(d['raw'],d['names'])]
    names=[v[1] for v in transformed];meta=[v[2] for v in transformed]
    # Identical operation and float32 boundaries to scale.prepare + ClipAsinh.
    z=transformed[1][0].copy();rawmap={n:i for i,n in enumerate(d['names'][1])}
    bounds=dict(zip(meta[1]['selected_raw'],zip(meta[1]['low'],meta[1]['high'])))
    stats=[]
    for j,name in enumerate(names[1]):
        if name.startswith('missing::'):continue
        raw=d['raw'][1][:,rawmap[name]].astype(float);observed=np.isfinite(raw);t=raw[train&observed]
        center=float(np.median(t));scale=float(np.quantile(t,.75)-np.quantile(t,.25));kind='IQR'
        if scale<=0:scale=float(1.4826*np.median(abs(t-center)));kind='MAD'
        if scale<=0:scale=float(t.std());kind='std'
        if scale<=0:scale=1.;kind='unit'
        lo,hi=bounds[name];v=((np.where(observed,np.clip(raw,lo,hi),center)-center)/scale).astype('f4')
        z[:,j]=np.arcsinh(v.astype(float)).astype('f4')
        stats.append({'field':name,'median':center,'scale':scale,'scale_kind':kind,'low':float(lo),'high':float(hi)})
    transformed[1]=(z,names[1],meta[1]);splits=[]
    for mask in (train,years==year-2,years==year-1,years==year):
        ix=np.flatnonzero(mask)
        splits.append({'xs':[v[0][ix] for v in transformed],**{k:d[k][ix] for k in ('q','state','industry','y','types')},'keys':d['keys'].iloc[ix].copy()})
    counts=[i for i,c in enumerate(names[2]) if c.removeprefix('missing::') in {'exec__events','exec__resign','exec__chair','exec__manager','exec__external'}]
    return {'splits':splits,'names':names,'events':(counts,[i for i in range(len(names[2])) if i not in counts]),'meta':meta,'robust_scale':stats}

def verify_old_cache(year):
    p=prepare(year,False);cache=np.load(ROOT/f'全局预测能力提升/cache/{year}.npz')
    for f,s in enumerate(p['splits']):
        for k in ('industry','q','state','y','types'):np.testing.assert_array_equal(s[k],cache[f'{f}_{k}'])
        for j,x in enumerate(s['xs']):np.testing.assert_array_equal(x,cache[f'{f}_x{j}'])
        np.testing.assert_array_equal(s['keys'].Symbol.to_numpy().astype(str),cache[f'{f}_symbol'])
        np.testing.assert_array_equal(s['keys'].economic_year,cache[f'{f}_year'])
    return p

def export(year,p):
    folder=HERE/'cache';folder.mkdir(exist_ok=True);data={}
    for f,s in enumerate(p['splits']):
        for k in ('industry','q','state','y','types'):data[f'{f}_{k}']=s[k]
        for j,x in enumerate(s['xs']):data[f'{f}_x{j}']=x
        data[f'{f}_symbol']=s['keys'].Symbol.to_numpy().astype(str);data[f'{f}_year']=s['keys'].economic_year.to_numpy()
    np.savez_compressed(folder/f'{year}.npz',**data)
    prev.save_json(folder/f'{year}.json',{'names':p['names'],'meta':p['meta'],'robust_scale':p['robust_scale'],'events':p['events'],'finance_path':str(FINANCE.resolve()),'finance_sha256':prev.sha(FINANCE)})

if __name__=='__main__':
    checks=[]
    for year in (2020,2021):
        old=verify_old_cache(year);new=prepare(year,True);export(year,new)
        assert old['names']==new['names'],'feature set changes require inspection'
        raw_old=load_data(year,False);raw_new=load_data(year,True)
        years=raw_old['keys'].economic_year.to_numpy()
        identical=(raw_old['raw'][1]==raw_new['raw'][1])|(np.isnan(raw_old['raw'][1])&np.isnan(raw_new['raw'][1]))
        for f,(a,b) in enumerate(zip(old['splits'],new['splits'])):
            for k in ('industry','state','y','types'):np.testing.assert_array_equal(a[k],b[k])
            for j in (0,2):np.testing.assert_array_equal(a['xs'][j],b['xs'][j])
            mask=[years<=year-3,years==year-2,years==year-1,years==year][f]
            changed=(~identical).any(1)&mask
            checks.append({'year':year,'fold':f,'rows':len(a['y']),'raw_changed_rows':int(changed.sum()),'raw_changed_cells':int((~identical)[mask].sum()),'transformed_changed_rows':int((a['xs'][1]!=b['xs'][1]).any(1).sum()),'q2_changed_rows':int((a['q'][:,1]!=b['q'][:,1]).sum())})
        print('Verified and exported',year,flush=True)
    pd.DataFrame(checks).to_csv(HERE/'data/cache_changes.csv',index=False)
    prev.save_json(HERE/'data/cache_verification.json',{'old_cache_reproduced_exactly':True,'unchanged_X1_X3_industry_labels_samples':True,'training_preprocessing_refitted':True,'same_feature_names':True,'years':[2020,2021]})
