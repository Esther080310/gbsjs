"""Eight fixed current-snapshot financial amounts, fitted using fold 0 only."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

COLUMNS=(('FS_Combas','A001000000'),('FS_Combas','A001111000'),('FS_Combas','A001123000'),('FS_Combas','A001212000'),('FS_Comins','B001101000'),('FS_Comins','B002000000'),('FS_Comscfd','C001000000'),('FS_Comscfd','C006000000'))

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def fit(raw):
 params=[]
 for j,(table,field) in enumerate(COLUMNS):
  x=np.asarray(raw[:,j],dtype=np.float64);x=x[np.isfinite(x)]
  if len(x)<100:raise ValueError(f'{table}.{field}: fold0 finite < 100 ({len(x)})')
  if np.all(x==x[0]):raise ValueError(f'{table}.{field}: fold0 constant')
  q01,q25,q75,q99=np.quantile(x,[.01,.25,.75,.99]);center=float(np.median(x));scale=float(q75-q25);method='IQR'
  if not np.isfinite(scale) or scale<=0:scale=float(1.4826*np.median(np.abs(x-center)));method='1.4826*MAD'
  if not np.isfinite(scale) or scale<=0:scale=float(np.std(x,ddof=0));method='std(ddof=0)'
  if not np.isfinite(scale) or scale<=0:scale=1.;method='1'
  params.append(dict(table=table,field=field,train_finite=len(x),q01=float(q01),q99=float(q99),center=center,scale=scale,scale_method=method))
 return params

def transform(raw,params):
 raw=np.asarray(raw,dtype=np.float64);finite=np.isfinite(raw);phi=np.zeros_like(raw)
 for j,p in enumerate(params):
  ok=finite[:,j];phi[ok,j]=np.arcsinh((np.clip(raw[ok,j],p['q01'],p['q99'])-p['center'])/p['scale'])
 phi=phi.astype(np.float32);missing=(~finite).astype(np.float32)
 if not np.isfinite(phi).all():raise ValueError('Nonfinite transformed financial amount')
 return phi,missing

def feature_variants(phi,missing):
 zero=np.zeros_like(phi)
 return dict(F0=np.concatenate([zero,zero],axis=1),FM=np.concatenate([zero,missing],axis=1),FV=np.concatenate([phi,missing],axis=1))

def load_financial_features(output_dir,target_year,fold,variant='FV'):
 """Return (n,16) float32 F block; concatenate with the trainer's frozen old X2."""
 if variant not in ('F0','FM','FV'):raise ValueError('variant must be F0, FM or FV')
 with np.load(Path(output_dir)/f'financial_features_{target_year}.npz',allow_pickle=False) as a:
  return a[f'{fold}_{variant}'].copy()

def boundary_checks():
 # Test all fallbacks and ensure zero is observed, negative amounts preserved, and validation never fits parameters.
 train=np.tile(np.arange(200,dtype=float)[:,None],(1,8))-100
 params=fit(train);p0=json.dumps(params,sort_keys=True)
 test=np.tile(np.array([0.,np.nan,-50.,np.inf])[:,None],(1,8));phi,m=transform(test,params)
 assert np.all(m[0]==0) and np.all(m[1]==1) and np.all(phi[1]==0)
 assert np.all(m[2]==0) and np.all(m[3]==1) and np.all(phi[2]<phi[0])
 assert json.dumps(params,sort_keys=True)==p0
 for v in feature_variants(phi,m).values():assert v.shape==(4,16) and v.dtype==np.float32
 sparse=np.zeros((200,8));sparse[-1]=1
 assert all(p['scale_method']=='std(ddof=0)' for p in fit(sparse))
 for bad in (np.zeros((200,8)),np.ones((99,8))):
  try:fit(bad)
  except ValueError:pass
  else:raise AssertionError('finite/constant guard missing')
 return True

def prepare(project_root):
 project_root=Path(project_root);source=project_root/'artifacts/followup/csmar/processed/20261016-v1/aligned';out=project_root/'artifacts/experiments/20261016-evidence-v1/financial_features';out.mkdir(parents=True,exist_ok=True)
 assert boundary_checks();report={'retrospective_only':True,'fit_fold':0,'boundary_checks_passed':True,'years':{}}
 names=np.array([f'{t}.{f}' for t,f in COLUMNS]);feature_names=np.array([f'phi__{n}' for n in names]+[f'missing__{n}' for n in names])
 for year in (2020,2021):
  path=source/f'raw_financial_alignment_{year}.npz';source_hash=sha256(path)
  with np.load(path,allow_pickle=False) as a:
   raw={}
   for fold in range(4):
    raw[fold]=np.column_stack([a[f'{fold}_{t}_values'][:,np.flatnonzero(a[t+'_names']==f).item()] for t,f in COLUMNS]).astype(np.float64)
   params=fit(raw[0]);arrays={'names':names,'feature_names':feature_names};diagnostics=[]
   for fold,x in raw.items():
    phi,missing=transform(x,params);arrays[f'{fold}_phi']=phi;arrays[f'{fold}_missing']=missing
    for variant,value in feature_variants(phi,missing).items():arrays[f'{fold}_{variant}']=value
    for key in ('symbol','year','y','types','row_index'):arrays[f'{fold}_{key}']=a[f'{fold}_{key}'].copy()
    assert np.array_equal(arrays[f'{fold}_row_index'],np.arange(len(x)))
    for j,n in enumerate(names):
     ok=np.isfinite(x[:,j]);observed=x[ok,j]
     diagnostics.append(dict(fold=fold,name=str(n),n=len(x),finite=int(ok.sum()),missing=int((~ok).sum()),zero=int(np.sum(observed==0)),negative=int(np.sum(observed<0)),below_q01=int(np.sum(observed<params[j]['q01'])),above_q99=int(np.sum(observed>params[j]['q99'])),transformed_zero=int(np.sum(phi[:,j]==0))))
   np.savez_compressed(out/f'financial_features_{year}.npz',**arrays)
   # Reopen the exact training interface, checking every fold's original sample references.
   with np.load(out/f'financial_features_{year}.npz',allow_pickle=False) as check:
    for fold in range(4):
     for key in ('symbol','year','y','types','row_index'):assert np.array_equal(check[f'{fold}_{key}'],a[f'{fold}_{key}'])
     assert check[f'{fold}_FV'].shape==(len(raw[fold]),16)
     assert np.array_equal(check[f'{fold}_missing'],(~np.isfinite(raw[fold])).astype(np.float32))
     assert np.all(check[f'{fold}_F0']==0)
   assert sha256(path)==source_hash
   detail=dict(source_path=str(path),source_sha256=source_hash,fit_fold=0,parameters=params,diagnostics=diagnostics,sample_order_Y_types_verified=True,retrospective_only=True)
   (out/f'fit_and_diagnostics_{year}.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2))
   report['years'][str(year)]={'rows_by_fold':{str(f):len(x) for f,x in raw.items()},'source_sha256':source_hash,'columns':len(names),'features_per_variant':16,'output_sha256':sha256(out/f'financial_features_{year}.npz')}
 (out/'preparation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 return report
