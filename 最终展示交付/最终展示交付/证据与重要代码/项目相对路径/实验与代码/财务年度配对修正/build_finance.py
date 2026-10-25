"""Build annual financial inputs independently of next-year market membership."""
from pathlib import Path
import json,hashlib,zipfile
import numpy as np
import pandas as pd
from lxml import etree
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'data'
NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main';T='{'+NS+'}'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def value(c,strings):
    if c.get('t')=='inlineStr':return ''.join(c.itertext())
    v=c.find(T+'v')
    if v is None:return ''
    s=v.text or '';return strings[int(s)] if c.get('t')=='s' else s
def number(s):
    try:return float(s)
    except (ValueError,TypeError):return np.nan
def scan(path,keys,prefix):
    best={};allrows=annualrows=0;header={};fields=[]
    with zipfile.ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            with z.open('xl/sharedStrings.xml') as stream:
                for _,e in etree.iterparse(stream,events=('end',),tag=T+'si'):
                    strings.append(''.join(e.itertext()));e.clear()
                    while e.getprevious() is not None:del e.getparent()[0]
        with z.open('xl/worksheets/sheet1.xml') as stream:
            for _,r in etree.iterparse(stream,events=('end',),tag=T+'row'):
                rn=int(r.get('r'));cells=list(r)
                if rn==1:
                    header={c.get('r').rstrip('0123456789'):value(c,strings) for c in cells}
                    assert header.get('A')=='Stkcd' and header.get('C')=='Accper'
                    fields=[name for name in header.values() if name.startswith('F')]
                elif rn>=4:
                    allrows+=1
                    first={c.get('r').rstrip('0123456789'):c for c in cells[:5]}
                    if 'A' in first and 'C' in first:
                        symbol=value(first['A'],strings).removesuffix('.0').zfill(6);period=value(first['C'],strings)
                        if len(period)>=10 and period[:4].isdigit() and period[5:10]=='12-31':
                            annualrows+=1;year=int(period[:4]);key=(symbol,year)
                            if key in keys:
                                typ=value(first['D'],strings) if 'D' in first else ''
                                source=value(first['E'],strings) if 'E' in first else ''
                                priority=(int(typ=='A'),int(number(source)==0))
                                if key not in best or priority>best[key][0]:
                                    values={header.get(c.get('r').rstrip('0123456789'),''):number(value(c,strings)) for c in cells if header.get(c.get('r').rstrip('0123456789'),'').startswith('F')}
                                    vals={prefix+'__'+name:values.get(name,np.nan) for name in fields}
                                    best[key]=(priority,{'Symbol':symbol,'economic_year':year,**vals},{'table':path.stem,'Symbol':symbol,'economic_year':year,'excel_row':rn,'Typrep':typ,'Source':source})
                r.clear()
                while r.getprevious() is not None:del r.getparent()[0]
    print('Scanned',path.stem,allrows,'retained',len(best),flush=True)
    frame=pd.DataFrame([b[1] for b in best.values()]);trace=[b[2] for b in best.values()]
    return frame,trace,{'table':path.stem,'rows_scanned':allrows,'annual_rows':annualrows,'retained_keys':len(best),'field_count':len(fields)}
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[next((ROOT/'大创数据集').rglob(name+'.xlsx')) for name in ['FI_T1','FI_T4','FI_T5']]
    sources=[*paths,ROOT/'data_v16_canonical/x1.csv.gz',ROOT/'data_v16_canonical/x2.csv.gz',HERE/'build_finance.py',HERE/'执行约定.md']
    before={str(p):sha(p) for p in sources}
    market=pd.read_csv(ROOT/'data_v16_canonical/x1.csv.gz',dtype={'Symbol':'string'},usecols=['Symbol','feature_year']).rename(columns={'feature_year':'economic_year'})
    market.Symbol=market.Symbol.str.zfill(6);assert not market.duplicated().any()
    keyset=set(map(tuple,market.to_numpy()));out=market.copy();traces=[];inventory=[]
    for path,prefix in zip(paths,['solv','oper','prof']):
        f,trace,stats=scan(path,keyset,prefix);out=out.merge(f,on=['Symbol','economic_year'],how='left',validate='one_to_one');traces+=trace;inventory.append(stats)
    names=[c for c in out if '__' in c];out[names]=out[names].replace([np.inf,-np.inf],np.nan)
    # Round-trip parsing preserves the stored float64 values when checking source identity.
    old=pd.read_csv(ROOT/'data_v16_canonical/x2.csv.gz',dtype={'Symbol':'string'},float_precision='round_trip');old['economic_year']=old.feature_year-1
    old=old[['Symbol','economic_year',*names]];compare=out.merge(old,on=['Symbol','economic_year'],suffixes=('_new','_old'),how='left',validate='one_to_one',indicator=True)
    existing=compare._merge.eq('both');a=compare[[c+'_new' for c in names]].to_numpy('f8');b=compare[[c+'_old' for c in names]].to_numpy('f8')
    identical=(a==b)|(np.isnan(a)&np.isnan(b));changes=(~identical)&existing.to_numpy()[:,None]
    details=[]
    for i,j in zip(*np.where(changes)):
        details.append({'Symbol':compare.iloc[i].Symbol,'economic_year':int(compare.iloc[i].economic_year),'field':names[j],'new':a[i,j],'old':b[i,j]})
    pd.DataFrame(details,columns=['Symbol','economic_year','field','new','old']).to_csv(OUT/'existing_value_changes.csv',index=False)
    assert not changes.any(),f'{int(changes.sum())} existing cells differ; inspect before training'
    recovered=(~existing)&np.isfinite(a).any(1)
    missing=compare.loc[~existing,['Symbol','economic_year']].copy();missing['recovered_numeric_fields']=np.isfinite(a[~existing]).sum(1)
    missing.to_csv(OUT/'missing_key_recovery.csv',index=False)
    out.to_csv(OUT/'x2_annual.csv.gz',index=False,compression='gzip');pd.DataFrame(traces).to_csv(OUT/'raw_finance_selected_rows.csv.gz',index=False,compression='gzip')
    assert before=={str(p):sha(p) for p in sources}
    stats={'sources':before,'scans':inventory,'samples':len(out),'raw_financial_fields':len(names),'existing_key_rows':int(existing.sum()),'existing_numeric_cells_checked':int(existing.sum()*len(names)),'existing_cell_changes':int(changes.sum()),'newly_recovered_company_years':int(recovered.sum()),'recovery_by_year':compare.loc[recovered].groupby('economic_year').size().to_dict(),'input_hashes_unchanged':True,'key_definition':'same market/sample year; no next-year membership condition'}
    (OUT/'verification.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    print('ANNUAL FINANCE COMPLETE',json.dumps({k:v for k,v in stats.items() if k not in ['sources','scans']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
