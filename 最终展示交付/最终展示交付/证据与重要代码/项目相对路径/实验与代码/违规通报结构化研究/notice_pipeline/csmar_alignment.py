"""Read-only CSMAR annual linkage; current exports are retrospective observations."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd

FINANCIAL = ('FS_Combas', 'FS_Comins', 'FS_Comscfd')
META = ('__source_member','__source_record_index','__line_start','__line_end','__raw_sha256','__parse_method','__content_sha256')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def complete_date(value):
    s=str(value).strip()[:10]
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',s): return None
    try: return pd.Timestamp(s).date().isoformat()
    except (ValueError,TypeError): return None

def correction_dates(value):
    parts=re.split(r'[,，;；、]+',str(value).strip())
    dates=[complete_date(p.strip()) for p in parts]
    return (max(dates),True) if dates and all(dates) else ('',False)

def symbol(s):
    s=str(s).strip()
    return s.zfill(6) if s.isdigit() and len(s)<=6 else s

def load(path):
    return pd.read_csv(path,dtype=str,keep_default_na=False)

def keyed(df,date='Accper'):
    df=df.copy(); df['Symbol']=df.Stkcd.map(symbol)
    df['economic_year']=pd.to_numeric(df[date].str[:4],errors='coerce').astype('Int64')
    return df

def deduplicate(df, keys):
    """Exact content duplicates collapse; different full contents never choose a winner."""
    df=df.copy()
    content=[c for c in df if not c.startswith('__') and c not in ('Symbol','economic_year')]
    if '__content_sha256' not in df:
        df['__content_sha256']=[hashlib.sha256(json.dumps(list(r),ensure_ascii=False).encode()).hexdigest() for r in df[content].itertuples(index=False,name=None)]
    df['source_id']=[json.dumps([r.get('__source_member',''),r.get('__source_record_index',''),r['__content_sha256']],ensure_ascii=False) for r in df.to_dict('records')]
    groups=df.groupby(keys+['__content_sha256'],dropna=False,sort=False)
    agg=groups.agg(duplicate_count=('source_id','size'),source_ids=('source_id',lambda x:json.dumps(list(x),ensure_ascii=False))).reset_index()
    unique=df.drop_duplicates(keys+['__content_sha256']).merge(agg,on=keys+['__content_sha256'],validate='one_to_one')
    unique['candidate_count']=unique.groupby(keys,dropna=False)['__content_sha256'].transform('size')
    unique['content_conflict']=unique.candidate_count.gt(1)
    return unique

def annual_candidates(df, financial=True):
    df=keyed(df)
    df=df[df.Accper.str[:10].str.endswith('-12-31')]
    if financial: df=df[df.Typrep.eq('A')]
    keys=['Stkcd','Accper']+(['Typrep'] if financial else [])
    return deduplicate(df,keys)

def publication_table(df):
    df=keyed(df); df=df[df.Reptyp.eq('4') & df.Accper.str[:10].str.endswith('-12-31')].copy()
    df['complete']=df.Annodt.map(complete_date)
    rows=[]
    for (sym,yr),g in df.groupby(['Symbol','economic_year'],sort=True):
        dates=sorted(set(g.complete.dropna()))
        rows.append(dict(Symbol=sym,economic_year=int(yr),publication_date_min=dates[0] if dates else '',publication_dates=json.dumps(dates),publication_incomplete=bool(g.complete.isna().any()),publication_raw_dates=json.dumps(sorted(set(g.Annodt))),publication_record_count=len(g),publication_source_ids=json.dumps(g[[c for c in META if c in g]].to_dict('records'),ensure_ascii=False)))
    return pd.DataFrame(rows,columns=['Symbol','economic_year','publication_date_min','publication_dates','publication_incomplete','publication_raw_dates','publication_record_count','publication_source_ids'])

def cache_samples(path, target):
    a=np.load(path,allow_pickle=False); rows=[]
    for fold in range(4):
        for i,(s,y,v,t) in enumerate(zip(a[f'{fold}_symbol'],a[f'{fold}_year'],a[f'{fold}_y'],a[f'{fold}_types'])):
            r=dict(target_year=target,fold=fold,row_index=i,Symbol=symbol(s),economic_year=int(y),Y=float(v))
            r.update({f'type_{j}':float(x) for j,x in enumerate(t)}); rows.append(r)
    return pd.DataFrame(rows)

def align_samples(samples,pubs,candidates):
    out=samples.merge(pubs,on=['Symbol','economic_year'],how='left',sort=False,validate='many_to_one')
    out['scoring_cutoff']=(out.economic_year+1).astype(str)+'-06-30'
    out['publication_gate']=out.publication_date_min.fillna('').ne('') & out.publication_date_min.fillna('9999-99-99').le(out.scoring_cutoff)
    for table,c in candidates.items():
        # Full key conflicts and multiple distinct report-period strings both exclude the one-to-one match.
        counts=c.groupby(['Symbol','economic_year']).size()
        count=out.set_index(['Symbol','economic_year']).index.map(counts).fillna(0).astype(int)
        out[table+'__candidate_count']=count
        selected=c[c.groupby(['Symbol','economic_year'])['__content_sha256'].transform('size').eq(1)].copy()
        selected['source_index']=selected.index
        cols=['Symbol','economic_year','source_index','duplicate_count','source_ids']
        if table in FINANCIAL: cols+=['IfCorrect','DeclareDate']
        selected=selected[cols].rename(columns={x:table+'__'+x for x in cols if x not in ['Symbol','economic_year']})
        out=out.merge(selected,on=['Symbol','economic_year'],how='left',sort=False,validate='many_to_one')
        out[table+'__report_period_paired']=out[table+'__source_index'].notna()
        if table in FINANCIAL:
            dates_info=out[table+'__DeclareDate'].fillna('').map(correction_dates)
            dates=dates_info.map(lambda x:x[0]); dates_complete=dates_info.map(lambda x:x[1])
            correction=out[table+'__IfCorrect'].fillna('')
            # No correction is a separate observed flag; it does not certify historical values.
            out[table+'__correction_status_known']=correction.isin(['0','1'])
            out[table+'__correction_time_gate']=out[table+'__report_period_paired'] & (correction.eq('0') | (correction.eq('1') & dates_complete & dates.le(out.scoring_cutoff)))
            out[table+'__observed_date_gates_pass']=out[table+'__correction_time_gate'] & out.publication_gate
    out['historical_value_version_verified']=False
    out['retrospective_only']=True
    return out

def save(df,path): df.to_csv(path,index=False,compression='gzip' if str(path).endswith('.gz') else None)

def related_links(normalized,out,table):
    # Bounded memory streaming: a disk SQLite table deduplicates full contents and keeps all provenance.
    import sqlite3
    db=sqlite3.connect(out/(table+'_links.sqlite'))
    db.executescript('DROP TABLE IF EXISTS records; DROP TABLE IF EXISTS sources; CREATE TABLE records (content_sha256 TEXT PRIMARY KEY, Symbol TEXT, economic_year INTEGER, Reptdt TEXT, related_party_id TEXT, announcement_date TEXT, announcement_complete INTEGER, strict_availability INTEGER, availability_reason TEXT, payload_json TEXT, duplicate_count INTEGER); CREATE TABLE sources (content_sha256 TEXT, source_id TEXT);')
    n=0
    for chunk in pd.read_csv(normalized/(table+'.csv.gz'),dtype=str,keep_default_na=False,chunksize=20000):
        values=[]; sources=[]
        for r in chunk.to_dict('records'):
            h=r['__content_sha256']; date=complete_date(r.get('Annodt','')); yr=r['Reptdt'][:4]
            # strict_availability means independently timed by the annual scoring cutoff, never value-version verification.
            valid=bool(date and yr.isdigit() and date<=f'{int(yr)+1}-06-30') if table=='RPT_Operation' else False
            reason=('published_by_cutoff' if valid else ('missing_independent_publication_time' if table=='RPT_Transfer' else ('incomplete_publication_date' if not date else 'published_after_cutoff')))
            values.append((h,symbol(r['Stkcd']),int(yr) if yr.isdigit() else None,r['Reptdt'],r.get('RalatedPartyID',''),r.get('Annodt',''),int(bool(date)),int(valid),reason,json.dumps({k:v for k,v in r.items() if not k.startswith('__')},ensure_ascii=False),1))
            sources.append((h,json.dumps({k:r.get(k,'') for k in META},ensure_ascii=False)))
        db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(content_sha256) DO UPDATE SET duplicate_count=duplicate_count+1',values)
        db.executemany('INSERT INTO sources VALUES (?,?)',sources); db.commit(); n+=len(chunk)
    db.execute('CREATE INDEX company_year ON records(Symbol,economic_year)');db.execute('CREATE INDEX source_hash ON sources(content_sha256)')
    # Same company/year/party/date is only a multiplicity diagnostic, not an asserted transaction identity.
    summary=pd.read_sql_query('SELECT Symbol,economic_year,COUNT(*) unique_content_records,SUM(duplicate_count) source_records,SUM(strict_availability) records_available_by_cutoff,COUNT(DISTINCT related_party_id) related_party_count FROM records GROUP BY Symbol,economic_year',db)
    save(summary,out/(table+'_annual_coverage.csv.gz'))
    print(table, 'related links complete', n, flush=True)
    diagnostic=pd.read_sql_query('SELECT Symbol,economic_year,Reptdt,related_party_id,COUNT(*) distinct_content_candidates FROM records GROUP BY Symbol,economic_year,Reptdt,related_party_id HAVING COUNT(*)>1',db)
    save(diagnostic,out/(table+'_multiplicity_diagnostic.csv.gz')); db.close()
    return summary,dict(source_rows=n,unique_contents=int(summary.unique_content_records.sum()),duplicate_rows=int(summary.source_records.sum()-summary.unique_content_records.sum()),multiplicity_groups=len(diagnostic))

def run(config_path):
    import tomllib
    config=tomllib.loads(Path(config_path).read_text()); base=Path(config['project_root']); normalized=base/config['normalized']; out=base/config['output']; out.mkdir(parents=True,exist_ok=True)
    caches={y:base/str(p) for y,p in zip(config['target_years'],config['cache_paths'])}
    protected=list((base/config['raw']).glob('*.zip'))+list(caches.values())+[base/config['old_financial']]
    before={str(p):sha256(p) for p in protected}
    pubs=publication_table(load(normalized/'IAR_Rept.csv.gz'));save(pubs,out/'annual_publication_dates.csv.gz')
    candidates={}; report={'label_snapshot_cutoff':'2023-12-31','target_years':config['target_years'],'retrospective_only':True,'historical_value_version_verified':False,'tables':{},'samples':{}}
    related={}
    for table in ['RPT_Operation','RPT_Transfer']:
        related[table],report['tables'][table]=related_links(normalized,out,table)
    numeric={}
    dictionary=load(base/config['dictionary'])
    for table in (*FINANCIAL,'FIN_Audit'):
        chunks=[]; normalized_rows=0
        for chunk in pd.read_csv(normalized/(table+'.csv.gz'),dtype=str,keep_default_na=False,chunksize=20000):
            normalized_rows+=len(chunk)
            keep=chunk.Accper.str[:10].str.endswith('-12-31')
            if table in FINANCIAL: keep &= chunk.Typrep.eq('A')
            chunks.append(chunk.loc[keep])
        full=pd.concat(chunks,ignore_index=True); c=annual_candidates(full,table in FINANCIAL); candidates[table]=c
        print(table,'annual candidates',len(c),flush=True)
        save(c,out/(table+'_annual_candidates.csv.gz'));save(c[c.content_conflict],out/(table+'_conflicts.csv.gz')); save(c[c.duplicate_count.gt(1)],out/(table+'_duplicates.csv.gz'))
        report['tables'][table]={'normalized_rows':normalized_rows,'annual_A_candidates':len(c),'conflict_candidate_rows':int(c.content_conflict.sum()),'duplicate_excess_rows':int((c.duplicate_count-1).sum()),'IfCorrect_counts':c.IfCorrect.value_counts().to_dict() if 'IfCorrect' in c else {}}
        if table in FINANCIAL:
            cols=dictionary.loc[dictionary.table.eq(table)&dictionary.data_type.str.lower().isin(['decimal','numeric','float','int','integer']), 'field'].tolist()
            cols=[x for x in cols if x in c and x not in ['IfCorrect']]
            converted=c[cols].apply(pd.to_numeric,errors='coerce')
            report['tables'][table]['numeric_parse_failures']={col:int((c[col].ne('') & converted[col].isna()).sum()) for col in cols if (c[col].ne('') & converted[col].isna()).any()}
            numeric[table]=(cols,converted.to_numpy(dtype=float))
    for target,path in caches.items():
        samples=cache_samples(path,target); aligned=align_samples(samples,pubs,candidates)
        for table,coverage in related.items():
            aligned=aligned.merge(coverage.add_prefix(table+'__').rename(columns={table+'__Symbol':'Symbol',table+'__economic_year':'economic_year'}),on=['Symbol','economic_year'],how='left',sort=False,validate='many_to_one')
        for table in related:
            aligned[table+'__linked']=aligned[table+'__unique_content_records'].fillna(0).gt(0)
            aligned[table+'__has_independently_timed_record']=aligned[table+'__records_available_by_cutoff'].fillna(0).gt(0)
        for col in samples:
            assert np.array_equal(samples[col].to_numpy(),aligned[col].to_numpy()),f'Cache order/labels changed: {col}'
        save(aligned,out/f'sample_alignment_{target}.csv.gz')
        arrays={}
        column_coverage=[]
        for fold,g in aligned.groupby('fold',sort=False):
            arrays[f'{fold}_symbol']=g.Symbol.to_numpy(dtype=str); arrays[f'{fold}_year']=g.economic_year.to_numpy(dtype=np.int64); arrays[f'{fold}_y']=g.Y.to_numpy(); arrays[f'{fold}_types']=g[[f'type_{i}' for i in range(10)]].to_numpy()
            for k in ['Symbol','economic_year','Y',*[f'type_{i}' for i in range(10)],'row_index']: arrays[f'{fold}_{k}']=g[k].to_numpy(dtype=str) if k=='Symbol' else g[k].to_numpy()
            for table,(cols,values) in numeric.items():
                idx=g[table+'__source_index'].fillna(-1).astype(int).to_numpy(); x=np.full((len(g),len(cols)),np.nan); ok=idx>=0; x[ok]=values[idx[ok]]
                column_coverage.extend(dict(target_year=target,fold=int(fold),table=table,field=field,n=len(g),observed_count=int(np.isfinite(x[:,j]).sum()),coverage=float(np.isfinite(x[:,j]).mean())) for j,field in enumerate(cols))
                arrays[f'{fold}_{table}_values']=x; arrays[f'{fold}_{table}_observed_mask']=np.isfinite(x);arrays[f'{fold}_{table}_source_index']=idx
                for gate in ['report_period_paired','correction_time_gate','observed_date_gates_pass']: arrays[f'{fold}_{table}_{gate}']=g[table+'__'+gate].to_numpy(dtype=bool)
                arrays[table+'_names']=np.array(cols,dtype=str)
            arrays[f'{fold}_publication_gate']=g.publication_gate.to_numpy(dtype=bool)
            arrays[f'{fold}_historical_value_version_verified']=np.zeros(len(g),dtype=bool)
            arrays[f'{fold}_retrospective_only']=np.ones(len(g),dtype=bool)
        np.savez_compressed(out/f'raw_financial_alignment_{target}.npz',**arrays)
        save(pd.DataFrame(column_coverage),out/f'numeric_column_coverage_{target}.csv.gz')
        indicators=['publication_gate',*[t+'__report_period_paired' for t in candidates],*[t+'__observed_date_gates_pass' for t in FINANCIAL],*[t+'__linked' for t in related],*[t+'__has_independently_timed_record' for t in related]]
        coverage=[]
        for fold,g in aligned.groupby('fold'):
            for name,sub in [('all',g),('Y=0',g[g.Y.eq(0)]),('Y=1',g[g.Y.eq(1)]),*[(f'type_{i}=1',g[g[f'type_{i}'].eq(1)]) for i in range(10)]]:
                coverage.append(dict(target_year=target,fold=int(fold),group=name,n=len(sub),**{k:int(sub[k].sum()) for k in indicators}))
        save(pd.DataFrame(coverage),out/f'coverage_by_fold_type_{target}.csv.gz')
        print('sample alignment complete',target,len(aligned),flush=True)
        report['samples'][str(target)]={str(f):dict(n=len(g),**{k:int(g[k].sum()) for k in indicators}) for f,g in aligned.groupby('fold')}
    after={str(p):sha256(p) for p in protected}; assert before==after,'Protected input changed'
    report['normalized_hashes']={str(p):sha256(p) for p in sorted(normalized.glob('*.csv.gz'))}
    report['protected_hashes_before']=before;report['protected_hashes_after']=after
    report['limitations']=['Current 2026 exports do not establish historical first-release financial values.','IAR publication gate and correction-time gate are independent of historical value verification.','FIN_Audit Annodt is audit date, not publication date.','RPT_Transfer has no independent announcement date and strict availability is false.','Related-party payloads retain currencies and amounts without aggregation; multiplicity is not a proven transaction conflict.','Old x2 contains ratios from different FI sources; raw statement amount columns cannot be asserted equal to old ratio columns.','No labels or type fields are included in new financial input arrays; copied labels only preserve frozen sample references.']
    (out/'alignment_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report
