"""汇总本轮浏览器保存的现行字典和查询页；不访问网站、不下载业务数据。"""
from pathlib import Path
import csv
import hashlib
import json
import re

P=Path(__file__).resolve().parents[1]
ROOT=P/'artifacts/followup/csmar'
TABLES=[
 ('FIN_Audit','财务报告审计意见','FIN_Audit_字段说明.txt','FIN_Audit_查询页.txt'),
 ('RPT_Transfer','关联资金往来','关联资金往来_字段说明.txt','关联资金往来_查询页.txt'),
 ('RPT_Operation','关联交易明细','关联交易情况_字段说明.txt','RPT_Operation_查询页.txt'),
 ('IAR_Rept','财报公布日期','报告公布日期_字段说明.txt','公布日期_查询页.txt'),
 ('FS_Combas','资产负债表','FS_Combas_字段说明.txt','财务报表_查询页.txt'),
 ('FS_Comins','利润表','FS_Comins_字段说明.txt','利润表_查询页.txt'),
 ('FS_Comscfd','现金流量表直接法','FS_Comscfd_字段说明.txt','现金流直接法_查询页.txt'),
 ('IC_EvaluationRepInfo','内控评价报告','内部控制_评价报告字段说明.txt','内部控制_查询页.txt'),
 ('IC_DeficiencyDetails','内控缺陷明细','IC_DeficiencyDetails_字段说明.txt','IC_DeficiencyDetails_查询页.txt'),
 ('STK_Guarantee_Main','对外担保','对外担保_字段说明.txt','对外担保_查询页.txt'),
 ('AEC_FinResAnn','财务重述公告','财务重述公告_字段说明.txt','会计差错更正_查询页.txt'),
 ('AEC_FinErrCrrct','更正前后科目对比','财务差错更正前后对比_字段说明.txt','AEC_FinErrCrrct_查询页.txt'),
]


def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    fields=[];tables=[]
    for table,name,dictionary,evidence in TABLES:
        dp=ROOT/'dictionaries'/dictionary; ep=ROOT/'evidence'/evidence
        raw=dp.read_text(); snap=ep.read_text()
        assert ('- generic: 已购买' in snap) != ('- generic: 无权限' in snap),table
        count=0
        for block in raw.split('\n\n\n'):
            if not block.strip(): continue
            cols=block.split('\n\t\n')
            assert len(cols) in (5,6),(table,cols)
            if len(cols)==5: cols.append('')
            index,code,dtype,title,unit,description=[c.strip() for c in cols]
            count+=1;assert int(index)==count
            fields.append(dict(table=table,field=code,data_type=dtype,title=title,unit=unit,
                               description=description,dictionary_file=dictionary,checked_on='2026-10-15'))
        match=re.search(r'本表数据总记录数：(\d+) 条，数据频率：([^，]+)，数据开始时间：\s*([^，]+)，数据结束时间：\s*([^\n]+)',snap)
        assert match,table
        m=re.search(r'特别提示：本表以\[([^\]]+)\]字段为时间查询基准',snap)
        tables.append(dict(table=table,name=name,status='已购买' if '- generic: 已购买' in snap else '无权限',
            fields=count,total_rows=int(match[1]),frequency=match[2],start=match[3].strip(),end=match[4].strip(),
            query_time_basis=m[1] if m else '',checked_on='2026-10-15',source='https://data.csmar.com/',
            dictionary_file='dictionaries/'+dictionary,evidence_file='evidence/'+evidence,
            dictionary_sha256=hashlib.sha256(dp.read_bytes()).hexdigest(),
            business_data_downloaded=False))
    save_csv(ROOT/'字段字典汇总.csv',fields)
    save_csv(ROOT/'表与授权核对.csv',tables)
    priorities={'FIN_Audit':1,'IAR_Rept':1,'RPT_Operation':2,'RPT_Transfer':2,
                'FS_Combas':3,'FS_Comins':3,'FS_Comscfd':3}
    conditions={t:'时间不限；全部证券代码；全部字段；保留全部可用历史记录，下载后本地筛选研究范围'
                for t in priorities}
    for t in ('FS_Combas','FS_Comins','FS_Comscfd'):
        conditions[t]+='；采集时保留全部季度与合并/母公司口径'
    joins={
      'FIN_Audit':'Stkcd+Accper连接IAR_Rept年报公开日；自身Annodt仅审计日',
      'IAR_Rept':'Stkcd+Accper+Reptyp；Annodt为报告公开日，不完整日期保留不确定性',
      'RPT_Operation':'Stkcd+Reptdt+关联方ID/名称+交易序号；Annodt控制公开时间；不能直接累加多公告重复',
      'RPT_Transfer':'Stkcd+Reptdt+关联方ID/名称+科目+交易序号；无独立公告日，来源时间待匹配',
      'FS_Combas':'Stkcd+Accper+Typrep；连IAR_Rept公开日；IfCorrect/DeclareDate先作版本审计',
      'FS_Comins':'Stkcd+Accper+Typrep；连IAR_Rept公开日；IfCorrect/DeclareDate先作版本审计',
      'FS_Comscfd':'Stkcd+Accper+Typrep；连IAR_Rept公开日；IfCorrect/DeclareDate先作版本审计',
    }
    tasks=[]
    for row in tables:
        if row['table'] not in priorities:continue
        table=row['table']
        tasks.append(dict(priority=priorities[table],table=table,name=row['name'],
            status='待额度恢复；按用户要求直接全量，取消小样本导出',symbols='全部代码',
            filter=conditions[table],output_format='CSV及网站提供的压缩包；保留原空白和字符编码',
            target_filename=f'{table}_all_available',field_count=row['fields'],
            alignment_rule=joins[table],catalog_rows_reference=row['total_rows'],
            expected_rows='以实际全量导出和平台上限为准；目录行数仅供核对',
            purpose='一次采集后本地抽查、验收、配对；建模仍按冻结标签与时间协议筛选',
            splitting_rule='先尝试整表；若平台要求分批，按不重叠期间最少拆分，不另做样本下载'))
    save_csv(ROOT/'待下载任务清单.csv',sorted(tasks,key=lambda r:(r['priority'],r['table'])))
    manifest=dict(checked_on='2026-10-15',method='用户已登录Chrome账户，官方界面字段说明与查询页',
        tables=len(tables),fields=len(fields),purchased_tables=sum(t['status']=='已购买' for t in tables),
        inaccessible_tables=sum(t['status']=='无权限' for t in tables),business_files_downloaded=0,
        audit_preview_rows=45,download_attempts=1,download_blocker='机构账号下载次数已达到今日上限',
        download_plan='用户要求7张已授权表全量采集；取消小样本下载；采集全历史，建模范围另行筛选',
        result='字典与账户权限已核对；业务数据未取得；不以样本预览推断总体缺失',
        future_dates='网页覆盖终点按原样保存；晚于查询日的终点不解释为未来实际交易已发生')
    (ROOT/'lookup_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
