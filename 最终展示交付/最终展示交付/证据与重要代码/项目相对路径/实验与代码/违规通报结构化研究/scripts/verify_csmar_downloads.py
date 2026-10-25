"""只读验收CSMAR原始压缩包：CSV、字段、条数、期间、文件哈希；不访问网站。"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import io
import json
import re
import zipfile

P=Path(__file__).resolve().parents[1]
ROOT=P/'artifacts/followup/csmar'
DATE='20261016'
TABLES={
    'FIN_Audit':('财务报告审计意见',80302,'Accper'),
    'IAR_Rept':('财报公布日期',309026,'Accper'),
    'RPT_Operation':('关联交易明细',2326911,'Reptdt'),
    'RPT_Transfer':('关联资金往来',1401718,'Reptdt'),
    'FS_Combas':('资产负债表',702706,'Accper'),
    'FS_Comins':('利润表',704402,'Accper'),
    'FS_Comscfd':('现金流量表直接法',679988,'Accper'),
}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    csv.field_size_limit(16*1024*1024)
    with (ROOT/'字段字典汇总.csv').open(encoding='utf-8-sig',newline='') as f:
        dictionary=list(csv.DictReader(f))
    result=[]
    for table,(name,catalog,period_field) in TABLES.items():
        path=ROOT/f'raw/{DATE}/{table}_all_available.zip'
        evidence=ROOT/f'evidence/{DATE}/{table}_导出概要.txt'
        page=evidence.read_text()
        exported=int(re.search(r'共\s+(\d+)\s+条',page).group(1))
        assert f'namePhy={table}' in page
        assert '时间不限' in page and '全部代码' in page and 'CSV格式' in page
        expected=[r['field'] for r in dictionary if r['table']==table]
        rows=0;bad_width=0;symbols=set();periods=Counter();types=Counter();partial_periods=0
        record_starts=0;prefix_failures=0
        with zipfile.ZipFile(path) as z:
            files=[i for i in z.infolist() if not i.is_dir()]
            csv_files=[i for i in files if i.filename.lower().endswith('.csv')]
            assert csv_files
            header=None;csv_parts=[]
            for member in csv_files:
                before_count=record_starts
                with z.open(member) as raw:
                    text=io.TextIOWrapper(raw,encoding='utf-8-sig',newline='')
                    part_header=next(csv.reader([next(text)]))
                    if header is None:header=part_header
                    assert part_header==header,(table,member.filename,'分片列头不同')
                    assert len(header)==len(set(header))
                    assert set(header)==set(expected),(table,set(header)-set(expected),set(expected)-set(header))
                    si=header.index('Stkcd');pi=header.index(period_field)
                    ti=header.index('Typrep') if 'Typrep' in header else None
                    def scan_lines():
                        nonlocal record_starts,prefix_failures,partial_periods
                        for line in text:
                            if re.match(r'^"\d{6}",',line):
                                record_starts+=1
                                prefix=next(csv.reader([line]))
                                required=max(si,pi,ti if ti is not None else 0)
                                if len(prefix)<=required:
                                    prefix_failures+=1
                                else:
                                    symbols.add(prefix[si]);period=prefix[pi]
                                    if re.fullmatch(r'\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])',period):
                                        periods[period]+=1
                                    else:partial_periods+=1
                                    if ti is not None:types[prefix[ti]]+=1
                            yield line
                    for row in csv.reader(scan_lines()):
                        rows+=1
                        if len(row)!=len(header):bad_width+=1
                csv_parts.append(dict(filename=member.filename,bytes=member.file_size,record_starts=record_starts-before_count))
            # CSV已完整读取，zipfile会校验其CRC；其余说明文件也读完以检查CRC。
            for item in files:
                if item.filename.lower().endswith('.csv'):continue
                with z.open(item) as f:
                    for _ in iter(lambda:f.read(1024*1024),b''):pass
        assert record_starts==exported,(table,record_starts,exported)
        item=dict(table=table,name=name,export_submissions=1,scope='时间不限、全部代码、全部字段、CSV',
                  catalog_rows=catalog,server_export_rows=exported,actual_record_starts=record_starts,
                  standard_csv_parsed_rows=rows,standard_csv_bad_width_rows=bad_width,
                  standard_csv_usable=(not bad_width and rows==exported),prefix_parse_failures=prefix_failures,csv_parts=csv_parts,
                  catalog_minus_export=catalog-exported,fields=len(header),headers=header,
                  dictionary_fields_match=True,record_starts_match_export=True,all_archive_members_crc_pass=True,
                  symbol_count=len(symbols),period_field=period_field,
                  min_period=min(periods) if periods else None,max_period=max(periods) if periods else None,
                  rows_period_through_2023=sum(n for date,n in periods.items() if date<='2023-12-31'),
                  partial_or_missing_period_rows=partial_periods,report_type_counts=dict(types),
                  file=str(path.resolve()),bytes=path.stat().st_size,sha256=digest(path),
                  evidence=str(evidence.resolve()))
        result.append(item)
        print(json.dumps({k:item[k] for k in ('table','actual_record_starts','fields','catalog_minus_export','standard_csv_bad_width_rows')},ensure_ascii=False),flush=True)
    summary=dict(date='2026-10-16',source='学校已登录Chrome中的CSMAR官方整表导出',
                 export_submissions=7,duplicate_export_submissions=0,sample_export_submissions=0,
                 quota_limit_observed=False,quota_remaining='页面未显示，不推断实际余额',
                 total_rows=sum(r['actual_record_starts'] for r in result),total_fields=sum(r['fields'] for r in result),
                 total_bytes=sum(r['bytes'] for r in result),downloaded=result,pending=[],
                 frozen_y_and_model_cache_modified=False,
                 acceptance='7个压缩包CRC、列头和证券代码起始记录计数通过；CSV标准解析异常单列，三表计数差异保留，不重复导出')
    (ROOT/f'下载台账_{DATE}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    columns=['table','name','actual_record_starts','fields','catalog_rows','catalog_minus_export',
             'standard_csv_parsed_rows','standard_csv_bad_width_rows','standard_csv_usable','symbol_count',
             'period_field','min_period','max_period','rows_period_through_2023','report_type_counts','bytes','sha256','file']
    with (ROOT/f'本地验收_{DATE}.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for item in result:
            line={k:item[k] for k in columns};line['report_type_counts']=json.dumps(line['report_type_counts'])
            writer.writerow(line)
    lines=['# CSMAR全量采集与本地验收','',
           '2026-10-16。7张已授权表均已保存原始压缩包及网页证据，每表仅提交一次全量导出；未做样本导出，未出现下载限额提示。完成后停止网站下载。','',
           '|表|实际记录数|字段|查询页与导出差额|期间范围|','|---|---:|---:|---:|---|']
    for r in result:lines.append(f"|{r['table']} {r['name']}|{r['actual_record_starts']:,}|{r['fields']}|{r['catalog_minus_export']}|{r['min_period']}—{r['max_period']}|")
    lines += ['',f"合计{summary['total_rows']:,}条、{summary['total_fields']}个字段位，压缩包共{summary['total_bytes']/1024**2:.1f} MiB。字段位是各表列数之和，不表示全部列都是模型特征。",'',
              '## 验收结果','',
              '7个压缩包全部成员CRC检查通过；CSV列头与已核对字典一致。按原始物理行开头“带引号的六位证券代码+逗号”计数，记录起始数与各导出概要一致。这项边界检查不等同于标准CSV解析通过；长文本内部引号/换行异常会造成普通读取器误分行或错列。原始证券代码按字符串保留；未改写、填补、丢弃或去重原始内容。', '',
              '三张原始财报的查询页各比导出包多15条。保留查询页计数、导出计数和实际文件哈希，暂不归因为特定缓存或更新时间机制，不以重复导出消耗下载机会。当前文件的验收结论按其实际快照表述。', '',
              '采集为全部代码、全部历史及全部字段；三表保留季度与报表口径。接下来在本地按模型研究期间、合并年报及实际公开时点筛选；原Y截至2023-12-31与原模型缓存继续冻结。报告期在2023年内不代表当时已公开，必须另用公布日期控制可用时间。', '',
              '## 后续处理顺序','',
              '1. 连接IAR_Rept与审计、财报，确认年报实际公布日及缺失/不完整日期。',
              '2. 检查关联交易的年度金额、期末余额、公告重复与关联方键；资金往来空值保留原样。',
              '3. 原始科目与旧X2配对，单独审计更正标记和未来更正信息，再固定新增观测版本。',
              '4. 与LLM行为监督分开检验新增观测的贡献；本次没有进行正式模型训练。','',
              f'原始包：`raw/{DATE}/`；网页概要和截图：`evidence/{DATE}/`；精确条数、字段、报表口径、哈希：`下载台账_{DATE}.json`及`本地验收_{DATE}.csv`。','']
    lines += ['## 标准CSV解析诊断','',
              '|表|普通读取器分出的行数|列数异常行|标准解析通过|',
              '|---|---:|---:|---|']
    for r in result:lines.append(f"|{r['table']}|{r['standard_csv_parsed_rows']:,}|{r['standard_csv_bad_width_rows']:,}|{'是' if r['standard_csv_usable'] else '需规范化后再用'}|")
    lines += ['','存在异常的表在建模导入前须单独规范化与逐记录验收；不使用跳过错误行的读取选项。当前验收不生成清洗后模型输入，也不以正文边界计数替代内容校验。','']
    (ROOT/f'全量采集与验收_{DATE}.md').write_text('\n'.join(lines))
    print(json.dumps({k:summary[k] for k in ('total_rows','total_fields','total_bytes','export_submissions')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
