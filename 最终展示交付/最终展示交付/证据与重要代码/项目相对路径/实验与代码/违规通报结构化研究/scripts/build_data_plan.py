"""将已读案件证据与待查询字段分开保存；不把候选字段写成账户确认。"""
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parents[1]
OUT = P/'artifacts/followup'
RUN = P/'artifacts/runs/pilot-20261015-snapshot2023-v1'

CASES = [
    dict(record_id='4052020',mechanism='年内占用与期末余额的信息差',
         evidence_substring='2014年度,三房巷集团及其子公司13次占用三房巷股份资金37,800万元,当年全部归还。',
         question='是否存在未入账的关联方年内资金使用；期末余额相同能否掩盖不同年内路径？',
         observations='逐笔日期/付款方/收款方/金额/对手身份；关联关系；账簿入账；合同和披露范围',
         current_x='年度比率及总量、公开事件；未覆盖完整逐笔流水及账簿',
         holders='企业财务部门；开户银行；交易对手；审计机构（合作候选）',
         privacy='流水、账簿、合同可能涉及非公开内容；实际授权和覆盖待机构确认',
         method='记录齐全时先核对流水—账簿—披露；信息不齐时再研究风险学习',
         public_proxy='关联交易明细、其他应收款附注、资金往来专项报告、审计/内控报告',
         missing='公开资料不能还原未披露的完整流水；期末其他应收款为零不能排除年内占用',
         direction='优先观测候选：年内路径与期末汇总的差异；尚未形成现实跨机构数据集'),
    dict(record_id='4053899',mechanism='循环支付与交易实质',
         evidence_substring='系通过星美系相关公司循环支付完成',
         question='收到的款项是否来自有独立交易实质的业务，还是同组主体的循环安排？',
         observations='多主体交易时序和金额；控制关系；合同实际签订时间；版权/履约证据；会计确认期间',
         current_x='单企业年度财务状态；不覆盖完整跨主体路径和履约证据',
         holders='企业与交易对手；各开户银行；审计机构；合同权利登记或履约证明持有方（候选）',
         privacy='账户路径和内部合同可能非公开；持有方及协作机制待确认',
         method='循环只提供核查线索；结合业务实质核验，不能把正常循环结算直接标违规',
         public_proxy='关联交易、关联关系、附注、非经常性损益及审计意见',
         missing='缺少完整银行交易和真实履约资料；原Y年度2016与正文2015确认/2016回款需分角色',
         direction='可支持机制模拟；不能把未来处罚还原路径作为事前模型输入'),
    dict(record_id='4095679',mechanism='担保对象和额度的审批一致性',
         evidence_substring='对金科嘉辰提供的担保金额超出担保额度20,000万元',
         question='实际担保的对象、额度、发生时间是否在有效审批和调剂范围内？',
         observations='批准对象集合/额度/有效期；实际担保对象/金额/日期；当期适用规则',
         current_x='无完整对象级批准和实际担保明细',
         holders='企业董事会/股东大会；债权银行；交易所披露平台（候选）',
         privacy='批准公告多为公开；完整合同履行记录是否非公开需逐项核对',
         method='对象和规则完整时优先集合/额度核验；潜在风险排序另设学习任务',
         public_proxy='对外担保明细、关联方、担保公告及审批公告原文',
         missing='CSMAR是否分别保留预计额度、实际额度及对象ID需要现行字典核对',
         direction='优先作公开代理核验样例；不单凭该案论证隐私学习必要'),
    dict(record_id='40103679',mechanism='任职与期限规则',
         evidence_substring='公司董事长未按规定及时代行董事会秘书职责',
         question='董秘空缺超过规定期限时，由谁代行职责？',
         observations='离职日；代行人职务与起止日；当期适用规则；首次公告时间',
         current_x='高管变更年度摘要；不能完整保留具体职位任职区间',
         holders='企业董事会办公室；交易所公开公告平台',
         privacy='主要资料可能公开；优先核查公开数据是否足够',
         method='任职和期限规则核验',
         public_proxy='高管变更任职明细及公司公告',
         missing='已有高管数据需核查是否含董秘和代行关系；不重复下载同样摘要',
         direction='作为无需隐私联合学习的对照案例'),
    dict(record_id='4089202',mechanism='身份关联与逐笔证券交易',
         evidence_substring='在买入后6个月内卖出,在卖出后6个月内又买入',
         question='相关人员的交易是否触发当期短线交易规定？',
         observations='人员身份关系；账户交易日期/方向/数量；当期规则与例外',
         current_x='年度股票行情与高管变更；不等于完整个人账户交易及关系',
         holders='上市公司人员登记部门；证券公司；证券登记结算机构；交易所（候选）',
         privacy='已披露持股变动部分公开；完整账户控制关系及交易可能非公开',
         method='身份和交易完整时先规则核验；未知控制关系不能用年度行情替代',
         public_proxy='董监高及相关人员持股变动、关系与公告日期',
         missing='已有持股表覆盖和填报时间限制；该案主体为监事配偶，不能自动归为公司行为监督',
         direction='用于主体边界对照；不优先增加日频行情'),
]


def write_csv(name, rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records={r['record_id']:r for r in map(json.loads,(RUN/'prepared.jsonl').read_text().splitlines())}
    registry=OUT/'csmar/表与授权核对.csv'
    checked={}
    if registry.exists():
        with registry.open(encoding='utf-8-sig',newline='') as f:
            checked={r['table']:r['status'] for r in csv.DictReader(f)}
    checked_groups={
        '审计/内控':['FIN_Audit','IC_EvaluationRepInfo','IC_DeficiencyDetails'],
        '关联交易/对外担保':['RPT_Operation','RPT_Transfer','STK_Guarantee_Main'],
        '原始三表/公布日/版本':['FS_Combas','FS_Comins','FS_Comscfd','IAR_Rept','AEC_FinResAnn','AEC_FinErrCrrct'],
    }
    case_rows=[]
    for case in CASES:
        source=records[case['record_id']]['source']
        quote=case['evidence_substring']
        assert quote in source['Activity'], (case['record_id'],quote)
        row=dict(case,company=source['CoFullName'],symbol=source['Symbol'],
                 original_year=source['ViolationYear'],declare_date=source['DeclareDate'],
                 evidence_field='Activity',evidence_start=source['Activity'].index(quote),
                 document_group=records[case['record_id']]['document_group'],
                 case_source='prepared原Y正文；研究分析不等于独立机构确认')
        case_rows.append(row)
    write_csv('机构边界案例表.csv',case_rows)
    mapping=[]
    datasets=[
        ('审计/内控','中国上市公司财务报告审计意见数据库；中国上市公司审计研究数据库；中国上市公司内部控制研究数据库',
         '证券代码、会计期、意见/缺陷、审计机构与变更、报告原文、首次公开日、版本',
         'FIN_Audit、Accper、Annodt、Audittyp来自2011旧手册；Annodt为审计日，非公开日'),
        ('关联交易/对外担保','中国上市公司关联交易研究数据库；中国上市公司对外担保研究数据库',
         '对象、关系、金额、发生日、有效期、批准额度/实际额度、公告日、修订',
         '准确库名有历史目录依据，当前表名及字段码待学校账户核查'),
        ('原始三表/公布日/版本','中国上市公司财务报表数据库；中国上市公司年、中、季报公布日期数据库；会计差错更正研究数据库',
         '证券代码、会计期、合并类型、来源、金额单位、独立科目、实际公开日、调整前后/版本',
         'FS_Combas/FS_Comins/FS_Comscfd等为历史表名；当前字段及首发版本待核查'),
    ]
    for case in case_rows:
        for category,db,fields,historical in datasets:
            if case['record_id'] in {'40103679','4089202'}: continue
            mapping.append(dict(record_id=case['record_id'],mechanism=case['mechanism'],
                question=case['question'],required_observations=case['observations'],current_x=case['current_x'],
                candidate_holders=case['holders'],public_private_boundary=case['privacy'],
                public_proxy=case['public_proxy'],candidate_database_group=category,database_names=db,
                requested_fields=fields,historical_evidence=historical,
                live_account_status=('；'.join(t+':'+checked[t] for t in checked_groups[category] if t in checked)
                    +'；现行字典已保存，机构今日下载次数达上限，业务数据未取得') if checked else
                    '尚无现行账户核对记录，待连接后查询',
                usable_time_rule='评分时已公开；未来处罚仅供训练监督或机制研究，不作当期输入',
                verification_or_learning=case['method'],remaining_gap=case['missing']))
    write_csv('mechanism_observation_map.csv',mapping)
    (OUT/'CSMAR最小下载清单.md').write_text('''# CSMAR全量采集与本地验收清单

2026-10-15。当前状态：Chrome连接超时，尚未进入学校账户，以下是待查询清单。

## 1．先做什么

按用户最新要求，现行字典确认后直接导出已授权表全量。每组保存字典、导出条件、原始文件和查询日期，抽查与验收在本地进行，不另用下载次数取小样本。

|顺序|在数据库中搜索|先核对什么|进入下一步的条件|
|---|---|---|---|
|1|审计意见、审计研究、内部控制|学校授权、表名、覆盖期间；意见和缺陷如何编码；审计日与公开日；是否有报告原文及版本|能区分报表期与实际可用时间；字段确有新信息|
|2|关联交易、对外担保|每条是否对应交易/年度汇总；对象标识；金额与单位；预计与实际额度；公告日、起止日、修订|能保留对象和期间；仅有年度总额需另补公告|
|3|财务报表、年中季报公布日期、会计差错更正|三表独立科目、合并/母公司、来源、实际公开时间、调整前后版本|能说明财务值的来源及版本；不能把事后重述当首发|

字典中没有的字段记录“无此字段/需原文补充”，不自行构造字段码。空白、未披露、无事件、未覆盖分别保留。

## 2．全量导出范围

7张已授权表：FIN_Audit、IAR_Rept、RPT_Operation、RPT_Transfer、FS_Combas、FS_Comins、FS_Comscfd。每表选择全部证券代码、可用历史范围和全部字段。

采集时保留季度、合并/母公司等原口径，下载后本地选择建模所需年份、年报和合并A。涉及公告保留实际公告日，不以报表期直接截掉次年发布记录；关联对象和对手明细保留。

优先CSV或网站压缩包。若平台要求拆分，按不重叠时间区间最少分批，不重复下载独立样本。原始采集范围不改变现有2023标签快照或年度评价协议。

每份原文件保存到本目录的 `csmar/raw/`，字典到 `csmar/dictionaries/`。`csmar/download_manifest.csv`记录库表、导出日期、筛选条件、文件名、行数、证券代码字段、期间字段、公开时间字段、版本字段及SHA256。未下载前不创建虚假的下载记录。

## 3．字段检查重点

- 审计：会计截止日、意见、事务所、审计日、公开日、报告文本、首次/修订。旧FIN_Audit中的Annodt是审计日，不当作公告日。
- 内控：披露/鉴证报告区分；缺陷类型、严重程度、整改及报告时间。未来查出的缺陷不倒填。
- 关联交易：交易主体和对手、关联关系、类型、金额、发生期间、报告公告、是否仅含已披露项目。
- 担保：批准对象/限额/期限、实际对象/金额/日期、可调剂条件；若缺批准信息先找公告原文。
- 三表：货币资金、应收账款/票据、合同资产、其他应收款、预付款、存货、减值、收入、利润、经营现金流等；先确认可用科目及单位，不把同源比率恢复量当新增观测。
- 版本：现行整理值、首发值、调整前后、修订日分别保存。公布日期库只证明时间，不自动证明三表值为首发版本。

## 4．本轮的研究判断

优先回答“年内关联资金使用，在年度公开汇总里丢失了多少信息”。三房巷原文明确记载年内多次占用并当年归还、相关往来未记账；这是研究信息边界的具体依据。公开CSMAR可能提供已披露关联交易、附注、审计和内控代理观测，但不能补出完整未披露流水。

担保和董秘期限的案例先做规则核验。这能帮助判断何处真正需要学习，而非为维持三个节点强行安排神经网络。

取得数据后先统计覆盖、时点、缺失和与原X重合程度，再固定是否开展补数实验。当前100条通报的候选监督量较小，先不训练。

## 5．当前连接阻碍

已确认Chrome运行、扩展安装并启用、连接配置检查通过。读取标签页两次超时；经用户同意打开空白Chrome窗口后仍超时。按Chrome插件故障说明，下一步由用户在应用插件界面重新安装Chrome插件，再恢复账户查询。不通过AppleScript、直接读取登录信息或其他方式绕过连接。

历史目录与字段依据见项目的“汇报材料/背景与数据路线重审/CSMAR官方目录核查备忘.md”。现行学校账户核查尚未完成。
''')
    if checked:
        path=OUT/'CSMAR最小下载清单.md'
        content=path.read_text()
        content=content.replace('2026-10-15。当前状态：Chrome连接超时，尚未进入学校账户，以下是待查询清单。',
            '2026-10-15更新：Chrome已恢复，已在学校账户核对12张表的字典和权限，7张已购买、5张无权限。机构当日下载次数达到上限，业务文件尚未取得。现行结果见 [账户核查结果与补数顺序](csmar/账户核查结果与补数顺序.md) 和 [待下载任务清单](csmar/待下载任务清单.csv)。')
        content=content.split('## 5．当前连接阻碍')[0]+'''## 5．账户核查与当前取数状态

Chrome连接已恢复，12张现行表字典、503个字段已保存。表级权限见`csmar/表与授权核对.csv`，具体任务见`csmar/待下载任务清单.csv`。

FIN_Audit曾按5家公司和2013—2021年预览45条；导出时机构账号今日下载次数已达上限，未取得业务文件。用户已取消小样本导出；恢复额度后直接全量采集审计+公布日、关联交易+资金余额、原始三表。内控、担保、更正轨迹已检查的表无权限，先保留所需表名。

现行字典特别确认：FIN_Audit的Annodt是审计日，IAR_Rept的同名字段是报告公开日；RPT_Transfer空值混合未披露/零/横杠，不填零；三表DeclareDate是更正披露日期，不当首次公开日。完整说明见账户核查结果。
'''
        path.write_text(content)
    print('已生成5个有原文定位的案例、9条机制—数据组映射和最小下载清单。')


if __name__=='__main__':
    main()
