"""Generate a plain-language account of the repair and matched evaluation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;O=H/'results'
def main():
    data=json.loads((H/'data/verification.json').read_text());cache=json.loads((H/'data/cache_verification.json').read_text())
    train=json.loads((O/'verification.json').read_text());ev=json.loads((O/'evaluation_verification.json').read_text())
    assert data['existing_cell_changes']==0 and cache['old_cache_reproduced_exactly'] and train['new_trainings']==24 and ev['full_sample_metrics_reproduced']
    changes=pd.read_csv(H/'data/cache_changes.csv');support=pd.read_csv(O/'group_support.csv');summary=pd.read_csv(O/'summary.csv');annual=pd.read_csv(O/'annual_summary.csv');ci=pd.read_csv(O/'AP_intervals.csv');paired=pd.read_csv(O/'paired_changes.csv')
    lines=['# 财务年度配对修正：处理过程、结果与下一步','',
    '2026-10-14。已完成原始年度财务重建、旧缓存逐值复现、24次正式匹配网络重训，以及固定年报后时点的未披露组评价。原始工作簿、旧缓存、原六种子模型保留。','',
    '## 1. 这次具体改了什么','',
    '现在直接用“公司＋报表实际年度”找财务指标。例如2020年的企业样本直接连接其2020年报表，不再要求这家公司同时存在于2021年的市场名单。原来的下一年市场名单条件，来自早期预测任务，现已在新版数据入口中去除。','',
    f'三张原始财务表共完整扫描{sum(s["rows_scanned"] for s in data["scans"]):,}条记录，按原规则选报表：合并报表优先，Source=0优先，同优先级保留原表首条。新版财务表共{data["samples"]:,}个公司年度、{data["raw_financial_fields"]}个原始指标。','',
    f'与旧表已有的{data["existing_key_rows"]:,}个键核对了{data["existing_numeric_cells_checked"]:,}个单元，值与缺失状态一致。源核查采用round_trip读取，避免把金额字段的浮点解析末位差异计为数据更改；此前检测到95处默认CSV解析差异，float32网络输入一致。','',
    f'本轮使用的2002—2021年范围共恢复{sum(int(v) for y,v in data["recovery_by_year"].items() if int(y)<=2021)}个企业年度。完整年度表还覆盖2024年的报表，但这些后续年度数据不参与本轮训练和评价。','',
    '训练、选择和校准期也使用相同修正规则。原始单元不变，但训练期补入少数原来缺失的值后，中心、尺度和缺失标记的标准化参数重新拟合，因此部分其他样本的变换值也会变化。这是正常的训练期预处理变化，不是加入新指标。','']
    for year,g in changes.groupby('year'):
        desc='；'.join(f'{["训练","选择","校准","开发评价"][int(r.fold)]}期恢复{r.raw_changed_rows}行、{r.raw_changed_cells}个单元' for r in g.itertuples())
        lines.append(f'- {year}折：{desc}。')
    lines+=['',
    '## 2. 如何保证比较对应同一问题','',
    '先用旧数据独立重建预处理，逐值复现既有四折缓存。再用新版财务输入重新拟合训练预处理。X1、X3、行业、样本名单、总体与十类型标签、四折划分保持一致，输出特征名保持一致。','',
    '全局网络与32维分布式网络沿用正式配置：代价敏感双头学习、proper loss选轮次、原校准折、10%名单及Top2。两个开发年度各六个原种子，共24次重训；初始参数与各自旧模型一致，重载后四折预测和校准核对通过。','',
    '结果中Old为原模型原输入；FrozenCorrected为原模型直接读取新版输入、保留旧校准；Corrected为新版输入上重训并重新校准。FrozenCorrected帮助区分输入修正与重新训练的作用，不另选最佳方法。','',
    '## 3. 原年度全样本任务的效果','']
    for model in ['GlobalNN','S32']:
        f=summary[(summary.model==model)&(summary.group=='All')].set_index('data_version');a=f.loc['Old'];b=f.loc['Corrected'];fixed=f.loc['FrozenCorrected']
        lines.append(f'- {model}：前10%召回{a.recall:.2%}→{b.recall:.2%}，变化{(b.recall-a.recall)*100:+.3f}个百分点；名单精确率{a.precision:.2%}→{b.precision:.2%}；AP {a.overall_ap:.4f}→{b.overall_ap:.4f}；类型宏覆盖{a.type_macro_coverage:.2%}→{b.type_macro_coverage:.2%}。仅更新输入的原模型召回为{fixed.recall:.2%}。')
        for year,g in annual[(annual.model==model)&(annual.group=='All')].groupby('year'):
            r=g.set_index('data_version');lines.append(f'  - {year}年度平均召回：{r.loc["Old","recall"]:.2%}→{r.loc["Corrected","recall"]:.2%}。')
        gap=ci[(ci.model==model)&(ci.group=='All')].iloc[0]
        lines.append(f'  - AP变化的固定模型企业重采样95%区间为[{gap.lower:+.4f}, {gap.upper:+.4f}]。')
    lines+=['',
    '以上平均值先在同年度六个种子上求均值，再对两个年度等权平均。恢复的行数和预测变化分别解释：修正消除了已定位的样本配对遗漏；预测是否提高由成对结果判断。开发年度已用于多轮研究，本轮评价用于数据修正后的开发判断。','',
    '## 4. 新增加的评价到底检查什么','',
    '以2020年度为例，把核查时间固定为2021年6月30日。先找出截至这一天具有同公司2020年年报公布日期的样本；再排除2020年度违规已于这一天披露的样本，得到“当年违规尚未披露组”。在这个组内重新按模型分数排序，检查前10%的企业。较早年度曾违规的企业仍保留，因为研究的是其2020年的新年度问题。','',
    '负例和后续才披露的正例使用同一截止规则，组别只由公告日期和事件披露日期确定。这里的“尚未披露”指公开记录状态，不认定监管当时完全不知道。','',
    '原总体标签仍固定到2023年底，训练期也沿用同一标签快照，以隔离本轮数据配对修正的影响。治理表公告日期是同公司年度时间参照，财务比率表没有完整历史版本；因此本轮结果定位为年报后未披露组的回顾性评价。复现历史部署还需要将每个训练折的标签截止到训练当时，并取得当时版本的特征。原全样本任务与这个组的分母不同，两组召回不能相减当作算法增益。','']
    for year,g in support.groupby('year'):
        r=g.set_index('group');a=r.loc['All'];available=r.loc['ReportAvailable'];u=r.loc['Undisclosed']
        lines.append(f'- {year}年度：原样本{int(a.rows)}家、正例{int(a.positives)}家；截止日已具备年报公布日期的样本{int(available.rows)}家；未披露组{int(u.rows)}家、后续记录正例{int(u.positives)}家，核查名额{int(u.budget10)}家。')
    for model in ['GlobalNN','S32']:
        f=summary[(summary.model==model)&(summary.group=='Undisclosed')].set_index('data_version');a=f.loc['Old'];b=f.loc['Corrected']
        lines.append(f'- {model}在未披露组：原模型召回{a.recall:.2%}、名单精确率{a.precision:.2%}；修正重训后召回{b.recall:.2%}、名单精确率{b.precision:.2%}、AP {b.overall_ap:.4f}。名单精确率相对该组正例比例为{b.precision_lift:.2f}倍，随机排序召回参照约10%。')
        for year,g in annual[(annual.model==model)&(annual.group=='Undisclosed')].groupby('year'):
            r=g.set_index('data_version');lines.append(f'  - {year}年度修正后：召回{r.loc["Corrected","recall"]:.2%}，名单精确率{r.loc["Corrected","precision"]:.2%}。')
    lines+=['',
    '## 5. 后续研究顺序','',
    '先依据上述未披露组支持量和筛查表现，确定实际核查对象与标签等待期限；随后检查哪类经济关系在这个对象上具有稳定预测信息。当前修正保留为数据入口更新，不包装为新的算法创新。','',
    '新版入口为pipeline.load_data/prepare(corrected=True)。后续发生年度实验显式使用本入口及cache，避免再从旧市场骨架移动财务键。旧发生年度data.py及历史结果保留，旧结果的数值和对应来源继续可追溯。','',
    '## 复算','',
    '按顺序运行build_finance.py、pipeline.py、train.py、evaluate.py、report.py。data保存年度财务及原始Excel行号；cache保存两个开发折的新版输入；results保存24份模型、旧模型参照、组别明细和逐case指标。两版本模型不共用结果目录。']
    (H/'结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('REPORT COMPLETE',H/'结果说明.md')
if __name__=='__main__':main()
