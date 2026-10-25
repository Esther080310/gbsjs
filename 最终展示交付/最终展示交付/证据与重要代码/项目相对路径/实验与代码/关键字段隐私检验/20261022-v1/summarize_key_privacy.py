from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;OUT=HERE/'results';ROOT=HERE.parents[2]

def main():
    assert (OUT/'completion.json').exists(),'Audit must finish before conclusions are written'
    manifest=json.loads((OUT/'manifest.json').read_text());completion=json.loads((OUT/'completion.json').read_text())
    a=pd.read_csv(OUT/'aggregate.csv');f=pd.read_csv(OUT/'fields.csv');c=pd.read_csv(OUT/'industry_type.csv');at=pd.read_csv(OUT/'attribution_recheck.csv')
    sets=json.loads((OUT/'field_sets.json').read_text())['sets'];key=sets['key_financial']
    paired=a.pivot(index=['seed','group','field_set'],columns='arm',values='nrmse').reset_index()
    paired['difference']=paired.D1-paired.S0;paired['relative_increase']=paired.D1/paired.S0-1
    paired.to_csv(OUT/'paired_aggregate.csv',index=False)
    fields=f[f.group=='all'].pivot(index=['seed','field'],columns='arm',values='nrmse_observed').reset_index()
    fields['difference']=fields.D1-fields.S0;fields['relative_increase']=fields.D1/fields.S0.replace(0,np.nan)-1
    summary=fields.groupby('field').agg(s0=('S0','mean'),dcor=('D1','mean'),difference=('difference','mean'),relative_increase=('relative_increase','mean'),positive_seeds=('difference',lambda x:int((x>0).sum()))).reset_index()
    public=f[(f.group=='all')&(f.arm=='S0')].groupby('field').agg(public=('public_nrmse','mean'),constant=('constant_nrmse','mean'),observed=('n_observed','first')).reset_index()
    summary=summary.merge(public,on='field');summary['remaining_gain_over_public']=summary.public-summary.dcor
    summary.to_csv(OUT/'paired_fields.csv',index=False)
    eligible_cells=c[~c.descriptive_only]
    cells=eligible_cells.groupby(['seed','arm','industry','type']).agg(nrmse=('nrmse','mean'),fields=('field','nunique'),min_observed=('n_observed','min')).reset_index()
    cellpairs=cells.pivot(index=['seed','industry','type'],columns='arm',values='nrmse').dropna().reset_index()
    cellpairs['relative_increase']=cellpairs.D1/cellpairs.S0-1
    cellpairs.to_csv(OUT/'paired_industry_type.csv',index=False)
    # Frozen task predictions must agree with the prior stability audit.
    taskchecks=[]
    for seed in completion['seeds']:
        for arm in ('S0','D1'):
            old=json.loads((ROOT/f'实验与代码/dCor稳定性复检/20261021-v1/results/{arm}_s{seed}/evaluation.json').read_text())
            row=a[(a.seed==seed)&(a.arm==arm)].iloc[0]
            assert abs(row.recall-old['recall'])<1e-12 and abs(row.ap-old['ap'])<1e-12
            taskchecks.append(dict(seed=seed,arm=arm,unchanged=True))
    # Relative sensitivities use matched cases and donors within each model pair.
    ats=at.pivot(index=['seed','industry','type','field'],columns='arm',values=['abs_overall_logit_delta','abs_type_logit_delta'])
    sens=[]
    for target in ['abs_overall_logit_delta','abs_type_logit_delta']:
        q=ats[target].reset_index();q['ratio']=q.D1/q.S0.replace(0,np.nan)
        q.to_csv(OUT/f'paired_{target}.csv',index=False)
        sens.append(dict(target=target,median_paired_ratio=float(q.ratio.median()),pairs=len(q)))
    names=pd.read_csv(ROOT/'实验与代码/经济结构依据/字段与公式.csv').set_index('field')['name'].to_dict()
    def cn(field):return names.get(field.split('::',1)[1],field.split('::',1)[1])
    overall=paired[(paired.group=='all')&(paired.field_set=='key_financial')]
    ks=summary[summary.field.isin(key)].copy()
    metrics=dict(seeds=completion['seeds'],financial_key_fields=len(key),key_mean_s0=float(overall.S0.mean()),key_mean_dcor=float(overall.D1.mean()),mean_paired_relative_increase=float(overall.relative_increase.mean()),positive_seed_pairs=int((overall.difference>0).sum()),fields_improved_on_average=int((ks.difference>0).sum()),fields_positive_all_five=int((ks.positive_seeds==5).sum()),mean_public_nrmse=float(ks.public.mean()),mean_remaining_gain_over_public=float(ks.remaining_gain_over_public.mean()),task_prediction_checks=taskchecks,sensitivity=sens)
    (OUT/'conclusions.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    lines=['# 关键字段隐私检验：操作与结论','', '2026-10-22。将当前模型的归因候选与字段级恢复攻击连接起来。复用五组无防护 S0 / dCor 模型；本轮没有训练或修改预测模型。','', '## 1. 这轮回答什么','', '重点是“任务相关的财务指标是否更难从上传表示中恢复”，并检查总体平均改善是否掩盖字段、行业及风险组差异。这里的数值是预处理后的模型输入，不是原始财务金额。','', '## 2. 实际操作','',f'- 固定上一轮全部 15 个唯一候选，其中 13 个财务字段进入主汇总；两项企业年限归入已知公共信息。另固定 13 个同经济组、近似观测比例的非候选对照；138 个数值输入均保留评价。',f'- 攻击拟合 {manifest["fit_rows"]:,} 行，按企业分离的内部选型 {manifest["inner_rows"]:,} 行，最终诊断 {manifest["validation_rows"]:,} 行（发生年 2018）。没有读取折 2/3。',f'- 最终诊断中 {manifest["validation_company_overlap_with_attack_fit"]:,} 家企业在攻击拟合历史中出现，属于后续年度的恢复检验，不是全新企业检验。', '- 模拟融合端有配对的历史辅助样本，可训练恢复器。公共知识设为行业、审计意见、完整率、治理状态和两项企业年限；完整攻击额外拥有节点消息，包括学习出的控制码。', '- 攻击库包括四个正则强度的 Ridge、全部 138 字段的深 MLP、固定 15 字段的专用深 MLP，并允许常数基线、已知年限直接读取。每字段按攻击内部选型集选择攻击器；诊断折不参与轮次和攻击器选择。', '- 深 MLP 在已观测字段上训练并回载内部选型最优权重。主要误差只统计真实观测值；另保存含填补值的误差。全体、行业、固定 S0 前 10% 风险名单和行业×违规类型分别输出。','', '## 3. 主要结果','', 'NRMSE 是按攻击拟合样本中已观测值标准差归一化的均方根误差；越高表示当前攻击器恢复越困难。不同字段先各自归一化，再等权平均。','', '|字段集合或企业组|S0 平均误差|dCor 平均误差|配对相对变化|误差升高种子数|','|---|---:|---:|---:|---:|']
    display=[('all','key_financial','13 个关键财务字段'),('all','matched_controls','13 个匹配对照字段'),('all','all_numeric','全部数值字段'),('C','key_financial','制造业／关键财务字段'),('I','key_financial','信息技术业／关键财务字段'),('S0_highrisk','key_financial','S0 前 10% 名单／关键财务字段'),('S0_other','key_financial','其余企业／关键财务字段')]
    for group,fs,label in display:
        r=paired[(paired.group==group)&(paired.field_set==fs)]
        if len(r):lines.append(f'|{label}|{r.S0.mean():.4f}|{r.D1.mean():.4f}|{100*r.relative_increase.mean():+.1f}%|{(r.difference>0).sum()}/{len(r)}|')
    lines+=['', f'13 个关键财务字段中，{metrics["fields_improved_on_average"]} 个的五种子平均误差升高，{metrics["fields_positive_all_five"]} 个在全部五个种子上同向。关键字段公共知识攻击平均误差 {metrics["mean_public_nrmse"]:.4f}；dCor 完整攻击平均误差 {metrics["key_mean_dcor"]:.4f}，上传表示仍带来 {metrics["mean_remaining_gain_over_public"]:+.4f} 的额外恢复改善。它是误差差值，不是泄露概率。','', '## 4. 每个关键财务字段','', '|指标|有效观测数|公共知识误差|S0 误差|dCor 误差|相对变化均值|同向种子|','|---|---:|---:|---:|---:|---:|---:|']
    for r in ks.sort_values('relative_increase').itertuples():lines.append(f'|{cn(r.field)}|{r.observed}|{r.public:.3f}|{r.s0:.3f}|{r.dcor:.3f}|{100*r.relative_increase:+.1f}%|{r.positive_seeds}/5|')
    lines+=['','企业成立年限和上市年限作为已知辅助知识直接给出，两种攻击下的误差均应为 0，不将这类公开字段的可恢复性记作新的财务保密风险。','','## 5. 与归因分析如何连接','','以下在相应行业、已记录相应违规类型的企业中，评价上一轮该组合的固定候选。每个字段至少 30 个实际观测才进入本表；不同组合的候选集合不同，绝对误差不用于横向比较哪个行业更安全。','', '|行业|违规类型|S0 误差|dCor 误差|配对相对变化|同向种子|','|---|---|---:|---:|---:|---:|']
    for (ind,code),rows in cellpairs.groupby(['industry','type']):
        label={'P2503':'虚假记载或误导性陈述','P2504':'推迟披露','P2505':'重大遗漏'}[code]
        lines.append(f'|{ind}|{label}|{rows.S0.mean():.3f}|{rows.D1.mean():.3f}|{100*rows.relative_increase.mean():+.1f}%|{(rows.D1>rows.S0).sum()}/{len(rows)}|')
    lines+=['','对上一轮固定案例和关键字段，使用成对相同的同行背景值，复查两个副模型的总体风险与类型 logit 替换敏感性。这里只解释模型依赖变化；logit 敏感性的比值不代表预测能力的保留比例。','']
    for item in sens:lines.append(f'- {item["target"]}：各配对单元的 dCor/S0 敏感性比值中位数 {item["median_paired_ratio"]:.3f}。')
    lines+=['','详细行业×类型结果在 industry_type.csv；少于 30 个实际观测的字段单元标为 descriptive_only。当前样本支持提出行业差异的检查线索，不能由单年度单元差值证明行业间保护效果显著不同。五组种子重复描述同一年度上的训练波动，不是五个独立年度的统计确认。','','## 6. 对项目的判断','', f'本轮支持的具体结论是：在固定的 13 个任务相关财务字段上，dCor 的平均恢复误差由 {metrics["key_mean_s0"]:.3f} 变为 {metrics["key_mean_dcor"]:.3f}，五组配对中的 {metrics["positive_seed_pairs"]} 组误差升高。是否所有字段都受益，以逐字段表为准。', '', '匹配对照字段的改善与关键字段相近，当前证据支持一般性的恢复抑制，不支持“dCor 专门保护了模型最重要的指标”。关键字段及优先名单的改善幅度低于全字段平均，说明汇报时必须同时展示这些分组，不能只引用整体平均改善。', '', '保护与预测效用必须一起呈现。此前五组配对的召回平均下降约 0.78 个百分点、AP 平均下降约 0.0139，效用保留门未通过。本轮逐一核对预测指标与此前记录一致，不据此把 dCor 替换为主模型。', '', '较高的重构误差表示当前恢复攻击下的经验改善。上传表示与公共信息基线之间仍有恢复差距时，仍存在额外泄露；本轮不提供差分隐私、不可逆或实际机构安全认证。', '', '当前合适的呈现：主模型保留，dCor 作为展示预测效用与数值恢复难度取舍的副模型；用逐字段结果解释其保护对象。后续若要确认跨年度成立，应另定独立年度，本轮不继续追加强度调参。','','## 7. 记录索引','','- 检测前约定.md：对象、辅助知识、攻击及判读口径。','- 运行日志.txt：实际执行记录。','- results/manifest.json、field_sets.json：样本、字段及源文件哈希。','- results/attackers/：恢复器选型、最优权重、训练历史、逐行预测及风险模型来源。','- results/fields.csv、paired_fields.csv：全部字段及配对汇总。','- results/industry_type.csv、paired_industry_type.csv：固定行业×违规类型的字段检测。','- results/attribution_recheck.csv：固定案例的模型依赖复核。','- results/conclusions.json、completion.json：机器可读结论与完成标记。']
    (HERE/'检测结果与操作记录.md').write_text('\n'.join(lines)+'\n')
    (HERE/'操作记录.md').write_text('# 操作记录\n\n1. 读取并冻结上一轮 15 个候选，写入检测前约定。\n2. 保留既有模型，用企业哈希建立攻击内部选型集，记录字段与分组。\n3. 训练公共知识攻击库及五组配对模型的完整表示攻击库；保存最优权重与预测。\n4. 仅在内部选型集选择字段攻击器，对最终诊断折计算观测值误差。\n5. 使用相同案例、相同背景复核字段依赖。\n6. 核对十个风险模型的任务指标与此前逐值一致；汇总字段、行业、风险组及类型结果。\n7. 未改变风险模型、未加载折 2/3、未启动新的防护强度搜索。\n\n复现入口：先 inspect_key_privacy.py，完成后 summarize_key_privacy.py。攻击已有产物可复用；训练完成标记和结论文件用于区分运行中与已完成状态。\n')
    print(json.dumps(metrics,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
