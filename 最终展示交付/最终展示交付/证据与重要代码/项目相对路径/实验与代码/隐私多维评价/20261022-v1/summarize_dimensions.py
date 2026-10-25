from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'results'

def main():
    complete=json.loads((OUT/'completion.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text())
    f=pd.read_csv(OUT/'fields.csv');a=pd.read_csv(OUT/'attributes.csv');t=pd.read_csv(OUT/'tasks.csv')
    f=f[f.n_observed>=30].copy();a=a[a.supported].copy()
    numerical=['nrmse','success_0.1','success_0.319','success_0.5']
    fm=f.groupby(['tag','arm','seed','view','group'])[numerical].mean().reset_index()
    am=a.groupby(['tag','arm','seed','view','group']).agg(auc=('auc','mean'),balanced_accuracy=('balanced_accuracy','mean'),n_attributes=('auc','count')).reset_index()
    joined=fm.merge(am,on=['tag','arm','seed','view','group'],how='left')
    joined.to_csv(OUT/'aggregate.csv',index=False)
    primary=joined[(joined.seed!=42)&(joined.group=='all')]
    primary.groupby(['arm','view'])[numerical+['auc','balanced_accuracy']].agg(['mean','std']).to_csv(OUT/'five_seed_summary.csv')
    pairs=[]
    for view in ['public','top2','outputs','messages']:
        v=primary[primary.view==view]
        for seed in manifest.get('seeds',[100,17,223,483,907]):
            s=v[(v.arm=='S0')&(v.seed==seed)].iloc[0];d=v[(v.arm=='D1')&(v.seed==seed)].iloc[0]
            pairs.append(dict(seed=seed,view=view,**{metric:float(d[metric]-s[metric]) for metric in numerical+['auc','balanced_accuracy']}))
    pd.DataFrame(pairs).to_csv(OUT/'paired_differences.csv',index=False)
    gaps=[]
    for (arm,seed),r in primary.groupby(['arm','seed']):
        r=r.set_index('view');m=r.loc['messages']
        for baseline in ['public','top2','outputs']:
            b=r.loc[baseline]
            gaps.append(dict(arm=arm,seed=int(seed),baseline=baseline,extra_success_0319=float(m['success_0.319']-b['success_0.319']),extra_attribute_auc=float(m.auc-b.auc),extra_balanced_accuracy=float(m.balanced_accuracy-b.balanced_accuracy),extra_rmse_reduction=float(b.nrmse-m.nrmse)))
    pd.DataFrame(gaps).to_csv(OUT/'incremental_disclosure.csv',index=False)
    # All task metrics for the five pairs must equal the previous independent audit.
    checks=[]
    for row in t[t.seed!=42].itertuples():
        old=json.loads((ROOT/f'实验与代码/dCor稳定性复检/20261021-v1/results/{row.tag}/evaluation.json').read_text())
        assert abs(row.recall-old['recall'])<1e-12 and abs(row.ap-old['ap'])<1e-12
        checks.append(dict(tag=row.tag,task_unchanged=True))
    fields=[]
    ff=f[(f.seed!=42)&(f.group=='all')&(f.view=='messages')]
    aa=a[(a.seed!=42)&(a.group=='all')&(a.view=='messages')]
    for name,cell in ff.groupby('field'):
        s=cell[cell.arm=='S0'];d=cell[cell.arm=='D1']
        hit=cell.pivot(index='seed',columns='arm',values='success_0.319')
        q=aa[aa.field==name].groupby('arm')[['auc','balanced_accuracy']].mean()
        fields.append(dict(field=name,S0_nrmse=float(s.nrmse.mean()),D1_nrmse=float(d.nrmse.mean()),S0_success0319=float(s['success_0.319'].mean()),D1_success0319=float(d['success_0.319'].mean()),success_reduced_seeds=int((hit.D1<hit.S0).sum()),S0_attribute_auc=float(q.loc['S0','auc']),D1_attribute_auc=float(q.loc['D1','auc'])))
    pd.DataFrame(fields).to_csv(OUT/'field_summary.csv',index=False)
    stats={}
    for arm in ['S0','D1']:
        stats[arm]={}
        for view in ['public','top2','outputs','messages']:
            rows=primary[(primary.arm==arm)&(primary.view==view)]
            stats[arm][view]={k:float(rows[k].mean()) for k in numerical+['auc','balanced_accuracy']}
    g=pd.DataFrame(gaps)
    stats['extra_disclosure']={}
    for arm in ['S0','D1']:
        stats['extra_disclosure'][arm]={b:{k:float(r[k].mean()) for k in ['extra_success_0319','extra_attribute_auc','extra_balanced_accuracy']} for b,r in g[g.arm==arm].groupby('baseline')}
    td=t[t.seed!=42].groupby('arm')[['recall','ap']].mean();stats['task']=td.to_dict('index')
    stats['task_checks']=checks
    stats['privacy_field_success_reduced_all_seeds']=sum(r['success_reduced_seeds']==5 for r in fields)
    (OUT/'conclusions.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    dictionary=pd.read_csv(ROOT/'实验与代码/经济结构依据/字段与公式.csv').set_index('field')['name'].to_dict()
    def cn(n):return dictionary.get(n.split('::',1)[1],n)
    viewnames={'public':'公共/辅助信息','top2':'辅助信息＋风险与Top2输出','outputs':'辅助信息＋全部类型输出','messages':'全部输出＋完整节点消息'}
    lines=['# 当前隐私保护的多维评价','', '2026-10-22。固定已有模型，补充数值恢复成功率、相对高低值状态推断和授权输出基线。本轮没有训练预测模型，也没有根据结果更改防护参数。','', '## 1. 先说明我们实际评价了什么','', '- 目标固定为上一轮归因选出的 13 个财务字段，只评价真实观测值；结果属于预处理后的模型数值，不直接等于原始财务比率的精度。', '- 主要证据为五组 S0/dCor 配对；仍是已参与开发的 2018 验证年度。五个种子描述训练波动，不代替独立年度复验。', '- 攻击拟合和选型按企业分离，最终验证折不参与恢复器轮次、容差攻击器及属性阈值选择。每种防护分别训练适应其表示的攻击器。', '- Top2 基线假设实际允许发布总体风险分数、Top2 身份及对应分数；另评估释放全部类型分数的影响。共享副模型本轮输出为既有验证使用的未校准分数。', '- 假设攻击者有带真实财务字段的历史辅助样本。噪声/MID仅模拟单次固定缓存消息，不评价反复请求后平均。固定模拟随机数用于复现，假设攻击者不获得随机数；当前按身份确定随机数的实验方式不能直接当作部署中的安全随机实现。','', '## 2. 平均误差与实际猜中比例','', '连续值落在真实值上下 0.1、0.319、0.5 个攻击训练标准差之内，记为恢复成功。0.319借鉴 TabLeak，属于评价容差，不是隐私达标线。每档容差在内部选型集分别挑选攻击器，成功率越低越有利于防护。','', '|模型|攻击者信息|NRMSE|容差0.1成功率|容差0.319成功率|容差0.5成功率|','|---|---|---:|---:|---:|---:|']
    for arm in ['S0','D1']:
        for view in ['public','top2','outputs','messages']:
            r=stats[arm][view]
            lines.append(f'|{arm}|{viewnames[view]}|{r["nrmse"]:.3f}|{r["success_0.1"]:.1%}|{r["success_0.319"]:.1%}|{r["success_0.5"]:.1%}|')
    lines+=['', '## 3. 不要求猜中精确数值，经济状态是否仍能推断','', '在攻击拟合数据中预先确定同行业25%/75%分位数，判断每字段是否相对低值/高值，共26个属性。样本不足行业使用拟合总体分位数。分位数遇到并列时实际比例不等于25%，因此不使用普通准确率作为主指标。AUC和平衡准确率越高，攻击越强。','', '|模型|攻击者信息|属性宏AUC|宏平衡准确率|','|---|---|---:|---:|']
    for arm in ['S0','D1']:
        for view in ['public','top2','outputs','messages']:
            r=stats[arm][view];lines.append(f'|{arm}|{viewnames[view]}|{r["auc"]:.3f}|{r["balanced_accuracy"]:.1%}|')
    lines+=['', 'AUC与平衡准确率分别在内部选型集选择攻击器；平衡准确率阈值也只在内部选型集选定。数值恢复器之外另训专用属性分类器，避免把回归模型分箱当作完整属性攻击。','','## 4. 中间消息在授权输出之外增加多少可推断信息','', '|模型|参照信息|容差0.319成功率额外增加|属性AUC额外增加|','|---|---|---:|---:|']
    for arm in ['S0','D1']:
        for base in ['public','top2','outputs']:
            r=stats['extra_disclosure'][arm][base];lines.append(f'|{arm}|{viewnames[base]}|{100*r["extra_success_0319"]:+.1f}个百分点|{r["extra_attribute_auc"]:+.3f}|')
    lines+=['','这是当前攻击库下的经验增量。发布结果本身支持推断，不意味着业务上自动允许该敏感属性暴露；输出风险和消息风险分别报告。更丰富视图可复用更少信息的攻击，但最终验证中仍可能因分布变化出现局部非单调，不截断负差。','', 'Top2与全部类型分数的比较描述输出侧披露：若外部接收者只拿到Top2，其信息少于拿到全部分数的接收者。好奇的融合端本来就能看见内部消息和模型计算，单纯在界面隐藏其他分数不能保护它所能访问的信息。授权输出基线用于分解泄露，尚不构成限制服务器访问的协议实现。','','## 5. 每个关键字段','', '|指标|S0恢复成功率|dCor恢复成功率|成功率降低种子|S0属性AUC|dCor属性AUC|','|---|---:|---:|---:|---:|---:|']
    for r in fields:lines.append(f'|{cn(r["field"])}|{r["S0_success0319"]:.1%}|{r["D1_success0319"]:.1%}|{r["success_reduced_seeds"]}/5|{r["S0_attribute_auc"]:.3f}|{r["D1_attribute_auc"]:.3f}|')
    lines+=['', '以上恢复成功率采用0.319标准差容差；字段属性AUC是其高/低值两个属性的平均。字段误差升高、成功率下降和属性AUC下降是三个不同判断。','','## 6. 行业、风险名单和违规类型分组','', '|企业组|S0成功率|dCor成功率|S0属性AUC|dCor属性AUC|','|---|---:|---:|---:|---:|']
    for group,label in [('C','制造业'),('I','信息技术业'),('S0_highrisk','固定S0前10%名单'),('S0_other','其余企业'),('C_P2503','制造业／虚假记载或误导性陈述'),('C_P2504','制造业／推迟披露'),('C_P2505','制造业／重大遗漏'),('I_P2503','信息技术业／虚假记载或误导性陈述'),('I_P2504','信息技术业／推迟披露'),('I_P2505','信息技术业／重大遗漏')]:
        r=joined[(joined.seed!=42)&(joined.view=='messages')&(joined.group==group)].groupby('arm')[['success_0.319','auc']].mean()
        if set(r.index)=={'S0','D1'}:lines.append(f'|{label}|{r.loc["S0","success_0.319"]:.1%}|{r.loc["D1","success_0.319"]:.1%}|{r.loc["S0","auc"]:.3f}|{r.loc["D1","auc"]:.3f}|')
    lines+=['', '行业×类型为已记录该类型正例中的描述。缺乏样本或单一类别的属性不进入宏平均；各组属性支持数见 aggregate.csv，逐属性观测数和正例比例见 attributes.csv。小组结果不用于显著性或安全达标判断。','','## 7. 已有其他防护的探索性参照','', '以下仅种子42，不合入五组配对主判断。Gaussian sigma=1 是对同一S0共享表示加入训练表示标准差尺度的噪声，再用同一消息计算输出；MID沿用已有0.1配置。这些是评价点，不是本轮调参选出的赢家。','', '|方法|前10%召回|AP|完整消息NRMSE|恢复成功率0.319|属性AUC|','|---|---:|---:|---:|---:|---:|']
    exp=joined[(joined.seed==42)&(joined.group=='all')&(joined.view=='messages')]
    for row in exp.itertuples():
        task=t[t.tag==row.tag].iloc[0];label={'S0':'无防护共享模型','D1':'dCor','M01':'MID 0.1','G1':'Gaussian sigma=1'}[row.arm]
        success=float(exp.loc[exp.tag==row.tag,'success_0.319'].iloc[0])
        lines.append(f'|{label}|{task.recall:.1%}|{task.ap:.3f}|{row.nrmse:.3f}|{success:.1%}|{row.auc:.3f}|')
    s=stats['S0']['messages'];d=stats['D1']['messages'];ds=stats['extra_disclosure']['D1']['top2']
    lines+=['', '## 8. 当前可以作出的判断','', f'1. 在当前攻击下，dCor使关键字段容差0.319内的恢复成功率由 {s["success_0.319"]:.1%} 变为 {d["success_0.319"]:.1%}；属性宏AUC由 {s["auc"]:.3f} 变为 {d["auc"]:.3f}。结合多档容差和逐字段结果判断，不能只引用均方误差变化。', f'2. dCor完整消息相对Top2授权输出基线，仍增加 {100*ds["extra_success_0319"]:.1f} 个百分点的容差恢复成功率、{ds["extra_attribute_auc"]:.3f} 的属性AUC。当前表示仍携带授权输出之外可被利用的财务信息。', f'3. 五组平均召回：S0 {td.loc["S0","recall"]:.2%}、dCor {td.loc["D1","recall"]:.2%}；AP分别为 {td.loc["S0","ap"]:.4f}、{td.loc["D1","ap"]:.4f}。已逐值确认与此前预测评估一致。', '4. 保留现有主模型，dCor继续作为经验防护副模型。是否适合实际使用还取决于业务允许暴露的信息、可接受效用损失与其他年度验证；本轮不赋予安全认证或差分隐私保证。', '5. 对其余防护只作探索比较。若某个点更有利，需同种子、同输出和同攻击口径复验后再选型；本轮不自动扩展新防护训练。','', '## 9. 操作与产物','',f'- 攻击拟合 {manifest["fit_rows"]:,} 行、内部选型 {manifest["inner_rows"]:,} 行、验证 {manifest["validation_rows"]:,} 行。加载折0/1，未加载折2/3。',f'- 复用 {complete["models"]} 个已有预测模型/扰动评价点。来源与哈希见 model_sources.json；扰动点没有训练新预测器。','- 每个视图均保存数值和属性攻击器、内部选择、最优轮次与逐行预测。完整消息还回放上一轮已训练的强恢复器，回放预测通过一致性检查。','- fields.csv、attributes.csv：全部字段/属性、视图、行业与类型分组；aggregate.csv、paired_differences.csv：配对汇总。','- incremental_disclosure.csv：授权输出与中间消息的增量。field_summary.csv：关键字段主表。','- conclusions.json：机器可读结论；completion.json：完成记录；运行日志.txt：实际运行过程。','', '文献与既有结果保留在上一轮目录，本报告没有把新增的恢复成功率容差当作安全阈值。']
    (HERE/'隐私多维评价报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
