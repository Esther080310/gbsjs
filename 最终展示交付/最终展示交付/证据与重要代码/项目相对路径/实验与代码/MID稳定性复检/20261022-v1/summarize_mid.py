from pathlib import Path
import json
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'results';OLD=ROOT/'实验与代码/隐私多维评价/20261022-v1/results'

def main():
    done=json.loads((OUT/'completion.json').read_text());seeds=done['seeds']
    f=pd.read_csv(OUT/'fields.csv');a=pd.read_csv(OUT/'attributes.csv');t=pd.read_csv(OUT/'tasks.csv');rep=pd.read_csv(OUT/'noise_task_repeats.csv')
    metrics=['nrmse','success_0.1','success_0.319','success_0.5']
    fm=f[f.n_observed>=30].groupby(['seed','view','group'])[metrics].mean().reset_index()
    am=a[a.supported].groupby(['seed','view','group']).agg(auc=('auc','mean'),balanced_accuracy=('balanced_accuracy','mean'),n_attributes=('auc','count')).reset_index()
    agg=fm.merge(am,on=['seed','view','group'],how='left');agg['arm']='MID'
    original=pd.read_csv(OLD/'aggregate.csv');original=original[(original.seed.isin(seeds))&(original.arm.isin(['S0','D1']))]
    combined=pd.concat([original,agg],ignore_index=True)
    combined.to_csv(OUT/'aggregate_comparison.csv',index=False)
    oldtask=pd.read_csv(OLD/'tasks.csv');oldtask=oldtask[(oldtask.seed.isin(seeds))&(oldtask.arm.isin(['S0','D1']))]
    t['arm']='MID';alltask=pd.concat([oldtask,t],ignore_index=True);alltask.to_csv(OUT/'task_comparison.csv',index=False)
    result={};paired=[]
    for seed in seeds:
        s=oldtask[(oldtask.seed==seed)&(oldtask.arm=='S0')].iloc[0];d=oldtask[(oldtask.seed==seed)&(oldtask.arm=='D1')].iloc[0];m=t[t.seed==seed].iloc[0]
        row=dict(seed=seed,S0_recall=s.recall,D1_recall=d.recall,MID_recall=m.recall,MID_minus_S0_recall=m.recall-s.recall,MID_minus_D1_recall=m.recall-d.recall,MID_minus_S0_ap=m.ap-s.ap,MID_minus_D1_ap=m.ap-d.ap)
        for arm,base in [('S0','S0'),('D1','D1')]:
            x=combined[(combined.seed==seed)&(combined.group=='all')&(combined.view=='messages')].set_index('arm')
            for metric in metrics+['auc','balanced_accuracy']:row[f'MID_minus_{base}_{metric}']=float(x.loc['MID',metric]-x.loc[arm,metric])
        paired.append(row)
    paired=pd.DataFrame(paired);paired.to_csv(OUT/'paired_results.csv',index=False)
    utility=bool(paired.MID_minus_S0_recall.mean()>=-.005 and (paired.MID_minus_S0_recall<-.01).sum()<=1 and paired.MID_minus_S0_ap.mean()>=-.01)
    for arm in ['S0','D1','MID']:
        result[arm]={}
        for view in ['public','top2','outputs','messages']:
            v=combined[(combined.arm==arm)&(combined.view==view)&(combined.group=='all')]
            result[arm][view]={k:float(v[k].mean()) for k in metrics+['auc','balanced_accuracy']}
        v=alltask[alltask.arm==arm];result[arm]['task']={k:float(v[k].mean()) for k in ['recall','ap']}
    result['utility_engineering_gate_passed']=utility
    result['paired']={k:float(paired[k].mean()) for k in paired.columns if k!='seed'}
    result['protection_direction_counts']={base:{metric:int((paired[f'MID_minus_{base}_{metric}']<0).sum()) for metric in ['success_0.319','auc','balanced_accuracy']} for base in ['S0','D1']}
    result['new_noise_draw_task']={'mean_recall':float(rep.recall.mean()),'mean_ap':float(rep.ap.mean()),'minimum_recall':float(rep.recall.min()),'maximum_recall':float(rep.recall.max()),'draws':len(rep)}
    runs=json.loads((OUT/'pair_checks.json').read_text());assert len(runs)==5 and all(r['initial_common_parameters_exact'] and r['common_batch_prefix_exact'] for r in runs)
    result['all_five_initialization_and_batch_checks_passed']=True
    (OUT/'conclusions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines=['# MID五种子复检结果','', '2026-10-22。固定MID强度0.1，新训练五个共享框架预测模型，与既有同种子S0/dCor比较。评价目标、信息视图及攻击器预算沿用上一轮，未新增强度搜索。','', '## 1. 预测表现与隐私表现放在一起看','', '|模型|前10%召回|AP|数值误差NRMSE|0.319容差猜中率|属性AUC|属性平衡准确率|','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ['S0','D1','MID']:
        s=result[arm]['messages'];u=result[arm]['task'];label={'S0':'同架构无防护','D1':'dCor','MID':'MID 0.1'}[arm]
        lines.append(f'|{label}|{u["recall"]:.2%}|{u["ap"]:.4f}|{s["nrmse"]:.3f}|{s["success_0.319"]:.1%}|{s["auc"]:.3f}|{s["balanced_accuracy"]:.1%}|')
    lines+=['','猜中率指13个关键财务字段的已观测数值，误差在真实值上下0.319个攻击训练标准差内的比例；不是违规识别准确率。属性为同行训练分布中的相对高/低值组。NRMSE越高、攻击成功率与AUC越低，当前攻击下越难恢复。','','## 2. 每组种子都列出','', '|种子|S0召回|dCor召回|MID召回|MID−S0|MID−dCor|MID−S0 AP|','|---|---:|---:|---:|---:|---:|---:|']
    for r in paired.itertuples():lines.append(f'|{r.seed}|{r.S0_recall:.2%}|{r.D1_recall:.2%}|{r.MID_recall:.2%}|{100*r.MID_minus_S0_recall:+.3f}pp|{100*r.MID_minus_D1_recall:+.3f}pp|{r.MID_minus_S0_ap:+.4f}|')
    lines+=['', f'按沿用的预测效用工程容差，本轮满足情况：{utility}。具体条件为平均召回差≥−0.5pp、召回差低于−1pp的种子至多1个、平均AP差≥−0.01。这不是隐私安全标准。', '', '|比较|0.319猜中率更低|属性AUC更低|属性平衡准确率更低|','|---|---:|---:|---:|']
    for base in ['S0','D1']:
        v=result['protection_direction_counts'][base];lines.append(f'|MID 对 {base}|{v["success_0.319"]}/5|{v["auc"]}/5|{v["balanced_accuracy"]}/5|')
    lines+=['','## 3. MID的保护收益覆盖哪些容差','', '|模型|0.1标准差容差|0.319标准差容差|0.5标准差容差|','|---|---:|---:|---:|']
    for arm in ['S0','D1','MID']:
        s=result[arm]['messages'];lines.append(f'|{arm}|{s["success_0.1"]:.1%}|{s["success_0.319"]:.1%}|{s["success_0.5"]:.1%}|')
    lines+=['', '容差在本轮开始前固定，不能解释为达标安全线。各档分别在攻击内部选型集选恢复器，不在评价集挑选。','','## 4. 相对授权输出还泄露多少','', '|MID攻击者信息|猜中率0.319|属性AUC|平衡准确率|','|---|---:|---:|---:|']
    for view,label in [('public','公共/辅助信息'),('top2','辅助信息＋风险分数与Top2'),('outputs','辅助信息＋全部类型输出'),('messages','全部输出＋节点消息')]:
        s=result['MID'][view];lines.append(f'|{label}|{s["success_0.319"]:.1%}|{s["auc"]:.3f}|{s["balanced_accuracy"]:.1%}|')
    m=result['MID']['messages'];o=result['MID']['top2'];full=result['MID']['outputs']
    lines+=['', f'MID节点消息相对风险+Top2基线，仍增加 {100*(m["success_0.319"]-o["success_0.319"]):.1f} 个百分点数值猜中率及 {m["auc"]-o["auc"]:.3f} 属性AUC；相对全部类型输出，增量分别为 {100*(m["success_0.319"]-full["success_0.319"]):.1f} 个百分点及 {m["auc"]-full["auc"]:.3f}。', '', '只在界面隐藏其他输出不能限制融合端访问内部消息。本表分解授权输出与消息信息，没有实现限制服务器访问的协议。','','## 5. 行业及固定名单','', '|企业组|MID猜中率0.319|dCor猜中率0.319|MID属性AUC|dCor属性AUC|','|---|---:|---:|---:|---:|']
    for group,label in [('C','制造业'),('I','信息技术业'),('S0_highrisk','固定S0前10%名单'),('S0_other','其余企业')]:
        q=combined[(combined.group==group)&(combined.view=='messages')].groupby('arm')[['success_0.319','auc']].mean()
        lines.append(f'|{label}|{q.loc["MID","success_0.319"]:.1%}|{q.loc["D1","success_0.319"]:.1%}|{q.loc["MID","auc"]:.3f}|{q.loc["D1","auc"]:.3f}|')
    lines+=['','行业×违规类型的逐字段结果也已保存，组内样本不足或属性单一类别不进入宏平均。这些单年度分组是描述性诊断。','','## 6. 对噪声抽样是否敏感','', f'另用预定三个噪声盐，五个模型共15个任务评价点：平均召回 {rep.recall.mean():.2%}，范围 {rep.recall.min():.2%}—{rep.recall.max():.2%}；平均AP {rep.ap.mean():.4f}。', '', '|种子|主抽样召回|三次新抽样召回均值|新抽样范围|','|---|---:|---:|---:|']
    for seed,rr in rep.groupby('seed'):
        main=t[t.seed==seed].iloc[0];lines.append(f'|{seed}|{main.recall:.2%}|{rr.recall.mean():.2%}|{rr.recall.min():.2%}—{rr.recall.max():.2%}|')
    lines+=['', '不挑选表现最好的抽样。该部分只检验任务对单次噪声的波动，隐私攻击仍只评价主缓存消息。重复请求后平均未被本轮覆盖。','','## 7. 方法与实现复核','', '- 共有参数初始逐值一致；共同训练轮的批次哈希与旧S0一致。训练MID噪声用独立生成器，避免其改变Dropout随机数流。', f'- S0参数 {runs[0]["baseline_parameters"]:,}，MID参数 {runs[0]["parameters"]:,}。额外方差层已计入；消息维度不变，新增参数不作为上传字段。', '- MID验证噪声、消息和预测来自同一抽样。随机数的可重复模拟不等于部署随机安全，实际部署须使用私有安全随机源。', '- 攻击数据目标、掩码、标准差和属性标签与前轮逐值一致。专用数值/属性攻击器之外，另补训138字段及15候选字段恢复器，使攻击预算与S0/dCor可比。', '- 每种视图分别训练和选择攻击器；完整视图允许使用低信息视图攻击。最优轮次、模型及攻击器哈希均保存。','','## 8. 当前判读','', f'MID对dCor的数值猜中率平均差 {100*paired["MID_minus_D1_success_0.319"].mean():+.2f} 个百分点，属性AUC平均差 {paired.MID_minus_D1_auc.mean():+.4f}；相应召回差 {100*paired.MID_minus_D1_recall.mean():+.3f} 个百分点、AP差 {paired.MID_minus_D1_ap.mean():+.4f}。这些差值共同描述取舍，不只选择有利指标。', '', f'对无防护S0的平均召回差为 {100*paired.MID_minus_S0_recall.mean():+.3f} 个百分点、AP差 {paired.MID_minus_S0_ap.mean():+.4f}。是否满足原工程容差以上文为准。', '', '当前结果仅用于决定后续是否值得独立年度确认，不自动替换已确定主模型。即便恢复成功率更低，只要属性仍可较强推断或相对授权输出有增量，就应保留剩余风险的具体说明。', '', '五个种子不是五个独立年度，本轮不能提供差分隐私保证或真实机构部署安全认证。','','## 9. 产物索引','', '- models/：五个MID模型、训练历史、共有参数与批次检查。','- attacks/、auxiliary_recovery/：视图攻击与额外强恢复器、内部选型、逐行预测。','- fields.csv、attributes.csv：字段、行业、名单及行业×类型评价。','- paired_results.csv、aggregate_comparison.csv：与S0/dCor的配对汇总。','- noise_task_repeats.csv：固定三次额外抽样任务结果。','- manifest.json、pair_checks.json、completion.json、conclusions.json：来源、检查和结论。']
    (HERE/'MID复检结果与操作说明.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
