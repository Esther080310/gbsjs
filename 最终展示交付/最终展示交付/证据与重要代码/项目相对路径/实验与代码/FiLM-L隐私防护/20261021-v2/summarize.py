"""Independent task metric recomputation and human-readable pilot findings."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import shared_privacy_experiment as E

def main():
 root=HERE/'results'; decision=json.loads((root/'selection.json').read_text()); end=json.loads((root/'final/summary.json').read_text())
 formal=E.C.shared.load_loader().formal; pack=E.C.pack_for(2020,(0,1,2,3)); s=pack['splits'][3]; y=np.asarray(s['y'])
 base=[];bt=[]
 for seed in E.C.SEEDS:
  with np.load(E.D.CONTROL/f'artifacts/2020/FiLM-L_s{seed}/fold3.npz') as z:base.append(z['a_logit']);bt.append(z['b_logit'])
 base_a=np.mean(base,0);base_t=np.mean(bt,0)
 base_metrics=E.EV.task_metrics(pack['splits'][0],s,base_a,base_t)
 base_record=json.loads((E.AUDIT/'results/primary-audit-v2-formal/records/2020_joint6_full_deployment.json').read_text())
 checks=[];rows=[]
 for result in end['results']:
  arm=result['arm']; z=np.load(root/'final'/arm/'predictions.npz'); logits=z['a_logit']
  frame=pd.DataFrame(dict(score=logits,y=y,symbol=s['keys'].Symbol.to_numpy()))
  selected=frame.sort_values('score',ascending=False,kind='stable').iloc[:int(np.ceil(len(frame)/10))]
  recall=int(selected.y.sum())/int(frame.y.sum());ap=float(average_precision_score(y,logits))
  assert recall==result['task']['recall_at_10pct']
  assert abs(ap-result['task']['overall_average_precision'])<1e-12
  checks.append(dict(arm=arm,independent_recall_exact=True,independent_ap_exact=True))
  full=next(r for r in result['privacy'] if r['view']=='full_deployment')
  public=next(r for r in result['privacy'] if r['view']=='public_side_only')
  rows.append(dict(arm=arm,recall=recall,ap=ap,recall_difference_pp=100*(recall-base_metrics['recall_at_10pct']),ap_difference=ap-base_metrics['overall_average_precision'],full_gain=full['gain'],full_auc=full['auc'],public_gain=public['gain'],public_auc=public['auc'],practical_leakage=full['practical_leakage']))
 verify=dict(checks=checks,selected_arms=end['results'] and [r['arm'] for r in end['results']],durable_selection_sha=E.D.file_digest(root/'selection.json'),final_selection_hash_matches=E.D.file_digest(root/'selection.json')==end['selection_sha'],baseline=base_metrics,rows=rows)
 assert verify['final_selection_hash_matches']
 E.write(root/'independent_verification.json',verify)
 E.write(root/'comparison.json',dict(baseline=base_metrics,baseline_full_gain=base_record['selected_gain_vs_constant']['point_estimate'],baseline_full_auc=base_record['attribute_macro_auc']['point_estimate'],rows=rows))
 text=['# 共享编码器与信息约束：2020年度小试验结果','','2026-10-21。固定开发年度，不是跨年度确认。','', '## 1. 研究问题与执行','', '本轮让三个节点各上传一份表示，六个融合头共用。先检验减少多模型投影是否改善联合泄漏，再分别加入距离相关约束和随机信息瓶颈。7个候选均仅用fold0训练、fold1选模；选择文件写定后，只有S0及符合门槛的候选进入fold2校准、fold3评价。原主模型未覆盖。','', '## 2. 验证折选择','','|臂|Recall@10%|AP|完整视图重构增益|效用合格|相对S0增益减少|','|---|---:|---:|---:|---|---:|']
 s0=next(r for r in decision['rows'] if r['arm']=='S0')['reconstruction_gain']
 for r in decision['rows']:
  text.append(f"|{r['arm']}|{r['recall_at_10pct']:.2%}|{r['overall_average_precision']:.4f}|{r['reconstruction_gain']:.4f}|{r['task_eligible']}|{s0-r['reconstruction_gain']:.4f}|")
 text+=['',f"进入最终评价：{', '.join(decision['selected'])}。D/M须同时通过验证折效用门，并相对S0减少完整重构增益至少0.10。未入选臂未读取测试折。",'', '## 3. 最终测试折','','|模型|Recall@10%|AP|重构增益|属性AUC|实用泄漏仍成立|','|---|---:|---:|---:|---:|---|',f"|原FiLM-L六模型|{base_metrics['recall_at_10pct']:.2%}|{base_metrics['overall_average_precision']:.4f}|{base_record['selected_gain_vs_constant']['point_estimate']:.4f}|{base_record['attribute_macro_auc']['point_estimate']:.4f}|True|"]
 for r in rows:text.append(f"|{r['arm']}|{r['recall']:.2%}|{r['ap']:.4f}|{r['full_gain']:.4f}|{r['full_auc']:.4f}|{r['practical_leakage']}|")
 text+=['', '重构增益为“常数预测NRMSE−攻击NRMSE”，越小越好；AUC越接近0.5，敏感属性越难从消息中恢复。保护目标是预处理后的模型输入，未声称恢复原始CSMAR数值。每个最终端点均重新拟合Ridge/MLP及直接属性攻击，攻击模型仅在训练折拟合、验证折选择。','', '## 4. 主动共享侧信息参照','','|模型|只看q/c/r/行业的重构增益|只看侧信息的属性AUC|看完整消息的重构增益|','|---|---:|---:|---:|']
 for r in rows:text.append(f"|{r['arm']}|{r['public_gain']:.4f}|{r['public_auc']:.4f}|{r['full_gain']:.4f}|")
 text+=['', '侧信息与原输入本身相关，单看其攻击也能恢复部分字段。完整与侧信息两组使用相同攻击协议，但增益相减只是经验比较，不是条件互信息估计。','', '## 5. 结果判读与采用决定','']
 for r in rows:
  task_ok=r['recall_difference_pp']>=-1 and r['ap_difference']>=-.01
  text.append(f"- {r['arm']}：相对原六模型召回差{r['recall_difference_pp']:+.3f}pp，AP差{r['ap_difference']:+.4f}；最终效用门{'通过' if task_ok else '未通过'}；完整视图实用泄漏{'仍成立' if r['practical_leakage'] else '未检出'}。")
 promote=[r for r in rows if r['recall_difference_pp']>=-1 and r['ap_difference']>=-.01 and not r['practical_leakage']]
 text+=['', '采用决定：'+('存在可进入跨年验证的候选，仍需完整多年度复验。' if promote else '本轮没有同时保留任务效用并通过隐私实用门的候选，主模型维持冻结。不能把训练正则下降或消息减少写成隐私保护通过。'),'', '## 6. 通信与范围','', '共享方案只上传一次137个槽位（h1=16、h2=88、h3=21、c=8、q=3、r=1），按136个FP32和r为uint8计545 B/企业；相对旧六模型FP32的3205 B下降83.0%。这是表示载荷结构核算，未包含行业公共字段、包头、加密、梯度；尚未对新权重执行混合精度名单保持测试，所以不能继承旧899 B候选的严格保真结论。','', '本轮随机瓶颈属于经验防御，不提供差分隐私保证。实验以可重放的键控伪随机数模拟单次私有消息；真实实现须使用私有随机性并缓存每企业年度消息。多次独立查询、恶意参与方、训练梯度和白盒反演尚待单独检验。','', '## 7. 复现与核对','', '独立用pandas稳定排序复算最终命中和召回，并用原始logit重新计算AP；选择文件哈希与最终评价逐项一致。详情见results/independent_verification.json。执行中仅在选择前修复旧六模型参照的校准聚合口径，见执行记录.md；没有根据测试结果调整候选或阈值。','', '候选定义与公式见训练前约定.md，全部逐字段攻击与企业簇区间位于results/final/各臂/{records,fields,bootstrap}。']
 # Descriptive follow-up only: disclose target composition and a simpler frozen reference.
 numeric_rows=[]
 for label,path in [('原六模型',E.AUDIT/'results/primary-audit-v2-formal/records/2020_joint6_full_deployment.json'),('共享S0完整消息',root/'final/S0/records/2020_shared6_full_deployment.json'),('共享S0仅侧信息',root/'final/S0/records/2020_shared6_public_side_only.json')]:
  rr=json.loads(path.read_text()); mm=rr['test_metrics']; method=rr['selected_attacker']
  numeric_rows.append(dict(model=label,numeric_gain=mm['constant']['macro_nrmse_numeric']-mm[method]['macro_nrmse_numeric'],binary_gain=mm['constant']['macro_nrmse_binary']-mm[method]['macro_nrmse_binary']))
 single_record=json.loads((E.AUDIT/'results/primary-audit-v2-formal/records/2020_single42_full_deployment.json').read_text())
 with np.load(E.D.CONTROL/'artifacts/2020/FiLM-L_s42/fold3.npz') as z:
  single_metrics=E.EV.task_metrics(pack['splits'][0],s,z['a_logit'],z['b_logit'])
 E.write(root/'descriptive_target_breakdown.json',dict(numeric_binary=numeric_rows,single42_task=single_metrics,single42_gain=single_record['selected_gain_vs_constant']['point_estimate'],descriptive_only=True,model_selection_changed=False))
 text+=['','## 8. 追加解释：哪些信息被恢复了','','以下为固定结果后的描述性拆分，不参与选择，也不改变原通过判据。原审计属性主要为缺失标志、市场类别和事件是否出现；它不是实际经济数值的高低分类。仅侧信息的高属性AUC主要反映缺失模式易于推断。','', '|视图|经济数值NRMSE增益|缺失/类别编码NRMSE增益|','|---|---:|---:|']
 for rr in numeric_rows:text.append(f"|{rr['model']}|{rr['numeric_gain']:.4f}|{rr['binary_gain']:.4f}|")
 text+=['', '完整消息对经济数值的恢复仍远强于仅侧信息，因此联合泄漏不是单纯由缺失标志造成的。下一轮应同时报告经济数值恢复、缺失模式恢复以及相对主动共享侧信息的增量，不能只用一个总AUC概括隐私。','', f"简单参照：原单模型seed42的Recall@10%为{single_metrics['recall_at_10pct']:.2%}，AP为{single_metrics['overall_average_precision']:.4f}，完整重构增益为{single_record['selected_gain_vs_constant']['point_estimate']:.4f}。共享S0未展示优于这一简单参照的隐私—效用折中。",'', '减少多套表示与重构降低的解释一致；但共享训练也改变参数量与优化，当前比较不能把全部变化单独归因于消息份数。当前不推广共享编码器，不扩大年度训练。']
 (HERE/'结果解读.md').write_text('\n'.join(text)+'\n')
 print(json.dumps(verify,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
