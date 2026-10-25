from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from scipy.stats import t
from sklearn.metrics import average_precision_score
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import stability as S


def main():
 root=HERE/'results';summary=json.loads((root/'summary.json').read_text());rows=summary['rows'];gate=summary['gate']
 pack=S.P.D.C.pack_for(2020,(0,1));target=pack['splits'][1]['y'].astype(int);budget=int(np.ceil(len(target)*.1));checks=[]
 for r in rows:
  with np.load(root/f"{r['arm']}_s{r['seed']}"/'validation_predictions.npz') as z:a=z['a_logit']
  frame=pd.DataFrame(dict(a=a,y=target));selected=frame.sort_values('a',ascending=False,kind='stable').iloc[:budget]
  assert int(selected.y.sum())/int(target.sum())==r['recall']
  assert abs(average_precision_score(target,a)-r['ap'])<1e-12
  assert min(r['attack']['attacker_nrmse'].values())==r['numeric_nrmse']
  checks.append(dict(seed=r['seed'],arm=r['arm'],recall_ap_exact=True,strongest_attacker_selected=True))
 S.P.write(root/'independent_verification.json',dict(checks=checks,all_passed=True,folds_loaded=[0,1]))
 text=['# dCor稳定性复检结果','','2026-10-21。2020开发年度验证折，未读取校准/测试折。','', '## 1. 为什么单次召回可能高于主模型','', 'dCor是额外正则项，可能压低与风险排序无关的细节、改善泛化，也可能损害有用信号。此前dCor比固定原六模型高约0.35pp，仅多命中3家，无法由一个点判断因果或稳定性；且两者架构不同。应以同共享框架配对比较隔离正则效应，原主模型保留为外部性能参照。','', '## 2. 更强攻击器复检','','保留Ridge与原20轮浅MLP，新增512/256两隐藏层、最多80轮、验证早停的独立攻击器。每个候选取所有攻击器中最低NRMSE。','', '|seed42模型|浅MLP误差|深MLP误差|Ridge误差|最强攻击误差|','|---|---:|---:|---:|---:|']
 for r in rows[:2]:
  a=r['attack']['attacker_nrmse'];text.append(f"|{r['arm']}|{a['shallow_mlp']:.4f}|{a['deep_mlp']:.4f}|{a['ridge']:.4f}|{r['numeric_nrmse']:.4f}|")
 text+=['',f"更强攻击下dCor相对无防护NRMSE提高{gate['stageA_relative_error_increase']:.2%}，阶段A{'通过' if gate['passed'] else '未通过'}。",'']
 if summary['status']!='complete':
  text+=['复检在攻击强度门停止，未启动额外预测网络训练。'];(HERE/'结果解读.md').write_text('\n'.join(text));return
 pairs=summary['pairs'];new=[r for r in pairs if r['seed']!=42]
 ref=json.loads((S.ROOT/'实验与代码/隐私效用统一比较/20261021-v1/results/summary.json').read_text())['external_independent_six_model']
 by={(r['seed'],r['arm']):r for r in rows}
 text+=['## 3. 配对种子结果','','seed42是先前挑选候选的探索点；另外5个是本轮新增训练，主判断只看这5个。每组初始参数与共同训练轮的批次顺序逐项一致。','', '|种子|S0召回|dCor召回|召回差pp|AP差|S0数值误差|dCor数值误差|误差相对提高|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in pairs:text.append(f"|{r['seed']}{'（探索）' if r['seed']==42 else ''}|{r['recall_S0']:.2%}|{r['recall_D1']:.2%}|{r['recall_difference']*100:+.3f}|{r['ap_difference']:+.4f}|{r['nrmse_S0']:.4f}|{r['nrmse_D1']:.4f}|{r['relative_error_increase']:.2%}|")
 vals=np.array([r['recall_difference'] for r in new]);mean=float(vals.mean());se=float(vals.std(ddof=1)/np.sqrt(len(vals)));ci=[float(mean-t.ppf(.975,len(vals)-1)*se),float(mean+t.ppf(.975,len(vals)-1)*se)]
 drec=np.array([r['recall_D1'] for r in new]);dmain=drec-ref['recall'];dm=dmain.mean();ds=dmain.std(ddof=1)/np.sqrt(len(dmain));dci=[float(dm-t.ppf(.975,4)*ds),float(dm+t.ppf(.975,4)*ds)]
 stats=dict(new_seeds=len(new),positive_recall_pairs=int((vals>0).sum()),mean_paired_recall_difference=mean,seed_t_interval95=ci,mean_S0_recall=float(np.mean([r['recall_S0'] for r in new])),mean_D1_recall=float(drec.mean()),mean_ap_difference=float(np.mean([r['ap_difference'] for r in new])),mean_relative_error_increase=float(np.mean([r['relative_error_increase'] for r in new])),privacy_positive_pairs=sum(r['relative_error_increase']>=.1 for r in new),external_main_recall=ref['recall'],D1_above_external_main=int((dmain>0).sum()),mean_D1_minus_external_main=float(dm),D1_minus_main_seed_t_interval95=dci,privacy_stable=summary['new_seed_privacy_stable'],utility_preserved=summary['new_seed_utility_preserved'],eligible_for_future_cross_year=summary['eligible_for_future_cross_year'])
 S.P.write(root/'interpretation.json',stats)
 text+=['', '## 4. 只看5个新种子的判断','',f"- 同框架平均召回：S0 {stats['mean_S0_recall']:.2%}，dCor {stats['mean_D1_recall']:.2%}，配对差{mean*100:+.3f}pp；{stats['positive_recall_pairs']}/5个种子为正。",f"- 配对召回差的训练种子t区间：[{ci[0]*100:+.3f}, {ci[1]*100:+.3f}]pp。它只描述同一验证集上的训练波动，不是跨企业/跨年总体推断。",f"- 平均AP差{stats['mean_ap_difference']:+.4f}。",f"- 数值重构误差平均相对提高{stats['mean_relative_error_increase']:.2%}；{stats['privacy_positive_pairs']}/5个新种子达到至少10%的工程改善门。",f"- 数值防护稳定门：{stats['privacy_stable']}；效用保留门：{stats['utility_preserved']}。",'', '## 5. 和原主模型比较','',f"原六独立模型验证召回{ref['recall']:.2%}；5个新种子的dCor平均{stats['mean_D1_recall']:.2%}，平均差{dm*100:+.3f}pp，{stats['D1_above_external_main']}/5个高于原主模型。该比较包含架构与训练随机性变化，不能把差值全归给dCor。",'', '## 6. 下一步','',('固定dCor=1，值得进入跨年度确认；当前仍不替换主模型。下一轮应固定协议，核对类型提示、校准、全消息属性泄漏与数值专门攻击，分别报告精度优先主模型和数值隐私优先副模型。' if stats['eligible_for_future_cross_year'] else '当前未同时满足数值防护稳定与效用保留，停止推广该dCor配置；不在这个验证年度追加强度补救。'), '', '数值NRMSE升高是当前攻击器下的恢复困难增加，不等同于差分隐私、不可逆或整体安全。旧全字段隐私判定未撤销。','', '![配对种子](figures/dCor配对复检.png)']
 (HERE/'结果解读.md').write_text('\n'.join(text)+'\n');plot(pairs)
 print(json.dumps(stats,ensure_ascii=False,indent=2))


def plot(pairs):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib import font_manager
 font_manager.fontManager.addfont('/Users/how/Library/Fonts/msyh.ttc')
 plt.rcParams.update({'font.family':font_manager.FontProperties(fname='/Users/how/Library/Fonts/msyh.ttc').get_name(),'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
 for i,r in enumerate(pairs):
  color='#BBBBBB' if r['seed']==42 else plt.get_cmap('tab10')(i)
  label=f"{r['seed']}"+('（探索）' if r['seed']==42 else '')
  axes[0].plot([0,1],[r['recall_S0']*100,r['recall_D1']*100],'-o',color=color,label=label,lw=1.4)
  axes[1].plot([0,1],[r['nrmse_S0'],r['nrmse_D1']],'-o',color=color,lw=1.4)
 for ax in axes:ax.set_xticks([0,1],['无防护S0','dCor=1']);ax.grid(axis='y',alpha=.2);ax.margins(x=.18)
 axes[0].set_ylabel('Recall@10%（%）');axes[1].set_ylabel('最强数值攻击宏NRMSE（越高越好）');axes[0].legend(ncol=2,fontsize=8,loc='best')
 fig.suptitle('dCor稳定性复检：同初始化、同批次的配对训练',fontsize=13);fig.supxlabel('2020验证折；灰色seed42为已见探索点，主判断仅看5个新增种子。',fontsize=9)
 dest=HERE/'figures';dest.mkdir(exist_ok=True);fig.savefig(dest/'dCor配对复检.png',dpi=200);fig.savefig(dest/'dCor配对复检.svg');plt.close(fig)
 (dest/'图表契约.md').write_text('论点：显示dCor相对同框架S0在训练种子变化下的任务与数值重构变化。每线为同初始化同批次配对，seed42旧探索点灰显，其余5个为新增训练。左召回右NRMSE，不混入不同攻击强度和年度；不把NRMSE称隐私概率。微软雅黑，PNG/SVG。\n')

if __name__=='__main__':main()
