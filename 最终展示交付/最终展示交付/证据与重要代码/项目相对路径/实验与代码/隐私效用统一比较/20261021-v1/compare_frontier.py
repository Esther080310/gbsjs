from pathlib import Path
import json,sys
import numpy as np,pandas as pd,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'实验与代码/FiLM-L随机扰动副模型/20261021-v1'))
import noise_pilot as P
sys.path.insert(0,str(ROOT/'实验与代码/FiLM-L隐私防护/20261021-v2'))
import shared_privacy_experiment as E


def evaluate(formal,model,pack,views,method,arm,strength,rep):
 splits=pack['splits'];task=P.EV.task_metrics(splits[0],splits[1],views[1]['a'],views[1]['b'])
 t0,names=P.numeric_targets(splits[0],pack);t1,_=P.numeric_targets(splits[1],pack)
 private=P.numeric_attack(views[0]['full_deployment'],views[1]['full_deployment'],t0,t1)
 result=dict(method=method,arm=arm,strength=strength,rep=rep,recall=task['recall_at_10pct'],ap=task['overall_average_precision'],numeric_nrmse=private['constant_nrmse']-private['gain'],normalized_error=(private['constant_nrmse']-private['gain'])/private['constant_nrmse'],gain=private['gain'],constant_nrmse=private['constant_nrmse'],task=task,attack=private,message_dimensions=views[0]['full_deployment'].shape[1],target_dimensions=len(names),same_shared_backbone=True)
 P.write(OUT/f'{arm}_r{rep}.json',result)
 print(arm,rep,'recall',result['recall'],'error',result['numeric_nrmse'],flush=True)
 return result

@torch.no_grad()
def noisy(formal,model,split,source,scales,sigma,rep):
 eps=P.noises(split,42,rep);keys=('h1','h2','h3');h={k:source[k]+np.float32(sigma)*scales[k]*eps[e] for k,e in zip(keys,('market','finance','governance'))}
 aa=[];bb=[]
 for start in range(0,len(split['y']),4096):
  sl=slice(start,start+4096);hs=[torch.tensor(h[k][sl]) for k in keys];c=torch.tensor(source['c'][sl]);r=torch.tensor(source['r'][sl]);q=torch.tensor(split['q'][sl]);ind=torch.tensor(split['industry'][sl])
  out=[head(hs,c,r,q,ind) for head in model.heads];aa.append(torch.stack([v[0] for v in out]).mean(0).numpy());bb.append(torch.stack([v[1] for v in out]).mean(0).numpy())
 return dict(a=np.concatenate(aa),b=np.concatenate(bb),full_deployment=np.concatenate([h['h1'],h['h2'],h['h3'],source['q'],source['c'],source['r'],source['industry']],1))

def main():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=False)
 formal=P.D.C.shared.load_loader().formal;pack=P.D.C.pack_for(2020,(0,1));rows=[]
 P.write(OUT/'manifest.json',dict(year=2020,folds_loaded=[0,1],fold2_loaded=False,fold3_loaded=False,predictor_training=False,protocol_sha=P.D.file_digest(HERE/'评价前约定.md'),source_sha=P.D.file_digest(Path(__file__))))
 s0=None;s0views=None
 for arm,(kind,strength) in E.ARMS.items():
  model=E.load(formal,pack,arm);views=[E.extract(model,s) for s in pack['splits']]
  rows.append(evaluate(formal,model,pack,views,'无防护' if arm=='S0' else ('dCor' if kind=='dcor' else 'MID'),arm,strength,42))
  if arm=='S0':s0=model;s0views=views
 scales={k:s0views[0][k].std(0,dtype=np.float64).astype('f4') for k in ('h1','h2','h3')}
 for sigma in (.2,.5,1.):
  for rep in P.REPS:
   views=[noisy(formal,s0,s,sv,scales,sigma,rep) for s,sv in zip(pack['splits'],s0views)]
   rows.append(evaluate(formal,s0,pack,views,'Gaussian',f'Gaussian{sigma:g}',sigma,rep))
 # Raw data table contains every realization; the plotted frontier uses method-strength means.
 frame=pd.DataFrame([{k:r[k] for k in ['method','arm','strength','rep','recall','ap','numeric_nrmse','normalized_error','gain','constant_nrmse','message_dimensions','target_dimensions']} for r in rows]);frame.to_csv(OUT/'all_points.csv',index=False)
 agg=frame.groupby(['method','arm','strength'],sort=False).agg(recall=('recall','mean'),recall_min=('recall','min'),recall_max=('recall','max'),numeric_nrmse=('numeric_nrmse','mean'),error_min=('numeric_nrmse','min'),error_max=('numeric_nrmse','max'),ap=('ap','mean'),n=('rep','count')).reset_index()
 pareto=[]
 for _,r in agg.iterrows():
  dominated=((agg.recall>=r.recall)&(agg.numeric_nrmse>=r.numeric_nrmse)&((agg.recall>r.recall)|(agg.numeric_nrmse>r.numeric_nrmse))).any();pareto.append(not dominated)
 agg['pareto_point_estimate']=pareto;agg.to_csv(OUT/'frontier.csv',index=False)
 baseline=json.loads((P.OUT/'selection.json').read_text())['reference']
 external=dict(recall=baseline['task']['recall_at_10pct'],ap=baseline['task']['overall_average_precision'],numeric_nrmse=baseline['privacy']['constant_nrmse']-baseline['privacy']['gain'],different_backbone=True)
 P.write(OUT/'summary.json',dict(status='complete_validation_only',rows=agg.to_dict('records'),external_independent_six_model=external,points=len(frame),method_strength_points=len(agg),predictor_training=False,folds_loaded=[0,1]))
 plot(frame,agg,external);report(agg,external)


def plot(frame,agg,external):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib import font_manager
 font_manager.fontManager.addfont('/Users/how/Library/Fonts/msyh.ttc')
 plt.rcParams.update({'font.family':font_manager.FontProperties(fname='/Users/how/Library/Fonts/msyh.ttc').get_name(),'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 colors={'无防护':'#555555','dCor':'#0072B2','MID':'#D55E00','Gaussian':'#009E73'};markers={'无防护':'s','dCor':'o','MID':'^','Gaussian':'D'}
 fig,ax=plt.subplots(figsize=(8.5,5.7),layout='constrained')
 for method,group in agg.groupby('method',sort=False):
  group=group.sort_values('strength');color=colors[method]
  if method not in ['无防护']:
   baseline=agg[agg.arm=='S0'].iloc[0]
   ax.plot([baseline.recall*100,*list(group.recall*100)],[baseline.numeric_nrmse,*list(group.numeric_nrmse)],color=color,alpha=.45,lw=1)
  ax.errorbar(group.recall*100,group.numeric_nrmse,xerr=np.stack([(group.recall-group.recall_min)*100,(group.recall_max-group.recall)*100]),yerr=np.stack([group.numeric_nrmse-group.error_min,group.error_max-group.numeric_nrmse]),fmt=markers[method],ms=7,color=color,capsize=3,label=method)
  for _,r in group.iterrows():
   label='共享编码器参照' if method=='无防护' else f'{method} {r.strength:g}'
   offsets={'S0':(6,-24),'D001':(9,-18),'D01':(-58,0),'D1':(8,8),'M0001':(-45,15),'M001':(9,8),'M01':(9,9),'Gaussian0.2':(-104,-18),'Gaussian0.5':(-103,10),'Gaussian1':(-91,12)}
   dx,dy=offsets[r.arm]
   ax.annotate(label,(r.recall*100,r.numeric_nrmse),xytext=(dx,dy),textcoords='offset points',fontsize=8.5,color=color)
 p=agg[agg.pareto_point_estimate].sort_values('recall');ax.plot(p.recall*100,p.numeric_nrmse,'--',color='#333333',lw=1.5,label='经验Pareto前沿')
 ax.scatter([external['recall']*100],[external['numeric_nrmse']],s=105,marker='*',color='#8B6BA8',label='原独立六模型（不同架构）',zorder=5)
 ax.annotate('原主模型',(external['recall']*100,external['numeric_nrmse']),xytext=(-5,-19),textcoords='offset points',fontsize=8.5,ha='right',color='#8B6BA8')
 ax.set_xlabel('Recall@10%（%，越高越好）');ax.set_ylabel('经济数值宏NRMSE（越高，重构越困难）');ax.grid(alpha=.15);ax.margins(x=.23,y=.20)
 ax.set_title('不同防护方法的筛查—重构权衡：2020验证折',fontsize=13)
 ax.text(.99,.98,'更理想 ↗',transform=ax.transAxes,ha='right',va='top',fontsize=10,color='#444444')
 ax.legend(loc='lower left',fontsize=8,framealpha=.88)
 fig.supxlabel('Gaussian误差线为3次扰动范围；其余点为单次模型/随机实现。前沿按点估计，不代表显著优越。',fontsize=8)
 dest=HERE/'figures';dest.mkdir(exist_ok=True)
 fig.savefig(dest/'Recall与数值保护_Pareto.png',dpi=200);fig.savefig(dest/'Recall与数值保护_Pareto.svg');plt.close(fig)
 (dest/'图表契约.md').write_text('论点：同共享架构下各防御族在Recall与数值NRMSE之间的经验折中，不预设赢家。数据all_points.csv及frontier.csv。Python散点图，各族连线表示强度，前沿仅同架构点估计，外部主模型星标。Gaussian三次范围，其余n1，训练/验证折一致。不把高NRMSE称安全概率。微软雅黑，PNG+可编辑SVG。\n')

def report(agg,external):
 text=['# 不同隐私防护方法的统一比较','','2026-10-21。仅2020验证折，不重训预测网络，不读取校准/测试折。','', '## 图如何读','', '每个点是一种算法及强度。横轴越右表示核查前10%企业覆盖越多违规；纵轴越高表示适应性攻击器恢复经济数值的误差越大。右上更理想。纵轴不是错误百分比，也不是隐私安全概率。', '', 'dCor、MID和Gaussian均基于同一共享编码器/六头框架。原独立六模型仅作外部参照，不参与同架构前沿。目标均为138个数值字段，攻击器在完整部署视图上按相同Ridge/MLP协议重训。','','## 全部强度点','','|方法|强度|Recall@10%|AP|数值NRMSE↑|重复数|经验前沿|','|---|---:|---:|---:|---:|---:|---|']
 for _,r in agg.iterrows():text.append(f'|{r.method}|{r.strength:g}|{r.recall:.2%}|{r.ap:.4f}|{r.numeric_nrmse:.4f}|{int(r.n)}|{r.pareto_point_estimate}|')
 text+=['',f"外部原主模型：Recall={external['recall']:.2%}，数值NRMSE={external['numeric_nrmse']:.4f}，AP={external['ap']:.4f}。",'', '## 如何据此做选择','', '经验前沿上的点意味着尚无另一个共享架构点在召回和数值重构误差上都超过它。前沿内的点无需因隐私目标继续推广。前沿并不保证差异超过训练/采样波动，也不保证已经阻止反演。','', '本轮Gaussian是三次扰动均值，其余模型为单次训练/随机实现。因此只能据此提出下一轮候选，不能直接宣布某个方法稳定最好。AP也应参与后续筛选，防止只顾Top10切点损伤总体风险排序。','', '下一步先明确可接受的召回代价，从前沿冻结1—2个候选；先进行多种子验证，再在新的年度检验，不在当前测试年反复挑强度。','', '![统一折中图](figures/Recall与数值保护_Pareto.png)','']
 (HERE/'结果解读.md').write_text('\n'.join(text))

if __name__=='__main__':main()
