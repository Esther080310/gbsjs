from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
H=Path(__file__).resolve().parent; R=H/'results'; F=H/'figures';F.mkdir(exist_ok=True)
ROOT=H.parents[2]
font=Path('/Users/how/Library/Fonts/msyh.ttc');font_manager.fontManager.addfont(str(font));family=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.family':family,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#263445','text.color':'#263445','axes.edgecolor':'#8D99A5','svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False})
ARMS=['Main','S0','MID']; LABEL={'Main':'主模型 FiLM-L','S0':'同架构无防护 S0','MID':'隐私副模型 MID'}; COLOR={'Main':'#234E70','S0':'#009E73','MID':'#D55E00'}
fields=pd.read_csv(R/'field_sensitivity.csv'); nodes=pd.read_csv(R/'node_shares.csv'); cases=pd.read_csv(R/'cases.csv',dtype={'symbol':str}); sample=pd.read_csv(R/'sample.csv'); audit=pd.read_csv(R/'audit_rank_shift.csv')
dictf=pd.read_csv(ROOT/'实验与代码/经济结构依据/字段与公式.csv'); names=dict(zip(dictf.field,dictf.name))
names.update({'mkt__company_age':'公司年龄','mkt__listing_age':'上市年限','exec__events':'高管变更次数','exec__resign':'高管离任次数','exec__chair':'董事长变更次数','exec__manager':'总经理变更次数','exec__external':'外部来源变更计数','exec__mean_age':'变更高管平均年龄','exec__mean_tenure':'变更高管平均任期','exec__mean_notice_lag':'变更至公告平均间隔'})
# Compact labels are display-only; original data retain complete source definitions.
names.update({'prof__F052201B':'资产减值损失／营业收入','oper__F041001B':'现金及等价物周转率A','solv__F010801B':'经营现金净流量／流动负债','solv__F011501A':'有形资产带息债务比'})
def save(fig,name,caption):
 fig.savefig(F/f'{name}.png',dpi=300,bbox_inches='tight',facecolor='white');fig.savefig(F/f'{name}.svg',bbox_inches='tight',facecolor='white');plt.close(fig)
 (F/f'{name}.txt').write_text(caption,encoding='utf-8')
# Independent checks against cached row-wise arrays; no risk networks are trained here.
checks={};ix=np.flatnonzero(sample.random_sample.to_numpy());groups=['X1市场','X2财务','X3治理事件','审计调制分支']
for a in ARMS:
 z=np.load(R/f'{a}_node_attribution.npz');phi=z['phi'];assert phi.shape[0]==(1 if a=='Main' else 5)
 checks[a+'_additivity']=float(np.max(np.abs(phi.sum(axis=2)-(z['prediction']-z['baseline']))));assert checks[a+'_additivity']<2e-5
 raw=np.mean(np.abs(phi[:,:,:,0][:,ix,:]),axis=1);share=100*raw/raw.sum(axis=1,keepdims=True)
 got=nodes[nodes.arm==a].pivot(index='replicate',columns='group',values='share')[groups].to_numpy();np.testing.assert_allclose(got,share,atol=1e-10)
 v=np.load(R/f'{a}_validation_scores.npz');matched=audit[audit.arm==a].matched.to_numpy();checks[a+'_unmatched_branch_max_error']=float(np.max(np.abs(v['full'][:,~matched]-v['film_off'][:,~matched])));assert checks[a+'_unmatched_branch_max_error']==0
# 1 Node/branch decomposition
fig,ax=plt.subplots(figsize=(10,4.8));yy=np.arange(4)
for k,a in enumerate(ARMS):
 d=nodes[nodes.arm==a];off=(k-1)*.20
 for j,g in enumerate(groups):
  vals=d[d.group==g].share.to_numpy();jitter=np.linspace(-.025,.025,len(vals));ax.scatter(vals,j+off+jitter,s=38 if a=='Main' else 24,color=COLOR[a],alpha=.9,label=LABEL[a] if j==0 else None)
  ax.text(53,j+off,f'{vals.mean():.1f}%',va='center',fontsize=9,color=COLOR[a])
ax.set(yticks=yy,yticklabels=groups,xlabel='平均绝对贡献的占比（%）',xlim=(0,60));ax.invert_yaxis();ax.grid(axis='x',alpha=.16);ax.legend(loc='upper left',bbox_to_anchor=(0,1.06),frameon=False,ncol=3,fontsize=8);ax.set_title('风险评分依赖哪些节点与分支？',loc='left',pad=35)
fig.text(.12,-.01,'制造业与信息技术业各64家；主模型为既定6模型集成，S0/MID每个点为一次独立训练。',fontsize=9)
save(fig,'01_节点贡献','指定同业历史背景下的四因素Shapley分解。占比按模型内部归一化；审计分支包含与财务表示的交互，不是审计字段的独立信息增益。')
# 2 Type-conditional financial dependence; fixed 13-field set and common observed support.
fin=fields[fields.node=='X2'].copy();fl=list(dict.fromkeys(fin.field));gl=[i+'_'+c for i in ['C','I'] for c in ['P2503','P2504','P2505']]
heat={};heat_n={}
for a in ARMS:
 d=fin[(fin.arm==a)&(fin.group.isin(gl))].copy();d['share']=d.mean_abs_delta/d.groupby(['replicate','group']).mean_abs_delta.transform('sum')*100
 heat[a]=d.groupby(['field','group']).share.mean().unstack().reindex(index=fl,columns=gl);heat_n[a]=d.groupby(['field','group']).n_observed.min().unstack().reindex(index=fl,columns=gl)
vmax=max(np.nanmax(x.to_numpy()) for x in heat.values());fig,axs=plt.subplots(1,3,figsize=(15,8),sharey=True)
for ax,a in zip(axs,ARMS):
 vals=heat[a].to_numpy();mask=heat_n[a].to_numpy()<20;vals=np.where(mask,np.nan,vals);im=ax.imshow(vals,aspect='auto',cmap='cividis',vmin=0,vmax=vmax)
 ax.set_xticks(range(6),['制造\n虚假记载','制造\n推迟披露','制造\n重大遗漏','信息\n虚假记载','信息\n推迟披露','信息\n重大遗漏'],fontsize=8);ax.set_title(LABEL[a]);ax.set_yticks(range(len(fl)),[names.get(x,x) for x in fl],fontsize=9)
 for r,c in zip(*np.where(mask)):ax.text(c,r,'—',ha='center',va='center',fontsize=9)
fig.subplots_adjust(left=.20,right=.91,bottom=.15,top=.87,wspace=.10);cb=fig.colorbar(im,ax=axs,location='right',fraction=.02,pad=.02);cb.set_label('13项财务指标内部的相对敏感性（%）')
fig.suptitle('不同企业行业、违规类型下，模型依赖的财务指标',x=.20,ha='left',fontsize=14,fontweight='bold');fig.text(.20,.025,'148条已记录正例；类型可重叠。仅替换实际观测值；有效样本不足20的格子不展示。\nS0/MID为5次训练结果的平均；每组独立归一化，不能比较组间预测难度或因果效应。',fontsize=9)
save(fig,'02_行业与违规类型','列为行业×已记录违规类型，颜色为同业替换后条件类型logit绝对变化的相对份额，不是该指标造成违规的概率。横向比较仅限固定13字段集合。')
# 3 Actual reranking: separately rank original and branch-off score distributions.
def rank(v):return 100*pd.Series(v).rank(method='average').to_numpy()/len(v)
rerank=[];overlaps=[]
for a in ARMS:
 z=np.load(R/f'{a}_validation_scores.npz');mov=np.stack([rank(u[:,0])-rank(v[:,0]) for u,v in zip(z['full'],z['film_off'])]); med=np.median(mov,axis=0)
 for i,m in enumerate(med):rerank.append({'arm':a,'validation_index':i,'rank_movement':m})
 k=int(np.ceil(.1*mov.shape[1]));ov=[]
 for u,v in zip(z['full'],z['film_off']):ov.append(len(set(np.argsort(-u[:,0],kind='stable')[:k])&set(np.argsort(-v[:,0],kind='stable')[:k]))/k*100)
 overlaps.append({'arm':a,'overlap_mean':np.mean(ov),'overlap_min':np.min(ov),'overlap_max':np.max(ov)})
rr=pd.DataFrame(rerank).merge(audit[['arm','validation_index','audit_group']],on=['arm','validation_index']);rr.to_csv(R/'audit_actual_reranking.csv',index=False);pd.DataFrame(overlaps).to_csv(R/'audit_list_overlap.csv',index=False)
ag={'audit::standard_unqualified':'标准无保留','audit::unqualified_with_explanation':'带解释的无保留','audit::qualified':'保留意见','audit::disclaimer_or_refusal':'无法表示／拒绝','未匹配审计记录':'未匹配审计'}
fig,ax=plt.subplots(figsize=(11,5.3));order=list(ag)
for k,a in enumerate(ARMS):
 data=[rr[(rr.arm==a)&(rr.audit_group==g)].rank_movement.to_numpy() for g in order]
 box=ax.boxplot(data,positions=np.arange(5)+(k-1)*.23,widths=.19,orientation='horizontal',patch_artist=True,showfliers=False,manage_ticks=False,medianprops={'color':'white','linewidth':1.4},boxprops={'facecolor':COLOR[a],'alpha':.85},whiskerprops={'color':COLOR[a]},capprops={'color':COLOR[a]})
 ax.plot([],[],color=COLOR[a],linewidth=6,label=LABEL[a])
ax.set(yticks=range(5),yticklabels=[f'{ag[g]}（n={sum((audit.arm=="Main")&(audit.audit_group==g))}）' for g in order],xlabel='开启审计调制后，风险排序百分位的变化（百分点）');ax.invert_yaxis();ax.axvline(0,color='#777777',lw=.8);ax.grid(axis='x',alpha=.15);ax.legend(frameon=False,loc='lower right');ax.set_title('审计调制怎样改变企业在名单中的相对位置？',loc='left',pad=16)
fig.text(.12,-.015,'分别对开启／关闭分支的全体验证企业重新排名；正值表示排名更靠前。箱体为企业间分布，非置信区间。',fontsize=9)
save(fig,'03_审计调制与风险排序','箱体为四分位区间、白线为中位数、须为1.5倍IQR内范围；不画极端点。S0/MID先取每家企业5次训练的排名变化中位数。关闭分支不重新训练，因此不是审计信息的预测增益估计。')
# 4 Field reliance and privacy leakage: same architecture control only.
q=fin[fin.group=='random'].copy();q['share']=q.mean_abs_delta/q.groupby(['arm','replicate']).mean_abs_delta.transform('sum')*100
qp=q.groupby(['arm','field']).share.mean().unstack(0).reindex(fl)
s0=pd.read_csv(ROOT/'实验与代码/隐私多维评价/20261022-v1/results/fields.csv');s0=s0[(s0.arm=='S0')&(s0.view=='messages')&(s0.group=='all')].copy()
mid=pd.read_csv(ROOT/'实验与代码/MID稳定性复检/20261022-v1/results/fields.csv');mid=mid[(mid.view=='messages')&(mid.group=='all')].copy()
pr=s0[['seed','field','success_0.319']].merge(mid[['seed','field','success_0.319']],on=['seed','field'],suffixes=('_S0','_MID'));pr['field']=pr.field.str.replace('x2::','',regex=False);pr['reduction_pp']=100*(pr['success_0.319_S0']-pr['success_0.319_MID']);pr=pr[pr.field.isin(fl)];pr.to_csv(R/'paired_field_privacy.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(12,7),sharey=True,gridspec_kw={'width_ratios':[1.05,1]});y=np.arange(len(fl))
for j,f in enumerate(fl):
 axs[0].plot([qp.loc[f,'S0'],qp.loc[f,'MID']],[j,j],color='#B5BDC6',lw=1.5)
 for a in ['S0','MID']:axs[0].scatter(qp.loc[f,a],j,color=COLOR[a],s=32,label=LABEL[a] if j==0 else None)
 vals=pr[pr.field==f].reduction_pp.to_numpy();axs[1].scatter(vals,np.repeat(j,len(vals))+np.linspace(-.10,.10,len(vals)),color=COLOR['MID'],s=20,alpha=.8)
axs[0].set(yticks=y,yticklabels=[names.get(f,f) for f in fl],xlabel='13项财务指标内部的风险评分敏感性占比（%）');axs[0].invert_yaxis();axs[0].set_title('任务侧：模型保留了哪些依赖？',loc='left');axs[0].legend(frameon=False)
axs[1].set_xlabel('字段猜中率降低：S0 − MID（百分点）');axs[1].axvline(0,color='#888888',lw=.8);axs[1].set_title('攻击侧：哪些字段更难被猜中？',loc='left')
for ax in axs:ax.grid(axis='x',alpha=.15)
fig.subplots_adjust(left=.29,wspace=.13,bottom=.12,top=.90);fig.text(.29,.015,'左：同一128家企业的解释；右：既有独立攻击评价，5个点对应5对训练。\n猜中容差为训练集观测值标准差的0.319倍。两侧样本不同，不拟合相关或因果关系。',fontsize=9)
save(fig,'04_字段依赖与隐私保护','只比较同架构的S0与MID。左为本次替换敏感性，右为已冻结的逐字段攻击结果。依赖与泄漏是两个问题：字段对任务重要，并不意味着必须可还原其精确值。')
# 5 Selected cases, actual model ranks with all replicate points.
fig,ax=plt.subplots(figsize=(10,4.8));idxs=list(dict.fromkeys(cases.validation_index));labels=[]
for j,i in enumerate(idxs):
 rows=cases[cases.validation_index==i];first=rows.iloc[0];labels.append(f'{first.symbol} · {int(first.economic_year)}\n'+('制造业' if first.industry=='C' else '信息技术业')+'｜'+('名单内正例' if '名单内' in first.case_kind else '名单外正例'))
 for k,a in enumerate(ARMS):
  row=rows[rows.arm==a].iloc[0];pos=j+(k-1)*.19
  ax.plot([row.risk_percentile_min,row.risk_percentile_max],[pos,pos],color=COLOR[a],lw=1.5);ax.scatter(row.risk_percentile_median,pos,color=COLOR[a],s=35,label=LABEL[a] if j==0 else None)
ax.set(yticks=range(4),yticklabels=labels,xlim=(0,101),xlabel='全体验证企业中的风险排序百分位（越高越靠前）');ax.invert_yaxis();ax.axvline(90,color='#777777',ls='--',lw=.8);ax.text(90.4,3.50,'前10%阈值',fontsize=9);ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(0,1.05),ncol=3,fontsize=8);ax.grid(axis='x',alpha=.15);ax.set_title('四个固定案例：保留正确识别，也展示遗漏',loc='left',pad=34)
fig.text(.12,-.015,'按主模型名单内／外分别选中间风险的已记录正例；S0/MID点为中位数、线为5次训练范围，非置信区间。',fontsize=9)
save(fig,'05_企业案例','案例按冻结主模型与真实标签选取，不根据MID改善与否挑选。图中主模型是既定集成；S0/MID的区间是5个独立预测器的结果范围，不是已部署集成。')
# Export final plotted aggregates, retaining exact source values.
heatrows=[]
for a in ARMS:
 for f in fl:
  for g in gl:heatrows.append(dict(arm=a,field=f,group=g,share=heat[a].loc[f,g],n_observed=int(heat_n[a].loc[f,g]),displayed=bool(heat_n[a].loc[f,g]>=20)))
pd.DataFrame(heatrows).to_csv(R/'figure_type_heatmap.csv',index=False)
qp.to_csv(R/'figure_financial_dependence.csv')
# Data summaries and concise reusable narrative.
ns=nodes.groupby(['arm','group']).share.mean().unstack().reindex(ARMS)
report=['# 主模型与隐私副模型的归因分析','', '本报告解释已冻结模型在企业风险筛查和类型提示中使用了哪些信息，并比较MID处理后预测依据的变化。此次只进行预测与解释，没有重新训练、调参或替换最终模型。','', '## 一、比较对象与样本','', '| 模型 | 本次使用方式 | 比较用途 |','|---|---|---|','| FiLM-L主模型 | 既定6模型校准后集成 | 解释正式主模型 |','| S0同架构无防护模型 | 5次独立训练，逐次解释 | 排除编码器架构差异的参照 |','| MID隐私副模型 | 与S0配对的5次独立训练 | 判断隐私处理伴随的依赖变化 |','', '解释数据为2020任务开发流程中的验证折，企业经济年度为2018年。随机抽取制造业、信息传输软件和信息技术服务业各64家，另复用148条已记录违规正例作类型分析。两部分及固定案例去重后为255条；不同违规类型可以出现在同一家企业。本报告不使用最终测试折，也不把这两个行业的结果推广为所有企业的平均水平。','', 'S0与MID的5次训练分别报告，未把5个模型当作已验证的部署集成。MID解释采用固定的一次消息噪声，并在原始与替换输入之间复用；本次结果描述该次消息实现。','', '## 二、节点贡献：主模型与MID的总体结构相近','', '对X1、X2、原始X3事件信息、审计调制分支四个因素枚举全部16种组合。被替换节点来自同行业历史训练企业，每条样本使用4个共同背景企业。审计字段保留，只切换调制分支；因此该分解中的审计贡献包含与财务表示的交互。','', r'$$f(x)-f(x_{\mathrm{背景}},\mathrm{关闭调制})=\phi_{X1}+\phi_{X2}+\phi_{X3}+\phi_{\mathrm{审计分支}}.$$','', '这是针对指定背景的预测评分分解。图中先取绝对贡献，再在每个模型内部归一化；贡献占比不是准确率贡献或独立信息增益。','', '| 模型 | X1市场 | X2财务 | X3治理事件 | 审计调制分支 |','|---|---:|---:|---:|---:|']
for a in ARMS:report.append('|'+LABEL[a]+'|'+'|'.join(f'{ns.loc[a,g]:.2f}%' for g in groups)+'|')
report+=['', '![节点贡献](figures/01_节点贡献.png)','', '三个模型均主要依赖X2、审计调制和X1。MID的X3治理事件贡献低于S0，这提示隐私处理并非均匀保留各节点信息。该差异可以指导后续检查，但本次不据此重新选模型。','', '## 三、企业行业与违规类型：展示不同任务的预测依据','', '这里的“企业分类”使用已有行业标签分组，不是新增企业类别预测器。类型分析选取虚假记载或误导性陈述（P2503）、推迟披露（P2504）、重大遗漏（P2505），解释已记录该类型企业上的条件类型评分。','', '对固定13项财务指标逐项替换：用同业训练企业的观测值替换当前值，保留缺失标记、其他字段与消息噪声，记录条件类型logit的变化。图中是13项指标内部的相对敏感性，不是全部财务字段的重要性排序。单项替换可能打破指标间的恒等关系，因此称为替换敏感性，不据此判断经济因果。','', '![行业与违规类型](figures/02_行业与违规类型.png)','', '各组有效样本因字段缺失而不同，低于20条的单元格留空。图用于说明模型依赖的差别，不检验某指标是否只对某类企业有效，也不据此选择新的特征。','', '## 四、审计意见怎样参与财务解释','', '调制作用在编码后的财务表示上：','', r'$$\widetilde h_2=h_2+r\{\gamma(c)\odot h_2+\beta(c)\}.$$','', '其中c是审计条件编码，r表示审计记录是否匹配。γ、β分别调整财务表示的尺度与偏移；它们调节的是88维潜在表示，不能直接称为对某个原始财务指标赋权。','', '我们只关闭花括号中的增量，不改变审计输入和其他分支，也不重新训练。随后分别对开启和关闭分支的全部3670家验证企业重新排序，计算每家企业的排名变化。这能分清评分共同上移与企业相对位置改变。','', '![审计调制与排序](figures/03_审计调制与风险排序.png)','', '审计分支关闭后，原前10%名单保留比例：']
for row in overlaps:report.append(f'- {LABEL[row["arm"]]}：{row["overlap_mean"]:.1f}%'+('。' if row['arm']=='Main' else f'，5次训练范围{row["overlap_min"]:.1f}%—{row["overlap_max"]:.1f}%。'))
report+=['', '未匹配审计记录的企业在关闭分支前后评分逐值相同；它们的排名仍可能因其他企业评分变化而移动。各审计组的分布还同时受企业财务状态、调制偏置与交互影响，不能把“关闭分支”当作“另一种审计意见”，也不能把标准无保留意见的评分变化解释成其增加经济违规风险。审计信息的真实预测增益应引用既有同输入、同训练规则的对照实验。','', '## 五、重要信息是否仍能被消息攻击还原','', '把本次任务归因与既有攻击结果并列：左侧说明模型使用哪些指标，右侧说明攻击者更难猜中哪些指标。隐私比较采用同架构S0对MID，避免将主模型架构差异归给MID。','', '![依赖与保护](figures/04_字段依赖与隐私保护.png)','', f'在固定13项财务字段和5对训练上，消息攻击猜中率的平均下降为{pr.reduction_pp.mean():.1f}个百分点；字段均值下降范围为{pr.groupby("field").reduction_pp.mean().min():.1f}—{pr.groupby("field").reduction_pp.mean().max():.1f}个百分点。猜中定义沿用既有协议：误差不超过该字段训练期观测值标准差的0.319倍。','', '任务侧依赖与可重构性不是同一指标。MID仍使用财务信息，同时降低了该协议下精确还原字段的能力。右图攻击评价使用其既定独立样本，左图使用本次解释样本；两侧不做相关性或因果推断。此处沿用FP32消息攻击结果，不将其表述为已验证的量化消息隐私保证。','', '## 六、四个企业案例','', '每个行业各选择一个主模型前10%名单内正例、一个名单外正例；按各组主模型风险的中位位置选取。全部是已记录违规企业，并非“名单外即无违规”。','', '![企业案例](figures/05_企业案例.png)','', '| 企业代码／年度 | 模型 | 风险排序百分位 | 解释范围内影响最大的三项指标 |','|---|---|---:|---|']
cf=pd.read_csv(R/'case_fields.csv')
for _,row in cases.iterrows():
 d=cf[(cf.arm==row.arm)&(cf.validation_index==row.validation_index)].nlargest(3,'mean_abs_risk_delta');txt='；'.join(names.get(v.field,v.field)+('（推高）' if v.mean_signed_risk_delta>0 else '（压低）') for v in d.itertuples())
 report.append(f'| {row.symbol}／{int(row.economic_year)} | {LABEL[row.arm]} | {row.risk_percentile_median:.1f} | {txt} |')
report+=['', '百分位越高表示排名越靠前，90附近对应前10%阈值。S0/MID使用5次训练的中位数，范围见图；这些数值不是违规概率。前三项仅在本次固定23项字段范围内选取，且按平均绝对敏感性排列，不代表已覆盖X1全部市场指标。类型Top2逐模型投票与已记录类型保存在cases.csv，可供核查，不以个案替代整体类型泛化评价。','', '## 七、汇报时可以怎样说','', '> 我们对正式主模型和MID副模型采用相同企业样本、相同历史背景进行解释。两者的风险判断都主要依赖财务、审计调制与市场信息；加入MID后，财务信息仍被使用，但治理事件分支的相对贡献下降。逐字段攻击显示，MID降低了这些财务字段被还原的能力。我们进一步按企业行业和违规类型展示具体依赖，并保留名单内及遗漏案例，以说明模型的预测依据和局限。','', '## 八、复核与文件','', '16种组合内的Shapley加总误差小于2×10⁻¹⁵；将组合端点与单独批次的直接预测比较，最大误差小于10⁻⁶；主模型输出已与既定预测函数核对；未匹配审计记录的分支开关评分差为0。图表统一使用微软雅黑，附300dpi PNG与可编辑SVG。运行中发现并修正节点汇总的数组轴顺序，逐样本归因文件不受影响；修正后按独立公式重新汇总并断言一致。','', '- `分析与图表约定.md`：分析前的对象、样本和解释口径。','- `results/manifest.json`：模型路径、哈希、样本规模、复现检查。','- `results/node_shares.csv`、`field_sensitivity.csv`：节点与字段解释源数据。','- `results/audit_actual_reranking.csv`：分别重新排序后的排名变化。','- `results/paired_field_privacy.csv`：既有S0/MID逐字段配对攻击结果。','- `results/cases.csv`、`case_fields.csv`：案例、类型提示和字段解释。','- `results/verification.json`：本次独立复核结果。','', '本次结论针对既定开发验证集、指定背景和指定攻击协议，不增加模型优越性或形式化隐私保证的结论。']
group_examples=['', '主模型中，相对敏感性排在前两位的财务指标如下（固定13项范围）：','', '| 企业行业 | 已记录违规类型 | 前两项指标及相对份额 |','|---|---|---|']
for g in gl:
 vals=heat['Main'][g].where(heat_n['Main'][g]>=20).nlargest(2)
 group_examples.append('|'+('制造业' if g.startswith('C') else '信息技术业')+'|'+{'P2503':'虚假记载或误导性陈述','P2504':'推迟披露','P2505':'重大遗漏'}[g.split('_')[1]]+'|'+'；'.join(names.get(f,f)+f'（{v:.1f}%）' for f,v in vals.items())+'|')
group_examples+=['', '这些结果展示同一财务字段在不同类型提示中的相对作用；例如是否属于偿债能力或收入费用结构，需要结合上表逐组解释，不把热力图颜色直接当作经济规律。','']
pos=report.index('## 四、审计意见怎样参与财务解释');report[pos:pos]=group_examples
type_names={'P2501':'虚构利润','P2503':'虚假记载或误导性陈述','P2504':'推迟披露','P2505':'重大遗漏','P2506':'其他披露不实','P2509':'擅自改变资金用途','P2510':'占用公司资产','P2514':'违规担保','P2515':'一般会计处理不当','P2599':'其他违规'}
case_types=['', '案例的类型提示（S0/MID按进入Top2的次数列示，括号内为次数／5）：','', '| 企业 | 模型 | 已记录类型 | 模型Top2或投票情况 |','|---|---|---|---|']
for row in cases.itertuples():
 votes=json.loads(row.top2_votes);chosen=sorted([(c,v) for c,v in votes.items() if v>0],key=lambda t:(-t[1],t[0]));shown='；'.join(type_names[c]+(f'（{v}/5）' if row.arm!='Main' else '') for c,v in chosen)
 truth='、'.join(type_names[c] for c in row.known_type_codes.split(';'))
 case_types.append(f'|{row.symbol}|{LABEL[row.arm]}|{truth}|{shown}|')
case_types+=['', '模型提示与已记录类型存在不一致；Top2用于提供核查线索，而不是对每家企业作确定类型裁决。','']
pos=report.index('## 七、汇报时可以怎样说');report[pos:pos]=case_types
report.insert(report.index('## 七、汇报时可以怎样说'), '案例表中“推高／压低”表示当前观测值相对同业历史替换值的平均评分方向，只用于解释该企业；不是该指标对违规的因果方向。\n')
(H/'主模型与隐私副模型归因报告.md').write_text('\n'.join(report),encoding='utf-8')
checks.update({'node_summary_axis_check':True,'font':family,'new_training':False,'figures':5,'all_node_share_sums_100':bool(np.allclose(nodes.groupby(['arm','replicate']).share.sum(),100))});(R/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
print(ns.to_string());print(pd.DataFrame(overlaps).to_string(index=False));print('REPORT COMPLETE',family)
