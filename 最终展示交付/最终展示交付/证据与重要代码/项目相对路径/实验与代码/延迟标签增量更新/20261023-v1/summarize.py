from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
H=Path(__file__).resolve().parent;R=H/'results';G=H/'figures';G.mkdir(exist_ok=True)
assert json.loads((R/'completion.json').read_text())['status']=='complete'
audit=json.loads((R/'data_audit.json').read_text());df=pd.read_csv(R/'real_metrics.csv');sim=pd.read_csv(R/'simulation_metrics.csv');comp=json.loads((R/'real_comparisons.json').read_text());font_manager.fontManager.addfont('/Users/how/Library/Fonts/msyh.ttc');font=font_manager.FontProperties(fname='/Users/how/Library/Fonts/msyh.ttc').get_name();plt.rcParams.update({'font.family':font,'font.size':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
arms=['NO_UPDATE','RETRAIN','FINETUNE','REPLAY'];labels={'NO_UPDATE':'不更新','RETRAIN':'全量重训','FINETUNE':'仅增量微调','REPLAY':'微调＋历史回放'};colors={'NO_UPDATE':'#7A8691','RETRAIN':'#234E70','FINETUNE':'#009E73','REPLAY':'#D55E00'}
summary=df.groupby(['subset','arm']).agg(recall10=('recall10','mean'),ap=('ap','mean'),conditional_top2=('conditional_top2','mean'),seconds=('update_seconds_including_cache','mean'),exposures=('exposures','mean'),parameters=('parameters','mean'),trainable=('trainable_parameters','mean'),logical_forward=('logical_train_forward_payload_bytes','mean'),logical_backward=('logical_train_backward_payload_bytes','mean'));summary.to_csv(R/'real_summary.csv')
for subset in ['future','old']:
 print(subset,summary.loc[subset][['recall10','ap','seconds']].round(4).to_string())
ss=sim.groupby(['mechanism','subset','arm'])[['recall10','ap','seconds']].mean();ss.to_csv(R/'simulation_summary.csv')
# Chart contract: paired seed points, not population confidence intervals; no inferred significance.
fig,axs=plt.subplots(1,3,figsize=(13,4.9),gridspec_kw={'width_ratios':[1.1,1.1,1]})
for panel,subset in enumerate(['future','old']):
 ax=axs[panel]
 for seed in sorted(df.seed.unique()):
  vals=[df[(df.seed==seed)&(df.arm==a)&(df.subset==subset)].recall10.iloc[0]*100 for a in arms];ax.plot(range(4),vals,color='#C1C8CE',lw=.7,zorder=1)
 for j,a in enumerate(arms):
  vals=df[(df.arm==a)&(df.subset==subset)].recall10.to_numpy()*100;ax.scatter(j+np.linspace(-.065,.065,len(vals)),vals,color=colors[a],s=32);ax.plot([j-.13,j+.13],[vals.mean()]*2,color=colors[a],lw=2)
 ax.set_xticks(range(4),[labels[a].replace('＋','\n＋') for a in arms],rotation=18,ha='right');ax.set_ylabel('前10%名单召回率（%）');ax.set_title('后续2019年度' if subset=='future' else '旧期2017年度保持');ax.grid(axis='y',alpha=.2)
ax=axs[2]
for j,a in enumerate(arms[1:]):
 vals=df[(df.arm==a)&(df.subset=='future')].update_seconds_including_cache.to_numpy();ax.scatter(j+np.linspace(-.065,.065,3),vals,color=colors[a],s=32);ax.text(j,vals.max()*1.25,f'{vals.mean():.2f}秒',ha='center',fontsize=9)
ax.set_yscale('log');ax.set_ylabel('更新时间（秒，对数轴）');ax.set_xticks(range(3),[labels[a].replace('＋','\n＋') for a in arms[1:]],rotation=18,ha='right');ax.set_title('实测更新计算成本');ax.grid(axis='y',alpha=.2)
fig.suptitle('真实历史回放：冻结节点表示的增量更新',fontweight='bold');fig.subplots_adjust(bottom=.25,wspace=.32,top=.84);fig.text(.08,.025,'每个点为一次训练，细线连接相同种子；短横线为3次均值。更新耗时计入缓存构建，不含初始模型训练、评估及文件写入。\n现有单机CPU实测；不是多机构网络部署耗时。固定30轮全量重训，两个增量臂各60步。',fontsize=9)
fig.savefig(G/'01_真实回放.png',dpi=300,bbox_inches='tight');fig.savefig(G/'01_真实回放.svg',bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(12,4.5),sharey=True);mechanisms={'stable':'关系稳定','head_shift':'风险读出关系变化','new_coordinates':'新坐标成为风险信号'}
for ax,(m,title) in zip(axs,mechanisms.items()):
 for seed in sorted(sim.seed.unique()):
  vals=[sim[(sim.mechanism==m)&(sim.subset=='future')&(sim.seed==seed)&(sim.arm==a)].recall10.iloc[0]*100 for a in arms];ax.plot(range(4),vals,color='#C1C8CE',lw=.7)
 for j,a in enumerate(arms):
  vals=sim[(sim.mechanism==m)&(sim.subset=='future')&(sim.arm==a)].recall10.to_numpy()*100;ax.scatter(j+np.linspace(-.065,.065,3),vals,s=32,color=colors[a]);ax.plot([j-.13,j+.13],[vals.mean()]*2,color=colors[a],lw=2)
 ax.set_title(title);ax.set_xticks(range(4),[labels[a].replace('＋','\n＋') for a in arms],rotation=18,ha='right');ax.grid(axis='y',alpha=.2)
axs[0].set_ylabel('模拟后续样本Recall@10%（%）');fig.subplots_adjust(bottom=.26,top=.84,wspace=.15);fig.suptitle('机制模拟：方法是否随更新条件改变表现？',fontweight='bold');fig.text(.08,.025,'经验特征重采样，人工设定响应关系与记录过程；每个点为一个生成种子。\n每种机制仅3次模拟，不作显著性或真实机制拟合结论。',fontsize=9);fig.savefig(G/'02_机制模拟.png',dpi=300,bbox_inches='tight');fig.savefig(G/'02_机制模拟.svg',bbox_inches='tight');plt.close(fig)
# Exact independent recomputation from saved row-level logits, keyed labels and indices.
z=np.load(R/'label_ledger.npz');weights=json.loads((R/'weights.json').read_text());checks=[]
for row in df.itertuples():
 pred=np.load(R/f's{row.seed}'/f'{row.arm}_scores.npz')['logits'];idx=z['target_index'] if row.subset=='future' else z['old_eval_index'];y=z['evaluation_y'][idx];s=pred[idx,0];k=int(np.ceil(len(idx)/10));pick=sorted(range(len(idx)),key=lambda i:(-float(s[i]),i))[:k];rec=float(y[pick].sum()/max(y.sum(),1));checks.append(abs(rec-row.recall10)<1e-12)
assert all(checks)
f=summary.loc['future'];old=summary.loc['old'];gate={}
for a in ['FINETUNE','REPLAY']:
 diff=(f.loc[a,'recall10']-f.loc['RETRAIN','recall10'])*100;ret=(old.loc[a,'recall10']-old.loc['NO_UPDATE','recall10'])*100;saving=100*(1-f.loc[a,'seconds']/f.loc['RETRAIN','seconds']);gate[a]=dict(retrain_recall_difference_pp=float(diff),old_retention_difference_pp=float(ret),time_saving_percent=float(saving),three_engineering_targets_met=bool(diff>=-1 and ret>=-1 and saving>=50))
write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
write(R/'decision.json',gate);write(R/'independent_verification.json',dict(exact_saved_prediction_checks=len(checks),all_exact=True,preexisting_checkpoint_paths_not_written=True,font=font,figure_count=2,source_sha256=hashlib.sha256((H/'run_pilot.py').read_bytes()).hexdigest()))
p=json.loads((R/'simulation_parameters.json').read_text());simtable=[]
for m,title in mechanisms.items():
 for a in arms:simtable.append(f'|{title}|{labels[a]}|{ss.loc[(m,"future",a),"recall10"]*100:.2f}%|{ss.loc[(m,"old",a),"recall10"]*100:.2f}%|')
lines=['# 延迟标签增量更新：小规模验证结果','', '## 本次回答什么','', '沿用FiLM-L结构，检验新通报和新年度数据到来时，能否保留节点编码器、只更新融合端。此次是新增维护能力的小试，不替换既定主模型，不改变MID副模型，也不继承其通信或隐私通过结论。','', '先验预期是：增量更新应比不更新更好；回放应减轻旧期退化；更新代价低于全量重训。下表原样检验这三点，不根据结果追加轮次或选择种子。','', '## 一、真实标签流如何构造','', f'- 初始时点：{audit["initial_cutoff"]}；训练企业年度不晚于2017，{audit["initial_rows"]:,}条。',f'- 更新时点：{audit["update_cutoff"]}；可用训练数据{audit["update_rows"]:,}条，其中新增{audit["new_rows"]:,}条，旧行标签更新{audit["revised_old_rows"]}条（首次记录违规{audit["newly_positive_old_rows"]}条，补充类型{audit["type_only_revisions"]}条）。',f'- 后续评价：2019年度{audit["target_rows"]:,}条；旧期保持：哈希保留企业的2017年度{audit["old_eval_rows"]}条，均未进入训练。','- 所有评价标签统一为截至2023-12-31的已记录违规，逐行总体及类型标签与冻结数据包一致。不是Y60实验。','- 年报日期用于训练样本可用性过滤。其他输入沿用冻结快照，因此本次隔离标签更新机制，不重建每个历史日期的财务修订版本或审计版本。','', '三个种子42、100、17。初始模型重新按初始标签训练，不加载已经见过后续标签的正式检查点。初始和全量重训各30轮；两个微调臂各60步、同学习率及代价权重，历史回放池固定2048行。没有用评价标签选择轮次。','', '## 二、真实后续年度的结果','', '|方法|前10%召回|AP|命中正例的条件Top2|更新时间（秒）|','|---|---:|---:|---:|---:|']
for a in arms:lines.append(f'|{labels[a]}|{f.loc[a,"recall10"]*100:.2f}%|{f.loc[a,"ap"]:.4f}|{f.loc[a,"conditional_top2"]*100:.2f}%|{f.loc[a,"seconds"]:.3f}|')
lines+=['', '表为三个独立种子的平均，不是三模型部署集成。不更新的更新时间为零；初始模型构建是所有方法共同的历史成本。类型Top2按共同类型代价权重校正，未做概率校准。','', '|配对比较|平均召回变化（百分点）|正向种子|','|---|---:|---:|']
for c in comp:lines.append(f'|{labels[c["a"]]} − {labels[c["b"]]}|{c["mean_recall_difference_pp"]:+.3f}|{c["positive_seeds"]}/3|')
lines+=['', '|方法|旧期2017召回|相对不更新变化（百分点）|','|---|---:|---:|']
for a in arms:lines.append(f'|{labels[a]}|{old.loc[a,"recall10"]*100:.2f}%|{(old.loc[a,"recall10"]-old.loc["NO_UPDATE","recall10"])*100:+.3f}|')
lines+=['', '![真实回放](figures/01_真实回放.png)','', '## 三、更新成本与节点通信','', '两种微调仅训练融合端的调制、行业嵌入、投影/门控、trunk和输出头；节点编码器与审计条件编码器哈希保持。缓存前向在相同批量下逐值复现原始前向。','', '|方法|可训练参数|更新样本曝光次数|相对重训耗时下降|','|---|---:|---:|---:|']
for a in arms[1:]:lines.append(f'|{labels[a]}|{int(f.loc[a,"trainable"]):,}|{int(f.loc[a,"exposures"]):,}|{100*(1-f.loc[a,"seconds"]/f.loc["RETRAIN","seconds"]):.1f}%|')
newbytes=audit['new_rows']*545
lines+=['', f'单次模型FP32逻辑接口为545字节/企业（128个表示与完整度浮点、8个审计条件浮点、1字节匹配标志）。有历史缓存时，{audit["revised_old_rows"]}条旧标签修正不需要重新上传旧表示；{audit["new_rows"]:,}条新增样本需上传约{newbytes/1e6:.3f} MB载荷。三个种子是三次独立试验，不将其合并后仍引用单模型账本。', '', '融合端训练60步不向节点传回梯度。全量重训的前向/反向张量逻辑字节见real_metrics.csv，来自完整表示和梯度计数；没有真实网络发送，不能称多机构网络测速，也不含加密、帧头及模型版本管理。缓存构建时间保守包含全部回放输入的编码，未只挑较小的新样本计时；评估与文件写入不计入训练耗时。', '', '使用缓存要获得保存节点表示的授权；表示仍有既有泄漏风险。节点编码器变化后旧缓存必须重建；本次不更新编码器。', '', '## 四、预定工程目标是否满足','', '|增量方法|对全量重训召回差|旧期对不更新召回差|计入缓存后时间下降|三项是否同时满足|','|---|---:|---:|---:|---|']
for a,v in gate.items():lines.append(f'|{labels[a]}|{v["retrain_recall_difference_pp"]:+.3f}pp|{v["old_retention_difference_pp"]:+.3f}pp|{v["time_saving_percent"]:.1f}%|'+('是' if v['three_engineering_targets_met'] else '否')+'|')
lines+=['', '参考带为：对重训平均召回差不低于−1pp、旧期不低于−1pp、更新时间至少减少50%。这是小规模工程筛选，不是统计非劣性或跨年度稳定性证明。全量重训只采用预定30轮，尚不是穷尽调参后的上限。','', '## 五、机制模拟','', f'从真实训练行联合重采样每节点8个固定数值坐标，保留经验相关结构，建立每节点8→2的轻量编码器。初始3000、新增1000、后续评价2000、旧机制评价1000条。有限快照正例率{p["finite_snapshot_positive_rate"]:.3f}，初始旧行记录覆盖{p["old_initial_recording_probability"]:.3f}、更新旧行覆盖{p["old_updated_recording_probability"]:.3f}、新增队列覆盖{p["new_recording_probability"]:.3f}，来自真实回放。将有限快照正例视为模拟真值仅用于构造可知真值的实验。', '', '三种响应机制预先固定：关系稳定、同坐标风险读出关系改变、原来未承担主要信号的新坐标成为风险信号。它们是受控假设，不声称已识别真实企业的生成机制。','', '|机制|方法|后续召回|旧机制召回|','|---|---|---:|---:|',*simtable,'', '![模拟](figures/02_机制模拟.png)','', '每个机制仅三个生成种子。模拟包含编码器可能失配的条件，不以构造对增量方法有利的结果作为采用依据。','', '## 六、选型判断','']
r0=f.loc['REPLAY','recall10']-f.loc['NO_UPDATE','recall10'];r1=f.loc['REPLAY','recall10']-f.loc['FINETUNE','recall10']
lines.append(('历史回放在后续年度平均召回高于不更新。' if r0>0 else '历史回放在后续年度没有提高不更新模型的平均召回。')+(' 相对普通微调，其平均召回也更高。' if r1>0 else ' 相对普通微调，回放未显示平均召回优势。'))
if gate['REPLAY']['three_engineering_targets_met']:lines.append('本轮回放方案通过三项工程参考目标，可保留为低成本更新候选；不据此替换正式主模型，亦不称新算法已跨年度验证。')
else:lines.append('本轮回放方案未同时满足三项工程参考目标，不纳入正式主模型；缓存表示更新能力作为已实现的工程原型保留，不继续扩大训练或调参救援。')
lines+=['', '这轮最应区分两件事：融合端更新确实减少计算/历史消息需求，是接口与实现事实；是否能持续保持排序，是需要结果支持的经验判断。“新增通报可利用”与“回放有独立增益”也分别看对照。冻结表示无法学习旧编码器未保留的新信号，其适用条件会受到限制。','', '## 七、核查和复现','', '- rows_and_labels.csv、label_ledger.npz：企业年度键、时间标签、训练/评价索引。','- real_metrics.csv、real_comparisons.json：逐种子真实结果与成对变化。','- simulation_parameters.json、simulation_metrics.csv：模拟参数与全部结果。','- 每个s*/目录：初始及各臂模型、逐行预测、缓存和训练记录。','- independent_verification.json：用保存预测和独立排序实现复算24个指标组，全部逐值一致。','- 操作记录.md：首轮批量形状数值核对修正。','', '正式FiLM-L、MID及旧通信检查点均未覆盖。结果只写入本次新目录。']
extra=['', '### 三个种子的后续年度召回','', '|种子|不更新|全量重训|普通微调|回放微调|','|---|---:|---:|---:|---:|']
for seed in sorted(df.seed.unique()):
 vals=[df[(df.seed==seed)&(df.arm==a)&(df.subset=='future')].recall10.iloc[0]*100 for a in arms]
 extra.append('|'+str(seed)+'|'+'|'.join(f'{v:.2f}%' for v in vals)+'|')
extra+=['', '回放相对不更新与普通微调的召回改善均为2/3种子，不能据平均点差称为稳定显著提升。旧期条件Top2由不更新的'+f'{old.loc["NO_UPDATE","conditional_top2"]*100:.2f}%降至回放的{old.loc["REPLAY","conditional_top2"]*100:.2f}%'+ '；两种方法命中的正例集合不同，因此此项同时受名单构成变化影响。当前保持目标主要约束总体筛查，未证实所有类型提示均保持。','']
pos=lines.index('## 三、更新成本与节点通信');lines[pos:pos]=extra
extra2=['', '模拟的具体判读：稳定机制中，回放后续召回为40.45%，全量重训41.11%；风险读出关系变化时，回放16.22%、普通微调17.87%、全量重训20.39%。这说明保留旧规律与适应新规律之间存在取舍，回放不应被叙述为普遍改善。新坐标机制中所有方法表现都较弱，全量重训仅13.40%；该构造没有给出强烈、专属于冻结编码器的失效证据，不能将其当作已证明的表示瓶颈。','']
pos=lines.index('## 六、选型判断');lines[pos:pos]=extra2
lines+=['', '成本下降同时来自冻结节点编码器与减少更新训练曝光量（15360对900810），不单独归因给历史回放算法。初始模型训练成本由各方案共同承担，长期多次更新的累计收益需另作评价。本次仅验证一个真实更新周期。']
(H/'结果与采用判断.md').write_text('\n'.join(lines));(H/'README.md').write_text('# 延迟标签增量更新小试\n\n入口：[结果与采用判断](结果与采用判断.md)。\n\n训练协议见训练前约定.md；模型、缓存、标签流和全部逐种子结果在results/；图在figures/。此轮不替换正式模型。\n')
print('GATE',json.dumps(gate));print('REPORT COMPLETE')
