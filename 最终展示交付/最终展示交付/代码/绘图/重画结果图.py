"""python 代码/绘图/重画结果图.py --only all|main|classification|communication|privacy|update|attribution|attribution-detail
所有路径相对于交付目录；只读冻结CSV/JSON，不导入训练模块、不修改原实验。
"""
from pathlib import Path
import argparse,importlib.util
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
H=Path(__file__).resolve().parents[2];D=H/'数据/绘图数据';F=H/'图表/结果图';F.mkdir(parents=True,exist_ok=True)
font=H/'字体/msyh.ttc'
if not font.exists():font=Path('/Users/how/Library/Fonts/msyh.ttc')
font_manager.fontManager.addfont(str(font));plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'font.size':11,'svg.fonttype':'none','axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
COL=['#234E70','#287C7E','#C45D35','#7C8795']
def save(fig,n):
    fig.savefig(F/(n+'.png'),dpi=300,bbox_inches='tight',facecolor='white');fig.savefig(F/(n+'.svg'),bbox_inches='tight');plt.close(fig)
def main():
    spec=importlib.util.spec_from_file_location('frozen_results',H/'代码/绘图/原始主结果绘图.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.DATA=D/'主模型';mod.ASSETS=F
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'font.sans-serif':[font_manager.FontProperties(fname=str(font)).get_name()],'svg.fonttype':'none'})
    mod.main()
def classification():
    d=pd.read_csv(D/'分类/annual_summary.csv');fig,ax=plt.subplots(figsize=(9,4.5));x=np.arange(len(d))
    for i,(c,l) in enumerate([('conditional_top2_hit','模型Top2'),('common_top2_hit','训练期最常见两类')]):ax.scatter(x+(i-.5)*.12,d[c]*100,s=60,label=l,color=COL[i])
    ax.set(xticks=x,xticklabels=d.year.astype(str),ylim=(0,100),ylabel='名单命中企业的条件Top2命中率（%）',title='类型提示与常见类别参照');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
    fig.text(.1,-.03,'条件Top2不包含未进入名单的企业；长尾端到端宏覆盖另见源表。',fontsize=9);save(fig,'08_类型提示对照')
def communication():
    import json
    d=json.loads((D/'通信/strict_candidate_selection.json').read_text())['selected'];keys=list(d['node_payload_bytes_fp32']);labels=['X1 市场','X2 财务','X3 治理与审计'];y=np.arange(3)
    fig,ax=plt.subplots(figsize=(9,4.5))
    for j,k in enumerate(keys):
        a=d['node_payload_bytes_fp32'][k];b=d['node_payload_bytes_selected'][k];ax.plot([b,a],[j,j],color='#ABB5BE',lw=2);ax.scatter([a,b],[j,j],c=[COL[0],COL[2]],s=55);ax.text(a+30,j,f'{a} → {b} B',va='center')
    ax.set(yticks=y,yticklabels=labels,xlabel='六模型预测消息载荷（B/企业）',xlim=(0,2650),title='各节点通信载荷');ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    fig.text(.1,-.04,'总计3205 → 899 B/企业；三个年度前10%名单及名单内Top2集合相同。帧头与尺度另计。',fontsize=9);save(fig,'09_节点通信载荷')
def privacy():
    t=pd.read_csv(D/'隐私/task_comparison.csv');a=pd.read_csv(D/'隐私/aggregate_comparison.csv');a=a[(a['view']=='messages')&(a['group']=='all')]
    z=t.merge(a[['arm','seed','success_0.319']],on=['arm','seed']);fig,ax=plt.subplots(figsize=(9,5))
    for i,(arm,label) in enumerate([('S0','S0无防护'),('D1','dCor'),('MID','MID 0.1')]):
        q=z[z.arm==arm];assert len(q)==5,(arm,len(q));xx=(1-q['success_0.319'])*100;yy=q.recall*100
        ax.scatter(xx,yy,s=32,c=COL[i],alpha=.55);ax.scatter(xx.mean(),yy.mean(),marker='D',s=80,c=COL[i],label=label)
    ax.set(xlabel='容差内未恢复比例（%，越高越好）',ylabel='Recall@10%（%，越高越好）',xlim=(0,100),ylim=(0,40),title='同架构下的隐私—效用取舍');ax.legend(frameon=False);ax.grid(alpha=.15)
    fig.text(.1,-.04,'小点为五次独立训练，菱形为均值；容差为训练期标准差的0.319倍。不是安全概率或DP保证。',fontsize=9);save(fig,'10_隐私效用取舍')
def update():
    d=pd.read_csv(D/'年度更新/rolling_metrics.csv');d=d[d.subset=='future'];names={'STATIC':'不更新','FULL':'每期全量','INCREMENT':'增量回放','ADAPTIVE':'监测选择'}
    fig,axs=plt.subplots(1,2,figsize=(12,5));
    for i,(a,l) in enumerate(names.items()):
        q=d[d.arm==a];m=q.groupby('evaluation_year').recall10.mean();axs[0].plot(m.index,m*100,marker='o',c=COL[i],label=l)
        for year,g in q.groupby('evaluation_year'):axs[0].scatter(np.full(len(g),year)+np.linspace(-.025,.025,len(g)),g.recall10*100,s=14,c=COL[i],alpha=.45)
        if a!='STATIC':
            v=q.groupby('seed').seconds.sum();j=i-1;axs[1].scatter(v,np.full(len(v),j)+np.linspace(-.08,.08,len(v)),c=COL[i]);axs[1].text(v.max()+.2,j,f'{v.mean():.2f}秒',va='center')
    axs[0].set(xticks=[2019,2020],xticklabels=['2019','2020'],ylim=(0,40),ylabel='Recall@10%（%）',title='两期回放效果');axs[0].legend(frameon=False,fontsize=9)
    axs[1].set(yticks=[0,1,2],yticklabels=list(names.values())[1:],xlabel='两次更新累计单机计算时间（秒）',xlim=(0,22),title='三种子分别计时');axs[1].invert_yaxis();fig.subplots_adjust(wspace=.4,bottom=.2)
    fig.text(.1,.02,'策略按当时保留集决定；后续评价使用固定2023年快照。计时不是实际跨机构网络耗时。',fontsize=9);save(fig,'11_年度更新效果与成本')
def attribution():
    d=pd.read_csv(D/'归因/node_shares.csv');groups=['X1市场','X2财务','X3治理事件','审计调制分支'];fig,ax=plt.subplots(figsize=(10,5))
    for i,(arm,label) in enumerate([('Main','FiLM-L主模型'),('S0','S0同架构参照'),('MID','MID副模型')]):
        for j,g in enumerate(groups):
            v=d[(d.arm==arm)&(d.group==g)].share;ax.scatter(v,np.full(len(v),j)+(i-1)*.2+np.linspace(-.025,.025,len(v)),c=COL[i],s=30,label=label if j==0 else None)
    ax.set(yticks=range(4),yticklabels=groups,xlim=(0,60),xlabel='模型内部平均绝对贡献占比（%）',title='节点与审计分支归因');ax.invert_yaxis();ax.legend(frameon=False,fontsize=9);ax.grid(axis='x',alpha=.15)
    fig.text(.1,-.03,'指定128家企业和同业背景；主模型为六模型集成，S0/MID为五次独立训练。占比不是独立精度贡献。',fontsize=9);save(fig,'12_节点归因')
def attribution_detail():
    names=pd.read_csv(H/'数据/字段与公式.csv');names=dict(zip(names.field,names.name))
    d=pd.read_csv(D/'归因/figure_type_heatmap.csv');fields=list(dict.fromkeys(d.field));groups=list(dict.fromkeys(d.group));vmax=d.share.max()
    fig,axs=plt.subplots(1,3,figsize=(15,8),sharey=True)
    for ax,arm,label in zip(axs,['Main','S0','MID'],['FiLM-L主模型','S0同构参照','MID副模型']):
        q=d[d.arm==arm].copy();q.loc[~q.displayed,'share']=np.nan
        z=q.pivot(index='field',columns='group',values='share').reindex(index=fields,columns=groups)
        im=ax.imshow(z,aspect='auto',vmin=0,vmax=vmax,cmap='cividis');ax.set_title(label)
        ax.set_xticks(range(len(groups)),['制造\n虚假记载','制造\n推迟披露','制造\n重大遗漏','信息\n虚假记载','信息\n推迟披露','信息\n重大遗漏'],fontsize=8)
        ax.set_yticks(range(len(fields)),[names.get(x,x) for x in fields],fontsize=8)
    fig.subplots_adjust(left=.26,right=.91,bottom=.17,wspace=.12);fig.colorbar(im,ax=axs,fraction=.02,pad=.025,label='固定13项指标内部相对敏感性（%）')
    fig.text(.26,.04,'每组独立归一化；低于20条观测的单元留空。原始字段替换敏感性不代表经济因果。',fontsize=9);save(fig,'13_行业与违规类型归因')
    q=pd.read_csv(D/'归因/audit_actual_reranking.csv');groups=list(dict.fromkeys(q.audit_group));labels={'audit::standard_unqualified':'标准无保留','audit::unqualified_with_explanation':'带解释的无保留','audit::qualified':'保留意见','audit::disclaimer_or_refusal':'无法表示／拒绝','未匹配审计记录':'未匹配审计'}
    fig,ax=plt.subplots(figsize=(11,5))
    for i,(arm,label) in enumerate([('Main','FiLM-L主模型'),('S0','S0'),('MID','MID')]):
        values=[q[(q.arm==arm)&(q.audit_group==g)].rank_movement.to_numpy() for g in groups]
        ax.boxplot(values,positions=np.arange(len(groups))+(i-1)*.22,widths=.18,vert=False,showfliers=False,patch_artist=True,manage_ticks=False,boxprops={'facecolor':COL[i]},medianprops={'color':'white'});ax.plot([],[],c=COL[i],label=label,lw=5)
    ax.set(yticks=range(len(groups)),yticklabels=[labels.get(g,g) for g in groups],xlabel='开启分支后风险排序百分位变化（百分点）');ax.axvline(0,c='#777',lw=.8);ax.invert_yaxis();ax.legend(frameon=False);fig.text(.1,-.04,'箱体为企业间四分位分布，须为1.5 IQR；S0/MID先对每家企业取五次中位数。不是置信区间。',fontsize=9);save(fig,'14_审计分支排序变化')
    a=pd.read_csv(D/'归因/figure_financial_dependence.csv');p=pd.read_csv(D/'归因/paired_field_privacy.csv');fields=a.field.tolist();fig,axs=plt.subplots(1,2,figsize=(13,7),sharey=True)
    for j,f in enumerate(fields):
        row=a[a.field==f].iloc[0];axs[0].plot([row.S0,row.MID],[j,j],c='#b9b9b9');axs[0].scatter([row.S0,row.MID],[j,j],c=[COL[1],COL[2]])
        v=p[p.field==f].reduction_pp;axs[1].scatter(v,j+np.linspace(-.08,.08,len(v)),c=COL[2],s=22)
    axs[0].set(yticks=range(len(fields)),yticklabels=[names.get(f,f) for f in fields],xlabel='财务敏感性相对份额（%）',title='任务侧依赖');axs[0].invert_yaxis();axs[1].set(xlabel='S0 − MID数值猜中率（百分点）',title='攻击侧恢复率下降');axs[1].axvline(0,c='#777',lw=.8);fig.subplots_adjust(left=.28,bottom=.18,wspace=.18)
    fig.text(.28,.04,'左图绿色S0、橙色MID；右图每字段五个配对点。两侧样本不同，不拟合因果关系。',fontsize=9);save(fig,'15_字段依赖与保护')
    d=pd.read_csv(D/'归因/cases.csv',dtype={'symbol':str});ix=list(dict.fromkeys(d.validation_index));fig,ax=plt.subplots(figsize=(10,5))
    for j,k in enumerate(ix):
        for i,(arm,label) in enumerate([('Main','FiLM-L'),('S0','S0'),('MID','MID')]):
            r=d[(d.validation_index==k)&(d.arm==arm)].iloc[0];y=j+(i-1)*.2;ax.plot([r.risk_percentile_min,r.risk_percentile_max],[y,y],c=COL[i]);ax.scatter(r.risk_percentile_median,y,c=COL[i],label=label if j==0 else None)
    labels=[d[d.validation_index==k].iloc[0].symbol+' / 2018' for k in ix];ax.set(yticks=range(len(ix)),yticklabels=labels,xlim=(0,101),xlabel='企业风险排序百分位（越高越靠前）');ax.axvline(90,c='#777',ls='--');ax.legend(frameon=False);ax.invert_yaxis();fig.text(.1,-.03,'包含名单内和遗漏的已记录正例；线为五次训练的范围，不是概率区间。',fontsize=9);save(fig,'16_固定企业案例')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--only',default='all',choices=['all','main','classification','communication','privacy','update','attribution','attribution-detail']);args=parser.parse_args()
    funcs={'main':main,'classification':classification,'communication':communication,'privacy':privacy,'update':update,'attribution':attribution,'attribution-detail':attribution_detail}
    for k,f in funcs.items():
        if args.only in ['all',k]:f()
    print('输出',F)
