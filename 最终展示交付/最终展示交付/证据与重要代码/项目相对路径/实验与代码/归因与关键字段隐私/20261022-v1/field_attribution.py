"""Exploratory observed-value replacement analysis of the frozen FiLM-L ensemble."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'实验与代码/FiLM-L随机扰动副模型/20261021-v1'))
import noise_pilot as P

CODES=['P2501','P2503','P2504','P2505','P2506','P2509','P2510','P2514','P2515','P2599']
LABELS={'P2503':'虚假记载或误导性陈述','P2504':'推迟披露','P2505':'重大遗漏'}

def take(s,ids):
    return {'xs':[v[ids].copy() for v in s['xs']],**{k:s[k][ids].copy() for k in ['industry','q','state']}}

def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-700,700)))

@torch.no_grad()
def predict(ms,s):
    aa=[];bb=[]
    for seed,(m,e) in ms.items():
        ap=[];bp=[]
        for lo in range(0,len(s['industry']),256):
            sl=slice(lo,lo+256)
            a,b=m([torch.from_numpy(v[sl]) for v in s['xs']],torch.from_numpy(s['industry'][sl]),torch.from_numpy(s['q'][sl]),torch.from_numpy(s['state'][sl]))
            ap.append(a.numpy().astype('f8'));bp.append(b.numpy().astype('f8'))
        w=e['weights'];c=e['calibration']
        a=np.concatenate(ap)-np.log(w['overall'])
        b=np.concatenate(bp)-np.log(np.asarray(w['types'],dtype='f4'))
        aa.append(c['overall']['slope']*a+c['overall']['intercept'])
        cb=c['types_shrink'];bb.append(cb['slope']*b+cb['intercept']+np.asarray(cb.get('delta',0)))
    a=np.stack(aa);b=np.stack(bb)
    pa=sigmoid(a);pb=sigmoid(b)
    # Axis 2: overall, 10 conditional types, 10 joint scores.
    per=np.concatenate([pa[:,:,None],pb,pa[:,:,None]*pb],axis=2)
    pa=sigmoid(a.mean(0));pb=sigmoid(b.mean(0))
    return np.column_stack([pa,pb,pa[:,None]*pb]),per

def observed(s,other,names,node,j):
    name=names[node][j];flag='missing::'+name
    if flag not in names[node]:return np.ones(len(s['industry']),bool)
    k=names[node].index(flag)
    u=np.unique(np.r_[s['xs'][node][:,k],other['xs'][node][:,k]])
    if len(u)==1:
        # An all-zero/constant numeric coordinate carries no numerical distinction.
        return np.full(len(s['industry']),np.std(np.r_[s['xs'][node][:,j],other['xs'][node][:,j]])>1e-8)
    assert len(u)==2,(name,u)
    return s['xs'][node][:,k]<(u[0]+u[1])/2

def main():
    torch.set_num_threads(4)
    out=HERE/'results-with-decomposition'
    assert not (out/'manifest.json').exists(),'Completed output already exists'
    out.mkdir(exist_ok=True)
    formal=P.D.C.shared.load_loader().formal;pack=P.D.C.pack_for(2020,(0,1))
    tr,va=pack['splits'];names=pack['names'];ms=P.models(formal,pack)
    rng=np.random.default_rng(20261022);cells=[];selected=[];supports=[]
    for ind,industry in [(2,'C'),(8,'I')]:
        for code in LABELS:
            typ=CODES.index(code);eligible=np.flatnonzero((va['industry']==ind)&(va['types'][:,typ]>0))
            supports.append(dict(industry=industry,type=code,support=len(eligible)))
            if len(eligible)<30:continue
            ids=np.sort(rng.choice(eligible,min(32,len(eligible)),replace=False))
            cells.append((industry,code,ids));selected.extend(ids.tolist())
    pd.DataFrame(supports).to_csv(out/'support.csv',index=False)
    ids=np.array(sorted(set(selected)));s=take(va,ids);base,seedbase=predict(ms,s)
    # Independently replay the existing prediction and calibration implementation.
    av=[];bv=[]
    for seed,(m,e) in ms.items():
        replay={**s,'y':va['y'][ids],'types':va['types'][ids]}
        a,b=formal.logits(m,replay);w=e['weights'];c=e['calibration']
        av.append(formal.exp.prob.apply_cal(a-np.log(w['overall']),c['overall']))
        bv.append(formal.exp.prob.apply_cal(b-np.log(np.asarray(w['types'],dtype='f4')),c['types_shrink']))
    np.testing.assert_allclose(base[:,0],sigmoid(np.mean(av,0)),atol=1e-7,rtol=1e-6)
    np.testing.assert_allclose(base[:,1:11],sigmoid(np.mean(bv,0)),atol=1e-7,rtol=1e-6)
    positions={v:i for i,v in enumerate(ids)}
    dictionary=pd.read_csv(ROOT/'实验与代码/经济结构依据/字段与公式.csv').set_index('field')['name'].to_dict()
    symbols_tr=tr['keys'].Symbol.astype(str).to_numpy();symbols=va['keys'].Symbol.astype(str).to_numpy()[ids]
    records=[];keyrows=[];conditionalrows=[]
    fields=[(node,j,n) for node,ns in enumerate(names) for j,n in enumerate(ns[:16] if node==2 else ns) if 'missing::' not in n and '=' not in n]
    for f,(node,j,name) in enumerate(fields):
        valid=observed(s,tr,names,node,j);validtr=observed(tr,s,names,node,j)
        if not valid.any() or tr['xs'][node][validtr,j].std()<1e-8:continue
        rep=take(s,np.repeat(np.arange(len(ids)),8))
        for row in range(len(ids)):
            if not valid[row]:continue
            donors=np.flatnonzero(validtr&(tr['industry']==s['industry'][row])&(symbols_tr!=symbols[row]))
            if not len(donors):valid[row]=False;continue
            donors=rng.choice(donors,8,replace=len(donors)<8)
            rep['xs'][node][row*8:(row+1)*8,j]=tr['xs'][node][donors,j]
        changed,seedchanged=predict(ms,rep)
        changed=changed.reshape(len(ids),8,21);seedchanged=seedchanged.reshape(6,len(ids),8,21)
        delta=base[:,None,:]-changed
        seed_delta=seedbase-seedchanged.mean(2)
        for industry,code,ci in cells:
            k=CODES.index(code);ix=np.array([positions[v] for v in ci]);ix=ix[valid[ix]]
            if not len(ix):continue
            target=11+k;dd=delta[ix,:,target].mean(1)
            sd=seed_delta[:,ix,target].mean(1)
            backgrounds=delta[ix,:,target].reshape(len(ix),2,4).mean((0,2))
            sign=np.sign(dd.mean())
            # Exact symmetric decomposition of a product difference, not causal mediation.
            riskpart=.5*(base[ix,None,1+k]+changed[ix,:,1+k])*delta[ix,:,0]
            typepart=.5*(base[ix,None,0]+changed[ix,:,0])*delta[ix,:,1+k]
            np.testing.assert_allclose(riskpart+typepart,delta[ix,:,target],atol=1e-14,rtol=1e-10)
            conditional_seed=seed_delta[:,ix,1+k].mean(1)
            conditional_bg=delta[ix,:,1+k].reshape(len(ix),2,4).mean((0,2))
            records.append(dict(industry=industry,type=code,type_name=LABELS[code],node=f'X{node+1}',field=name,name=dictionary.get(name,{'mkt__company_age':'企业成立年限','mkt__listing_age':'上市年限'}.get(name,name)),n_observed=len(ix),mean_joint_delta=float(dd.mean()),mean_abs_joint_delta=float(np.abs(dd).mean()),conditional_delta=float(delta[ix,:,1+k].mean()),mean_abs_conditional_delta=float(np.abs(delta[ix,:,1+k].mean(1)).mean()),overall_delta=float(delta[ix,:,0].mean()),risk_component=float(riskpart.mean()),type_component=float(typepart.mean()),seed_sign_agreement=float(np.mean(np.sign(sd)==sign)),conditional_seed_sign_agreement=float(np.mean(np.sign(conditional_seed)==np.sign(delta[ix,:,1+k].mean()))),background_sign_agreement=bool(np.sign(backgrounds[0])==np.sign(backgrounds[1])),conditional_background_sign_agreement=bool(np.sign(conditional_bg[0])==np.sign(conditional_bg[1])),mean_original_score=float(base[ix,target].mean())))
        if f%20==0:print('fields',f,'/',len(fields),flush=True)
    df=pd.DataFrame(records);df.to_csv(out/'field_sensitivity.csv',index=False)
    for (industry,code),frame in df.groupby(['industry','type']):
        eligible=frame[(frame.n_observed>=20)&(frame.seed_sign_agreement>=5/6)&frame.background_sign_agreement]
        keyrows.append(eligible.nlargest(5,'mean_abs_joint_delta'))
        conditional=frame[(frame.n_observed>=20)&(frame.conditional_seed_sign_agreement>=5/6)&frame.conditional_background_sign_agreement]
        conditionalrows.append(conditional.nlargest(5,'mean_abs_conditional_delta'))
    keys=pd.concat(keyrows,ignore_index=True);keys.to_csv(out/'exploratory_key_fields.csv',index=False)
    conditionalkeys=pd.concat(conditionalrows,ignore_index=True);conditionalkeys.to_csv(out/'conditional_type_key_fields.csv',index=False)
    pd.concat([keys,conditionalkeys]).drop_duplicates(['industry','type','field']).to_csv(out/'privacy_candidate_union.csv',index=False)
    np.savez_compressed(out/'sample_predictions.npz',validation_indices=ids,predictions=base)
    manifest={'task_year':2020,'validation_economic_years':sorted(va['keys'].economic_year.unique().tolist()),'sample_rows':len(ids),'fields_considered':len(fields),'models':list(ms),'scope':'recorded positive cases within selected industry/type cells; exploratory numeric replacement; no new training','folds_loaded':[0,1],'protocol_sha256':hashlib.sha256((HERE/'分析约定.md').read_bytes()).hexdigest(),'prediction_replay_check':True,'product_decomposition_check':True,'additional_analysis':'After first exploratory pass, split joint sensitivity into overall-risk and conditional-type components; retain first pass files. Conditional candidate union is exploratory, not preregistered.'}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    lines=['# 当前模型的关键字段归因：首轮结果','',f'解释当前 FiLM-L 六种子模型，2020 开发任务的 {manifest["validation_economic_years"]} 年验证折；合并去重 {len(ids)} 个已记录违规案例。未训练模型。','', '## 行业与类型样本量','']
    for r in supports:lines.append(f'- {r["industry"]} / {LABELS[r["type"]]}：验证折该类型正例 {r["support"]} 个，抽取最多 32 个。')
    lines+=['','## 可进入隐私复核的探索候选','', '字段需有至少 20 个有效数值案例、至少 5/6 个种子的方向一致、两批背景方向一致，再按平均绝对敏感性各组取前五。方向指相对同行背景推高或压低联合类型分数，不表示指标升高的因果效应。','']
    for (industry,code),frame in keys.groupby(['industry','type']):
        lines+= [f'### {industry}：{LABELS[code]}','', '|节点|字段|有效案例|联合分数差（百分点）|平均绝对差（百分点）|种子方向一致|','|---|---|---:|---:|---:|---:|']
        for r in frame.itertuples():lines.append(f'|{r.node}|{r.name} ({r.field})|{r.n_observed}|{100*r.mean_joint_delta:+.3f}|{100*r.mean_abs_joint_delta:.3f}|{r.seed_sign_agreement:.0%}|')
        lines.append('')
    lines+=['## 总体风险与类型提示需分开解释','','联合分数 p×q 的变化，可精确分解为“平均 q × p 的变化”与“平均 p × q 的变化”。这只是乘积差的代数分解。前者表示总体风险输出变化，后者表示条件类型输出变化。已逐值校验两部分之和等于联合分数差。不能把共同的总体风险信号说成某一种违规特有的指标。','','在第一轮结果后，增补了条件类型输出的单独排名，属于探索性分析。以下每组列出平均绝对条件敏感性最高且满足同样样本及方向筛选的三个字段：','']
    for (industry,code),frame in conditionalkeys.groupby(['industry','type']):
        lines.append(f'- {industry} / {LABELS[code]}：'+ '；'.join(f'{r.name}（条件概率平均绝对变化 {100*r.mean_abs_conditional_delta:.3f} 个百分点）' for r in frame.head(3).itertuples())+'。')
    lines+=['','两种排名的并集已固定在 results-with-decomposition/privacy_candidate_union.csv，避免后续只挑保护效果好的字段。旧首轮文件保留在 results/。','','## 如何使用这份结果','','这些是已有正例中的模型数值依赖，不代表指标能区分违规与未违规，不代表经济因果关系。同行字段替换仍可能打破财务科目关系；单字段较弱也可能由相关指标替代。不能根据本表立即给训练特征加权。','','下一步固定候选字段并用同架构 S0/dCor 的强攻击逐字段比较：重构误差、相对公共信息攻击的增量、行业差异；复核候选在两个副模型中的任务依赖，防止把“模型不再使用指标”误写成无代价的保护。所有字段及业务指定字段仍需同时评价。','','本轮没有执行新的隐私攻击，没有跨年确认。验证折结果已经用于探索，后续确认须明确独立年度及数据可用性。']
    (HERE/'结果解读.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
