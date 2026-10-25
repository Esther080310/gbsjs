"""检查文件、冻结指标、图表索引；--portable-smoke另在临时独立目录重画。"""
from pathlib import Path
import argparse,csv,hashlib,json,subprocess,sys,tempfile,shutil
import numpy as np,pandas as pd
H=Path(__file__).resolve().parents[1];O=H/'检查记录';O.mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--portable-smoke',action='store_true');args=parser.parse_args()
    source=pd.read_csv(H/'文件来源与校验.csv');bad=[]
    for r in source.itertuples(index=False):
        p=H/r.交付路径
        if not p.exists() or sha(p)!=r.sha256:bad.append(r.交付路径)
    assert not bad,bad
    a=pd.read_csv(H/'数据/绘图数据/分类/annual_summary.csv');maincheck=[]
    for row in a.itertuples():
        p=H/f'数据/绘图数据/主模型/FiLM-L_{row.year}_排名.csv';d=pd.read_csv(p,dtype={'symbol':str})
        y=d.recorded_positive_by_2023.to_numpy();k=int(np.ceil(.1*len(d)));found=int(y[:k].sum())
        assert (len(d),int(y.sum()),k,found)==(row.rows,row.positives,row.budget,row.found)
        assert abs(found/y.sum()-row.recall)<1e-12
        maincheck.append({'year':row.year,'rows':len(d),'positives':int(y.sum()),'budget':k,'found':found})
    c=json.loads((H/'数据/绘图数据/通信/strict_candidate_selection.json').read_text())['selected']
    assert sum(c['node_payload_bytes_fp32'].values())==3205
    assert sum(c['node_payload_bytes_selected'].values())==899
    assert all(y['strict.all_required_pass'] for y in c['yearly'])
    t=pd.read_csv(H/'数据/绘图数据/隐私/task_comparison.csv');p=pd.read_csv(H/'数据/绘图数据/隐私/aggregate_comparison.csv');p=p[(p['view']=='messages')&(p['group']=='all')]
    priv=[]
    for arm in ['S0','D1','MID']:
        v=t[t.arm==arm];q=p[p.arm==arm];assert len(v)==len(q)==5
        priv.append({'arm':arm,'recall':float(v.recall.mean()),'guess_success':float(q['success_0.319'].mean())})
    idx=[]
    def add(path,source,script,command,scope):
        assert (H/path).is_file(),path
        idx.append({'图表':path,'源数据':source,'修改代码':script,'重画命令':command,'解释口径':scope})
    cmd='python 代码/绘图/重画结果图.py --only '
    for n in ['04_模型效果对照','05_增量与排序质量','06_前10前5前1名单密度','06_名单密度_2021','06_名单密度_2022','07_预算与命中率']:
        add('图表/结果图/'+n+'.png','数据/绘图数据/主模型/展示指标.json；各年完整排名CSV','代码/绘图/原始主结果绘图.py',cmd+'main','六模型校准logit平均；固定2023标签快照；预算ceil')
    defs=[('08_类型提示对照','分类/annual_summary.csv','classification','条件Top2；不是全体分类准确率'),('09_节点通信载荷','通信/strict_candidate_selection.json','communication','同FiLM-L六模型接口载荷；元数据另计'),('10_隐私效用取舍','隐私/task_comparison.csv；隐私/aggregate_comparison.csv','privacy','同构S0/dCor/MID；五次独立训练；FP32攻击'),('11_年度更新效果与成本','年度更新/rolling_metrics.csv','update','两期三种子；同机计算时间'),('12_节点归因','归因/node_shares.csv','attribution','指定128家；按模型内部归一化'),('13_行业与违规类型归因','归因/figure_type_heatmap.csv','attribution-detail','固定13字段；小样本留空'),('14_审计分支排序变化','归因/audit_actual_reranking.csv','attribution-detail','全体验证企业重新排序；箱体为企业间分布'),('15_字段依赖与保护','归因/figure_financial_dependence.csv；归因/paired_field_privacy.csv','attribution-detail','两侧样本不同；不拟合因果'),('16_固定企业案例','归因/cases.csv','attribution-detail','名单内外固定正例；不是违规概率')]
    for n,src,only,scope in defs:add('图表/结果图/'+n+'.png','数据/绘图数据/'+src,'代码/绘图/重画结果图.py',cmd+only,scope)
    for n,scope in [('01_通信缩减原理','按训练尺度量化并解码；不改变模型参数'),('02_MID隐私副模型原理','内容表示采样；审计等附加消息仍可见'),('03_年度更新与回退原理','增量回放加当前保留集；失败重训或保留')]:
        add('图表/原理图/'+n+'.png','冻结协议及代码；无新的模拟数值','代码/绘图/绘制三个原理图.py','python 代码/绘图/绘制三个原理图.py',scope)
    for n in ['01_研究路径图_专业版','02_FiLM-L模型结构_专业版']:
        add('图表/原理图/'+n+'.png','FilmKeepX3及实际协议账本','代码/绘图/原始主模型原理绘图.py；重画主模型原理.py','python 代码/绘图/重画主模型原理.py','原布局经包装入口更新545/3205字节账本')
    with (H/'图表与结果索引.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(idx[0]));w.writeheader();w.writerows(idx)
    prior=O/'交付核查.json'
    portable=json.loads(prior.read_text()).get('portable_plot_smoke') if prior.exists() else None
    if args.portable_smoke:
        with tempfile.TemporaryDirectory(prefix='project_plot_portable_') as tmp:
            p=Path(tmp)/'交付';p.mkdir()
            for rel in ['代码/绘图','数据/绘图数据','字体']:
                shutil.copytree(H/rel,p/rel)
            (p/'数据').mkdir(exist_ok=True);shutil.copy2(H/'数据/字段与公式.csv',p/'数据/字段与公式.csv')
            subprocess.run([sys.executable,str(p/'代码/绘图/绘制三个原理图.py')],cwd=tmp,check=True)
            subprocess.run([sys.executable,str(p/'代码/绘图/重画主模型原理.py')],cwd=tmp,check=True)
            subprocess.run([sys.executable,str(p/'代码/绘图/重画结果图.py'),'--only','all'],cwd=tmp,check=True)
            files=list((p/'图表').rglob('*.png'));assert len(files)==20,len(files)
            portable={'independent_temporary_directory':True,'without_training_tree':True,'png_count':len(files),'all_scripts_completed':True}
    result={'source_files_checked':len(source),'source_hash_mismatches':bad,'main_ranking_matches_classification':maincheck,'communication_payload_checked':[3205,899],'privacy_five_seed_means':priv,'figure_index_rows':len(idx),'portable_plot_smoke':portable,'new_training':False}
    (O/'交付核查.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    files=[]
    for p in sorted(H.rglob('*')):
        if p.is_file() and '检查记录' not in p.parts and '__pycache__' not in p.parts:
            files.append({'路径':str(p.relative_to(H)),'字节':p.stat().st_size,'sha256':sha(p)})
    with (O/'最终文件清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['路径','字节','sha256']);w.writeheader();w.writerows(files)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
