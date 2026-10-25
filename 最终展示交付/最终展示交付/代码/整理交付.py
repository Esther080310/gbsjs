"""组装冻结展示资料。只复制指定目录的结果和代码；不训练、不删除原件。"""
from pathlib import Path
import csv,hashlib,json,shutil

H=Path(__file__).resolve().parents[1];R=H.parent;E=R/'实验与代码';S=H/'证据与重要代码/项目相对路径'
registry=[]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def copy(src,dst,role):
    if not src.is_file():raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists() or sha(src)!=sha(dst):shutil.copy2(src,dst)
    registry.append({'用途':role,'源路径':str(src.relative_to(R)) if src.is_relative_to(R) else str(src),'交付路径':str(dst.relative_to(H)),'字节':dst.stat().st_size,'sha256':sha(dst)})
def tree(src,dst,role,predicate=lambda p:True):
    if not src.exists():raise FileNotFoundError(src)
    for p in sorted(src.rglob('*')):
        if p.is_file() and not any(x in p.parts for x in ['__pycache__','.git','.venv-literature']) and not p.name.startswith(('~$','.env')) and predicate(p):copy(p,dst/p.relative_to(src),role)
def small(p):return p.suffix.lower() in ['.py','.md','.json','.csv','.toml','.txt','.sh','.yaml','.yml']
def snapshot(rel,predicate=small):
    p=R/rel;tree(p,S/rel,'核心代码和结果证据',predicate)

def main():
    # 每种图的数据与训练目录解耦，使重画入口无需torch或历史绝对路径。
    groups={
        '主模型':R/'汇报材料/最终展示-20261018/data',
        '分类':E/'FiLM-L分类闭环/20261020-v1',
        '通信':E/'FiLM-L通信优化/20261020-v1/results',
        '隐私':E/'MID稳定性复检/20261022-v1/results',
        '归因':E/'主模型与隐私副模型归因展示/20261023-v1/results',
        '年度更新':E/'年度更新机制收束/20261023-v1/results'}
    for label,p in groups.items():
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix in ['.csv','.json']:copy(f,H/'数据/绘图数据'/label/f.name,'独立重画的数据')
    copy(R/'汇报材料/最终展示-20261018/scripts/draw_results.py',H/'代码/绘图/原始主结果绘图.py','主模型图原始计算代码')
    copy(R/'汇报材料/中期汇报20261019/scripts/draw_professional_diagrams.py',H/'代码/绘图/原始主模型原理绘图.py','主模型原理图的布局源码；用配套包装入口更新账本')
    for f in ['msyh.ttc','msyhbd.ttc']:
        copy(Path('/Users/how/Library/Fonts')/f,H/'字体'/f,'本地内部使用字体；外发前按授权处理')
    # 原始公开数据与最终使用的对齐缓存。
    tree(E/'大创数据集',H/'数据/原始与对齐/原始X1_X2_X3_Y','原始数据')
    for rel in ['data_v16_canonical','财务年度配对修正/cache']:
        tree(E/rel,H/'数据/原始与对齐'/rel,'规范数据与实际训练缓存')
    P='实验与代码/违规通报结构化研究'
    for rel in ['artifacts/experiments/20261016-evidence-v1/audit_information/features',
                'artifacts/experiments/20261016-audit-year2022-v1/cache',
                'artifacts/experiments/20261016-audit-year2022-v1/features']:
        tree(R/P/rel,H/'数据/原始与对齐/审计与2022'/Path(rel).parent.name/Path(rel).name,'审计对齐与2022实际输入')
    for rel in ['artifacts/followup/csmar/processed/20261016-v1/normalized/FIN_Audit.csv.gz',
                'artifacts/followup/csmar/processed/20261016-v1/aligned/FIN_Audit_annual_candidates.csv.gz',
                'artifacts/followup/csmar/dictionaries/FIN_Audit_字段说明.txt']:
        copy(R/P/rel,H/'数据/原始与对齐/审计来源'/Path(rel).name,'审计来源')
    copy(E/'经济结构依据/字段与公式.csv',H/'数据/字段与公式.csv','财务字段名称与公式')
    code_dirs=['控制消息架构/20261018-v1','FiLM-L分类闭环/20261020-v1','FiLM-L通信优化/20261020-v1',
       'MID稳定性复检/20261022-v1','dCor稳定性复检/20261021-v1','隐私多维评价/20261022-v1',
       '隐私效用统一比较/20261021-v1','FiLM-L隐私防护/20261021-v2','FiLM-L隐私审计/20261020-v1',
       '归因与关键字段隐私/20261022-v1','关键字段隐私检验/20261022-v1',
       '隐私副模型收束与通信兼容/20261023-v1','主模型与隐私副模型归因展示/20261023-v1',
       '延迟标签增量更新/20261023-v1','年度更新机制收束/20261023-v1',
       '违规通报结构化研究/scripts','违规通报结构化研究/notice_pipeline','违规通报结构化研究/configs',
       '财务年度配对修正','经济结构依据','最终版本冻结']
    for rel in code_dirs:
        if (E/rel).exists():snapshot('实验与代码/'+rel)
    # 重要权重、校准预测与逐行归因数组；不复制大规模失败候选的消息缓存。
    modeldirs=[E/'控制消息架构/20261018-v1/artifacts',E/'MID稳定性复检/20261022-v1/results/models',
               E/'dCor稳定性复检/20261021-v1/results',E/'年度更新机制收束/20261023-v1/results']
    for p in modeldirs:
        tree(p,S/p.relative_to(R),'最终或同构对照权重及预测',lambda f:f.suffix in ['.pt','.npz'] and (p.name!='artifacts' or 'FiLM-L_' in str(f)))
    p=E/'主模型与隐私副模型归因展示/20261023-v1/results'
    tree(p,S/p.relative_to(R),'逐行归因与缓存分数',lambda f:f.suffix=='.npz')
    # 展示已有图与说明，不把旧演示PPT误当成已同步新内容的最终PPT。
    for rel,label in [('汇报材料/中期汇报20261019/assets','主模型既有图'),
                      ('实验与代码/主模型与隐私副模型归因展示/20261023-v1/figures','归因既有图'),
                      ('实验与代码/年度更新机制收束/20261023-v1/figures','年度更新既有图')]:
        tree(R/rel,H/'图表/已验证原图'/label,'已验证原图',lambda f:f.suffix in ['.png','.svg','.txt'])
    for p in [R/'汇报材料/项目最终成果与收束索引-20261023.md',R/'项目主线.md']:
        copy(p,H/'报告/证据说明'/p.name,'冻结结论入口')
    for name in ['官方文件和国家新闻(1).docx','核查流程叙事_背景切入版.md','金融领域隐私保护研究综述(1).docx']:
        p=Path('/Users/how/Desktop')/name
        if p.exists():copy(p,H/'报告/背景资料'/name,'组员收集的背景资料；按原稿保存')
    for p in [R/'汇报材料/中期汇报20261023/给老师的中期研究汇报_背景与扩展完整版.docx',R/'汇报材料/中期汇报20261023/build_revised_report.py']:
        copy(p,H/'报告/上一版与来源'/p.name,'上一版保留供比较')
    # 全量原数据目录索引只记录，不将7GB补数中未进入最终模型的表复制入展示。
    extra=E/'违规通报结构化研究/artifacts/followup/csmar'
    entries=[{'项目相对路径':str(p.relative_to(R)),'字节':p.stat().st_size,'是否复制':p.name.startswith('FIN_Audit')} for p in extra.rglob('*') if p.is_file() and p.suffix in ['.gz','.zip','.xlsx']]
    (H/'数据/补充数据原位置索引.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2))
    with (H/'文件来源与校验.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['用途','源路径','交付路径','字节','sha256']);w.writeheader();w.writerows(registry)
    print('整理完成',len(registry),'份文件',round(sum(x['字节'] for x in registry)/1024**2,1),'MiB')

if __name__=='__main__':main()
