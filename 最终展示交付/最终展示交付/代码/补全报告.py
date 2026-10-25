"""保留用户当前Word正文，仅在三个相应章节插入原理图和说明。"""
from pathlib import Path
import importlib.util,re
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
H=Path(__file__).resolve().parents[1];R=H.parent
SOURCE=H/'报告/上一版与来源/给老师的中期研究汇报_背景与扩展完整版.docx'
OUT=H/'报告/中期研究汇报_原理图与完整成果版.docx'

def insert_after(doc,anchor,text=None,image=None,caption=False):
    p=doc.add_paragraph();anchor._p.addnext(p._p)
    p.paragraph_format.space_after=Pt(7);p.paragraph_format.line_spacing=1.4
    if image:
        p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run().add_picture(str(image),width=Inches(6.3));p.paragraph_format.keep_with_next=True
    else:
        run=p.add_run(text);run.font.name='Microsoft YaHei';run._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Microsoft YaHei');run.font.size=Pt(9 if caption else 11);run.font.color.rgb=RGBColor.from_string('555555' if caption else '222222')
        if caption:p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    return p

def main():
    d=Document(SOURCE)
    # 用已核对的真实字节账本替换主模型图中的早期全浮点近似。
    drawings=[p for p in d.paragraphs if p._p.xpath('.//w:drawing')]
    main_image=drawings[1]
    for run in list(main_image.runs):
        if run._r.xpath('.//w:drawing'):main_image._p.remove(run._r)
    main_image.add_run().add_picture(str(H/'图表/原理图/02_FiLM-L模型结构_专业版.png'),width=Inches(6.3))
    sections=[
      ('六、通信量优化','01_通信缩减原理.png','通信缩减的编码、传输与融合端还原',[
      '这一部分改变的是消息的存储精度。训练好的编码器、财务调制和融合网络保持固定，节点先生成原来的浮点表示，再将部分表示转成INT8发送。融合端用预先约定的尺度还原为浮点数，执行原有预测。X1采用FP16，X2和X3内容及审计控制码采用INT8，审计匹配标志精确传送。',
      'INT8可理解为把连续数值放到有限的网格点上：q = clip(round(h/s), −127, 127)，解码值为ĥ = s×q；s按字段维度仅用训练数据拟合。数值会有小误差，但我们逐家比较前10%名单及名单内Top2类型集合，只有最终决策都保持的组合才作为采用方案。它不需要重新训练模型。']),
      ('七、归因分析与隐私保护副模型','02_MID隐私副模型原理.png','随机信息瓶颈与攻击评价',[
      'MID副模型在X1、X2和X3三个内容编码器之后加入随机瓶颈。原表示作为均值μ，另用线性层学习log σ²，生成z = μ + σ⊙ε，其中ε服从标准正态分布。相同企业年度的一份随机表示供六个融合头共同使用，融合端接收采样后的表示；审计控制码等附加信息也属于攻击者视图。',
      '训练同时降低风险与类型预测损失，以及强度为0.1的KL正则。KL衡量所学随机表示分布与标准正态分布的差异，按维度再按三个节点取平均。预测损失促使模型保留风险线索，瓶颈正则限制携带过多细节。是否真正减少泄漏，还要由攻击器检验：在相同辅助样本和预算下，比较无防护S0与MID恢复具体财务值、推断经济属性的能力。']),
      ('八、年度更新与版本回退','03_年度更新与回退原理.png','年度增量回放、重训与版本回退',[
      '这里采用按年度到达信息的批量增量更新。新通报先定位企业和违规发生年度，修正该行标签或补充违规类型。同一企业年度的特征未变、编码器和预处理版本未变时，可继续使用已有表示；新年度只需生成新键的表示。',
      '普通更新时固定节点与审计编码器，只学习融合端。每批混合128条新增或修正样本与128条历史回放样本，共训练60步。历史样本使用最新标签，它们提供旧任务的约束，减少模型只适应新样本而遗忘旧信号的情况。这里的各半是采样比例，具体风险与类型损失仍按已有代价敏感规则计算。',
      '候选用当前已经公开的标签检查固定历史组和最新训练年度。任一组召回下降超过1个百分点，或AP下降超过0.01，就转入全量重训评估；全量候选也未通过时保留父模型。政策事件在登记后、且政策后的有效监督足够时优先进入全量分支。保留集用于当时选择，后续年度标签只用于最终评价。'])]
    for heading,filename,cap,paras in sections:
        anchor=next(p for p in d.paragraphs if p.text==heading)
        # 在章节原来的首段之后解释，随后接原结果表，保留原稿结构。
        idx=d.paragraphs.index(anchor) if False else None
        next_el=anchor._p.getnext()
        from docx.text.paragraph import Paragraph
        anchor=Paragraph(next_el,anchor._parent)
        if heading.startswith('七、'):
            # 先讲攻击威胁，再在副模型介绍中展开MID，而非在威胁说明前插入算法。
            anchor=next(p for p in d.paragraphs if p.text.startswith('在与主任务兼容的共享编码器框架下'))
        for text in paras:anchor=insert_after(d,anchor,text=text)
        anchor=insert_after(d,anchor,image=H/'图表/原理图'/filename)
        insert_after(d,anchor,text='图 '+cap,caption=True)
    # 全文图号按出现顺序重排，避免新增图与原编号冲突。
    n=0
    for p in d.paragraphs:
        if re.match(r'^图(?:\d+| )',p.text):
            n+=1;t=re.sub(r'^图(?:\d+)?\s*',f'图{n} ',p.text);p.runs[0].text=t
            for run in p.runs[1:]:run.text=''
    # 明确主表是冻结快照回顾性评价，与更新实验的历史标签回放区别。
    anchor=next(p for p in d.paragraphs if p.text=='四、风险排序结果')
    insert_after(d,anchor,text='主表采用截至2023年12月31日的违规记录，按企业与违规发生年度配对，是固定快照下的回顾性比较。年度更新部分另外按历史通报截止日构造训练标签，两个实验分别报告。')
    for p in d.paragraphs:
        if p.text.startswith('样本单位为企业年度。评分时点设在年报'):
            p.runs[0].text=p.text.replace('使用当时能够取得的市场、财务、治理、审计和企业行业信息，预测后续被记录的违规风险。','使用年度配对的市场、财务、治理、审计和企业行业信息，评估相应发生年度的已记录违规风险。')
            for run in p.runs[1:]:run.text=''
    # 表格不拆行；保留原列宽与已获认可版式。
    from docx.oxml import OxmlElement
    for table in d.tables:
        for row in table.rows:
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
    for section in d.sections:
        p=section.footer.paragraphs[0];p.clear();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');p._p.append(field)
    for p in d.paragraphs:
        if re.match(r'^\[\d+\]',p.text):
            p.paragraph_format.space_after=Pt(2);p.paragraph_format.line_spacing=1.12
            p.paragraph_format.first_line_indent=Pt(0)
        if '数值恢复成功率下降20.5个百分点' in p.text:
            # 用未四舍五入的配对均值计算差值，避免直接相减展示小数。
            p.runs[0].text=p.text.replace('数值恢复成功率下降20.5个百分点','数值恢复成功率下降20.4个百分点')
            for run in p.runs[1:]:run.text=''
    OUT.parent.mkdir(parents=True,exist_ok=True);d.save(OUT)
    # 同步纯文本与表格，便于组员的AI逐段查阅；Word仍为正式排版稿。
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    lines=[]
    for child in d._element.body:
        if child.tag==qn('w:p'):
            p=Paragraph(child,d);t=p.text.strip()
            if t:
                prefix='# ' if p.style.name=='Title' else '## ' if p.style.name.startswith('Heading 1') else '### ' if p.style.name.startswith('Heading') else ''
                lines.extend([prefix+t,''])
        elif child.tag==qn('w:tbl'):
            tb=Table(child,d)
            for i,row in enumerate(tb.rows):
                lines.append('| '+' | '.join(c.text.replace('\n',' / ') for c in row.cells)+' |')
                if i==0:lines.append('|'+'---|'*len(row.cells))
            lines.append('')
    lines.extend(['图片文件及重画入口见交付目录的图表与结果索引。',''])
    OUT.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    print(OUT,'原理图3张；共',len(d.inline_shapes),'张图')

if __name__=='__main__':main()
