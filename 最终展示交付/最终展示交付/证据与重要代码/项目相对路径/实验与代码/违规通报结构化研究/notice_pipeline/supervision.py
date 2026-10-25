"""把已准入事实转成无实体的组合式行为说明；不推断未提及行为不存在。"""
from __future__ import annotations

import re

SUPERVISION_VERSION = 'behavior-summary-v1.1'

# 只从单项behavior匹配，不把整案证据或LLM机制枚举传播到每项事实。
# 匹配后输出固定短语，因而公司名、数值、处分和日期不会进入模型目标。
PROPOSITIONS = (
    ('fabrication', '虚构或伪造业务事项', r'虚构|伪造|捏造|虚假交易'),
    ('overstatement', '虚增收入或利润', r'虚增.*(?:收入|利润)|多计.*(?:收入|利润)'),
    ('understatement', '少计收入、成本或利润', r'少计.*(?:收入|成本|利润)|虚减'),
    ('early_recognition', '提前确认会计事项', r'提前确认'),
    ('delayed_recognition', '延后确认或入账', r'延[后迟].*(?:确认|入账)|延至.*入账'),
    ('recognition_mismatch', '未按实际业务进度核算', r'未按实际.*(?:进度|核算)|不符合.*收入确认'),
    ('impairment_failure', '计提或减值处理不当', r'(?:未|少|多|不当|不充分|不合理).*(?:计提|减值)|(?:计提|减值).*(?:不当|不充分|不足|不合理)'),
    ('account_classification', '款项性质或报表列报不当', r'性质.*有误|应在.*列报|列报.*(?:错误|不当)|会计科目.*错误'),
    ('off_book', '资金往来未记账或未入账', r'未将.*(?:记账|入账)|未(?:记账|入账)'),
    ('concealment', '隐瞒事项', r'隐瞒'),
    ('fund_occupation', '资金占用事项', r'资金占用|占用.*资金'),
    ('approval_missing', '未履行审批或审议程序', r'未履行.*(?:审议|审批)|未经.*(?:审议|审批)|未.*提交.*(?:董事会|股东大会).*审议'),
    ('approval_exceeded', '对象或金额超出批准范围', r'超出.*(?:额度|范围)|超过.*(?:额度|审批|批准|授权)|未在.*担保对象'),
    ('disclosure_late', '未及时披露事项', r'未及时.*披露|延[迟后].*披露|迟至.*披露'),
    ('disclosure_missing', '未披露或存在重大遗漏', r'未(?:予|按规定|进行|对外)?披露|重大遗漏'),
    ('disclosure_inaccurate', '披露内容与实际不符', r'披露.*(?:不符|不实|有误)|不准确|虚假记载|误导性'),
    ('governance_duty', '未按期限履行治理职责', r'未.*代行|空缺超过|未.*勤勉|未.*尽职'),
    ('securities_window', '违反证券交易期间或短线交易规定', r'敏感期|窗口期|短线交易'),
)

OBJECTS = (
    ('related_party', '关联方往来', r'关联方|关联交易|控股股东|实际控制人'),
    ('guarantee', '对外担保', r'担保|保证合同|保证人'),
    ('fundraising', '募集资金用途', r'募集资金|募投'),
    ('equity', '股权与证券交易', r'股权|股票|新股|证券'),
    ('revenue', '收入与成本确认', r'收入|成本|营业利润'),
    ('receivables', '应收款项及其列报', r'应收款|应收账款|应收票据|预付'),
    ('copyright', '版权交易', r'版权'),
    ('governance', '公司治理与任职', r'董事|监事|经理|秘书|薪酬|任职|代行'),
    ('reporting', '定期报告披露', r'年报|年度报告|季度报告|定期报告'),
)


def behavioral_target(behavior: str) -> dict:
    propositions = [(k,text) for k,text,pattern in PROPOSITIONS if re.search(pattern, behavior)]
    objects = [(k,text) for k,text,pattern in OBJECTS if re.search(pattern, behavior)]
    # 只找到事项而没有不当行为时弃权，不能以“提到担保”认定违规担保。
    if not propositions:
        return {'text': '', 'proposition_ids': [], 'object_ids': [], 'status': 'no_specific_proposition'}
    text = '行为：'+'；'.join(t for _,t in propositions)+'。'
    if objects:
        text += '涉及事项：'+'；'.join(t for _,t in objects)+'。'
    return {'text': text, 'proposition_ids': [k for k,_ in propositions],
            'object_ids': [k for k,_ in objects], 'status': 'usable'}


def merge_targets(items: list[dict]) -> dict:
    """保持同项行为的组合；重复事实与重复文书不增加重复文本的权重。"""
    sentences = sorted({item['text'] for item in items if item['text']})
    return {'text':'\n'.join(sentences), 'unique_behavior_summaries':len(sentences)}


def load_auxiliary_texts(directory, evaluation_year, protocol, fold, symbols, years):
    """返回对齐的B/C文本及mask；symbols/years应直接取正在训练的冻结缓存。"""
    import json
    from pathlib import Path
    import numpy as np
    if protocol not in {'retrospective_snapshot2023','asof_training_year_end'}:
        raise ValueError('必须明确选择已生成的时间协议')
    directory=Path(directory)
    targets=[json.loads(line) for line in (directory/'targets.jsonl').read_text().splitlines()]
    with np.load(directory/f'{evaluation_year}_{protocol}.npz',allow_pickle=False) as pack:
        if not (np.array_equal(np.asarray(symbols).astype(str),pack[f'{fold}_symbol'])
                and np.array_equal(years,pack[f'{fold}_year'])):
            raise ValueError('辅助监督与当前样本键或顺序不一致')
        mask=pack[f'{fold}_aux_mask'].copy()
        indices=pack[f'{fold}_target_index'].copy()
    if fold!=0 and mask.any():
        raise ValueError('辅助监督只能用于训练折')
    if np.any(indices[~mask]!=-1):
        raise ValueError('无监督样本必须使用-1索引')
    b,c=['']*len(mask),['']*len(mask)
    for i in np.flatnonzero(mask):
        index=int(indices[i])
        if not 0<=index<len(targets): raise ValueError('目标索引越界')
        target=targets[index]
        if not (target['target_index']==index and target['protocol']==protocol and
                target['evaluation_year']==evaluation_year and target['fold']==fold and
                target['row_index']==int(i) and target['symbol']==str(symbols[i]) and
                target['economic_year']==int(years[i])):
            raise ValueError('目标来源与样本或时间协议不一致')
        b[i],c[i]=target['B_coarse_text'],target['C_behavior_text']
    return {'B_text':b,'C_text':c,'aux_mask':mask}


def load_auxiliary_vectors(directory,evaluation_year,protocol,fold,symbols,years):
    """检查向量来源，返回小型目标库和逐样本索引，不扩成整张稠密矩阵。"""
    import json,hashlib
    from pathlib import Path
    import numpy as np
    directory=Path(directory)
    verified=load_auxiliary_texts(directory,evaluation_year,protocol,fold,symbols,years)
    manifest=json.loads((directory/'embeddings/encoder_manifest.json').read_text())
    if hashlib.sha256((directory/'targets.jsonl').read_bytes()).hexdigest()!=manifest['input_sha256']:
        raise ValueError('文本目标已改变，须使用同一冻结编码器重新编码')
    with np.load(directory/f'{evaluation_year}_{protocol}.npz',allow_pickle=False) as pack:
        indices=pack[f'{fold}_target_index'].copy()
    with np.load(directory/'embeddings/embeddings.npz',allow_pickle=False) as pack:
        b=pack['B_embeddings'].copy();c=pack['C_embeddings'].copy()
        if not np.array_equal(pack['target_index'],np.arange(len(b))):
            raise ValueError('目标向量行索引有误')
    if b.shape!=c.shape or b.shape[1]!=manifest['dimension'] or not np.isfinite(b).all() or not np.isfinite(c).all():
        raise ValueError('目标向量形状或数值错误')
    return dict(aux_mask=verified['aux_mask'],target_index=indices,B_vectors=b,C_vectors=c,
                dimension=manifest['dimension'],model_revision=manifest['revision'])


def alignment_batch(aux,indices,arm='C'):
    """未知目标填0但关闭mask；从不将-1索引为最后一条语义目标。"""
    import numpy as np
    if arm not in ('B','C'):raise ValueError('此接口只读取B/C；D须通过案件置换准入')
    indices=np.asarray(indices)
    mask=aux['aux_mask'][indices]
    vectors=np.zeros((len(indices),aux['dimension']),dtype=np.float32)
    ids=aux['target_index'][indices][mask]
    if np.any(ids<0):raise ValueError('打开掩码的目标索引不能是-1')
    vectors[mask]=aux[f'{arm}_vectors'][ids]
    return vectors,mask
