"""对正式S32做单批次初始化/更新一致性及辅助梯度检查；不是效果训练。"""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
import torch
from torch import nn
P=Path(__file__).resolve().parents[1];E=P.parent
sys.path.insert(0,str(P))
from notice_pipeline.alignment import make_alignment_head,combined_loss,clip_separately
from notice_pipeline.supervision import load_auxiliary_vectors,alignment_batch
from notice_pipeline.storage import write_json,file_digest


def main():
    torch.set_num_threads(4)
    spec=importlib.util.spec_from_file_location('corrected_loader',E/'财务年度配对修正/pipeline.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    formal=loader.formal;prev=formal.prev
    cachepath=E/'财务年度配对修正/cache/2020.npz'
    protected=[cachepath,P/'artifacts/followup/supervision_targets/targets.jsonl']
    before={str(p):file_digest(p) for p in protected}
    meta=json.loads((E/'财务年度配对修正/cache/2020.json').read_text())
    with np.load(cachepath,allow_pickle=False) as z:
        splits=[dict(xs=[z[f'{f}_x{i}'].copy() for i in range(3)],
                     **{k:z[f'{f}_{k}'].copy() for k in ('industry','q','state','y','types')}) for f in range(4)]
        aux=load_auxiliary_vectors(P/'artifacts/followup/supervision_targets',2020,
            'retrospective_snapshot2023',0,z['0_symbol'],z['0_year'])
    pack=dict(meta,splits=splits)
    a=formal.RiskModel(pack,32,'Plain',42);b=formal.RiskModel(pack,32,'Plain',42)
    assert all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values()))
    rng=torch.random.get_rng_state().clone();head=make_alignment_head(1042,aux['dimension'])
    assert torch.equal(rng,torch.random.get_rng_state())
    data=prev.tensor_split(splits[0],'cpu');aw,tw=map(torch.as_tensor,prev.cost_weights(splits[0]))
    selected=np.concatenate([np.flatnonzero(aux['aux_mask'])[:4],np.flatnonzero(~aux['aux_mask'])[:12]])
    ix=torch.as_tensor(selected)
    vectors,mask=alignment_batch(aux,selected,'C')
    targets=torch.as_tensor(vectors);mask=torch.as_tensor(mask)
    def main_loss(model):
        x,y=model(*prev.batch(data,ix));truth=data['y'][ix]
        typ=nn.functional.binary_cross_entropy_with_logits(y,data['types'][ix],pos_weight=tw,reduction='none').mean(1)
        task=nn.functional.binary_cross_entropy_with_logits(x,truth,pos_weight=aw)+(typ*truth).sum()/truth.sum().clamp_min(1)
        return x,y,task+model.penalty()
    oa=torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=.0001)
    ob=torch.optim.AdamW(list(b.parameters())+list(head.parameters()),lr=.001,weight_decay=.0001)
    prev.seed_all(42);xa,ya,la=main_loss(a);la.backward();clip_separately(a,None);oa.step()
    prev.seed_all(42);xb,yb,lb=main_loss(b)
    combined_loss(lb,b.messages,targets,mask,head,0).backward();clip_separately(b,head);ob.step()
    logits_equal=torch.equal(xa,xb) and torch.equal(ya,yb)
    update_equal=all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values()))
    assert logits_equal and update_equal
    # 相同更新后的两个模型、相同Dropout流，只改变辅助损失。
    a.zero_grad();b.zero_grad();head.zero_grad()
    prev.seed_all(43);_,_,la=main_loss(a);la.backward()
    ga=[p.grad.clone() for p in a.base.enc.parameters() if p.grad is not None]
    prev.seed_all(43);_,_,lb=main_loss(b)
    combined_loss(lb,b.messages,targets,mask,head,.1).backward()
    gb=[p.grad.clone() for p in b.base.enc.parameters() if p.grad is not None]
    difference=sum(float((x-y).abs().sum()) for x,y in zip(ga,gb))
    assert difference>0 and all(p.grad is not None for p in head.parameters())
    assert before=={str(p):file_digest(p) for p in protected}
    out=dict(backbone='RiskModel(corrected cache,32,Plain,42)',batch_rows=len(selected),
        supervised_rows=int(mask.sum()),embedding_dimension=aux['dimension'],
        initial_backbones_equal=True,head_initialization_preserves_rng=True,
        lambda0_logits_equal=logits_equal,lambda0_first_update_equal=update_equal,
        auxiliary_gradient_reaches_node_encoders=True,encoder_gradient_abs_difference=difference,
        inputs_unchanged=True,formal_training_runs=0,effect_metrics_computed=False,api_calls=0,
        note='单批次工程检查；不产生召回/AP提升结果。')
    write_json(P/'artifacts/followup/supervision_setup/s32_alignment_check.json',out)
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
