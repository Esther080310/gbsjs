"""训练期辅助对齐的小组件；原骨干、主损失和选模不在这里改写。"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def make_alignment_head(seed: int, output_dim: int = 512, input_dim: int = 32):
    """用私有CPU随机数初始化，恢复Linear构造消耗的全局CPU随机流。"""
    state=torch.random.get_rng_state()
    try:
        head=nn.Linear(input_dim,output_dim)
        generator=torch.Generator(device='cpu').manual_seed(seed)
        nn.init.kaiming_uniform_(head.weight,a=math.sqrt(5),generator=generator)
        bound=1/math.sqrt(input_dim)
        nn.init.uniform_(head.bias,-bound,bound,generator=generator)
    finally:
        torch.random.set_rng_state(state)
    return head


def alignment_loss(messages, targets, mask, head):
    """每样本L2平方距离再按可用样本平均；不额外除文本维数。"""
    representation=torch.cat(messages,dim=1)
    mask=mask.to(device=representation.device,dtype=torch.bool)
    if not mask.any():
        return representation.sum()*0.0
    pred=F.normalize(head(representation[mask]),p=2,dim=-1,eps=1e-8)
    wanted=F.normalize(targets[mask],p=2,dim=-1,eps=1e-8)
    return (pred-wanted).square().sum(dim=-1).mean()


def combined_loss(main_loss, messages, targets, mask, head, weight):
    # 关闭辅助项时不执行分支，便于逐更新复现原模型。
    if weight==0:
        return main_loss
    return main_loss+weight*alignment_loss(messages,targets,mask,head)


def clip_separately(backbone,head,max_norm=1.):
    """辅助头不参与原骨干梯度范数，保持A/B/C/D骨干裁剪规则一致。"""
    backbone_norm=nn.utils.clip_grad_norm_(backbone.parameters(),max_norm)
    head_norm=nn.utils.clip_grad_norm_(head.parameters(),max_norm) if head is not None else None
    return backbone_norm,head_norm
