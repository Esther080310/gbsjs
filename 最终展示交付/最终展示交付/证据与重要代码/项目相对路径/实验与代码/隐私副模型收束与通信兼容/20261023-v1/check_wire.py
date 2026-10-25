from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'实验与代码/MID稳定性复检/20261022-v1'))
import check_mid as T
K,P,E,ST=T.K,T.P,T.E,T.ST
sys.path.insert(0,str(ROOT/'实验与代码/FiLM-L通信优化/20261020-v1'))
import protocol as W

@torch.no_grad()
def fuse(model,ex,blocks):
    aa=[];bb=[]
    for lo in range(0,len(ex['a']),4096):
        sl=slice(lo,lo+4096)
        hs=[torch.tensor(blocks[k][sl,:d]) for k,d in [('market',16),('finance',88),('governance',21)]]
        q=torch.tensor(np.column_stack([blocks[k][sl,-1] for k in ['market','finance','governance']]))
        ac=torch.tensor(blocks['audit_control'][sl]);ind=torch.tensor(ex['industry'][sl,0].astype('i8'))
        pred=[head(hs,ac[:,:8],ac[:,8:9],q,ind) for head in model.heads]
        aa.append(torch.stack([p[0] for p in pred]).mean(0).numpy());bb.append(torch.stack([p[1] for p in pred]).mean(0).numpy())
    return np.concatenate(aa),np.concatenate(bb)

def messages(ex):
    return {'market':np.column_stack([ex['h1'],ex['q'][:,0]]),'finance':np.column_stack([ex['h2'],ex['q'][:,1]]),'governance':np.column_stack([ex['h3'],ex['q'][:,2]]),'audit_control':np.column_stack([ex['c'],ex['r']])}

def decision(a,b):
    k=int(np.ceil(.1*len(a)));chosen=np.argsort(-a,kind='stable')[:k]
    top=np.sort(np.argsort(-b,axis=1,kind='stable')[:,:2],axis=1)
    return set(chosen.tolist()),top

def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    formal=P.D.C.shared.load_loader().formal;pack=P.D.C.pack_for(2020,(0,1));tr,va=pack['splits'];records=[]
    candidates={'fp32':{k:'fp32' for k in W.BLOCKS},'fp16':{k:'fp16' for k in W.BLOCKS},'mixed':{'market':'fp16','finance':'int8','governance':'int8','audit_control':'int8'}}
    for seed in T.SEEDS:
        model=T.construct(formal,pack,seed);path=T.OUT/f'models/MID_s{seed}/model.pt';model.load_state_dict(torch.load(path,weights_only=True));model.eval()
        ex0,ex1=[E.extract(model,s) for s in [tr,va]];m0,m1=messages(ex0),messages(ex1);scales=W.fit_scales(m0)
        a0,b0=fuse(model,ex1,m1);np.testing.assert_array_equal(a0,ex1['a']);np.testing.assert_array_equal(b0,ex1['b'])
        expected,top0=decision(a0,b0);base=P.EV.task_metrics(tr,va,a0,b0);seedout=OUT/f'seed{seed}';seedout.mkdir(exist_ok=True)
        np.savez_compressed(seedout/'scales.npz',**scales)
        for name,prec in candidates.items():
            decoded={};frames={};payload={}
            for k in W.BLOCKS:
                frame=W.serialize_block(k,m1[k],prec[k],scales[k]);block,precision,decoded[k]=W.deserialize_block(frame,scales[k])
                assert block==k and precision==prec[k]
                np.testing.assert_array_equal(decoded[k],W.quantize_dequantize(k,m1[k],prec[k],scales[k]))
                frames[k]=len(frame);payload[k]=len(frame)-W.HEADER.size
            a,b=fuse(model,ex1,decoded);selected,top=decision(a,b);rows=np.array(sorted(expected))
            pa,pb=P.EV.sigmoid(a),P.EV.sigmoid(b);pa0,pb0=P.EV.sigmoid(a0),P.EV.sigmoid(b0)
            err_a=float(np.max(np.abs(pa-pa0)));err_b=float(np.max(np.abs(pb-pb0)))
            listexact=selected==expected;topexact=bool(np.array_equal(top[rows],top0[rows]));strict=bool(listexact and topexact and err_a<=.005 and err_b<=.005)
            task=P.EV.task_metrics(tr,va,a,b)
            row=dict(seed=seed,candidate=name,rows=len(va['y']),top10_exact=listexact,top2_on_reference_list_exact=topexact,strict_pass=strict,top10_jaccard=len(selected&expected)/len(selected|expected),top2_agreement_reference_list=float(np.mean(np.all(top[rows]==top0[rows],axis=1))),max_overall_score_error=err_a,max_type_score_error=err_b,recall=task['recall_at_10pct'],recall_difference=task['recall_at_10pct']-base['recall_at_10pct'],ap=task['overall_average_precision'],ap_difference=task['overall_average_precision']-base['overall_average_precision'],payload_bytes_per_company=sum(payload.values())/len(va['y']),header_bytes=sum(frames.values())-sum(payload.values()),scale_metadata_bytes=sum(scales[k].nbytes for k in W.BLOCKS if prec[k]=='int8'),node_x1_payload=payload['market']/len(va['y']),node_x2_payload=payload['finance']/len(va['y']),node_x3_payload=(payload['governance']+payload['audit_control'])/len(va['y']),model_sha256=T.M.sha(path),same_cached_noise_for_all_precisions=True)
            np.savez_compressed(seedout/f'{name}_scores.npz',a=a,b=b)
            writepath=seedout/f'{name}_frames.json';K.write(writepath,dict(precision=prec,frame_bytes=frames,payload_bytes=payload))
            records.append(row);print(seed,name,strict,row['payload_bytes_per_company'],err_a,err_b,flush=True)
    df=pd.DataFrame(records);df.to_csv(OUT/'compatibility.csv',index=False)
    eligible=[c for c in ['fp16','mixed'] if df[df.candidate==c].strict_pass.all()]
    selected=min(eligible,key=lambda c:df[df.candidate==c].payload_bytes_per_company.iloc[0]) if eligible else None
    K.write(OUT/'summary.json',dict(status='complete',seeds=list(T.SEEDS),selected_under_strict_rule=selected,all_candidates=records,new_training=False,privacy_attacks_on_quantized_view=False,folds_loaded=[0,1],protocol_sha256=T.M.sha(HERE/'收束决定与后续顺序.md'),source_sha256=T.M.sha(__file__)))

if __name__=='__main__':main()
