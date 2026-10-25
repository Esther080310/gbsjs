"""仅画原理图，不读取标签、不训练模型。位置/文字直接在本文件修改。"""
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch

HOME=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('PROJECT_FIGURE_OUT', str(HOME/'图表/原理图')))
OUT.mkdir(parents=True,exist_ok=True)
fonts=[HOME/'字体/msyh.ttc',Path('/Users/how/Library/Fonts/msyh.ttc')]
for font in fonts:
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
        break
else: raise FileNotFoundError('请在交付目录字体文件夹放置msyh.ttc，或安装微软雅黑')
plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False})
NAVY='#17324D'; TEAL='#287C7E'; ORANGE='#BC6138'; GRAY='#5D6977'

def canvas(title,subtitle,h=6):
    fig,ax=plt.subplots(figsize=(13,h));ax.set(xlim=(0,13),ylim=(0,h));ax.axis('off')
    ax.text(.3,h-.32,title,fontsize=18,weight='bold',color=NAVY,va='top')
    ax.text(.3,h-.8,subtitle,fontsize=10.4,color=GRAY,va='top')
    return fig,ax

def box(ax,x,y,w,h,title,body,color=NAVY):
    ax.add_patch(Rectangle((x,y),w,h,fc='white',ec=color,lw=1.4))
    ax.text(x+w/2,y+h-.25,title,ha='center',va='center',fontsize=12,weight='bold',color=color)
    ax.text(x+w/2,y+(h-.48)/2,body,ha='center',va='center',fontsize=10.4,color=NAVY,linespacing=1.6)

def arrow(ax,start,end,label=None,color=GRAY,path=None):
    if path:
        xs,ys=zip(*([start]+path+[end]));ax.plot(xs,ys,c=color,lw=1.15)
        start=path[-1]
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,lw=1.2,color=color))
    if label:ax.text((start[0]+end[0])/2,(start[1]+end[1])/2+.12,label,ha='center',fontsize=9,color=color)

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def communication():
    fig,ax=canvas('通信缩减  保留任务决策并降低消息数值精度','参数和模型结构固定；只改变三个节点送往融合端的消息编码',6)
    for y,t,b,c in [(3.35,'X1 市场节点','16维表示 + 完整率\n传输精度：FP16',NAVY),(1.95,'X2 财务节点','88维表示 + 完整率\n传输精度：INT8',TEAL),(.55,'X3 治理与审计','21维治理表示 + 审计控制码\n传输精度：INT8；匹配位精确发送',ORANGE)]:
        box(ax,.3,y,3.3,1.12,t,b,c);arrow(ax,(3.6,y+.56),(4.35,y+.56),color=c)
    box(ax,4.35,.55,3.8,3.92,'消息编码与还原','INT8尺度只由训练数据确定\n\nq = clip(round(h / s), −127, 127)\n\n融合端还原：ĥ = s × q\n\nFP16采用半精度存储与解码',TEAL)
    arrow(ax,(8.15,3.7),(8.8,3.7))
    box(ax,8.8,2.95,3.85,1.52,'原融合网络与六模型集成','重新计算风险排序\n重新计算名单企业的Top2类型',NAVY)
    box(ax,8.8,.55,3.85,1.8,'逐企业核对结果','2020—2022年前10%名单相同\n名单内Top2类型集合相同\n六模型载荷：3205 → 899 B/企业',ORANGE)
    arrow(ax,(10.72,2.95),(10.72,2.35))
    ax.text(.3,.07,'s为逐维量化尺度；原始信息没有被加密。字节为预测载荷，帧头与尺度元数据另计。',fontsize=9.3,color=GRAY)
    save(fig,'01_通信缩减原理')

def privacy():
    fig,ax=canvas('隐私副模型  学习带随机性的节点表示','MID随机瓶颈用于三个内容编码器；审计控制码等消息也纳入攻击者可见范围',6.6)
    box(ax,.3,3.15,2.6,1.9,'本地原始数据','市场 / 财务 / 治理\n编码得到均值 μ = h',NAVY)
    arrow(ax,(2.9,4.1),(3.5,4.1))
    box(ax,3.5,3.15,4.5,1.9,'随机信息瓶颈','学习 log σ²，采样 ε ~ N(0, I)\nz = μ + σ ⊙ ε\n每个企业年度形成一份随机消息',TEAL)
    arrow(ax,(8,4.1),(8.65,4.1))
    box(ax,8.65,3.15,4.0,1.9,'融合端','同一组随机表示供六个融合头使用\n预测企业风险与违规类型\n评价筛查效用',NAVY)
    box(ax,.3,.9,7.7,1.6,'训练时共同优化','总损失 = 风险与类型预测损失 + 0.1 × KL正则\nKL约束随机表示分布，使模型压缩非必要信息；三节点按维度与节点取平均',TEAL)
    box(ax,8.65,.9,4,1.6,'模拟融合端攻击','读取消息并训练攻击器\n恢复13项财务数值 / 推断属性\n与相同架构S0配对比较',ORANGE)
    arrow(ax,(10.65,3.15),(10.65,2.5),'同样的消息',ORANGE)
    ax.text(.3,.35,'五次独立复检：数值猜中率 75.2% → 54.7%；Recall@10% 30.13% → 29.32%。',fontsize=10.5,color=NAVY)
    ax.text(.3,.02,'评价为指定攻击下的经验防护；随机瓶颈不自动提供差分隐私保证。五个种子不组成部署集成。',fontsize=9.3,color=GRAY)
    save(fig,'02_MID隐私副模型原理')

def update():
    fig,ax=canvas('年度更新  冻结节点编码器并回放历史样本','新通报按企业与违规发生年度修正标签；增量候选通过当前保留集检查后才接受',7.4)
    box(ax,.3,4.95,3.25,1.1,'新信息到达','新年度特征 / 后续违规通报',NAVY)
    box(ax,.3,3.35,3.25,1.15,'标签与表示缓存','旧键修正标签，新键计算表示\n编码器版本变化则重建缓存',NAVY)
    arrow(ax,(1.92,4.95),(1.92,4.5))
    box(ax,4.3,3.35,4.05,2.7,'融合端增量训练','冻结：节点与审计编码器\n更新：融合端参数\n\n每批128条新增 / 修正样本\n+ 128条历史回放样本\n固定训练60步',TEAL)
    arrow(ax,(3.55,3.92),(4.3,3.92))
    box(ax,9.1,3.35,3.55,2.7,'保留集检查','固定历史组 + 最新训练年度\n使用当前已经公开的标签\n\n召回下降 > 1个百分点\n或 AP下降 > 0.01：不通过',ORANGE)
    arrow(ax,(8.35,4.7),(9.1,4.7))
    box(ax,.3,.9,3.25,1.6,'优先评估全量重训','重大政策后新监督充足\n或增量候选未通过\n重新学习编码器与融合端',ORANGE)
    box(ax,4.3,.9,4.05,1.6,'全量候选也按同一规则检查','通过：保存新版本并重建缓存\n不通过：保留父版本\n记录文件哈希与父版本供回退',NAVY)
    box(ax,9.1,.9,3.55,1.6,'保存增量候选版本','增量检查通过\n保存新融合端并复用编码器\n正式发布另做原接口兼容复核',TEAL)
    arrow(ax,(10.87,3.35),(10.87,2.5),'通过',TEAL)
    arrow(ax,(9.1,3.7),(1.92,2.5),color=ORANGE,path=[(8.7,3.7),(8.7,2.88),(1.92,2.88)])
    ax.text(6.3,3.0,'增量未通过',ha='center',fontsize=9,color=ORANGE)
    arrow(ax,(3.55,1.7),(4.3,1.7))
    ax.text(.3,.28,'回放目的是兼顾新信号与历史表现。两期实验显示成本下降，后续召回仍低于每期全量重训。',fontsize=10,color=GRAY)
    save(fig,'03_年度更新与回退原理')

if __name__=='__main__':
    communication();privacy();update()
    print('原理图已输出至',OUT)
