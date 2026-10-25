from pathlib import Path
import importlib.util
from matplotlib import font_manager
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parents[2]
p=H/'代码/绘图/原始主模型原理绘图.py'
spec=importlib.util.spec_from_file_location('architecture_source',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
font=H/'字体/msyh.ttc';font_manager.fontManager.addfont(str(font));plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'figure.dpi':300,'savefig.dpi':300})
m.OUT=H/'图表/原理图';m.OUT.mkdir(exist_ok=True,parents=True)
old=m.text
def text(ax,x,y,s,*args,**kwargs):
    if s.startswith('部署接口：'):
        s='单模型载荷：内容与完整率128 floats + 审计控制码8 floats + 匹配标志1 byte = 545 B；正式六模型FP32为3205 B / 企业'
    return old(ax,x,y,s,*args,**kwargs)
m.text=text
m.research_path();m.model_architecture()
print('主模型与研究路径原理图已重画')
