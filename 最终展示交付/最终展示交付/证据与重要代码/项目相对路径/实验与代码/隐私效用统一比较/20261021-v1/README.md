# 隐私—效用统一比较

已完成16个共享框架模型/扰动实现的数值专门攻击评价，汇总为10个方法强度点；另展示原六独立模型外部参照。只使用fold0/1，不重训主模型，不重开测试折。

- [评价前约定](评价前约定.md)
- [结果解读](结果解读.md)
- [Recall—数值重构误差图](figures/Recall与数值保护_Pareto.png)
- `results/all_points.csv`：全部实现
- `results/frontier.csv`：方法强度均值及经验前沿
- `results/verification.json`：均值与前沿独立复算

前沿包括dCor=1、MID=0.1和Gaussian=1三个点。不同重复次数及AP差异详见结果解读，不能把开发年度点估计视为跨年确认。dCor/MID源模型沿用前轮，单次训练；Gaussian各3次扰动。
