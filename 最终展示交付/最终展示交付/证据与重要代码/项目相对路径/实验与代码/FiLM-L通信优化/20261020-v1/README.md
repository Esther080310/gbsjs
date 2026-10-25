# FiLM-L通信优化（20261020-v1）

协议预检、校准逐值回放和81种精度组合的正式评价已经完成。

```bash
cd "/Users/how/Desktop/分布式企业违规检测/实验与代码/FiLM-L通信优化/20261020-v1"
../../.venv-literature/bin/python ledger.py
../../.venv-literature/bin/python preflight.py
../../.venv-literature/bin/python candidate_matrix.py
```

- `训练前约定.md`：拓扑、六模型口径、候选协议和停止边界；
- `protocol_contract.json`：正式评价前冻结的机器可读判据；
- `预检结论.md`：已核实事实、字节口径和遗留问题；
- `protocol.py`：四类逻辑消息提取、融合端重放和固定精度帧；
- `ledger.py`：单模型/六模型及旧训练账本修正；
- `preflight.py`：无标签小烟测；
- `candidate_matrix.py`：只生成81种候选及其字节账本，不读取标签；
- `evaluate_candidates.py`：严格复现逐种子校准与六校准logit平均，评价81种候选；
- `select_strict_candidate.py`：在全部严格门通过者中选择最终六模型payload最小者；
- `results/formal_candidate_*.csv`：逐年、三年和字节账本；
- `results/strict_candidate_selection.json`：最终选择与节点分解。

54/81 个候选通过三年严格门。最小方案为市场FP16、财务INT8、治理INT8、审计控制INT8：最终六模型payload由3205降至899字节/企业（下降71.95%），且三个年度的前10%名单和名单内Top2类型集合逐项相同。总体解释见 `../../阶段扩展汇总/20261021-v1/结果总览.md`。
