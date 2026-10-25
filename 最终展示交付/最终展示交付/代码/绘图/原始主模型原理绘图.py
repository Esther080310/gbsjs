from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle


OUT = Path(__file__).resolve().parents[1] / "assets"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "Microsoft YaHei",
        "font.sans-serif": ["Microsoft YaHei"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.dpi": 160,
    }
)

NAVY = "#17324D"
BLUE = "#3A6EA5"
TEAL = "#3A7D78"
ORANGE = "#C66A2B"
INK = "#20262E"
MID = "#5F6B76"
LIGHT = "#E9EEF2"
PALE_BLUE = "#EDF3F8"
PALE_TEAL = "#EDF6F4"
PALE_ORANGE = "#FBF2EA"
WHITE = "#FFFFFF"


def save(fig, stem):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def text(ax, x, y, s, size=10, color=INK, weight="normal", ha="left", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, **kw)


def rect(ax, xy, w, h, face=WHITE, edge=LIGHT, lw=1.0):
    p = Rectangle(xy, w, h, facecolor=face, edgecolor=edge, linewidth=lw)
    ax.add_patch(p)
    return p


def arrow(ax, start, end, color=MID, lw=1.2, style="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            linestyle=style,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )


def research_path():
    fig, ax = plt.subplots(figsize=(13.2, 6.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    text(ax, 2, 58.5, "研究设计与证据收束", 17, NAVY, "bold")
    text(ax, 2, 55.1, "从现实时间边界出发，依次完成数据审计、候选检验、归因比较与原型定型", 9.5, MID)
    ax.plot([2, 98], [52.5, 52.5], color=NAVY, lw=1.6)

    cols = [2, 21.5, 41, 60.5, 80]
    widths = [16.5] * 5
    heads = ["任务界定", "数据与时间", "候选机制", "归因对照", "最终原型"]
    bodies = [
        ["年报与审计信息发布后", "形成企业风险排序", "主指标：Recall@10%"],
        ["企业—年度对齐", "市场 / 财务 / 治理 / 审计", "按时间顺序训练与回放"],
        ["经济结构与表格网络", "类别不均衡与案例学习", "延迟标签与通信压缩"],
        ["U0：无审计信息", "UA7：审计普通接入", "FiLM-L：条件调节", "集中式网络参照"],
        ["三节点本地编码", "审计条件调节财务表示", "六模型固定集成", "名单与类型提示"],
    ]
    fills = [PALE_BLUE, PALE_BLUE, "#F5F6F7", PALE_TEAL, PALE_ORANGE]
    accents = [BLUE, BLUE, MID, TEAL, ORANGE]

    for i, (x, w, head, lines, fill, accent) in enumerate(zip(cols, widths, heads, bodies, fills, accents), 1):
        rect(ax, (x, 27), w, 21.5, fill, "#CBD4DC", 0.9)
        ax.add_patch(Rectangle((x, 46.5), w, 2.0, facecolor=accent, edgecolor=accent, lw=0))
        ax.add_patch(Circle((x + 2.3, 43.3), 1.45, facecolor=WHITE, edgecolor=accent, lw=1.1))
        text(ax, x + 2.3, 43.3, str(i), 9, accent, "bold", ha="center")
        text(ax, x + 4.5, 43.3, head, 11.5, NAVY, "bold")
        y = 38.9
        for line in lines:
            ax.plot([x + 2.1, x + 2.7], [y, y], color=accent, lw=1.5)
            text(ax, x + 3.4, y, line, 8.8, INK)
            y -= 3.5
        if i < 5:
            arrow(ax, (x + w + 0.4, 37.5), (cols[i] - 0.6, 37.5), color="#7C8994", lw=1.1)

    text(ax, 2, 22.5, "证据如何使用", 10.5, NAVY, "bold")
    ax.plot([2, 98], [20.6, 20.6], color="#CBD4DC", lw=0.8)
    bands = [
        (2, 30, BLUE, "真实数据诊断", "决定任务口径、时间边界和可用信息"),
        (34.5, 30, TEAL, "受控模拟", "检验机制在什么条件下有效；不替代真实数据结果"),
        (67, 31, ORANGE, "真实回放与工程核对", "选择最终原型；通信实测与模型精度分别报告"),
    ]
    for x, w, c, a, b in bands:
        ax.plot([x, x + 3.5], [16.8, 16.8], color=c, lw=3.0, solid_capstyle="butt")
        text(ax, x + 4.6, 16.8, a, 9.5, NAVY, "bold")
        text(ax, x + 4.6, 12.9, b, 8.4, MID)

    text(ax, 2, 6.7, "收束原则", 9.3, NAVY, "bold")
    text(ax, 12.8, 6.7, "保留可复现的增量证据；把未通过的候选作为选型依据；不把模拟优势写成真实场景优势。", 9.1, INK)
    save(fig, "01_研究路径图_专业版")


def model_architecture():
    fig, ax = plt.subplots(figsize=(13.8, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 86)
    ax.axis("off")
    text(ax, 2, 82.2, "三节点本地编码与审计条件调节", 17, NAVY, "bold")
    text(ax, 2, 78.5, "FiLM-L：审计意见按维调节财务编码 h2；原始财务指标 x2、治理表示 h3 均予以保留", 9.5, MID)
    ax.plot([2, 98], [75.8, 75.8], color=NAVY, lw=1.6)

    # Zone labels and communication boundary
    text(ax, 2, 72.8, "三个节点：本地编码", 9.5, NAVY, "bold")
    text(ax, 64.5, 72.8, "融合端：条件调节与预测", 9.5, NAVY, "bold")
    ax.plot([61.5, 61.5], [18.5, 73.8], color="#9BA7B0", lw=1.0, ls=(0, (3, 3)))
    text(ax, 60.4, 70.4, "消息传输边界", 7.5, MID, ha="right")

    # Node 1: market and trading
    y = 65.0
    text(ax, 2, y, "节点 1  市场与交易", 9.1, NAVY, "bold")
    rect(ax, (15, y - 3.5), 10.5, 7.0, WHITE, "#B9C4CD", 0.9)
    text(ax, 20.25, y, "市场指标 x1", 8.6, INK, ha="center")
    rect(ax, (28.5, y - 3.5), 11.5, 7.0, PALE_BLUE, BLUE, 1.0)
    text(ax, 34.25, y, "本地 MLP", 8.8, NAVY, "bold", ha="center")
    rect(ax, (44, y - 3.7), 14.0, 7.4, WHITE, BLUE, 1.1)
    text(ax, 51, y + 1.2, "m1 = [h1; q1]", 8.2, BLUE, "bold", ha="center")
    text(ax, 51, y - 1.5, "16 + 1 = 17 floats", 7.4, MID, ha="center")
    arrow(ax, (25.7, y), (28.2, y), BLUE)
    arrow(ax, (40.2, y), (43.7, y), BLUE)

    # Node 2: finance. The transmitted h2 is modulated only after the boundary.
    y = 52.0
    text(ax, 2, y, "节点 2  财务报表", 9.1, NAVY, "bold")
    rect(ax, (15, y - 3.5), 10.5, 7.0, WHITE, "#B9C4CD", 0.9)
    text(ax, 20.25, y, "财务指标 x2", 8.6, INK, ha="center")
    rect(ax, (28.5, y - 3.5), 11.5, 7.0, PALE_TEAL, TEAL, 1.0)
    text(ax, 34.25, y + 1.1, "经济分组编码", 8.4, NAVY, "bold", ha="center")
    text(ax, 34.25, y - 1.5, "30 + 29 + 29", 7.3, MID, ha="center")
    rect(ax, (44, y - 3.7), 14.0, 7.4, WHITE, TEAL, 1.1)
    text(ax, 51, y + 1.2, "m2 = [h2; q2]", 8.2, TEAL, "bold", ha="center")
    text(ax, 51, y - 1.5, "88 + 1 = 89 floats", 7.4, MID, ha="center")
    arrow(ax, (25.7, y), (28.2, y), TEAL)
    arrow(ax, (40.2, y), (43.7, y), TEAL)

    # Audit control branch: a separate 9-float deployment message.
    y = 39.0
    text(ax, 2, y, "节点 3  审计条件支路", 9.1, NAVY, "bold")
    rect(ax, (15, y - 3.5), 13.5, 7.0, WHITE, "#B9C4CD", 0.9)
    text(ax, 21.75, y + 1.1, "审计向量 a（7维）", 8.2, INK, ha="center")
    text(ax, 21.75, y - 1.5, "匹配位 r + 6类意见", 7.3, MID, ha="center")
    rect(ax, (32, y - 3.5), 12.0, 7.0, PALE_ORANGE, ORANGE, 1.0)
    text(ax, 38, y, "条件编码 c\n8维", 8.2, NAVY, "bold", ha="center")
    rect(ax, (47.5, y - 3.7), 10.5, 7.4, WHITE, ORANGE, 1.0)
    text(ax, 52.75, y + 1.2, "控制消息", 8.2, ORANGE, "bold", ha="center")
    text(ax, 52.75, y - 1.5, "c(8) + r(1)", 7.4, MID, ha="center")
    arrow(ax, (28.7, y), (31.7, y), ORANGE)
    arrow(ax, (44.2, y), (47.2, y), ORANGE)

    # Node 3 governance/event content remains a separate message.
    y = 26.0
    text(ax, 2, y, "节点 3  治理与事件", 9.1, NAVY, "bold")
    rect(ax, (15, y - 3.5), 10.5, 7.0, WHITE, "#B9C4CD", 0.9)
    text(ax, 20.25, y, "治理事件 x3", 8.6, INK, ha="center")
    rect(ax, (28.5, y - 3.5), 11.5, 7.0, "#F3F4F5", MID, 1.0)
    text(ax, 34.25, y + 1.1, "事件编码器", 8.6, NAVY, "bold", ha="center")
    text(ax, 34.25, y - 1.5, "12 + 9", 7.3, MID, ha="center")
    rect(ax, (44, y - 3.7), 14.0, 7.4, WHITE, MID, 1.1)
    text(ax, 51, y + 1.2, "m3 = [h3; q3]", 8.2, MID, "bold", ha="center")
    text(ax, 51, y - 1.5, "21 + 1 = 22 floats", 7.4, MID, ha="center")
    arrow(ax, (25.7, y), (28.2, y), MID)
    arrow(ax, (40.2, y), (43.7, y), MID)

    # FiLM is placed at the fusion side: no node-to-node link is implied.
    rect(ax, (65, 47.6), 9.5, 8.8, PALE_ORANGE, ORANGE, 1.2)
    text(ax, 69.75, 53.4, "FiLM", 9.5, ORANGE, "bold", ha="center")
    text(ax, 69.75, 50.3, "h2 → h2′", 8.0, NAVY, "bold", ha="center")
    text(ax, 69.75, 48.2, "q2 保留", 7.2, MID, ha="center")
    arrow(ax, (58.2, 52.0), (64.7, 52.0), TEAL)
    arrow(ax, (58.2, 39.0), (67.0, 47.4), ORANGE, style=(0, (3, 2)))

    # Gated fusion uses three main messages plus industry context.
    rect(ax, (79, 43.0), 11.0, 17.0, PALE_BLUE, BLUE, 1.1)
    text(ax, 84.5, 56.2, "门控融合", 9.8, NAVY, "bold", ha="center")
    text(ax, 84.5, 52.6, "α = softmax(s)", 7.9, BLUE, "bold", ha="center")
    text(ax, 84.5, 49.0, "m1, m2′, m3", 8.0, NAVY, "bold", ha="center")
    text(ax, 84.5, 45.5, "+ 行业嵌入", 7.7, MID, ha="center")
    arrow(ax, (58.2, 65.0), (78.7, 56.0), BLUE)
    arrow(ax, (74.7, 52.0), (78.7, 52.0), ORANGE)
    arrow(ax, (58.2, 26.0), (78.7, 46.0), MID)

    rect(ax, (93, 53.0), 5.0, 7.0, PALE_ORANGE, ORANGE, 1.1)
    text(ax, 95.5, 56.5, "风险排序", 8.1, NAVY, "bold", ha="center")
    rect(ax, (93, 40.0), 5.0, 7.0, WHITE, "#B9C4CD", 1.0)
    text(ax, 95.5, 43.5, "Top2\n类型提示", 7.8, NAVY, "bold", ha="center")
    arrow(ax, (90.2, 54.0), (92.7, 56.2), ORANGE)
    arrow(ax, (90.2, 48.0), (92.7, 43.8), MID)

    # Exact operation used by FilmKeepX3.forward. Plain text keeps the full figure in YaHei.
    text(ax, 15, 17.0, "部署接口：m1(17) + m2(89) + m3(22) = 128 floats；另传 c(8) 与 r(1)，共 137 floats = 548 B / 企业", 7.7, MID)
    rect(ax, (2, 3.0), 96, 11.2, "#FAFBFC", "#CBD4DC", 0.9)
    text(ax, 4, 10.7, "财务表示调节公式", 9.2, NAVY, "bold")
    text(ax, 18, 10.7, "c = tanh(Wa · a + ba) ∈ R8", 8.1, INK)
    text(ax, 40, 10.7, "[γ(c); β(c)] = Wf · c + bf ∈ R176", 8.1, INK)
    text(ax, 69, 10.7, "h2′ = h2 + r[γ(c) ⊙ h2 + β(c)]", 8.2, ORANGE, "bold")
    text(ax, 18, 6.2, "γ(c), β(c) ∈ R88；r = 0 时 h2′ = h2。调节对象是编码后的财务表示，不是原始 x2 的变量权重，也不是融合权重 α。", 7.6, MID)

    save(fig, "02_FiLM-L模型结构_专业版")


if __name__ == "__main__":
    research_path()
    model_architecture()

