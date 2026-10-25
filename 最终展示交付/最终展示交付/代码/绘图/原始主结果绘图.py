"""Presentation figures from frozen display data; no fitting or model selection.

Contracts:
04 compares exact ensemble Recall@10% and paired uncertainty.
05 separates same-structure information evidence from same-information wiring evidence.
06 shows one company per dot in the actual risk order, not a smoothed density estimate.
07 computes every budget point from the saved full ranking, not interpolation.
Exports use a presentation-wide palette, editable SVG text, and 300-dpi PNG.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parents[1]
ASSETS = HERE / 'assets'
DATA = HERE / 'data'
COLOR = {'U0': '#96A4B4', 'UA7': '#377DA6', 'FiLM-L': '#C6533C', 'GLOBAL': '#C99A37'}
RED, BLUE = COLOR['FiLM-L'], COLOR['UA7']
LABELS = {'U0': '无审计参照 U0', 'UA7': '原接入方式 UA7',
          'FiLM-L': '最终原型 FiLM-L', 'GLOBAL': '集中式全局网络'}
YEARS = ('2020', '2021', '2022')
plt.rcParams.update({
    'font.sans-serif': ['Heiti TC', 'PingFang HK', 'Arial Unicode MS', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'font.size': 12, 'axes.titlesize': 14,
    'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'figure.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
})


def save(fig, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(ASSETS / f'{name}.{ext}', dpi=300, facecolor='white')
    plt.close(fig)


def foot(fig, message, y=.022):
    fig.text(.04, y, message, ha='left', va='bottom', fontsize=9, color='#596675')


def styled(ax):
    ax.grid(axis='y', color='#E6EBF0', lw=.7)
    ax.set_axisbelow(True)


def effects_figure(payload):
    fig = plt.figure(figsize=(14.5, 7.0))
    gs = fig.add_gridspec(1, 2, left=.065, right=.975, bottom=.23, top=.79,
                          width_ratios=[1.65, 1], wspace=.30)
    ax = fig.add_subplot(gs[0, 0]); forest = fig.add_subplot(gs[0, 1])
    x = np.arange(3); width = .185
    arms = ('U0', 'UA7', 'FiLM-L', 'GLOBAL')
    for j, arm in enumerate(arms):
        valid = [i for i, year in enumerate(YEARS) if arm in payload['ensemble_metrics'][year]]
        values = [100*payload['ensemble_metrics'][YEARS[i]][arm]['lists']['10%']['recall'] for i in valid]
        lower = [payload['ensemble_recall_ci'][YEARS[i]][arm]['lower_pp'] for i in valid]
        upper = [payload['ensemble_recall_ci'][YEARS[i]][arm]['upper_pp'] for i in valid]
        positions = x[valid]+(j-1.5)*width
        ax.bar(positions, values, width, color=COLOR[arm], label=LABELS[arm],
               yerr=[np.array(values)-lower, np.array(upper)-values], capsize=3,
               error_kw={'elinewidth': 1, 'ecolor': '#344250'})
        for xx, v, hi in zip(positions, values, upper):
            ax.text(xx, hi+.8, f'{v:.1f}', ha='center', fontsize=9, color='#344250')
    ax.set_xticks(x); ax.set_xticklabels(YEARS)
    ax.set_ylim(0, 42); ax.set_ylabel('前10%名单召回（%）')
    ax.set_title('a  相同年度、相同标签下的模型对照', loc='left', pad=18)
    ax.text(2.26, 2.0, '全局未训练', ha='center', rotation=90, color='#768391', fontsize=9)
    styled(ax)
    comparisons = [('FiLM-L_minus_UA7', '相对原接入方式\n三年等权', .72),
                   ('FiLM-L_minus_GLOBAL', '相对全局网络\n两年等权', .22)]
    for key, label, yy in comparisons:
        s = payload['comparisons'][key]['equal_year']
        value, lo, hi = s['exact_pp'], s['lower_pp'], s['upper_pp']
        forest.errorbar(value, yy, xerr=[[value-lo], [hi-value]], fmt='o', color=RED,
                        elinewidth=2.2, markersize=8, capsize=5)
        forest.text(-1.65, yy+.17, label, fontsize=12, va='center')
        forest.text(-1.65, yy-.13, f'{value:+.3f}pp  [{lo:+.3f}, {hi:+.3f}]', fontsize=11,
                    color='#526172', va='center')
    forest.axvline(0, color='#758291', ls='--', lw=1.1)
    forest.set_xlim(-1.8, 1.6); forest.set_ylim(-.08, 1.14)
    forest.set_yticks([]); forest.set_xlabel('名单召回差（百分点，FiLM-L − 对照）')
    forest.set_title('b  配对差值及95%区间', loc='left', pad=18)
    forest.spines['left'].set_visible(False)
    fig.suptitle('分节点原型的名单召回与集中式网络平均点值接近',
                 x=.04, y=.965, ha='left', fontsize=21, fontweight='bold')
    fig.text(.04, .884, '主口径：六模型校准后logit均值 → 一张固定风险名单；2022年没有全局对照',
             fontsize=12, color='#526172')
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLOR[a]) for a in arms]
    fig.legend(handles, [LABELS[a] for a in arms], loc='lower center', bbox_to_anchor=(.38, .12),
               ncol=2, frameon=False, fontsize=11)
    foot(fig, '标签快照：2023-12-31；N=4,292 / 4,794 / 5,199，正例=787 / 747 / 629。')
    foot(fig, '误差线：企业聚类配对bootstrap 2,000次；同一企业跨年共用抽样，区间条件于已保存的模型预测。', y=.058)
    save(fig, '04_模型效果对照')


def increments_figure(payload):
    fig = plt.figure(figsize=(14.5, 7.0))
    gs = fig.add_gridspec(1, 2, left=.08, right=.965, bottom=.23, top=.79,
                          width_ratios=[1.18, 1], wspace=.32)
    ax = fig.add_subplot(gs[0, 0]); ap = fig.add_subplot(gs[0, 1])
    ys = np.arange(3)[::-1]
    for key, offset, color, label in [('UA7_minus_U0', .11, BLUE, '增加审计信息：UA7 − U0'),
                                       ('FiLM-L_minus_UA7', -.11, RED, '改变接入结构：FiLM-L − UA7')]:
        per = payload['comparisons'][key]['per_year']
        vals = np.array([per[y]['exact_pp'] for y in YEARS])
        lo = np.array([per[y]['lower_pp'] for y in YEARS]); hi = np.array([per[y]['upper_pp'] for y in YEARS])
        ax.errorbar(vals, ys+offset, xerr=[vals-lo, hi-vals], fmt='o', ms=7,
                    capsize=4, elinewidth=1.8, color=color, label=label)
    ax.axvline(0, color='#758291', ls='--', lw=1)
    ax.set_yticks(ys); ax.set_yticklabels(YEARS); ax.set_ylim(-.5, 2.5)
    ax.set_xlim(-2.3, 5.7); ax.set_xlabel('前10%名单召回差（百分点）')
    ax.set_title('a  分开回答“信息增量”与“接入结构”', loc='left', pad=18)
    ax.grid(axis='x', color='#E6EBF0', lw=.7)
    values = np.array([payload['comparisons']['FiLM-L_minus_UA7']['per_year'][y]['ap_difference_pp'] for y in YEARS])
    ap.bar(np.arange(3), values, color=RED, width=.50)
    for i, v in enumerate(values):
        ap.text(i, v+.055, f'{v:+.3f}pp', ha='center', fontsize=12, color=RED)
    ap.set_xticks(np.arange(3)); ap.set_xticklabels(YEARS)
    ap.set_ylim(0, 1.45); ap.set_ylabel('AP差（百分点，FiLM-L − UA7）')
    ap.set_title('b  整体排序质量的次要证据', loc='left', pad=18)
    styled(ap)
    fig.suptitle('信息收益与结构收益分别验证', x=.04, y=.965, ha='left',
                 fontsize=21, fontweight='bold')
    fig.text(.04, .884, '审计信息带来可重复的净增量；条件调制提供另一种保留原信息通道的接入方式',
             fontsize=12, color='#526172')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.50, .122),
               ncol=2, frameon=False, fontsize=10)
    foot(fig, '六模型固定集成；固定2023年末标签快照。a：公司聚类配对95%区间；b：精确AP点差，不附显著性结论。')
    foot(fig, 'UA7−U0保持同一接入结构；FiLM-L−UA7比较同信息的两种接法。FiLM-L−U0不作纯审计归因。', y=.058)
    save(fig, '05_增量与排序质量')


def ranked(year):
    frame = pd.read_csv(DATA/f'FiLM-L_{year}_排名.csv', dtype={'symbol': str})
    assert np.array_equal(frame['rank'].to_numpy(), np.arange(1, len(frame)+1))
    assert np.all(np.diff(frame['mean_calibrated_logit'].to_numpy()) <= 0)
    return frame


def density_figure(payload, year):
    frame = ranked(year)
    y = frame['recorded_positive_by_2023'].to_numpy(int)
    n, positives = len(y), int(y.sum())
    meta = payload['ensemble_metrics'][str(year)]['FiLM-L']['lists']
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 7.8))
    fig.subplots_adjust(left=.075, right=.97, top=.75, bottom=.23, wspace=.28)
    for j, (ax, pct) in enumerate(zip(axes, (10, 5, 1))):
        k = math.ceil(pct/100*n)
        st = meta[f'{pct}%']; assert k == st['budget'] and int(y[:k].sum()) == st['hits']
        rng = np.random.default_rng(20261018)
        xx = rng.uniform(-.35, .35, k)
        ranks = np.arange(1, k+1)
        for label, color in ((0, BLUE), (1, RED)):
            pick = y[:k] == label
            ax.scatter(xx[pick], ranks[pick], s=22 if k>100 else 38, color=color,
                       linewidths=.3, edgecolors='white', alpha=.91)
        ax.set_xlim(-.55,.55); ax.set_ylim(k+max(2, .02*k), -max(2,.02*k))
        ax.set_xticks([])
        ax.set_yticks(np.unique(np.rint(np.linspace(1, k, 6)).astype(int)))
        ax.set_ylabel('名单内风险排名（1为最高）')
        ax.grid(axis='y', color='#E6EBF0', lw=.6)
        ax.set_axisbelow(True)
        ax.spines['bottom'].set_visible(False)
        ax.set_title(f'前{pct}%｜{k}家\n命中{st["hits"]}家，精确率{100*st["precision"]:.1f}%',
                     fontsize=15, pad=18, linespacing=1.55)
        ax.text(.5, -.11, f'覆盖全部记录正例的{100*st["recall"]:.1f}%\n较随机名单浓度高{st["lift_over_random"]:.2f}倍',
                transform=ax.transAxes, ha='center', va='top', fontsize=11, color='#526172', linespacing=1.55)
    fig.suptitle(f'{year}年：风险名单中的记录正例密度', x=.04, y=.965, ha='left',
                 fontsize=22, fontweight='bold')
    fig.text(.04, .884, f'FiLM-L固定六模型集成；目标企业{n:,}家，记录正例{positives:,}家，基准率{100*positives/n:.1f}%',
             fontsize=12, color='#526172')
    legend = [Line2D([],[],marker='o',linestyle='',color=RED,markersize=9,
                     label='截至2023年末已记录正例'),
              Line2D([],[],marker='o',linestyle='',color=BLUE,markersize=9,
                     label='截至该快照尚未记录为正例')]
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5,.075), ncol=2,
               fontsize=11, frameon=False)
    foot(fig, '每个圆点代表一家企业，纵向保持真实风险排名；三个窗口分别放大，纵轴尺度不同。横向抖动仅用于避免遮挡。')
    name = '06_前10前5前1名单密度' if year==2020 else f'06_名单密度_{year}'
    save(fig, name)


def budget_figure(payload):
    frame=ranked(2020); y=frame['recorded_positive_by_2023'].to_numpy(int)
    n=len(y); P=int(y.sum()); base=P/n
    pcts=np.arange(1,41); ks=np.ceil(pcts/100*n).astype(int)
    hits=np.cumsum(y)[ks-1]; prec=hits/ks; recall=hits/P
    fig, axes=plt.subplots(1,2,figsize=(14.5,6.8))
    fig.subplots_adjust(left=.075,right=.96,top=.78,bottom=.20,wspace=.23)
    axes[0].plot(pcts,100*prec,color=RED,lw=2.4,label='FiLM-L名单精确率')
    axes[0].axhline(100*base,color='#66717D',ls='--',lw=1.2,label=f'随机名单期望 {100*base:.1f}%')
    axes[1].plot(pcts,100*recall,color=RED,lw=2.4,label='FiLM-L名单召回')
    axes[1].plot(pcts,100*ks/n,color='#66717D',ls='--',lw=1.2,label='随机排序的期望覆盖')
    for pct in (1,5,10):
        k=math.ceil(pct/100*n); h=int(y[:k].sum())
        for ax,value in ((axes[0],100*h/k),(axes[1],100*h/P)):
            ax.scatter(pct,value,s=48,color=RED,zorder=3)
            dx,dy=(9,9)
            ax.annotate(f'{pct}%预算：{value:.1f}%',(pct,value),textcoords='offset points',
                        xytext=(dx,dy),fontsize=10,color=RED)
    for ax in axes:
        ax.set_xlim(0,40);ax.set_ylim(0,100);ax.set_xlabel('检查预算：目标企业比例（%）')
        ax.legend(frameon=False,loc='upper right',fontsize=10);styled(ax)
    axes[0].set_ylabel('名单中记录正例占比（精确率，%）')
    axes[1].set_ylabel('全部记录正例被覆盖的比例（召回率，%）')
    axes[0].set_title('a  缩小名单，提高核查线索浓度',loc='left',pad=15)
    axes[1].set_title('b  扩大名单，覆盖更多记录正例',loc='left',pad=15)
    fig.suptitle('核查预算与名单效用：浓度和覆盖同时报告', x=.04,y=.965,ha='left',
                 fontsize=21,fontweight='bold')
    fig.text(.04,.884,'2020年案例｜固定六模型集成｜精确率回答“查中的比例”，召回率回答“覆盖的比例”',
             fontsize=12,color='#526172')
    foot(fig,'N=4,292，记录正例787。每个预算点均由完整真实排名计算；随机线为期望参照，不代表实际监管核查效果。')
    foot(fig,'标签截至2023-12-31；本图展示回顾性排序效用，预算取整采用ceil，后续年度密度见配套备查图。', y=.058)
    save(fig,'07_预算与命中率')


def main():
    payload=json.loads((DATA/'展示指标.json').read_text(encoding='utf8'))
    assert payload['new_training_runs']==0
    effects_figure(payload);increments_figure(payload)
    for year in (2020,2021,2022): density_figure(payload,year)
    budget_figure(payload)
    print('Generated 6 data figures, each in PNG(300dpi), SVG(editable text), PDF.')


if __name__=='__main__':main()
