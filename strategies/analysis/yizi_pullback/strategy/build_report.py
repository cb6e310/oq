"""Build the final report for the one-word limit-up pullback study."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
STRATEGY = ROOT / "strategies" / "analysis" / "yizi_pullback"
RESULTS = STRATEGY / "results"
FIGURES = STRATEGY / "figures"
DOCS = STRATEGY / "docs"
REPORT = DOCS / "backtest_report.md"
PATH = "ret_close_to_best_close_25"
EXCESS = "ex_close_to_best_close_25"

def pct(value):
    return "—" if pd.isna(value) else "{:+.2%}".format(value)

def stats(frame):
    values = frame[PATH].dropna()
    excess = frame.loc[values.index, EXCESS].dropna()
    return dict(n=len(values), mean=values.mean(), median=values.median(), p10=values.quantile(.1), p90=values.quantile(.9), win=(values > 0).mean(), excess=excess.mean())

def table(frame, column):
    rows = []
    for key, group in frame.groupby(column, observed=True, dropna=False):
        item = stats(group)
        rows.append("| {} | {:,} | {} | {} | {} | {} | {} |".format(key, item["n"], pct(item["mean"]), pct(item["median"]), pct(item["p10"]), pct(item["p90"]), pct(item["excess"])))
    return ["| 分组 | 样本数 | 区间均值 | 中位数 | P10 | P90 | 路径平均超额 |", "|---|---:|---:|---:|---:|---:|---:|"] + rows

def make_figures(events):
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    values = events[PATH].dropna() * 100
    fig, axis = plt.subplots(figsize=(9.5, 4.7))
    axis.hist(values.clip(upper=80), bins=48, color="#4c78a8", alpha=.85)
    axis.axvline(values.mean(), color="#d62728", label="均值 {:.2f}%".format(values.mean()))
    axis.axvline(values.median(), color="#2ca02c", label="中位数 {:.2f}%".format(values.median()))
    axis.set(title="D1-D3最低价K线收盘至T+25最高收盘收益分布", xlabel="区间收益（%，80%以上仅截尾显示）", ylabel="样本数")
    axis.legend(); fig.tight_layout(); fig.savefig(FIGURES / "return_distribution.png", dpi=180); plt.close(fig)
    yearly = events.groupby("year", observed=True)[PATH].agg(["count", "mean"])
    fig, axis = plt.subplots(figsize=(10, 4.8))
    bars = axis.bar(yearly.index.astype(str), yearly["mean"] * 100, color="#d95f59")
    axis.set(title="区间收益年度均值", xlabel="信号年份", ylabel="平均区间收益（%）")
    for bar, count in zip(bars, yearly["count"]): axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .3, "n={}".format(int(count)), ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "yearly_return.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, pair in zip(axes.flat, [("gap_bars", "信号位置"), ("drop_bucket", "信号日回调幅度"), ("d1_direction", "D1方向"), ("board", "板块")]):
        column, title = pair
        grouped = events.groupby(column, observed=True)[PATH].agg(["count", "mean"])
        axis.bar([str(value) for value in grouped.index], grouped["mean"] * 100, color="#d95f59")
        axis.set_title(title); axis.set_ylabel("平均区间收益（%）"); axis.tick_params(axis="x", rotation=20)
        for index, pair2 in enumerate(zip(grouped["mean"] * 100, grouped["count"])): axis.text(index, pair2[0] + .25, "n={}".format(int(pair2[1])), ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "group_comparison.png", dpi=180); plt.close(fig)

def build():
    dates = ["signal_date", "yizi_date", "d1_date", "return_start_date", "return_end_date"]
    events = pd.read_csv(RESULTS / "events.csv", parse_dates=dates)
    regime = pd.read_csv(RESULTS / "market_regime_events.csv", parse_dates=dates + ["t25_date"])
    make_figures(events)
    valid = events.dropna(subset=[PATH]).copy()
    mean = valid[PATH].mean()
    representative = pd.concat([valid.nlargest(10, PATH).assign(sample_group="高收益"), valid.assign(_gap=(valid[PATH] - mean).abs()).nsmallest(10, "_gap").assign(sample_group="接近均值"), valid.nsmallest(10, PATH).assign(sample_group="低收益")])
    representative.to_csv(RESULTS / "representative_samples.csv", index=False, encoding="utf-8-sig")
    bull = regime[(regime["t25_regime"] == "牛市") & (regime[PATH] <= 0)].nsmallest(10, PATH).assign(sample_group="牛市非正收益")
    bear = regime[(regime["t25_regime"] == "熊市") & (regime[PATH] > 0)].nlargest(10, PATH).assign(sample_group="熊市正收益")
    pd.concat([bull, bear], ignore_index=True).to_csv(RESULTS / "market_regime_samples.csv", index=False, encoding="utf-8-sig")
    overall = stats(events)
    lines = ["# 一字板回调策略：最终回测报告", "", "## 策略规则", "", "研究对象是一字涨停后两根有效K线内首次跌破板价的回调形态。D0前紧邻3根有效K线均不得涨停；排除新股期与疑似ST；同时使用月度涨幅和近三个月低点约束。", "", "## 收益口径", "", "买入点为D0后第1至第3根有效K线中最低价最低的K线收盘价；卖出点为买入后T+1至T+25内最高后复权收盘价。同日不可卖出，T+25从买入日计算。终点是事后窗口最高收盘价，表示策略的客观路径能力，不代表可预知成交收益。", "", "## 核心结果", "", "| 事件数 | 有效路径样本 | 区间均值 | 中位数 | P10 | P90 | 正收益占比 | 路径平均超额 |", "|---:|---:|---:|---:|---:|---:|---:|---:|", "| {:,} | {:,} | {} | {} | {} | {} | {:.1%} | {} |".format(len(events), overall["n"], pct(overall["mean"]), pct(overall["median"]), pct(overall["p10"]), pct(overall["p90"]), overall["win"], pct(overall["excess"])), "", "有效路径样本为{:,}个；无效样本包括买入点早于信号日或后续行情不足。".format(overall["n"]), "", "![区间收益分布](../figures/return_distribution.png)", "", "## 分组统计", "", "### 信号位置"] + table(events, "gap_bars") + ["", "### 信号日回调幅度"] + table(events, "drop_bucket") + ["", "### D1方向"] + table(events, "d1_direction") + ["", "### 板块"] + table(events, "board") + ["", "![年度区间收益](../figures/yearly_return.png)", "", "![主要分组对比](../figures/group_comparison.png)", "", "## 市场阶段", "", "![市场阶段概览](../figures/market_regime_overview.png)", "", "![市场阶段分布](../figures/market_regime_distribution.png)", "", "[市场阶段样本图](../figures/market-regime-samples.html)", "", "## 样本图", "", "[代表性样本图](../figures/representative-samples.html)", "", "[暴力与中等回调样本图](../figures/pullback-samples.html)", "", "## 结论与限制", "", "D1-D3买入收盘至T+25最佳收盘均值{}、中位数{}。这反映窗口内的反弹路径空间，不等于可直接执行的正期望。".format(pct(overall["mean"]), pct(overall["median"])), "", "买入K线和最佳卖出日均为事后识别；结果未计佣金、滑点、涨跌停排队、仓位和资金占用。", "", "## 文件", "", "事件明细：results/events.csv；统计结果：results/summary.csv；策略入口：strategy/pattern.py；市场阶段分析：strategy/market_regime_analysis.py；报告生成：strategy/build_report.py"]
    DOCS.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)

if __name__ == "__main__": build()
