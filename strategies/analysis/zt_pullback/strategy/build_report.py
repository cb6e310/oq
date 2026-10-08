"""生成 ZT 回调公式的唯一正式统计报告。"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
STRATEGY = ROOT / "strategies" / "analysis" / "zt_pullback"
RESULTS, FIGURES, DOCS = STRATEGY / "results", STRATEGY / "figures", STRATEGY / "docs"
PATH, EXCESS = "ret_close_to_best_close_25", "ex_close_to_best_close_25"


def pct(value): return "—" if pd.isna(value) else f"{value:+.2%}"


def stats(frame):
    values = frame[PATH].dropna(); excess = frame.loc[values.index, EXCESS].dropna()
    return dict(n=len(values), mean=values.mean(), median=values.median(), p10=values.quantile(.1), p90=values.quantile(.9), win=(values > 0).mean(), excess=excess.mean())


def table(frame, column):
    rows = ["| 分组 | 样本数 | 区间均值 | 中位数 | P10 | P90 | 路径平均超额 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, group in frame.groupby(column, observed=True, dropna=False):
        item = stats(group); rows.append(f"| {key} | {item['n']:,} | {pct(item['mean'])} | {pct(item['median'])} | {pct(item['p10'])} | {pct(item['p90'])} | {pct(item['excess'])} |")
    return rows


def make_figures(events):
    FIGURES.mkdir(exist_ok=True); plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
    values = events[PATH].dropna() * 100
    fig, axis = plt.subplots(figsize=(9.5, 4.7)); axis.hist(values.clip(upper=80), bins=48, color="#4c78a8", alpha=.85); axis.axvline(values.mean(), color="#d62728", label=f"均值 {values.mean():.2f}%"); axis.axvline(values.median(), color="#2ca02c", label=f"中位数 {values.median():.2f}%"); axis.set(title="信号日收盘至T+25最高收盘收益分布", xlabel="区间收益（%，80%以上仅截尾显示）", ylabel="样本数"); axis.legend(); fig.tight_layout(); fig.savefig(FIGURES / "return_distribution.png", dpi=180); plt.close(fig)
    yearly = events.groupby("year", observed=True)[PATH].agg(["count", "mean"])
    fig, axis = plt.subplots(figsize=(10, 4.8)); bars = axis.bar(yearly.index.astype(str), yearly["mean"] * 100, color="#d95f59"); axis.set(title="区间收益年度均值", xlabel="信号年份", ylabel="平均区间收益（%）")
    for bar, count in zip(bars, yearly["count"]): axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .3, f"n={int(count)}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "yearly_return.png", dpi=180); plt.close(fig)
    pairs = [("t", "T位置"), ("pullback_bucket", "回调幅度"), ("signal_direction", "信号日方向"), ("board", "板块")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (column, title) in zip(axes.flat, pairs):
        grouped = events.groupby(column, observed=True)[PATH].agg(["count", "mean"]); axis.bar([str(x) for x in grouped.index], grouped["mean"] * 100, color="#d95f59"); axis.set_title(title); axis.set_ylabel("平均区间收益（%）"); axis.tick_params(axis="x", rotation=20)
        for index, (mean, count) in enumerate(zip(grouped["mean"] * 100, grouped["count"])): axis.text(index, mean + .25, f"n={int(count)}", ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "group_comparison.png", dpi=180); plt.close(fig)


def main():
    dates = ["signal_date", "zt_date", "return_start_date", "return_end_date"]
    events = pd.read_csv(RESULTS / "events.csv", parse_dates=dates); regime = pd.read_csv(RESULTS / "market_regime_events.csv", parse_dates=dates + ["t25_date"]); make_figures(events)
    valid = events.dropna(subset=[PATH]).copy(); mean = valid[PATH].mean()
    representative = pd.concat([valid.nlargest(10, PATH).assign(sample_group="高收益"), valid.assign(_gap=(valid[PATH] - mean).abs()).nsmallest(10, "_gap").assign(sample_group="接近均值"), valid.nsmallest(10, PATH).assign(sample_group="低收益")]); representative.to_csv(RESULTS / "representative_samples.csv", index=False, encoding="utf-8-sig")
    bull = regime[(regime.t25_regime.eq("牛市")) & (regime[PATH] <= 0)].nsmallest(10, PATH).assign(sample_group="牛市非正收益"); bear = regime[(regime.t25_regime.eq("熊市")) & (regime[PATH] > 0)].nlargest(10, PATH).assign(sample_group="熊市正收益"); pd.concat([bull, bear]).to_csv(RESULTS / "market_regime_samples.csv", index=False, encoding="utf-8-sig")
    overall = stats(events)
    lines = ["# ZT回调公式：最终回测报告", "", "## 公式规则", "", "严格按用户给定公式执行：ZT为收盘相对前一交易日上涨超过9%；T为最近一次ZT至当前K线的有效交易日间隔；信号要求T在1至7之间，并同时满足回调区间高点不超过涨幅日收盘的108%、低点不低于涨幅前后收盘中点、区间平均收盘不低于98%、区间收盘低点不低于96%、当前收盘不低于97%，且不存在跌幅达到5%的阴线。原式定义IS_MAIN但未写入XG，因此全样本为正式口径，主板仅作子集统计。", "", "## 收益口径", "", "买入点为信号日收盘；卖出点为买入后T+1至T+25内最高后复权收盘价。T+25观察边界按个股有效K线计数，路径终点为事后最优，不代表可预知成交收益。", "", "## 核心结果", "", "| 事件数 | 有效路径样本 | 区间均值 | 中位数 | P10 | P90 | 正收益占比 | 路径平均超额 |", "|---:|---:|---:|---:|---:|---:|---:|---:|", f"| {len(events):,} | {overall['n']:,} | {pct(overall['mean'])} | {pct(overall['median'])} | {pct(overall['p10'])} | {pct(overall['p90'])} | {overall['win']:.1%} | {pct(overall['excess'])} |", "", f"![区间收益分布](../figures/return_distribution.png)", "", "## 分组统计", "", "### T位置"]
    lines += table(events, "t") + ["", "### 回调幅度"] + table(events, "pullback_bucket") + ["", "### 信号日方向"] + table(events, "signal_direction") + ["", "### 板块"] + table(events, "board")
    lines += ["", "![年度区间收益](../figures/yearly_return.png)", "", "![主要分组对比](../figures/group_comparison.png)", "", "## 主板子集", ""] + table(events[events.is_main], "board")
    lines += ["", "## 市场阶段", "", "[市场阶段分析](market_regime.md)", "", "![市场阶段概览](../figures/market_regime_overview.png)", "", "![市场阶段分布](../figures/market_regime_distribution.png)", "", "## 样本图", "", "[代表性样本图](../figures/representative-samples.html)", "", "[回调分组样本图](../figures/pullback-samples.html)", "", "[牛熊反例样本图](../figures/market-regime-samples.html)", "", "## 结论与限制", "", f"信号日收盘至T+25最佳收盘的路径收益均值为{pct(overall['mean'])}、中位数为{pct(overall['median'])}；这描述的是公式事件后的历史价格空间，不等同于实盘可实现正期望。", "", "结果未计佣金、滑点、涨跌停排队、仓位和资金占用；最佳买入/卖出位置均为事后识别。", "", "## 文件", "", "事件明细：results/events.csv；统计结果：results/summary.csv；策略入口：strategy/pattern.py；市场阶段分析：strategy/market_regime_analysis.py；报告生成：strategy/build_report.py"]
    (DOCS / "backtest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); print(DOCS / "backtest_report.md")


if __name__ == "__main__": main()
