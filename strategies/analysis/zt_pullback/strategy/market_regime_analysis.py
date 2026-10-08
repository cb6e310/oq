"""分析公式信号的 T+25 路径收益与项目固定牛熊区间的关系。"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
STRATEGY = ROOT / "strategies" / "analysis" / "zt_pullback"
RESULTS = STRATEGY / "results"
FIGURES = STRATEGY / "figures"
DOCS = STRATEGY / "docs"
sys.path.insert(0, str(ROOT / "scripts"))
from market_regimes import market_regime


def regime_span(start, end):
    labels = {market_regime(d) for d in pd.date_range(start, end, freq="D")}
    return next(iter(labels)) if len(labels) == 1 else "跨牛熊"


def load_events():
    ev = pd.read_csv(RESULTS / "events.csv", parse_dates=["signal_date", "zt_date", "return_start_date", "return_horizon_date"])
    ev["t25_date"] = ev.return_horizon_date
    ev = ev.dropna(subset=["return_start_date", "t25_date", "ret_close_to_best_close_25"]).copy()
    ev["t25_regime"] = [regime_span(a, b) for a, b in zip(ev.return_start_date, ev.t25_date)]
    ev["return_sign"] = ev.ret_close_to_best_close_25.map(lambda x: "正收益" if x > 0 else "负收益")
    ev["signal_regime"] = ev.signal_date.map(market_regime)
    ev["exit_regime"] = ev.t25_date.map(market_regime)
    return ev


def main():
    ev = load_events()
    grouped = ev.groupby(["t25_regime", "return_sign"], observed=True)
    detail = grouped.agg(samples=("ret_close_to_best_close_25", "size"), share=("ret_close_to_best_close_25", lambda x: len(x) / len(ev)), mean_return=("ret_close_to_best_close_25", "mean"), median_return=("ret_close_to_best_close_25", "median")).reset_index()
    totals = ev.groupby("t25_regime", observed=True).agg(samples=("ret_close_to_best_close_25", "size"), mean_return=("ret_close_to_best_close_25", "mean"), median_return=("ret_close_to_best_close_25", "median"), positive_rate=("ret_close_to_best_close_25", lambda x: (x > 0).mean())).reset_index()
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    order = [x for x in ["牛市", "熊市", "跨牛熊"] if x in ev.t25_regime.unique()]
    pivot = detail.pivot(index="t25_regime", columns="return_sign", values="samples").reindex(order).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    pivot[[x for x in ["正收益", "负收益"] if x in pivot.columns]].plot.bar(ax=axes[0], color=["#e15759", "#59a14f"], width=.72)
    axes[0].set(title="T+25正负收益样本数", xlabel="T+25持有区间市场阶段", ylabel="样本数")
    rate = totals.set_index("t25_regime").reindex(order)
    axes[1].bar(rate.index, rate.positive_rate * 100, color=["#d95f59" if x == "牛市" else "#4c78a8" if x == "熊市" else "#8c8c8c" for x in order], width=.62)
    axes[1].axhline(50, color="#777", lw=.8, ls="--")
    axes[1].set(title="不同市场阶段的T+25胜率", xlabel="T+25持有区间市场阶段", ylabel="正收益占比（%）")
    for i, value in enumerate(rate.positive_rate * 100): axes[1].text(i, value + 1, f"{value:.1f}%", ha="center")
    fig.tight_layout(); overview = FIGURES / "market_regime_overview.png"; fig.savefig(overview, dpi=180); plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.boxplot([ev.loc[ev.t25_regime.eq(x), "ret_close_to_best_close_25"] * 100 for x in order], tick_labels=order, patch_artist=True, boxprops={"facecolor": "#d9e6f2"}, medianprops={"color": "#d62728"})
    axis.axhline(0, color="#777", lw=.8); axis.set(title="信号日收盘至T+25最佳收盘收益分布（按市场阶段）", xlabel="T+25持有区间市场阶段", ylabel="收益率（%）")
    fig.tight_layout(); distribution = FIGURES / "market_regime_distribution.png"; fig.savefig(distribution, dpi=180); plt.close(fig)
    RESULTS.mkdir(exist_ok=True); ev.to_csv(RESULTS / "market_regime_events.csv", index=False, encoding="utf-8-sig"); detail.to_csv(RESULTS / "market_regime_summary.csv", index=False, encoding="utf-8-sig")
    lines = ["# ZT回调公式：市场阶段分析", "", "以信号日收盘为起点，向后取第25根有效K线作为观察边界；T+1至T+25最高后复权收盘路径收益仅用于描述窗口空间，不代表可实现收益。", "", "| T+25持有区间 | 样本数 | 正收益数 | 负收益数 | 正收益占比 | 平均收益 | 中位数 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for regime in order:
        total = totals[totals.t25_regime.eq(regime)].iloc[0]
        pos = detail[(detail.t25_regime.eq(regime)) & detail.return_sign.eq("正收益")].samples.sum()
        neg = detail[(detail.t25_regime.eq(regime)) & detail.return_sign.eq("负收益")].samples.sum()
        lines.append(f"| {regime} | {int(total.samples):,} | {int(pos):,} | {int(neg):,} | {total.positive_rate:.1%} | {total.mean_return:+.2%} | {total.median_return:+.2%} |")
    lines += ["", f"![市场阶段概览](../figures/{overview.name})", "", f"![市场阶段分布](../figures/{distribution.name})", ""]
    DOCS.mkdir(exist_ok=True); (DOCS / "market_regime.md").write_text("\n".join(lines), encoding="utf-8")
    print(totals.to_string(index=False))


if __name__ == "__main__":
    main()
