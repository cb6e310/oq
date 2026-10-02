"""分析一字板回调形态买入后 T+25 收益与固定牛熊区间的关系。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
STRATEGY = ROOT / "strategies" / "analysis" / "yizi_pullback"
RESULTS = STRATEGY / "results"
FIGURES = STRATEGY / "figures"
DOCS = STRATEGY / "docs"
sys.path.insert(0, str(ROOT / "scripts"))
from market_regimes import market_regime


def regime_span(start, end):
    labels = {market_regime(d) for d in pd.date_range(start, end, freq="D")}
    if labels == {"熊市"}:
        return "熊市"
    if labels == {"牛市"}:
        return "牛市"
    return "跨牛熊"


def load_events():
    ev = pd.read_csv(RESULTS / "events.csv",
                     parse_dates=["signal_date", "yizi_date", "d1_date",
                                  "return_start_date", "return_horizon_date"])
    ev["t25_date"] = ev.return_horizon_date
    ev = ev.dropna(subset=["return_start_date", "t25_date", "ret_close_to_best_close_25"]).copy()
    ev["t25_regime"] = [regime_span(a, b) for a, b in zip(ev.return_start_date, ev.t25_date)]
    ev["return_sign"] = ev.ret_close_to_best_close_25.map(lambda x: "正收益" if x > 0 else "负收益")
    ev["signal_regime"] = ev.signal_date.map(market_regime)
    ev["exit_regime"] = ev.t25_date.map(market_regime)
    return ev


def summary(ev):
    g = ev.groupby(["t25_regime", "return_sign"], observed=True)
    out = g.agg(samples=("ret_close_to_best_close_25", "size"),
                share=("ret_close_to_best_close_25", lambda x: len(x) / len(ev)),
                mean_return=("ret_close_to_best_close_25", "mean"),
                median_return=("ret_close_to_best_close_25", "median"),
                win_rate=("ret_close_to_best_close_25", lambda x: (x > 0).mean())).reset_index()
    totals = ev.groupby("t25_regime", observed=True).agg(
        samples=("ret_close_to_best_close_25", "size"), mean_return=("ret_close_to_best_close_25", "mean"),
        median_return=("ret_close_to_best_close_25", "median"), positive_rate=("ret_close_to_best_close_25", lambda x: (x > 0).mean())
    ).reset_index()
    return out, totals


def make_figures(ev, out, totals):
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    order = [x for x in ["牛市", "熊市", "跨牛熊"] if x in ev.t25_regime.unique()]
    colors = {"牛市": "#d95f59", "熊市": "#4c78a8", "跨牛熊": "#8c8c8c"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    pivot = out.pivot(index="t25_regime", columns="return_sign", values="samples").reindex(order).fillna(0)
    pivot[[x for x in ["正收益", "负收益"] if x in pivot.columns]].plot.bar(
        ax=axes[0], color=["#e15759", "#59a14f"], width=.72)
    axes[0].set_title("T+25 正负收益样本数")
    axes[0].set_xlabel("T+25 持有区间市场阶段")
    axes[0].set_ylabel("样本数")
    axes[0].legend(title="收益方向")
    rate = totals.set_index("t25_regime").reindex(order)
    axes[1].bar(rate.index, rate.positive_rate * 100,
                 color=[colors[x] for x in order], width=.62)
    axes[1].axhline(50, color="#777", lw=.8, ls="--")
    axes[1].set_title("不同市场阶段的 T+25 胜率")
    axes[1].set_xlabel("T+25 持有区间市场阶段")
    axes[1].set_ylabel("正收益占比（%）")
    for i, v in enumerate(rate.positive_rate * 100):
        axes[1].text(i, v + 1.5, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    overview = FIGURES / "market_regime_overview.png"
    fig.savefig(overview, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    data = [ev.loc[ev.t25_regime.eq(x), "ret_close_to_best_close_25"] * 100 for x in order]
    ax.boxplot(data, tick_labels=order, patch_artist=True,
               boxprops={"facecolor": "#d9e6f2"}, medianprops={"color": "#d62728"})
    ax.axhline(0, color="#777", lw=.8)
    ax.set_title("D1-D3最低价K线收盘至T+25最佳收盘收益分布（按市场阶段）")
    ax.set_xlabel("T+25 持有区间市场阶段")
    ax.set_ylabel("收益率（%）")
    fig.tight_layout()
    box = FIGURES / "market_regime_distribution.png"
    fig.savefig(box, dpi=180)
    plt.close(fig)
    return overview, box


def write_report(ev, out, totals, overview, box):
    report = DOCS / "backtest_report.md"
    text = report.read_text(encoding="utf-8")
    marker = "## 统计结果可视化：T+25 与牛熊区间相关性"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    order = [x for x in ["牛市", "熊市", "跨牛熊"] if x in totals.t25_regime.tolist()]
    lines = [marker, "", "### 匹配口径", "",
             "以D1-D3最低价K线的收盘（买入点）为起点，向后取该股票第 25 根有效 K 线作为 T+25 观察边界；将买入日至 T+25 观察边界的整个区间与项目固定牛熊日历匹配。区间内只包含牛市记为“牛市”，只包含熊市记为“熊市”，跨越边界记为“跨牛熊”。收益采用买入收盘至其后第 1 至 25 根有效 K 线最高后复权收盘价的路径收益 `ret_close_to_best_close_25`。", "",
             f"有效样本数：{len(ev):,}。正收益定义为 `ret_close_to_best_close_25 > 0`，负收益定义为 `ret_close_to_best_close_25 <= 0`。该指标是路径潜力，不是可直接成交的胜率。", "",
             "### 分组统计", "", "| T+25持有区间 | 样本数 | 正收益数 | 负收益数 | 正收益占比 | 平均收益 | 中位数 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for regime in order:
        x = totals[totals.t25_regime.eq(regime)].iloc[0]
        pos = out[(out.t25_regime.eq(regime)) & (out.return_sign.eq("正收益"))].samples.sum()
        neg = out[(out.t25_regime.eq(regime)) & (out.return_sign.eq("负收益"))].samples.sum()
        lines.append(f"| {regime} | {int(x.samples):,} | {int(pos):,} | {int(neg):,} | {x.positive_rate:.1%} | {x.mean_return:+.2%} | {x.median_return:+.2%} |")
    lines += ["", "### 图表", "", f"![T+25正负收益与胜率]({overview.relative_to(ROOT).as_posix()})", "", f"![T+25收益分布]({box.relative_to(ROOT).as_posix()})", "", "### 结论", ""]
    overall = ev.ret_close_to_best_close_25.mean()
    best = totals.sort_values("positive_rate", ascending=False).iloc[0]
    worst = totals.sort_values("positive_rate", ascending=True).iloc[0]
    lines += [f"- 全样本路径收益均值为 **{overall:+.2%}**。终点取买入后T+1至T+25最高收盘价，路径收益不是可实现胜率。", f"- 市场阶段之间的差异只能作为路径空间的条件相关性，不能解释为实际交易胜率：最高阶段为 **{best.t25_regime}**（{best.positive_rate:.1%}），最低阶段为 **{worst.t25_regime}**（{worst.positive_rate:.1%}）。", "", "", "明细结果：`results/market_regime_events.csv`；分组统计：`results/market_regime_summary.csv`。"]
    report.write_text(text + "\n".join(lines) + "\n", encoding="utf-8")


def main():
    ev = load_events()
    out, totals = summary(ev)
    overview, box = make_figures(ev, out, totals)
    ev.to_csv(RESULTS / "market_regime_events.csv", index=False, encoding="utf-8-sig")
    out.to_csv(RESULTS / "market_regime_summary.csv", index=False, encoding="utf-8-sig")
    write_report(ev, out, totals, overview, box)
    print(totals.to_string(index=False))
    print(f"report={DOCS / 'backtest_report.md'}")


if __name__ == "__main__":
    main()
