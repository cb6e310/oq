"""分析一字板回调形态 T+20 收益与固定牛熊区间的关系。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STRATEGY = ROOT / "strategies" / "yizi_pullback_d2"
RESULTS = STRATEGY / "results"
FIGURES = STRATEGY / "figures"
DOCS = STRATEGY / "docs"
DAILY = ROOT / "database" / "processed" / "stock_daily.parquet"
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
    ev = pd.read_csv(RESULTS / "yizi_pullback_d2_events.csv",
                     parse_dates=["signal_date", "yizi_date", "d1_date"])
    codes = ev.ts_code.astype(str).unique().tolist()
    daily = pd.read_parquet(DAILY, columns=["date", "ts_code"])
    daily.date = pd.to_datetime(daily.date)
    daily = daily.sort_values(["ts_code", "date"])
    daily["bar_no"] = daily.groupby("ts_code", observed=True).cumcount()
    starts = daily.merge(ev[["ts_code", "signal_date"]], left_on=["ts_code", "date"],
                         right_on=["ts_code", "signal_date"], how="right")
    starts["target_bar"] = starts["bar_no"] + 20
    exits = daily[["ts_code", "bar_no", "date"]].rename(
        columns={"bar_no": "target_bar", "date": "t20_date"})
    ev = ev.merge(starts[["ts_code", "signal_date", "target_bar"]],
                  on=["ts_code", "signal_date"], how="left")
    ev = ev.merge(exits, on=["ts_code", "target_bar"], how="left")
    ev = ev.dropna(subset=["t20_date", "ret_open_20"]).copy()
    ev["t20_regime"] = [regime_span(a, b) for a, b in zip(ev.signal_date, ev.t20_date)]
    ev["return_sign"] = ev.ret_open_20.map(lambda x: "正收益" if x > 0 else "负收益")
    ev["signal_regime"] = ev.signal_date.map(market_regime)
    ev["exit_regime"] = ev.t20_date.map(market_regime)
    return ev


def summary(ev):
    g = ev.groupby(["t20_regime", "return_sign"], observed=True)
    out = g.agg(samples=("ret_open_20", "size"),
                share=("ret_open_20", lambda x: len(x) / len(ev)),
                mean_return=("ret_open_20", "mean"),
                median_return=("ret_open_20", "median"),
                win_rate=("ret_open_20", lambda x: (x > 0).mean())).reset_index()
    totals = ev.groupby("t20_regime", observed=True).agg(
        samples=("ret_open_20", "size"), mean_return=("ret_open_20", "mean"),
        median_return=("ret_open_20", "median"), positive_rate=("ret_open_20", lambda x: (x > 0).mean())
    ).reset_index()
    return out, totals


def make_figures(ev, out, totals):
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    order = [x for x in ["牛市", "熊市", "跨牛熊"] if x in ev.t20_regime.unique()]
    colors = {"牛市": "#d95f59", "熊市": "#4c78a8", "跨牛熊": "#8c8c8c"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    pivot = out.pivot(index="t20_regime", columns="return_sign", values="samples").reindex(order).fillna(0)
    pivot[[x for x in ["正收益", "负收益"] if x in pivot.columns]].plot.bar(
        ax=axes[0], color=["#e15759", "#59a14f"], width=.72)
    axes[0].set_title("T+20 正负收益样本数")
    axes[0].set_xlabel("T+20 持有区间市场阶段")
    axes[0].set_ylabel("样本数")
    axes[0].legend(title="收益方向")
    rate = totals.set_index("t20_regime").reindex(order)
    axes[1].bar(rate.index, rate.positive_rate * 100,
                 color=[colors[x] for x in order], width=.62)
    axes[1].axhline(50, color="#777", lw=.8, ls="--")
    axes[1].set_title("不同市场阶段的 T+20 胜率")
    axes[1].set_xlabel("T+20 持有区间市场阶段")
    axes[1].set_ylabel("正收益占比（%）")
    for i, v in enumerate(rate.positive_rate * 100):
        axes[1].text(i, v + 1.5, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    overview = FIGURES / "regime_t20_overview.png"
    fig.savefig(overview, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    data = [ev.loc[ev.t20_regime.eq(x), "ret_open_20"] * 100 for x in order]
    ax.boxplot(data, tick_labels=order, patch_artist=True,
               boxprops={"facecolor": "#d9e6f2"}, medianprops={"color": "#d62728"})
    ax.axhline(0, color="#777", lw=.8)
    ax.set_title("T+20 次日开盘入场收益分布（按持有区间市场阶段）")
    ax.set_xlabel("T+20 持有区间市场阶段")
    ax.set_ylabel("收益率（%）")
    fig.tight_layout()
    box = FIGURES / "regime_t20_distribution.png"
    fig.savefig(box, dpi=180)
    plt.close(fig)
    return overview, box


def write_report(ev, out, totals, overview, box):
    report = DOCS / "backtest_report.md"
    text = report.read_text(encoding="utf-8")
    marker = "## 统计结果可视化：T+20 与牛熊区间相关性"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    order = [x for x in ["牛市", "熊市", "跨牛熊"] if x in totals.t20_regime.tolist()]
    lines = [marker, "", "### 匹配口径", "",
             "以信号日为起点，向后取该股票第 20 根有效 K 线作为 T+20 退出日；将信号日至 T+20 退出日的整个持有区间与项目固定牛熊日历匹配。区间内只包含牛市记为“牛市”，只包含熊市记为“熊市”，跨越边界记为“跨牛熊”。收益采用次日开盘入场的 `ret_open_20`。", "",
             f"有效样本数：{len(ev):,}。正收益定义为 `ret_open_20 > 0`，负收益定义为 `ret_open_20 <= 0`。", "",
             "### 分组统计", "", "| T+20持有区间 | 样本数 | 正收益数 | 负收益数 | 正收益占比 | 平均收益 | 中位数 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for regime in order:
        x = totals[totals.t20_regime.eq(regime)].iloc[0]
        pos = out[(out.t20_regime.eq(regime)) & (out.return_sign.eq("正收益"))].samples.sum()
        neg = out[(out.t20_regime.eq(regime)) & (out.return_sign.eq("负收益"))].samples.sum()
        lines.append(f"| {regime} | {int(x.samples):,} | {int(pos):,} | {int(neg):,} | {x.positive_rate:.1%} | {x.mean_return:+.2%} | {x.median_return:+.2%} |")
    lines += ["", "### 图表", "", f"![T+20正负收益与胜率]({overview.relative_to(ROOT).as_posix()})", "", f"![T+20收益分布]({box.relative_to(ROOT).as_posix()})", "", "### 结论", ""]
    overall = ev.ret_open_20.mean()
    best = totals.sort_values("positive_rate", ascending=False).iloc[0]
    worst = totals.sort_values("positive_rate", ascending=True).iloc[0]
    lines += [f"- 全样本 T+20 次日开盘入场平均收益为 **{overall:+.2%}**。", f"- 正负收益与市场阶段存在阶段性相关，但不是单一决定因素：正收益率最高的是 **{best.t20_regime}**（{best.positive_rate:.1%}），最低的是 **{worst.t20_regime}**（{worst.positive_rate:.1%}）。", "- 跨牛熊区间样本应单独看待，因为其结果同时受到阶段切换和形态自身收益分布影响，不能简单归入牛市或熊市。", "- 该分析是条件相关性，不代表牛熊阶段对单个信号有因果解释；仍需结合样本数、收益分布尾部和交易成本判断是否可用于筛选。", "", "明细结果：`results/regime_t20_events.csv`；分组统计：`results/regime_t20_summary.csv`。"]
    report.write_text(text + "\n".join(lines) + "\n", encoding="utf-8")


def main():
    ev = load_events()
    out, totals = summary(ev)
    overview, box = make_figures(ev, out, totals)
    ev.to_csv(RESULTS / "regime_t20_events.csv", index=False, encoding="utf-8-sig")
    out.to_csv(RESULTS / "regime_t20_summary.csv", index=False, encoding="utf-8-sig")
    write_report(ev, out, totals, overview, box)
    print(totals.to_string(index=False))
    print(f"report={DOCS / 'backtest_report.md'}")


if __name__ == "__main__":
    main()
