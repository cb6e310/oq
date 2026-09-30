"""从正式事件明细重建一字板回调策略的唯一回测报告。"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STRATEGY = ROOT / "strategies" / "yizi_pullback_d2"
RESULTS, FIGURES, DOCS = STRATEGY / "results", STRATEGY / "figures", STRATEGY / "docs"
REPORT = DOCS / "backtest_report.md"
HORIZONS = [1, 3, 5, 10, 20]

def pct(x): return "—" if pd.isna(x) else f"{x:+.2%}"
def rate(x): return "—" if pd.isna(x) else f"{x:.1%}"

def stats(x, ret="ret_open_20", ex="ex_open_20"):
    y = x[ret].dropna(); z = x.loc[y.index, ex].dropna()
    return {"n": len(y), "mean": y.mean(), "median": y.median(), "win": (y > 0).mean(),
            "p10": y.quantile(.1), "p90": y.quantile(.9), "ex": z.mean()}

def table_by(ev, col, labels=None):
    rows = []
    for key, x in ev.groupby(col, observed=True, dropna=False):
        rows.append([labels.get(key, key) if labels else key, stats(x)])
    rows.sort(key=lambda r: str(r[0]))
    out = ["| 分组 | 样本数 | 平均收益 | 中位数 | 胜率 | P10 | P90 | 平均超额 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    out += [f"| {k} | {s['n']:,} | {pct(s['mean'])} | {pct(s['median'])} | {rate(s['win'])} | {pct(s['p10'])} | {pct(s['p90'])} | {pct(s['ex'])} |" for k, s in rows]
    return out

def plot_all(ev):
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for label, entry, style in [("次日开盘", "open", "-o"), ("信号日收盘", "close", "--o")]:
        ax.plot(HORIZONS, [ev[f"ret_{entry}_{h}"].mean() * 100 for h in HORIZONS], style, label=label)
    ax.axhline(0, color="#666", lw=.8); ax.set(title="不同持有期平均收益", xlabel="持有个股有效K线数", ylabel="平均收益（%）", xticks=HORIZONS); ax.legend(); fig.tight_layout()
    p1 = FIGURES / "final_horizon_returns.png"; fig.savefig(p1, dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.7)); v = ev.ret_open_20.dropna() * 100
    ax.hist(v.clip(-50, 80), bins=52, color="#4c78a8", alpha=.85); ax.axvline(v.mean(), color="#d62728", label=f"均值 {v.mean():.2f}%"); ax.axvline(v.median(), color="#2ca02c", label=f"中位数 {v.median():.2f}%")
    ax.set_title("T+20 次日开盘入场收益分布"); ax.set_xlabel("T+20收益（%，显示截尾-50至80）"); ax.set_ylabel("样本数"); ax.legend(); fig.tight_layout()
    p2 = FIGURES / "final_return_distribution.png"; fig.savefig(p2, dpi=180); plt.close(fig)

    yearly = ev.groupby("year", observed=True).ret_open_20.agg(["count", "mean", lambda x: (x > 0).mean()]); yearly.columns = ["n", "mean", "win"]
    fig, ax = plt.subplots(figsize=(10, 4.8)); bars = ax.bar(yearly.index.astype(str), yearly["mean"] * 100, color=np.where(yearly["mean"] >= 0, "#d95f59", "#59a14f")); ax.axhline(0, color="#666", lw=.8); ax.set(title="T+20 年度平均收益", xlabel="信号年份", ylabel="平均收益（%）")
    for b, n in zip(bars, yearly.n): ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (0.4 if b.get_height() >= 0 else -1.1), str(int(n)), ha="center", fontsize=8)
    fig.tight_layout(); p3 = FIGURES / "final_yearly_returns.png"; fig.savefig(p3, dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8)); specs = [("gap_bars", "信号位置"), ("drop_bucket", "信号日相对D0跌幅"), ("d1_direction", "D1方向"), ("board", "板块")]
    for ax, (col, title) in zip(axes.flat, specs):
        g = ev.groupby(col, observed=True).ret_open_20.agg(["count", "mean", lambda x: (x > 0).mean()]); g.columns = ["n", "mean", "win"]
        ax.bar([str(x) for x in g.index], g["mean"] * 100, color=np.where(g["mean"] >= 0, "#d95f59", "#59a14f")); ax.axhline(0, color="#777", lw=.7); ax.set_title(title); ax.set_ylabel("T+20平均收益（%）"); ax.tick_params(axis="x", rotation=20)
        for i, (v, n) in enumerate(zip(g["mean"] * 100, g.n)): ax.text(i, v + (0.3 if v >= 0 else -0.8), f"n={int(n)}", ha="center", fontsize=7)
    fig.tight_layout(); p4 = FIGURES / "final_group_comparison.png"; fig.savefig(p4, dpi=180); plt.close(fig)
    return p1, p2, p3, p4

def build():
    ev = pd.read_csv(RESULTS / "yizi_pullback_d2_events.csv", parse_dates=["signal_date", "yizi_date", "d1_date"])
    regime = pd.read_csv(RESULTS / "regime_t20_events.csv", parse_dates=["signal_date", "yizi_date", "d1_date", "t20_date"])
    bull_negative = regime[regime.t20_regime.eq("牛市")].nsmallest(10, "ret_open_20").assign(sample_group="牛市负收益")
    bear_positive = regime[regime.t20_regime.eq("熊市")].nlargest(10, "ret_open_20").assign(sample_group="熊市正收益")
    extreme_samples = pd.concat([bull_negative, bear_positive], ignore_index=True)
    extreme_samples.to_csv(RESULTS / "regime_t20_extreme_samples.csv", index=False, encoding="utf-8-sig")
    charts = plot_all(ev); valid = ev.dropna(subset=["ret_open_20"]).copy(); mean = valid.ret_open_20.mean()
    samples = pd.concat([valid.nlargest(10, "ret_open_20").assign(sample_group="高收益"), valid.assign(_gap=(valid.ret_open_20 - mean).abs()).nsmallest(10, "_gap").assign(sample_group="接近均值"), valid.nsmallest(10, "ret_open_20").assign(sample_group="低收益")])
    so, sc = stats(ev), stats(ev, "ret_close_20", "ex_close_20")
    lines = ["# 一字板回调策略：回测与可视化报告", "", "回测日期：2026-09-30  ", "数据截止：2026-09-29  ", "研究区间：近10年", "", "## 1. 最终策略规则", "", "研究对象是一字涨停后两根有效K线内首次跌破板价的回调形态。以下为唯一正式口径：", "", "1. D0前紧邻3根个股有效K线均不得收于涨停价。", "2. D0为准一字涨停，OHLC与涨停价最多允许1分钱误差。", "3. 排除上市不足60根K线、尚未首次开板的新股期，并剔除主板5%一字涨停近似识别的疑似ST。", "4. D1、D2中首次后复权收盘低于D0后复权板价的交易日为信号日；每个D0最多一次信号。", "5. 当前月截至信号日涨幅、前一个完整自然月涨幅均不得超过50%，不使用信号日后的月内数据。", "6. D0后复权收盘价不得超过此前63根有效日线最低后复权低点的1.2倍，即相对前三个月最低点涨幅不得超过20%；窗口不含D0。", "", "## 2. 回测口径", "", "- 入场：信号日收盘，或次日开盘；次日一字涨停无法买入的样本从开盘口径剔除。", "- 持有期：1、3、5、10、20根个股有效K线，停牌日不计。", "- 收益：后复权个股收益；超额收益相对同期全市场每日再平衡等权组合。", "- T+20市场阶段：信号日至第20根有效K线退出日全处于牛市/熊市则归入对应阶段，跨边界单列。", "- 本报告是事件研究，不是含仓位、资金占用和交易成本的组合净值回测。", "", "## 3. 核心回测结果", "", "| 事件数 | 有效T+20样本 | 收盘入场T+20均值 | 次日开盘T+20均值 | 开盘中位数 | 开盘胜率 | 开盘超额 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    lines.append(f"| {len(ev):,} | {so['n']:,} | {pct(sc['mean'])} | {pct(so['mean'])} | {pct(so['median'])} | {rate(so['win'])} | {pct(so['ex'])} |")
    lines += ["", "### 3.1 各持有期完整统计", "", "| 入场 | 持有期 | 样本数 | 平均收益 | 中位数 | 胜率 | 平均超额 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for entry, label in [("close", "信号日收盘"), ("open", "次日开盘")]:
        for h in HORIZONS:
            s = stats(ev, f"ret_{entry}_{h}", f"ex_{entry}_{h}"); lines.append(f"| {label} | {h} | {s['n']:,} | {pct(s['mean'])} | {pct(s['median'])} | {rate(s['win'])} | {pct(s['ex'])} |")
    lines += ["", f"![不同持有期平均收益](../figures/{charts[0].name})", "", f"![T+20收益分布](../figures/{charts[1].name})", "", "## 4. T+20分组统计", "", "以下表格统一采用次日开盘入场、持有20根有效K线。", "", "### 4.1 信号位于D1或D2"] + table_by(ev, "gap_bars", {1: "D1首次跌破", 2: "D2首次跌破"})
    for heading, col, labels in [("4.2 信号日回调幅度", "drop_bucket", None), ("4.3 D1方向", "d1_direction", None), ("4.4 D1是否涨停", "d1_limit_up", {True: "D1涨停", False: "D1非涨停"}), ("4.5 板块", "board", None), ("4.6 年度", "year", None)]: lines += ["", f"### {heading}"] + table_by(ev, col, labels)
    lines += ["", f"![年度平均收益](../figures/{charts[2].name})", "", f"![主要分组对比](../figures/{charts[3].name})", "", "## 5. T+20与固定牛熊区间", "", "固定市场阶段来自项目级 `scripts/market_regimes.py`。正收益为 `ret_open_20 > 0`，负收益为 `ret_open_20 <= 0`。", "", "| T+20持有区间 | 样本数 | 正收益数 | 负收益数 | 正收益占比 | 平均收益 | 中位数 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, x in regime.groupby("t20_regime", observed=True):
        s = stats(x); lines.append(f"| {key} | {s['n']:,} | {(x.ret_open_20 > 0).sum():,} | {(x.ret_open_20 <= 0).sum():,} | {rate(s['win'])} | {pct(s['mean'])} | {pct(s['median'])} |")
    lines += ["", "![牛熊阶段正负收益与胜率](../figures/regime_t20_overview.png)", "", "![牛熊阶段收益分布](../figures/regime_t20_distribution.png)", "", "牛市平均收益高于熊市，但牛市中位数仍为负；跨牛熊样本仅13个，均值受少数高收益尾部影响，不能视为稳定优势。", "", "### 5.1 牛市负收益与熊市正收益样本", "", "以下样本采用当前正式策略、次日开盘入场、持有20根有效K线；牛市负收益取收益最低10个，熊市正收益取收益最高10个，用于观察阶段内的反例与尾部，不代表阶段整体分布。", "", "| 类型 | 股票 | D0 | 信号日 | T+20退出日 | D1/D2 | T+20收益 | 超额收益 |", "|---|---|---|---|---|---:|---:|---:|"]
    for r in extreme_samples.itertuples(): lines.append(f"| {r.sample_group} | {r.ts_code} | {r.yizi_date:%Y-%m-%d} | {r.signal_date:%Y-%m-%d} | {r.t20_date:%Y-%m-%d} | D{r.gap_bars} | {pct(r.ret_open_20)} | {pct(r.ex_open_20)} |")
    lines += ["", "明细：`../results/regime_t20_extreme_samples.csv`。", "", "可视化：[牛市负收益与熊市正收益多周期K线长图](../figures/regime-t20-extreme-k-lines.html)。每个样本同时展示日线、周线、个股月线、创业板指月线和上证指数月线，统一使用对数价格轴；日线前后各取44根，周线和月线的辅助范围均映射到日线完整视窗。", "", "## 6. 典型样本与多周期K线", "", "样本按次日开盘入场、持有20根K线选择：最高10个、最接近全样本均值10个、最低10个。", "", "| 类型 | 股票 | D0 | 信号日 | D1/D2 | T+20收益 | 超额收益 |", "|---|---|---|---|---:|---:|---:|"]
    for r in samples.itertuples(): lines.append(f"| {r.sample_group} | {r.ts_code} | {r.yizi_date:%Y-%m-%d} | {r.signal_date:%Y-%m-%d} | D{r.gap_bars} | {pct(r.ret_open_20)} | {pct(r.ex_open_20)} |")
    lines += ["", "[打开30个典型样本的日线、周线、个股月线、创业板指月线和上证指数月线长图](../figures/yizi-mobile-k-lines.html)", "", "长图采用对数价格坐标、线性成交量坐标；日线前后各取44根，周线及所有月线的黄色辅助范围均映射到日线完整视窗。", "", "[打开随机正收益50个与负收益50个样本长图](../figures/yizi-random-positive-negative-k-lines.html)", "", "随机图使用固定种子 `20260930`，每组50个有效 T+20 样本，沿用同一套日线44/44、周线/月线22/22及指数月线绘图规范。", "", "## 7. 规则边界复核", "", "- `002428.SZ`：2026-03-20准一字板，开盘/最低仅比涨停价低1分钱；2026-03-23（D1）首次跌破。当前月截至信号日-15.87%，前月+37.34%，满足过滤。", "- `600223.SH`：2019-12-20之前更早阶段虽有连板，但紧邻D0的前三根有效K线均非涨停，按最终规则保留。", "", "## 8. 最终结论", "", f"该形态整体更接近一字板失守后的弱势延续，而不是稳定反弹买点。当前正式策略次日开盘持有20根有效K线平均收益{pct(so['mean'])}，中位数{pct(so['median'])}，胜率{rate(so['win'])}，平均超额收益{pct(so['ex'])}。", "", "市场阶段能够解释一部分差异：熊市表现弱于牛市，但牛市本身仍无稳定正期望。牛市负收益和熊市正收益反例显示，阶段标签不能替代个股形态后的风险控制；后续研究应叠加板块强度、成交量结构与反转确认。", "", "## 9. 风险与限制", "", "- 信号日收盘入场存在收盘确认后的成交时点问题，实际执行优先参考次日开盘。", "- 未计佣金、印花税、滑点、涨跌停排队成交、仓位上限、信号重叠和资金占用。", "- 数据无正式ST标签，主板5%一字板仅是近似排除。", "- 少量重整转增、特殊除权事件可能导致涨停参考价误判。", "- 分组结果是条件相关性，不代表因果；小样本分组尤其容易受极端值影响。", "", "## 10. 唯一报告与数据产物", "", "本文件是该策略唯一的结果、结论与可视化报告。", "", "- 正式事件明细：`../results/yizi_pullback_d2_events.csv`、`../results/yizi_pullback_d2_summary.csv`", "- 牛熊分析：`../results/regime_t20_events.csv`、`../results/regime_t20_summary.csv`", "- 牛熊反例：`../results/regime_t20_extreme_samples.csv`、`../figures/regime-t20-extreme-k-lines.html`", "- 随机正负收益样本：`../figures/yizi-random-positive-negative-k-lines.html`", "- 策略代码：`../strategy/pattern_yizi_pullback.py`", "- 报告生成：`../strategy/build_backtest_report.py`", "- 多周期图生成：`../strategy/make_yizi_mobile_charts.py`、`../strategy/make_yizi_random_charts.py`"]
    REPORT.write_text("\n".join(map(str, lines)) + "\n", encoding="utf-8")
    print(REPORT)

if __name__ == "__main__": build()
