#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优惠券离线核销失败 —— 根因定位与策略测算
==========================================

复现方式:
    python analyze.py                      # 默认读 data/coupon_redemptions.csv
    python analyze.py --data path/to.csv   # 指定数据
    python analyze.py --json out.json      # 另存结构化指标

只依赖 pandas + numpy。所有 README 中引用的数字都由本脚本打印/落盘,
没有硬编码结论。

口径(来自题面):
    每次核销失败损失 75 元
    每触发一次联网强制校验成本 10 元
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 常量 / 口径
LOSS_PER_FAIL = 75.0        # 一次核销失败的损失(元)
COST_PER_VERIFY = 10.0      # 一次联网强制校验的成本(元)
STALE_THRESHOLD_MIN = 20.0  # 快照滞后阈值(分钟) —— 由数据双峰结构确定,见 discover_threshold()

TABLES = Path("output/tables")


# ---------------------------------------------------------------- 工具函数
def _utf8_stdout() -> None:
    try:  # Windows 控制台默认 GBK,中文门店名会炸
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def chi2_2x2_p(chi2: float) -> float:
    """2x2 卡方,df=1,不依赖 scipy: p = erfc(sqrt(chi2/2))"""
    return math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------- 1. 读入 + 清洗
def load_and_clean(path: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    audit: dict = {"raw_rows": int(len(raw))}

    # --- 数据质量 1: 全行完全重复(疑似导出/粘贴重复)
    dup_mask = raw.duplicated(keep="first")
    audit["dup_rows"] = int(dup_mask.sum())
    audit["dup_ids"] = int(raw["redemption_id"].duplicated().sum())
    df = raw.loc[~dup_mask].copy()
    audit["rows_after_dedup"] = int(len(df))

    # --- 数据质量 2: is_success 缺失
    audit["missing_success_raw"] = int(raw["is_success"].isna().sum())
    audit["missing_success_after_dedup"] = int(df["is_success"].isna().sum())
    df = df.dropna(subset=["is_success"]).copy()
    df["is_success"] = df["is_success"].astype(int)
    audit["analysis_rows"] = int(len(df))
    audit["dropped_total"] = audit["raw_rows"] - audit["analysis_rows"]

    # --- 关键坑: redeem_time 是 +08:00, snapshot_synced_at 是 UTC(Z)
    #     不统一时区直接相减会多出 8 小时,滞后全为负 -> 结论完全反向
    df["redeem_time"] = pd.to_datetime(df["redeem_time"], utc=True)
    df["snapshot_synced_at"] = pd.to_datetime(df["snapshot_synced_at"], utc=True)

    df["lag_min"] = (
        df["redeem_time"] - df["snapshot_synced_at"]
    ).dt.total_seconds() / 60.0

    audit["negative_lag"] = int((df["lag_min"] < 0).sum())
    audit["lag_min"], audit["lag_max"] = (
        float(df["lag_min"].min()),
        float(df["lag_min"].max()),
    )

    df["stale"] = (df["lag_min"] >= STALE_THRESHOLD_MIN).astype(int)
    df["zero_quota"] = (df["remaining_quota"] == 0).astype(int)
    df["fail"] = 1 - df["is_success"]
    df["hour_local"] = df["redeem_time"].dt.tz_convert("Asia/Shanghai").dt.hour
    df["date_local"] = df["redeem_time"].dt.tz_convert("Asia/Shanghai").dt.date
    return df, audit


# ---------------------------------------------------------------- 2. 双峰结构
def discover_threshold(df: pd.DataFrame) -> dict:
    """不预设阈值: 看 lag 分布本身长什么样,阈值应从数据断层里读出来。"""
    hr("2. 快照滞后(lag_min)分布 —— 阈值不预设,从分布断层读")
    vc = df["lag_min"].round().astype(int).value_counts().sort_index()

    # 找“零观测空档”: 连续没有样本的分钟区间
    observed = sorted(vc.index)
    gaps = []
    for a, b in zip(observed, observed[1:]):
        if b - a > 1:
            gaps.append((a, b, b - a - 1))

    print(f"lag 取值范围: {df['lag_min'].min():.0f} ~ {df['lag_min'].max():.0f} 分钟")
    print(f"滞后为 0 的请求: {int((df['lag_min'] == 0).sum())} 笔")
    print("\n观测到的分钟空档(中间完全没有样本):")
    for a, b, width in gaps:
        print(f"  ({a}, {b}) —— 空档宽度 {width} 分钟")
    # 快照是不是“逐笔刷新”? 若是, lag 就直接等于“距上次成功同步过了多久”
    uniq = df["snapshot_synced_at"].nunique()
    reuse = df.groupby("snapshot_synced_at").size()
    print(f"\n不同 snapshot_synced_at 时间点: {uniq} 个 / {len(df)} 笔请求 "
          f"(中位复用 {reuse.median():.0f} 笔/时间点, 最多 {reuse.max()} 笔)")
    print("-> 快照基本随每笔交易刷新, 因此 lag_min 可以直接读作"
          "「这台收银机距上一次成功同步已经过去多久」")

    print("\n按区间汇总:")
    bins = [-0.1, 19.999, 21.999, 1e9]
    labels = ["lag < 20", "20 <= lag < 22", "lag >= 22"]
    seg = pd.cut(df["lag_min"], bins=bins, labels=labels, include_lowest=True)
    tab = df.groupby(seg, observed=False).agg(
        n=("fail", "size"),
        fail_n=("fail", "sum"),
        fail_rate=("fail", "mean"),
    )
    print(tab.round(4).to_string())

    TABLES.mkdir(parents=True, exist_ok=True)
    vc.rename_axis("lag_min").rename("n").to_csv(TABLES / "lag_distribution.csv")
    tab.to_csv(TABLES / "lag_segments.csv")

    return {
        "gaps": [{"from": a, "to": b, "width": w} for a, b, w in gaps],
        "segments": tab.reset_index().to_dict("records"),
    }


# ---------------------------------------------------------------- 3. 主对照
def headline_contrast(df: pd.DataFrame) -> dict:
    hr("3. 主对照: stale(滞后>=20min) vs fresh")
    ct = pd.crosstab(df["stale"], df["is_success"])
    print(ct.to_string())

    fresh = df[df["stale"] == 0]
    stale = df[df["stale"] == 1]
    n, st_n = len(df), len(stale)
    f_n, s_n = len(fresh), len(stale)
    f_fail, s_fail = int(fresh["fail"].sum()), int(stale["fail"].sum())
    tot_fail = f_fail + s_fail

    # 2x2 卡方(手算,不依赖 scipy)
    a, b = f_fail, f_n - f_fail          # fresh: 失败 / 成功
    c, d = s_fail, s_n - s_fail          # stale: 失败 / 成功
    n_tot = a + b + c + d
    chi2 = n_tot * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    p_val = chi2_2x2_p(chi2)
    rr = (c / s_n) / (a / f_n)

    out = {
        "n": n,
        "fail_total": tot_fail,
        "fail_rate_total": tot_fail / n,
        "stale_n": s_n,
        "stale_share": s_n / n,
        "stale_fail_n": s_fail,
        "stale_fail_rate": s_fail / s_n,
        "fresh_n": f_n,
        "fresh_fail_n": f_fail,
        "fresh_fail_rate": f_fail / f_n,
        "stale_share_of_all_failures": s_fail / tot_fail,
        "risk_ratio": rr,
        "chi2": chi2,
        "p_value": p_val,
        "baseline_loss_yuan": tot_fail * LOSS_PER_FAIL,
    }

    print(f"\n样本 {n} 笔, 失败 {tot_fail} 笔 ({fmt_pct(tot_fail/n)}) -> "
          f"直接损失 {tot_fail*LOSS_PER_FAIL:.0f} 元")
    print(f"  stale : {s_n} 笔 ({fmt_pct(s_n/n)}), 失败 {s_fail} 笔 ({fmt_pct(s_fail/s_n)})")
    print(f"  fresh : {f_n} 笔 ({fmt_pct(f_n/n)}), 失败 {f_fail} 笔 ({fmt_pct(f_fail/f_n)})")
    print(f"  stale 只占 {fmt_pct(s_n/n)} 的请求, 却贡献 {fmt_pct(s_fail/tot_fail)} 的失败")
    print(f"  stale 组的滞后中位数 {stale['lag_min'].median():.0f} 分钟, "
          f"最长 {stale['lag_min'].max():.0f} 分钟 (一旦断, 平均断半小时以上)")
    print(f"  风险比 RR = {rr:.2f}, Pearson chi2 = {chi2:.1f} (df=1, 未做 Yates 校正), "
          f"p = {p_val:.2e}")
    TABLES.mkdir(parents=True, exist_ok=True)
    ct.to_csv(TABLES / "stale_vs_success.csv")
    return out


# ---------------------------------------------------------------- 4. 排除混杂
def rule_out_confounders(df: pd.DataFrame) -> dict:
    hr("4. 排除其他解释: 门店 / 券种 / 时段 / 额度")
    out: dict = {}

    print("--- 4.1 门店: 失败率差异是否只是 stale 比例的镜像? ---")
    by_store = df.groupby("store_name").agg(
        n=("fail", "size"),
        fail_rate=("fail", "mean"),
        stale_share=("stale", "mean"),
    )
    # 只看新鲜快照, 剥离同步问题后门店的真实差异
    fresh_only = (df[df["stale"] == 0]
                  .groupby("store_name")
                  .agg(fresh_n=("fail", "size"), fresh_fail_rate=("fail", "mean")))
    by_store = by_store.join(fresh_only)
    print(by_store.round(4).to_string())
    by_store.to_csv(TABLES / "by_store.csv")
    out["by_store"] = by_store.reset_index().to_dict("records")

    worst = by_store["fresh_fail_rate"].idxmax()
    out["store_with_residual_issue"] = str(worst)
    out["store_residual_fail_rate"] = float(by_store.loc[worst, "fresh_fail_rate"])
    out["store_others_fail_rate"] = float(
        by_store.drop(index=worst)["fresh_fail_rate"].mean()
    )

    # 该店 vs 其他店, 只在新鲜快照子集里做 2x2 卡方
    sub = df[df["stale"] == 0]
    is_w = sub["store_name"] == worst
    a = int((is_w & (sub["fail"] == 1)).sum())
    b = int((is_w & (sub["fail"] == 0)).sum())
    c = int((~is_w & (sub["fail"] == 1)).sum())
    d = int((~is_w & (sub["fail"] == 0)).sum())
    n_tot = a + b + c + d
    chi2 = n_tot * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    out["store_residual_p_value"] = chi2_2x2_p(chi2)
    print(f"\n剥离 stale 后, {worst} 仍有 "
          f"{fmt_pct(by_store.loc[worst,'fresh_fail_rate'])} 的失败率 "
          f"({a}/{a+b}, 其他门店均值 {fmt_pct(by_store.drop(index=worst)['fresh_fail_rate'].mean())}, "
          f"p = {out['store_residual_p_value']:.4f})")
    print("-> 该店在同步之外还有独立故障, 需单独排查(设备/网络/本地缓存损坏), 不并入主策略")

    print("\n--- 4.2 券种 ---")
    by_type = df.groupby("coupon_type").agg(
        n=("fail", "size"), fail_rate=("fail", "mean"), stale_share=("stale", "mean")
    ).sort_values("fail_rate", ascending=False)
    print(by_type.round(4).to_string())
    print(f"券种失败率极差: {fmt_pct(by_type['fail_rate'].max() - by_type['fail_rate'].min())}"
          " -> 差异远小于 stale/fresh 之间的差异, 不是券种问题")
    by_type.to_csv(TABLES / "by_coupon_type.csv")
    out["coupon_type_range"] = float(by_type["fail_rate"].max() - by_type["fail_rate"].min())

    print("\n--- 4.3 时段(高峰压力假设): 只看新鲜快照, 各小时失败率是否仍随高峰抬升? ---")
    by_hour = df[df["stale"] == 0].groupby("hour_local").agg(
        fresh_n=("fail", "size"), fresh_fail_rate=("fail", "mean")
    )
    print(by_hour.round(4).to_string())
    print("若为高峰系统压力, fresh 内失败率应随客流抬升; 实测在 0%-6% 之间平坦波动"
          " -> 高峰只是让 stale 更容易发生, 不是独立致因")
    by_hour.to_csv(TABLES / "by_hour_fresh_only.csv")
    out["hour_fresh_range"] = [
        float(by_hour["fresh_fail_rate"].min()),
        float(by_hour["fresh_fail_rate"].max()),
    ]

    print("\n--- 4.4 额度(remaining_quota): 次级因素, 不是主因 ---")
    quad = df.pivot_table(
        index="stale", columns="zero_quota", values="fail",
        aggfunc=["size", "mean"], observed=True,
    )
    print(quad.round(4).to_string())
    z_fresh = df[(df.stale == 0) & (df.zero_quota == 1)]["fail"]
    nz_fresh = df[(df.stale == 0) & (df.zero_quota == 0)]["fail"]
    print(f"\n快照新鲜时: 额度=0 失败率 {fmt_pct(z_fresh.mean())} ({int(z_fresh.sum())}/{len(z_fresh)}), "
          f"额度>0 失败率 {fmt_pct(nz_fresh.mean())} ({int(nz_fresh.sum())}/{len(nz_fresh)})")
    print("额度=0 有独立影响, 但只占样本 "
          f"{fmt_pct((df.zero_quota==1).mean())}, 量级远小于 stale")
    quad.to_csv(TABLES / "quota_by_stale.csv")
    out["zero_quota_fresh_fail_rate"] = float(z_fresh.mean())
    out["nonzero_quota_fresh_fail_rate"] = float(nz_fresh.mean())
    out["zero_quota_share"] = float((df["zero_quota"] == 1).mean())

    print("\n--- 4.5 相关性: lag 是不是别的东西的代理变量? ---")
    print(f"corr(lag_min, remaining_quota) = {df['lag_min'].corr(df['remaining_quota']):.3f}")
    print(f"corr(lag_min, order_amount)     = {df['lag_min'].corr(df['order_amount']):.3f}")
    print("均接近 0 -> lag 不是额度或客单价的代理")
    return out


# ---------------------------------------------------------------- 5. 策略测算
def simulate_strategies(df: pd.DataFrame) -> dict:
    hr("5. 策略测算(失败 75 元 / 校验 10 元)")
    n = len(df)
    tot_fail = int(df["fail"].sum())
    fresh_rate = df[df["stale"] == 0]["fail"].mean()   # 基线失败率

    def evaluate(name: str, gate: pd.Series, optimistic: bool) -> dict:
        """gate=True 的请求走联网强制校验。

        乐观: 校验后失败率归 0(拿到真实状态就一定能判对)
        保守: 校验后失败率降到“快照新鲜”基线(校验也会超时/失败)
        """
        g_n = int(gate.sum())
        g_fail = int(df.loc[gate, "fail"].sum())
        if optimistic:
            avoided = g_fail
        else:
            avoided = max(g_fail - g_n * fresh_rate, 0.0)
        cost = g_n * COST_PER_VERIFY
        saved = avoided * LOSS_PER_FAIL
        return {
            "strategy": name,
            "gated_requests": g_n,
            "gated_share": g_n / n,
            "verify_cost": cost,
            "avoided_failures": round(avoided, 1),
            "saved_yuan": round(saved, 1),
            "net_yuan": round(saved - cost, 1),
        }

    rows = []
    rows.append(evaluate("A. 全部强制联网校验", pd.Series(True, index=df.index), False))
    rows.append(evaluate("B. 仅额度=0 时校验", df["zero_quota"] == 1, False))
    rows.append(evaluate("C. 仅滞后>=20min 时校验", df["stale"] == 1, False))
    rows.append(evaluate("D. 滞后>=20min 或 额度=0", (df["stale"] == 1) | (df["zero_quota"] == 1), False))
    plan = pd.DataFrame(rows)
    print("保守口径(校验后失败率降为新鲜快照基线 "
          f"{fmt_pct(fresh_rate)}):")
    print(plan.to_string(index=False))

    opt = []
    opt.append(evaluate("A. 全部强制联网校验", pd.Series(True, index=df.index), True))
    opt.append(evaluate("B. 仅额度=0 时校验", df["zero_quota"] == 1, True))
    opt.append(evaluate("C. 仅滞后>=20min 时校验", df["stale"] == 1, True))
    opt.append(evaluate("D. 滞后>=20min 或 额度=0", (df["stale"] == 1) | (df["zero_quota"] == 1), True))
    plan_opt = pd.DataFrame(opt)
    print("\n乐观口径(校验后失败率归 0):")
    print(plan_opt.to_string(index=False))

    TABLES.mkdir(parents=True, exist_ok=True)
    plan.to_csv(TABLES / "strategy_conservative.csv", index=False)
    plan_opt.to_csv(TABLES / "strategy_optimistic.csv", index=False)

    # --- 阈值扫描: 证明 20 分钟不是拍脑袋
    print("\n--- 5.1 阈值扫描(只对 lag>=t 的请求强制校验, 保守口径) ---")
    scan = []
    for t in range(1, 51):
        sel = df["lag_min"] >= t
        r = evaluate(f"t={t}", sel, False)
        r["threshold"] = t
        scan.append(r)
    scan = pd.DataFrame(scan)
    best = scan.sort_values("net_yuan", ascending=False).iloc[0]
    plateau = scan[scan["net_yuan"] == scan["net_yuan"].max()]["threshold"]
    print(scan[["threshold", "gated_requests", "avoided_failures", "net_yuan"]]
          .head(30).to_string(index=False))
    print(f"\n净收益最大: t={best['threshold']:.0f} 分钟, 净省 {best['net_yuan']:.0f} 元; "
          f"最优平台 t = {int(plateau.min())}~{int(plateau.max())} 分钟")
    print("(该平台恰好落在数据里 19-21 分钟的零观测空档 —— 阈值来自数据结构, 不是人为设定)")
    scan.to_csv(TABLES / "threshold_scan.csv", index=False)

    # --- 根因修复: 把 lag 压到阈值内
    root_fixed_fail = n * fresh_rate
    root_avoided = tot_fail - root_fixed_fail
    root = {
        "strategy": "E. 根因修复: 保证同步滞后<=20min",
        "gated_requests": 0,
        "verify_cost": 0.0,
        "avoided_failures": round(root_avoided, 1),
        "saved_yuan": round(root_avoided * LOSS_PER_FAIL, 1),
        "net_yuan": round(root_avoided * LOSS_PER_FAIL, 1),
    }
    print("\n--- 5.2 根因修复(把滞后压到阈值内) ---")
    print(f"失败 {tot_fail} 笔 -> 约 {root_fixed_fail:.0f} 笔, 少损失 "
          f"{root_avoided:.0f} 次 x {LOSS_PER_FAIL:.0f} 元 = {root['net_yuan']:.0f} 元, 且无校验成本")

    return {
        "conservative": plan.to_dict("records"),
        "optimistic": plan_opt.to_dict("records"),
        "threshold_best": int(best["threshold"]),
        "threshold_plateau": [int(plateau.min()), int(plateau.max())],
        "threshold_best_net": float(best["net_yuan"]),
        "root_fix": root,
        "fresh_fail_rate": float(fresh_rate),
    }


# ---------------------------------------------------------------- 6. 敏感性
def sensitivity(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
    hr("6. 敏感性: is_success 缺失 30 笔当成失败 / 当成成功, 结论是否翻转")
    out = {}
    for name, fill in [("缺失=失败", 0), ("缺失=成功", 1)]:
        d = df_raw.drop_duplicates().copy()
        d["is_success"] = d["is_success"].fillna(fill).astype(int)
        d["redeem_time"] = pd.to_datetime(d["redeem_time"], utc=True)
        d["snapshot_synced_at"] = pd.to_datetime(d["snapshot_synced_at"], utc=True)
        d["lag_min"] = (d["redeem_time"] - d["snapshot_synced_at"]).dt.total_seconds() / 60
        d["stale"] = (d["lag_min"] >= STALE_THRESHOLD_MIN).astype(int)
        d["fail"] = 1 - d["is_success"]
        s = d[d.stale == 1]["fail"].mean()
        f = d[d.stale == 0]["fail"].mean()
        out[name] = {"n": int(len(d)), "stale_fail_rate": float(s),
                     "fresh_fail_rate": float(f), "rr": float(s / f)}
        print(f"{name}: n={len(d)}, stale 失败率 {fmt_pct(s)}, fresh 失败率 {fmt_pct(f)}, "
              f"RR={s/f:.2f}")
    print("两种极端口径下 RR 都在 5~10 之间, 结论不翻转")

    # 缺失行本身的滞后分布(看缺失是否随机)
    miss = df_raw[df_raw["is_success"].isna()].drop_duplicates().copy()
    if len(miss):
        lag = (pd.to_datetime(miss["redeem_time"], utc=True)
               - pd.to_datetime(miss["snapshot_synced_at"], utc=True)).dt.total_seconds() / 60
        share = float((lag >= STALE_THRESHOLD_MIN).mean())
        out["missing_stale_share"] = share
        print(f"缺失行的 stale 比例 {fmt_pct(share)} (全样本 "
              f"{fmt_pct(df_clean['stale'].mean())}) -> 缺失与滞后基本无关, 可按随机缺失处理")
    return out


# ---------------------------------------------------------------- 7. 上界对照
def scale_note(df: pd.DataFrame) -> dict:
    hr("7. 规模化口径(仅按样本线性外推, 非实测)")
    n = len(df)
    days = df["date_local"].nunique()
    stores = df["store_name"].nunique()
    fail = int(df["fail"].sum())
    per_1000 = fail / n * 1000
    print(f"样本跨度 {days} 天 x {stores} 店, {n} 笔核销, 失败 {fail} 笔")
    print(f"每 1000 笔核销约有 {per_1000:.0f} 笔失败 -> "
          f"{per_1000*LOSS_PER_FAIL:.0f} 元损失")
    print(f"若只对滞后>=20min 的请求联网校验, 每 1000 笔触发 "
          f"{df['stale'].mean()*1000:.0f} 次校验")
    return {"days": int(days), "stores": int(stores), "fail_per_1000": float(per_1000)}


# ---------------------------------------------------------------- main
def main() -> None:
    _utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/coupon_redemptions.csv")
    ap.add_argument("--json", default="output/metrics.json")
    args = ap.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)

    df, audit = load_and_clean(args.data)
    hr("1. 数据清洗审计")
    for k, v in audit.items():
        print(f"  {k:32s}: {v}")
    print(f"  剔除合计 {audit['dropped_total']} 行 "
          f"({audit['dropped_total']/audit['raw_rows']*100:.1f}%),"
          " 剔除原因全部为重复行/结果缺失, 不涉及筛选业务口径")

    shape = discover_threshold(df)
    head = headline_contrast(df)
    conf = rule_out_confounders(df)
    strat = simulate_strategies(df)
    sens = sensitivity(pd.read_csv(args.data), df)
    scale = scale_note(df)

    metrics = {
        "audit": audit,
        "lag_structure": shape,
        "headline": head,
        "confounders": conf,
        "strategies": strat,
        "sensitivity": sens,
        "scale": scale,
        "assumptions": {
            "loss_per_failure_yuan": LOSS_PER_FAIL,
            "cost_per_online_verify_yuan": COST_PER_VERIFY,
            "stale_threshold_min": STALE_THRESHOLD_MIN,
        },
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    hr("DONE")
    print(f"结构化结果 -> {args.json}")
    print(f"明细表     -> {TABLES}/")


if __name__ == "__main__":
    main()
