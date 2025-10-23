# -*- coding: utf-8 -*-
import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--per-sample", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.summary, "r", encoding="utf-8") as f:
        S = json.load(f)
    df = pd.read_csv(args.per_sample)

    # 1) per-timestep RMSE
    vt_t = np.asarray(S["per_timestep_rmse"]["vt"], dtype=float)
    eu_t = np.asarray(S["per_timestep_rmse"]["euclid"], dtype=float)
    T = len(vt_t)
    plt.figure(figsize=(7,4), dpi=160)
    plt.plot(range(T), vt_t, label="VT", linewidth=2)
    plt.plot(range(T), eu_t, label="Euclid", linewidth=2)
    plt.xlabel("timestep"); plt.ylabel("RMSE"); plt.title("Per-timestep RMSE")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(args.outdir, "rmse_per_timestep.png"), bbox_inches="tight"); plt.close()

    # 2) per-dimension RMSE
    vt_d = np.asarray(S["per_dimension_rmse"]["vt"], dtype=float)
    eu_d = np.asarray(S["per_dimension_rmse"]["euclid"], dtype=float)
    D = len(vt_d); x = np.arange(D); w = 0.4
    plt.figure(figsize=(8,4), dpi=160)
    plt.bar(x - w/2, vt_d, width=w, label="VT")
    plt.bar(x + w/2, eu_d, width=w, label="Euclid")
    plt.xlabel("dimension (0..13)"); plt.ylabel("RMSE"); plt.title("Per-dimension RMSE")
    plt.xticks(x); plt.grid(True, axis="y", alpha=0.3); plt.legend()
    plt.savefig(os.path.join(args.outdir, "rmse_per_dimension.png"), bbox_inches="tight"); plt.close()

    # 3) improvement histogram
    imp = df["improve_%_euclid_vs_vt"].values
    plt.figure(figsize=(7,4), dpi=160)
    plt.hist(imp, bins=50)
    plt.axvline(0.0, linestyle="--")
    mean_imp = np.nanmean(imp); med_imp = np.nanmedian(imp)
    plt.title(f"Per-sample RMSE improvement (%): mean={mean_imp:.2f}, median={med_imp:.2f}")
    plt.xlabel("improvement % (positive = Euclid better)"); plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.outdir, "rmse_improvement_hist.png"), bbox_inches="tight"); plt.close()

    # 4) ADE/FDE bars
    ade_vt = S["ade_fde_ev_only"]["vt"]["ADE"]; fde_vt = S["ade_fde_ev_only"]["vt"]["FDE"]
    ade_eu = S["ade_fde_ev_only"]["euclid"]["ADE"]; fde_eu = S["ade_fde_ev_only"]["euclid"]["FDE"]
    labels = ["ADE", "FDE"]; x = np.arange(len(labels)); w = 0.35
    plt.figure(figsize=(5.5,4), dpi=160)
    plt.bar(x - w/2, [ade_vt, fde_vt], width=w, label="VT")
    plt.bar(x + w/2, [ade_eu, fde_eu], width=w, label="Euclid")
    plt.xticks(x, labels); plt.ylabel("error"); plt.title("EV-only ADE/FDE")
    plt.grid(True, axis="y", alpha=0.3); plt.legend()
    plt.savefig(os.path.join(args.outdir, "ade_fde_bars.png"), bbox_inches="tight"); plt.close()

    # 5) VT vs Euclid scatter
    plt.figure(figsize=(5.2,5.2), dpi=160)
    plt.scatter(df["rmse_vt"].values, df["rmse_euclid"].values, s=6, alpha=0.4)
    lim = [0, max(df["rmse_vt"].max(), df["rmse_euclid"].max())*1.05]
    plt.plot(lim, lim, linestyle="--")
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("RMSE (VT)"); plt.ylabel("RMSE (Euclid)"); plt.title("Per-sample RMSE scatter")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.outdir, "rmse_scatter_vt_vs_euclid.png"), bbox_inches="tight"); plt.close()

    print("[OK] 全部图片已保存到：", args.outdir)

if __name__ == "__main__":
    main()
