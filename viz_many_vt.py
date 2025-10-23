#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量可视化（适配 lstmkf_train_vt.py 的模型与权重）

用法示例：
conda activate lstm-kf
REPO="$(git rev-parse --show-toplevel)"; export REPO
cd "$REPO"
export MPLBACKEND=Agg
OUT="$REPO/outputs/lstmkf_vt_128000"; export OUT

python viz_many_vt.py \
  --csv "$REPO/data/ngsim_processed_samples_128000.csv" \
  --stats-json "$REPO/data/ngsim_stats.json" \
  --splits-json "$OUT/splits.json" \
  --resume "$OUT/best.pt" \
  --outdir "$OUT/figs_val_topk" \
  --split val --select topk --num 50 \
  --anchor hist_last --smooth-win 5 --jump-ft 30 --y-view-half 80 --x-view-half 25
"""

import os, sys, json, argparse, random
import numpy as np
import torch
from types import SimpleNamespace

# 关键：使用“VT”版本训练脚本
import lstmkf_train_vt as m

# --------- 日志 ---------
def log_info(*a):  print("[INFO]", *a)
def log_warn(*a):  print("[WARN]", *a)
def log_err(*a):   print("[ERROR]", *a)

# --------- 工具：读 hparams、恢复模型、lane centers ---------
def _load_hparams(outdir: str):
    hp_path = os.path.join(outdir, "hparams.json")
    if os.path.isfile(hp_path):
        with open(hp_path, "r", encoding="utf-8") as f:
            hp = json.load(f)
        log_info(f"Loaded hparams from {hp_path}")
        return hp
    log_warn(f"hparams.json not found in {outdir}, will use defaults.")
    return {}

def _build_model_from_hp(hp: dict, device: torch.device):
    args_like = SimpleNamespace(
        f_hidden   = int(hp.get("f_hidden", 64)),
        f_layers   = int(hp.get("f_layers", 2)),
        q_hidden   = int(hp.get("q_hidden", 64)),
        r_hidden   = int(hp.get("r_hidden", 32)),
        dropout    = float(hp.get("dropout", 0.2)),
        jitter     = float(hp.get("jitter", 1e-3)),
        p0_scale   = float(hp.get("p0_scale", 1.0)),
        min_q      = float(hp.get("min_q", 1e-5)),
        min_r      = float(hp.get("min_r", 1e-4)),
        ev_x_idx   = int(hp.get("ev_x_idx", 0)),
        ev_y_idx   = int(hp.get("ev_y_idx", 1)),
        dp         = False,
        compile    = False,
        future_len = int(hp.get("future_len", 25)),
    )
    model = m.build_model(args_like, device)
    return model

def _lane_centers_from_hp(hp: dict, stats_json: str) -> torch.Tensor:
    args_like = SimpleNamespace(
        lane_centers_data = hp.get("lane_centers_data", ""),
        lane_centers_feet = hp.get("lane_centers_feet", "-12,0,12"),
    )
    return m.make_lane_centers_data_units(args_like, stats_json)  # (3,)

# --------- 评分与选样 ---------
def _score_samples_one(full_item) -> float:
    x_hist_full21 = full_item[3].numpy()  # (H,21)
    y_fut_full21  = full_item[2].numpy()  # (F,21)
    H = x_hist_full21.shape[0]; F = y_fut_full21.shape[0]
    mask_hist7 = x_hist_full21.reshape(H, 7, 3)[:, :, 2]
    mask_fut7  = y_fut_full21.reshape(F, 7, 3)[:, :, 2]
    mask_all   = np.concatenate([mask_hist7, mask_fut7], axis=0)
    ev_ratio   = mask_all[:, 0].mean()
    sv_ratio   = mask_all[:, 1:].mean(axis=0)
    good_svs   = (sv_ratio >= 0.20).sum()
    return float(good_svs * 1.0 + sv_ratio.mean() * 0.1 + ev_ratio * 0.1)

def _select_indices(dataset_subset, how: str, num: int, seed: int = 42):
    rng = random.Random(seed)
    base_indices = list(range(len(dataset_subset)))
    if len(base_indices) == 0:
        return []
    if how == "sequential":
        return base_indices[:min(num, len(base_indices))]
    if how == "random":
        return rng.sample(base_indices, k=min(num, len(base_indices)))
    # topk
    scored = []
    for i in base_indices:
        try:
            item = dataset_subset[i]
            scored.append((_score_samples_one(item), i))
        except Exception as e:
            log_warn(f"scoring sample {i} failed: {e}")
    scored.sort(reverse=True)
    return [i for _, i in scored[:min(num, len(scored))]]

# --------- splits 读取并容错 ---------
def _subset_from_split(full_dataset, splits_json: str, split: str):
    with open(splits_json, "r", encoding="utf-8") as f:
        sp = json.load(f)
    raw_idx = {"train": sp["train_idx"], "val": sp["val_idx"], "test": sp["test_idx"]}[split]
    n = len(full_dataset)
    ok_idx = [i for i in raw_idx if 0 <= i < n]
    bad = len(raw_idx) - len(ok_idx)
    if bad > 0:
        log_warn(f"{bad} indices out of range for split={split}; filtered.")
    from torch.utils.data import Subset
    return Subset(full_dataset, ok_idx), ok_idx

# --------- 主程序 ---------
def main():
    ap = argparse.ArgumentParser("Export many figures from VT LSTM-KF (lstmkf_train_vt)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--stats-json", required=True)
    ap.add_argument("--splits-json", required=True)
    ap.add_argument("--resume", required=True, help="path to best.pt (state_dict) or last.pth (dict)")
    ap.add_argument("--outdir", required=True, help="directory to save figures")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--num", type=int, default=24)
    ap.add_argument("--select", choices=["topk", "random", "sequential"], default="topk")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    # 画图风格
    ap.add_argument("--anchor", choices=["hist_last", "future_first"], default="hist_last")
    ap.add_argument("--smooth-win", type=int, default=5)
    ap.add_argument("--jump-ft", type=float, default=30.0)
    ap.add_argument("--y-view-half", type=float, default=80)
    ap.add_argument("--x-view-half", type=float, default=25)
    ap.add_argument("--lane-width", type=float, default=12.0)
    args = ap.parse_args()

    # 基本检查
    for path, name in [(args.csv, "csv"), (args.stats_json, "stats-json"),
                       (args.splits_json, "splits-json"), (args.resume, "resume (weights)")]:
        if not os.path.isfile(path):
            log_err(f"{name} file not found: {path}")
            sys.exit(1)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if (args.device in ["auto", "cuda"] and torch.cuda.is_available()) else "cpu")
    log_info(f"Device: {device}")

    # 从权重所在目录读取 hparams
    out_root = os.path.dirname(os.path.abspath(args.resume))
    hp = _load_hparams(out_root)
    hist_len   = int(hp.get("hist_len", 25))
    future_len = int(hp.get("future_len", 25))
    dt         = float(hp.get("dt", 0.1))
    log_info(f"hist_len={hist_len}, future_len={future_len}, dt={dt}")

    # 数据集
    full = m.ProcessedNGSIMDataset(args.csv, hist_len, future_len)
    log_info(f"Dataset loaded: {len(full)} samples (after hist/fut filtering)")

    subset, used_idx = _subset_from_split(full, args.splits_json, args.split)
    log_info(f"Split '{args.split}': {len(used_idx)} indices, subset size={len(subset)}")

    if len(subset) == 0:
        log_err("Subset is empty. Check splits.json and dataset filtering (hist_len/future_len).")
        sys.exit(1)

    # 构建模型并加载权重
    model = _build_model_from_hp(hp, device)
    # 兼容 last.pth（dict）或 best.pt（state_dict）
    try:
        sd = torch.load(args.resume, map_location=device, weights_only=True)
        log_info("Loaded weights with weights_only=True")
    except TypeError:
        sd = torch.load(args.resume, map_location=device)
        log_info("Loaded weights without weights_only=True (older PyTorch)")

    # 判断权重格式
    if isinstance(sd, dict) and "state_dict" not in sd and "model_state" not in sd:
        state = sd
    else:
        state = sd.get("model_state", sd.get("state_dict", sd))

    # 宽松加载并打印差异（修正：不要解包；使用 res.missing_keys / res.unexpected_keys）
    res = model.load_state_dict(state, strict=False)
    log_info(f"load_state_dict: missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
    if res.missing_keys:
        log_warn("Missing keys (show up to 8): " + ", ".join(list(res.missing_keys)[:8]))
    if res.unexpected_keys:
        log_warn("Unexpected keys (show up to 8): " + ", ".join(list(res.unexpected_keys)[:8]))
    model.eval()

    # lane centers（数据单位）
    lane_centers_tensor = _lane_centers_from_hp(hp, args.stats_json).to(device)
    log_info(f"lane_centers (data units): {lane_centers_tensor.tolist()}")

    # 选样
    sel_local_idx = _select_indices(subset, args.select, args.num, seed=args.seed)
    if len(sel_local_idx) == 0:
        log_err("No samples selected. Try --select random --num 1")
        sys.exit(1)
    log_info(f"Selected {len(sel_local_idx)} samples (strategy={args.select})")

    # 聚合 batch
    items = [subset[i] for i in sel_local_idx]
    x_hist_xy     = torch.stack([it[0] for it in items], dim=0).float().to(device)
    y_fut_xy      = torch.stack([it[1] for it in items], dim=0).float().to(device)
    y_fut_full21  = torch.stack([it[2] for it in items], dim=0).float().to(device)
    x_hist_full21 = torch.stack([it[3] for it in items], dim=0).cpu()
    vids          = torch.stack([it[4] for it in items], dim=0).cpu().numpy().tolist()
    starts        = torch.stack([it[5] for it in items], dim=0).cpu().numpy().tolist()
    log_info(f"Batch tensor shapes: x_hist={tuple(x_hist_xy.shape)}, y_fut={tuple(y_fut_xy.shape)}")

    # 前向
    with torch.no_grad():
        env = {"lane_centers": lane_centers_tensor}
        out = model(x_hist_xy, y_fut_xy, z_mask_full21=y_fut_full21, P0=None, env=env, T_s=dt)
    log_info("Forward pass done.")

    # 逐样本绘图
    saved, failed = 0, 0
    for k in range(len(items)):
        try:
            x1  = x_hist_xy[k:k+1].cpu()
            y1  = y_fut_xy[k:k+1].cpu()
            y21 = y_fut_full21[k:k+1].cpu()
            o1  = {kk: (vv[k:k+1].detach().cpu() if isinstance(vv, torch.Tensor) else vv) for kk, vv in out.items()}
            meta = {"veh": int(vids[k]), "frame": int(starts[k])}

            save_path = os.path.join(
                args.outdir, f"{args.split}_{args.select}_{k:03d}_veh{vids[k]}_frame{starts[k]}.png"
            )
            m.plot_anchor_scene(
                x1, y1, o1,
                stats_json_path=args.stats_json,
                sample=0,
                lane_width_ft=args.lane_width,
                save_path=save_path,
                y_view_half=args.y_view_half,
                x_view_half=args.x_view_half,
                y_fut_full21=y21,
                x_hist_full21=x_hist_full21[k:k+1],
                anchor_where=args.anchor,
                smooth_win=args.smooth_win,
                jump_break_ft=args.jump_ft,
                meta_title=meta
            )
            print("Saved:", save_path)
            saved += 1
        except Exception as e:
            log_warn(f"Save failed for sample {k}: {e}")
            failed += 1

    log_info(f"Done. Saved {saved} figures to {args.outdir}. Failed: {failed}.")

if __name__ == "__main__":
    main()
