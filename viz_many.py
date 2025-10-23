# viz_many.py

"""
conda activate lstm-kf
REPO="$(git rev-parse --show-toplevel)"; export REPO
cd "$REPO"
export MPLBACKEND=Agg
OUT="$REPO/outputs/lstmkf_64000"; export OUT
python viz_many.py \
  --csv "$REPO/data/ngsim_processed_samples_64000.csv" \
  --stats-json "$REPO/data/ngsim_stats.json" \
  --splits-json "$OUT/splits.json" \
  --resume "$OUT/best.pt" \
  --outdir "$OUT/figs_val_topk" \
  --split val --select topk --num 24 \
  --anchor hist_last --smooth-win 5 --jump-ft 30 --y-view-half 80 --x-view-half 25"""
import os, json, argparse, random
import numpy as np
import torch
import lstmkf_train as m  # 依赖你现有的脚本
from types import SimpleNamespace

def _load_hparams(outdir):
    hp_path = os.path.join(outdir, "hparams.json")
    if os.path.isfile(hp_path):
        with open(hp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _build_model_from_hp(hp, device):
    # 从 hparams 恢复关键结构超参；缺的用你训练脚本里的默认
    args_like = SimpleNamespace(
        f_hidden     = int(hp.get("f_hidden", 64)),
        f_layers     = int(hp.get("f_layers", 2)),
        q_hidden     = int(hp.get("q_hidden", 64)),
        r_hidden     = int(hp.get("r_hidden", 32)),
        fdiag_hidden = int(hp.get("fdiag_hidden", 64)),
        no_fdiag     = bool(hp.get("no_fdiag", False)),
        dropout      = float(hp.get("dropout", 0.2)),
        jitter       = float(hp.get("jitter", 1e-3)),
        p0_scale     = float(hp.get("p0_scale", 1.0)),
        min_q        = float(hp.get("min_q", 1e-5)),
        min_r        = float(hp.get("min_r", 1e-4)),
        dp           = False,
        compile      = False,
        future_len   = int(hp.get("future_len", 25)),  # 仅为 f_net 构图一致
    )
    return m.build_model(args_like, device)

def _subset_from_split(full, splits_json, split):
    with open(splits_json, "r", encoding="utf-8") as f:
        sp = json.load(f)
    idx = {"train": sp["train_idx"], "val": sp["val_idx"], "test": sp["test_idx"]}[split]
    from torch.utils.data import Subset
    return Subset(full, idx), idx

def _score_samples_one(full_item):
    # 与 pick_good_sample 一致的打分：优先 EV 可见、SV 数量多
    *_unused, = ()
    H = full_item[3].shape[0]  # x_hist_full21: (H,21)
    F = full_item[2].shape[0]  # y_fut_full21 : (F,21)
    x_hist_full21 = full_item[3].numpy()  # (H,21)
    y_fut_full21  = full_item[2].numpy()  # (F,21)
    mask_hist7 = x_hist_full21.reshape(H, 7, 3)[:, :, 2]  # (H,7)
    mask_fut7  = y_fut_full21.reshape(F, 7, 3)[:, :, 2]  # (F,7)
    mask_all   = np.concatenate([mask_hist7, mask_fut7], axis=0)  # (H+F,7)
    ev_ratio   = mask_all[:, 0].mean()
    sv_ratio   = mask_all[:, 1:].mean(axis=0)  # (6,)
    good_svs   = (sv_ratio >= 0.20).sum()
    score = good_svs * 1.0 + sv_ratio.mean() * 0.1 + ev_ratio * 0.1
    return float(score)

def _select_indices(dataset_subset, how, num, seed=42):
    rng = random.Random(seed)
    base_indices = list(range(len(dataset_subset)))
    if how == "sequential":
        return base_indices[:num]
    if how == "random":
        return rng.sample(base_indices, k=min(num, len(base_indices)))
    # topk：逐个打分（简单稳妥）
    scored = []
    for i in base_indices:
        item = dataset_subset[i]  # (x_hist_xy, y_fut_xy, y_fut_full21, x_hist_full21, vid, start)
        scored.append(( _score_samples_one(item), i))
    scored.sort(reverse=True)  # 分高在前
    return [i for _, i in scored[:num]]

def main():
    ap = argparse.ArgumentParser("Export many figures from trained LSTM-KF")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--stats-json", required=True)
    ap.add_argument("--splits-json", required=True)
    ap.add_argument("--resume", required=True, help="path to best.pt")
    ap.add_argument("--outdir", required=True, help="directory to save figures")
    ap.add_argument("--split", choices=["train","val","test"], default="val")
    ap.add_argument("--num", type=int, default=24, help="number of samples to export")
    ap.add_argument("--select", choices=["topk","random","sequential"], default="topk")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    # 画图风格参数
    ap.add_argument("--anchor", choices=["hist_last","future_first"], default="hist_last")
    ap.add_argument("--smooth-win", type=int, default=5)
    ap.add_argument("--jump-ft", type=float, default=30.0)
    ap.add_argument("--y-view-half", type=float, default=80)
    ap.add_argument("--x-view-half", type=float, default=25)
    ap.add_argument("--lane-width", type=float, default=12.0)
    args = ap.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if (args.device in ["auto","cuda"] and torch.cuda.is_available()) else "cpu")

    # 读 hparams
    out_root = os.path.dirname(os.path.abspath(args.resume))
    hp = _load_hparams(out_root)
    hist_len  = int(hp.get("hist_len", 25))
    future_len= int(hp.get("future_len", 25))

    # 数据
    full = m.ProcessedNGSIMDataset(args.csv, hist_len, future_len)
    subset, original_idx = _subset_from_split(full, args.splits_json, args.split)

    # 模型
    model = _build_model_from_hp(hp, device)
    sd = torch.load(args.resume, map_location=device)
    if isinstance(sd, dict) and "state_dict" not in sd and "model_state" not in sd:
        model.load_state_dict(sd)
    else:
        model.load_state_dict(sd.get("model_state", sd))
    model.eval()

    # 选样
    sel_local_idx = _select_indices(subset, args.select, args.num, seed=args.seed)
    # 聚合为一个 batch
    items = [subset[i] for i in sel_local_idx]
    x_hist_xy     = torch.stack([it[0] for it in items], dim=0).float().to(device)
    y_fut_xy      = torch.stack([it[1] for it in items], dim=0).float().to(device)
    y_fut_full21  = torch.stack([it[2] for it in items], dim=0).float().to(device)
    x_hist_full21 = torch.stack([it[3] for it in items], dim=0).cpu()
    vids          = torch.stack([it[4] for it in items], dim=0).cpu().numpy().tolist()
    starts        = torch.stack([it[5] for it in items], dim=0).cpu().numpy().tolist()

    with torch.no_grad():
        out = model(x_hist_xy, y_fut_xy, z_mask_full21=y_fut_full21, P0=None)

    # 逐样本出图
    for k in range(len(items)):
        x1  = x_hist_xy[k:k+1].cpu()
        y1  = y_fut_xy[k:k+1].cpu()
        y21 = y_fut_full21[k:k+1].cpu()
        o1  = {kk: (vv[k:k+1].detach().cpu() if isinstance(vv, torch.Tensor) else vv) for kk, vv in out.items()}
        meta = {"veh": vids[k], "frame": starts[k]}

        save_path = os.path.join(
            args.outdir,
            f"{args.split}_{args.select}_{k:03d}_veh{vids[k]}_frame{starts[k]}.png"
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
            jump_ft=args.jump_ft,
            meta_title=meta
        )
        print("Saved:", save_path)

if __name__ == "__main__":
    main()
