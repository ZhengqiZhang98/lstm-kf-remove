import os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND","Agg"))
import matplotlib.pyplot as plt
from matplotlib import animation
import lstmkf_train as m

def pick_writer():
    avail = animation.writers.list()
    print("[INFO] available writers:", avail)
    if "pillow" in avail:
        return animation.PillowWriter(fps=8)
    if "ffmpeg" in avail:
        return animation.FFMpegWriter(fps=8, codec="libx264")
    if "imagemagick" in avail:
        return animation.ImageMagickWriter(fps=8)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--stats-json", required=True)
    ap.add_argument("--splits-json", required=True)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--out", required=True)  # e.g., $OUT/anim_ev.gif or .mp4
    ap.add_argument("--split", choices=["val","test","train"], default="val")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    outdir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(outdir, exist_ok=True)
    print("[INFO] save to:", args.out)

    # 读hp
    hp_path = os.path.join(os.path.dirname(os.path.abspath(args.resume)), "hparams.json")
    hp = json.load(open(hp_path,"r",encoding="utf-8")) if os.path.isfile(hp_path) else {}
    H = int(hp.get("hist_len",25)); F = int(hp.get("future_len",25))
    print(f"[INFO] H={H} F={F}")

    # 数据 & split
    full = m.ProcessedNGSIMDataset(args.csv, H, F)
    with open(args.splits_json,"r",encoding="utf-8") as f: sp = json.load(f)
    idx_map = {"train":sp["train_idx"],"val":sp["val_idx"],"test":sp["test_idx"]}
    sel_idx = idx_map[args.split]
    print(f"[INFO] split {args.split} size =", len(sel_idx))
    if not sel_idx:
        raise SystemExit("[ERROR] 选定split为空，请检查splits.json或split名称。")
    if args.index < 0 or args.index >= len(sel_idx):
        print("[WARN] --index 超界，重置为0")
        args.index = 0

    from torch.utils.data import Subset
    subset = Subset(full, sel_idx)
    x_hist_xy, y_fut_xy, y_fut_full21, x_hist_full21, vid, start = subset[args.index]
    print(f"[INFO] using sample: veh={int(vid)} frame={int(start)}")

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = m.build_model(type("A",(object,),dict(
        f_hidden=int(hp.get("f_hidden",64)), f_layers=int(hp.get("f_layers",2)),
        q_hidden=int(hp.get("q_hidden",64)), r_hidden=int(hp.get("r_hidden",32)),
        fdiag_hidden=int(hp.get("fdiag_hidden",64)), no_fdiag=bool(hp.get("no_fdiag",False)),
        dropout=float(hp.get("dropout",0.2)), jitter=float(hp.get("jitter",1e-3)),
        p0_scale=float(hp.get("p0_scale",1.0)), min_q=float(hp.get("min_q",1e-5)),
        min_r=float(hp.get("min_r",1e-4)), dp=False, compile=False, future_len=F
    )), device)
    sd = torch.load(args.resume, map_location=device)
    model.load_state_dict(sd.get("model_state", sd))
    model.eval()

    # 前向
    x = x_hist_xy.unsqueeze(0).to(device).float()
    z = y_fut_xy.unsqueeze(0).to(device).float()
    m21 = y_fut_full21.unsqueeze(0).to(device).float()
    with torch.no_grad():
        out = model(x, z, z_mask_full21=m21, P0=None)

    # 读取标准差（英尺还原）
    sj = json.load(open(args.stats_json,"r",encoding="utf-8"))
    std_x = float(sj.get("Local_X",{}).get("std",1.0))
    std_y = float(sj.get("Local_Y",{}).get("std",1.0))

    gt = z.squeeze(0).cpu().numpy().reshape(F,7,2)
    pr = out["x_post"].squeeze(0).cpu().numpy().reshape(F,7,2)
    for arr in (gt, pr):
        arr[...,0] *= std_x; arr[...,1] *= std_y
        arr[...,[0,1]] = arr[...,[1,0]]

    fig, ax = plt.subplots(figsize=(6,2.2), dpi=150)
    ax.set_xlim(-80,80); ax.set_ylim(-25,25)
    (gt_line,) = ax.plot([],[], '-', lw=2, label="EV GT")
    (pr_line,) = ax.plot([],[], '--', lw=2, label="EV Pred")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=8)

    def init():
        gt_line.set_data([],[]); pr_line.set_data([],[])
        return gt_line, pr_line

    def update(t):
        gt_line.set_data(gt[:t+1,0,0], gt[:t+1,0,1])
        pr_line.set_data(pr[:t+1,0,0], pr[:t+1,0,1])
        return gt_line, pr_line

    ani = animation.FuncAnimation(fig, update, frames=F, init_func=init, interval=120, blit=True)

    writer = pick_writer()
    if writer is not None:
        ani.save(args.out, writer=writer)
        print("[OK] Saved:", args.out)
    else:
        # 回退：导出PNG序列
        pngdir = os.path.join(outdir, "frames_png")
        os.makedirs(pngdir, exist_ok=True)
        for t in range(F):
            update(t)
            fig.savefig(os.path.join(pngdir, f"frame_{t:03d}.png"), dpi=150, bbox_inches="tight")
        print("[WARN] 没有可用动画写入器。已导出 PNG 序列到:", pngdir)
        print("可用命令合成为GIF：")
        print(f"  convert -delay 12 -loop 0 {pngdir}/frame_*.png {args.out}")
        print(f"或用 ffmpeg：")
        print(f"  ffmpeg -y -framerate 8 -i {pngdir}/frame_%03d.png -vf palettegen {outdir}/palette.png")
        print(f"  ffmpeg -y -framerate 8 -i {pngdir}/frame_%03d.png -i {outdir}/palette.png -lavfi paletteuse {args.out}")

if __name__ == "__main__":
    main()
