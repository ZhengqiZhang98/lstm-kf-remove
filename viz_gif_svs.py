# viz_gif_svs.py
import os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND","Agg"))
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
import lstmkf_train as m  # 复用你的数据集与建模函数

NAMES = ["E","F","B","LF","LB","RF","RB"]

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

def to_feet_and_swap(arr, std_x, std_y):
    # arr: (T,7,2)  内部是 (x_lateral_norm, y_long_norm)
    a = arr.copy()
    a[...,0] *= std_x   # lateral
    a[...,1] *= std_y   # longitudinal
    a = a[..., [1,0]]   # (y, x) for plotting
    return a

def masked_until_t(arr, mask, t):
    """把 t 之后的点置为 NaN；mask==0 的点也置 NaN。arr:(T,2), mask:(T,)"""
    out = arr.copy()
    valid = (np.arange(out.shape[0]) <= t) & (mask > 0.5)
    out[~valid] = np.nan
    return out

def main():
    ap = argparse.ArgumentParser("Export GIF with EV+SV trajectories")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--stats-json", required=True)
    ap.add_argument("--splits-json", required=True)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["val","test","train"], default="val")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--show-sv-pred", action="store_true", help="同时显示SV预测（浅蓝点划线）")
    # 视野/样式
    ap.add_argument("--y-view-half", type=float, default=80)
    ap.add_argument("--x-view-half", type=float, default=25)
    ap.add_argument("--lane-width", type=float, default=12.0)
    args = ap.parse_args()

    outdir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(outdir, exist_ok=True)
    print("[INFO] save to:", args.out)

    # 读 hparams 以获取 H/F
    hp_path = os.path.join(os.path.dirname(os.path.abspath(args.resume)), "hparams.json")
    hp = json.load(open(hp_path,"r",encoding="utf-8")) if os.path.isfile(hp_path) else {}
    H = int(hp.get("hist_len", 25)); F = int(hp.get("future_len", 25))
    print(f"[INFO] H={H} F={F}")

    # 数据与 split
    full = m.ProcessedNGSIMDataset(args.csv, H, F)
    with open(args.splits_json,"r",encoding="utf-8") as f: sp = json.load(f)
    idx_map = {"train":sp["train_idx"],"val":sp["val_idx"],"test":sp["test_idx"]}
    sel_idx = idx_map[args.split]
    print(f"[INFO] split {args.split} size =", len(sel_idx))
    if not sel_idx: raise SystemExit(f"[ERROR] 选定split为空：{args.split}")
    if args.index < 0 or args.index >= len(sel_idx):
        print("[WARN] --index 超界，重置为 0"); args.index = 0

    from torch.utils.data import Subset
    subset = Subset(full, sel_idx)
    x_hist_xy, y_fut_xy, y_fut_full21, x_hist_full21, vid, start = subset[args.index]
    print(f"[INFO] using sample: veh={int(vid)} frame={int(start)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型
    model = m.build_model(type("A",(object,),dict(
        f_hidden=int(hp.get("f_hidden",64)), f_layers=int(hp.get("f_layers",2)),
        q_hidden=int(hp.get("q_hidden",64)), r_hidden=int(hp.get("r_hidden",32)),
        fdiag_hidden=int(hp.get("fdiag_hidden",64)), no_fdiag=bool(hp.get("no_fdiag",False)),
        dropout=float(hp.get("dropout",0.2)), jitter=float(hp.get("jitter",1e-3)),
        p0_scale=float(hp.get("p0_scale",1.0)), min_q=float(hp.get("min_q",1e-5)),
        min_r=float(hp.get("min_r",1e-4)), dp=False, compile=False, future_len=F
    )), device)
    sd = torch.load(args.resume, map_location=device)
    model.load_state_dict(sd.get("model_state", sd)); model.eval()

    # 前向（预测）
    x = x_hist_xy.unsqueeze(0).to(device).float()
    z = y_fut_xy.unsqueeze(0).to(device).float()
    m21 = y_fut_full21.unsqueeze(0).to(device).float()
    with torch.no_grad():
        out = model(x, z, z_mask_full21=m21, P0=None)

    # 标准差（英尺复原）
    sj = json.load(open(args.stats_json,"r",encoding="utf-8"))
    std_x = float(sj.get("Local_X",{}).get("std",1.0))
    std_y = float(sj.get("Local_Y",{}).get("std",1.0))

    # 整形：(H/F,7,2)
    xh = x_hist_xy.numpy().reshape(H,7,2)
    yf = y_fut_xy.numpy().reshape(F,7,2)
    pr = out["x_post"].squeeze(0).cpu().numpy().reshape(F,7,2)
    # 掩码：(H/F,7)
    mask_hist7 = x_hist_full21.numpy().reshape(H,7,3)[:,:,2]
    mask_fut7  = y_fut_full21.numpy().reshape(F,7,3)[:,:,2]

    # 还原英尺&坐标交换
    xh_s = to_feet_and_swap(xh, std_x, std_y)
    yf_s = to_feet_and_swap(yf, std_x, std_y)
    pr_s = to_feet_and_swap(pr, std_x, std_y)

    # 画布
    fig, ax = plt.subplots(figsize=(9.2, 3.2), dpi=160)
    ax.set_title(f"Vehicle_ID={int(vid)} | Frame_ID={int(start)}")
    ax.set_xlabel("y (feet)"); ax.set_ylabel("x (feet)")
    ax.set_xlim(-args.y_view_half, args.y_view_half)
    ax.set_ylim(-args.x_view_half, args.x_view_half)
    ax.grid(True, alpha=0.3)

    # 车道线
    w = args.lane_width
    for x_off in (-1.5*w, -0.5*w, +0.5*w, +1.5*w):
        ax.plot([-args.y_view_half, args.y_view_half], [x_off, x_off], '-', color='#BDBDBD', lw=1.2, zorder=1)

    # 车辆矩形（锚在历史最后一帧）
    anchor = xh_s[-1]              # (7,2) -> (y,x)
    anchor_mask = mask_hist7[-1]   # (7,)
    for j in range(7):
        if anchor_mask[j] < 0.5: continue
        y0, x0 = anchor[j,0], anchor[j,1]
        rect = Rectangle((y0-7/2, x0-4/2), 7, 4,
                         edgecolor=('k' if j==0 else 'gray'),
                         facecolor=('yellow' if j==0 else 'white'),
                         lw=1.0, alpha=0.85, zorder=3)
        ax.add_patch(rect)
        ax.text(y0, x0+4*0.7, NAMES[j], ha='center', va='bottom',
                fontsize=8, color=('k' if j==0 else 'gray'), weight='bold', zorder=4)

    # —— 准备线对象 —— #
    # 历史线（静态）：EV 黑、SV 灰
    (ev_hist_line,) = ax.plot(xh_s[:,0,0], xh_s[:,0,1], '-', lw=2.2, color='black', label='EV history')
    sv_hist_lines = []
    for j in range(1,7):
        (l,) = ax.plot(xh_s[:,j,0], xh_s[:,j,1], '-', lw=1.2, color='gray',
                       alpha=0.95, label='SV history' if j==1 else None)
        sv_hist_lines.append(l)

    # 未来 GT（动画）：EV 绿、SV 灰虚线
    (ev_gt_line,) = ax.plot([], [], '-', lw=2.2, color='green', label='EV future GT')
    sv_gt_lines = []
    for j in range(1,7):
        (l,) = ax.plot([], [], '--', lw=1.2, color='gray', alpha=0.9,
                       label='SV future' if j==1 else None)
        sv_gt_lines.append(l)

    # 预测（动画）：EV 蓝虚线；可选 SV 预测（浅蓝点划线）
    (ev_pred_line,) = ax.plot([], [], '--', lw=2.2, color='blue', label='EV future Pred')
    sv_pred_lines = []
    if args.show_sv_pred:
        for j in range(1,7):
            (l,) = ax.plot([], [], ':', lw=1.2, color='#3da5ff', alpha=0.9,
                           label='SV future Pred' if j==1 else None)
            sv_pred_lines.append(l)

    # 图例
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc='upper left', fontsize=9, frameon=True)

    # 遮挡历史中不可见的点（用 NaN 断开）
    for j in range(7):
        mh = mask_hist7[:,j] > 0.5
        xh_plot = xh_s[:,j,:].copy()
        xh_plot[~mh] = np.nan
        if j==0:
            ev_hist_line.set_data(xh_plot[:,0], xh_plot[:,1])
        else:
            sv_hist_lines[j-1].set_data(xh_plot[:,0], xh_plot[:,1])

    # —— 动画更新函数 —— #
    def update(t):
        # EV GT
        ev_gt = masked_until_t(yf_s[:,0,:], mask_fut7[:,0], t)
        ev_gt_line.set_data(ev_gt[:,0], ev_gt[:,1])
        # EV Pred
        ev_pr = masked_until_t(pr_s[:,0,:], mask_fut7[:,0], t)
        ev_pred_line.set_data(ev_pr[:,0], ev_pr[:,1])

        # SVs
        for j in range(1,7):
            gt = masked_until_t(yf_s[:,j,:], mask_fut7[:,j], t)
            sv_gt_lines[j-1].set_data(gt[:,0], gt[:,1])
            if sv_pred_lines:
                prj = masked_until_t(pr_s[:,j,:], mask_fut7[:,j], t)
                sv_pred_lines[j-1].set_data(prj[:,0], prj[:,1])
        return (ev_gt_line, ev_pred_line, *sv_gt_lines, *sv_pred_lines)

    ani = animation.FuncAnimation(fig, update, frames=F, interval=120, blit=True)
    writer = pick_writer()
    if writer is not None:
        ani.save(args.out, writer=writer)
        print("[OK] Saved:", args.out)
    else:
        # 回退：导出 PNG 序列
        pngdir = os.path.join(outdir, "frames_png")
        os.makedirs(pngdir, exist_ok=True)
        for t in range(F):
            update(t)
            fig.savefig(os.path.join(pngdir, f"frame_{t:03d}.png"), dpi=160, bbox_inches="tight")
        print("[WARN] 无可用动画写入器，已导出 PNG 序列到:", pngdir)
        print("可用 ffmpeg 或 convert 合成为 GIF。")

if __name__ == "__main__":
    main()
