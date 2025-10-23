# lstmkf_train.py (GPU-optimized + robust KF)
"""
Train LSTM-KF with Euclidean loss (Eq.21):
  L = (1/T) * sum_t ||y_t - x_post_t||^2 + lambda * ||y_t - x_prior_t||^2

REPO="$(git rev-parse --show-toplevel)"; cd "$REPO"
conda activate lstm-kf
OUT="$REPO/outputs/lstmkf_64000"; LOG="$OUT/train.log"; mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0
nohup python -u lstmkf_train.py \
  --csv "$REPO/data/ngsim_processed_samples_64000.csv" \
  --stats-json "$REPO/data/ngsim_stats.json" \
  --hist-len 25 --future-len 25 \
  --batch-size 64 --epochs 20 \
  --lr 3e-4 \
  --f-hidden 64 --q-hidden 64 --r-hidden 32 --f-layers 2 \
  --device cuda \
  --out "$OUT" \
  --split-by vehicle --val-split 0.1 --test-split 0.1 \
  --save-splits \
  > "$LOG" 2>&1 &

  tail -n 50 -f "$LOG"

  实时日志：tail -n 100 -f "$REPO/outputs/lstmkf_64000/train.log"

最佳权重：$REPO/outputs/lstmkf_64000/best.pt

最终可视化：$REPO/outputs/lstmkf_64000/final_plot.png

TensorBoard：
tensorboard --logdir "$REPO/outputs/lstmkf_640/runs" --port 6006 --bind_all

test
python lstmkf_train.py \
  --csv "$REPO/data/ngsim_processed_samples_640.csv" \
  --stats-json "$REPO/data/ngsim_stats.json" \
  --hist-len 25 --future-len 25 \
  --batch-size 64 --device cuda \
  --out "$REPO/outputs/lstmkf_640_eval" \
  --splits-json "$REPO/outputs/lstmkf_640/splits.json" \
  --eval-test --eval-only --resume "$REPO/outputs/lstmkf_640/best.pt"

"""


import os
import json
import argparse
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ====== AMP 推荐 API ======
from torch.amp import GradScaler, autocast


# ---------------------------
# Dataset utilities
# ---------------------------

def _xy_from_21(arr_21: np.ndarray) -> np.ndarray:
    """(T,21) -> (T,14)  取每帧7个槽位的 (x,y)，丢弃 mask。"""
    T, D = arr_21.shape
    assert D == 21, f"expect 21, got {D}"
    out = np.zeros((T, 14), dtype=np.float32)
    idx3 = np.arange(0, 21, 3)   # [0,3,6,9,12,15,18]
    idx2 = np.arange(0, 14, 2)   # [0,2,4,6,8,10,12]
    for k in range(7):
        out[:, idx2[k]    ] = arr_21[:, idx3[k]    ]
        out[:, idx2[k] + 1] = arr_21[:, idx3[k] + 1]
    return out


def _ensure_2d(arr: np.ndarray, T: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        D = arr.size // T
        return arr.reshape(T, D)
    return arr


class ProcessedNGSIMDataset(Dataset):
    """
    读取预处理CSV（每行一个样本）。
    返回：
      x_hist_xy      (H,14)
      y_fut_xy       (F,14)
      y_fut_full21   (F,21)  用于mask
      x_hist_full21  (H,21)  可视化/选样
      vid            ()
      start_frame    ()
    """
    def __init__(self, csv_path, hist_len, fut_len):
        df = pd.read_csv(csv_path)
        if 'hist_len' in df.columns:
            df = df[df['hist_len'] == hist_len]
        if 'fut_len' in df.columns:
            df = df[df['fut_len'] == fut_len]
        self.df = df.reset_index(drop=True)
        self.H, self.F = hist_len, fut_len
        self.xcol = self.df["x_hist_flat"].tolist()
        self.ycol = self.df["y_fut_flat"].tolist()
        self.has_vid   = "Vehicle_ID"  in self.df.columns
        self.has_start = "Start_Frame" in self.df.columns
        self.vids   = self.df["Vehicle_ID"].astype(int).to_numpy() if self.has_vid else np.full(len(self.df), -1, dtype=int)
        self.starts = self.df["Start_Frame"].astype(int).to_numpy() if self.has_start else np.full(len(self.df), -1, dtype=int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_raw = np.fromstring(self.xcol[idx], sep=' ', dtype=np.float32)
        y_raw = np.fromstring(self.ycol[idx], sep=' ', dtype=np.float32)
        x_any = _ensure_2d(x_raw, self.H)     # (H, Dx)
        y_any = _ensure_2d(y_raw, self.F)     # (F, Dy)

        Dx = x_any.shape[1]; Dy = y_any.shape[1]

        # history
        if Dx == 21:
            x_hist_full21 = x_any
            x_hist_xy     = _xy_from_21(x_any)
        elif Dx == 14:
            x_hist_xy     = x_any
            x_hist_full21 = np.zeros((self.H, 21), dtype=np.float32)
            idx3 = np.arange(0, 21, 3)
            idx2 = np.arange(0, 14, 2)
            for k in range(7):
                x_hist_full21[:, idx3[k]    ] = x_any[:, idx2[k]    ]
                x_hist_full21[:, idx3[k] + 1] = x_any[:, idx2[k] + 1]
                x_hist_full21[:, idx3[k] + 2] = 1.0
        else:
            raise ValueError(f"Unsupported x per-frame dim {Dx}. Expect 14 or 21.")

        # future
        if Dy == 21:
            y_fut_full21 = y_any
            y_fut_xy     = _xy_from_21(y_any)
        elif Dy == 14:
            y_fut_xy     = y_any
            y_fut_full21 = np.zeros((self.F, 21), dtype=np.float32)
            idx3 = np.arange(0, 21, 3)
            idx2 = np.arange(0, 14, 2)
            for k in range(7):
                y_fut_full21[:, idx3[k]    ] = y_any[:, idx2[k]    ]
                y_fut_full21[:, idx3[k] + 1] = y_any[:, idx2[k] + 1]
                y_fut_full21[:, idx3[k] + 2] = 1.0
        else:
            raise ValueError(f"Unsupported y per-frame dim {Dy}. Expect 14 or 21.")

        vid = int(self.vids[idx]); stf = int(self.starts[idx])
        return (
            torch.from_numpy(x_hist_xy),       # (H,14)
            torch.from_numpy(y_fut_xy),        # (F,14)
            torch.from_numpy(y_fut_full21),    # (F,21)
            torch.from_numpy(x_hist_full21),   # (H,21)
            torch.tensor(vid, dtype=torch.long),
            torch.tensor(stf, dtype=torch.long),
        )


# ---------------------------
# Model (LSTM-KF)
# ---------------------------

class TransitionLSTM(nn.Module):
    def __init__(self, input_dim, state_dim, hidden_dim, num_layers, future_len, dropout=0.0):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(state_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.hidden2state = nn.Linear(hidden_dim, state_dim)
        self.post_linear_dropout = nn.Dropout(p=dropout)
        self.future_len = future_len
        self.state_dim = state_dim

    def forward(self, x_hist_xy):
        _, (h_n, c_n) = self.encoder(x_hist_xy)
        dec_in = x_hist_xy[:, -1:, :self.state_dim]
        outs, h, c = [], h_n, c_n
        for _ in range(self.future_len):
            o, (h, c) = self.decoder(dec_in, (h, c))
            s = self.hidden2state(o)
            s = self.post_linear_dropout(s)
            outs.append(s)
            dec_in = s
        return torch.cat(outs, dim=1)  # (B,T,state_dim)


class LSTM_F_Diag(nn.Module):
    """F_diag = 1 + alpha * tanh(raw) 让 F 近似单位阵，稳定一些。"""
    def __init__(self, input_dim, hidden, layers, state_dim, alpha=0.3, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=0.0)
        self.proj = nn.Linear(hidden, state_dim)
        self.post_linear_dropout = nn.Dropout(p=dropout)
        self.alpha = alpha
    def forward(self, x_seq):
        h, _ = self.lstm(x_seq)
        raw = self.proj(h)
        raw = self.post_linear_dropout(raw)
        return 1.0 + self.alpha * torch.tanh(raw)


class LSTM_Q_Noise(nn.Module):
    def __init__(self, input_dim, q_hidden, q_layer, state_dim, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, q_hidden, q_layer, batch_first=True)
        self.hidden2state = nn.Linear(q_hidden, state_dim)
        self.post_linear_dropout = nn.Dropout(p=dropout)
    def forward(self, x_seq):
        out, _ = self.lstm(x_seq)
        q_lin = self.hidden2state(out)
        q_lin = self.post_linear_dropout(q_lin)
        return F.softplus(q_lin) + 1e-6  # 下界会再由 clamp 控，避免过小


class LSTM_R_Noise(nn.Module):
    def __init__(self, input_dim, r_hidden, r_layer, state_dim, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, r_hidden, r_layer, batch_first=True)
        self.hidden2state = nn.Linear(r_hidden, state_dim)
        self.post_linear_dropout = nn.Dropout(p=dropout)
    def forward(self, z_xy):
        out, _ = self.lstm(z_xy)
        r_lin = self.hidden2state(out)
        r_lin = self.post_linear_dropout(r_lin)
        return F.softplus(r_lin) + 1e-6  # 下界会再由 clamp 控，避免过小


# ====== 数值稳健求解 S X = B ======
def _solve_spd_safe(S, B, base_eps=1e-6, max_tries=5):
    """
    期望 S 为对称正定，但可能数值不佳。
    策略：对称化 -> 按迹规模化对角加载 -> Cholesky -> 自适应增大 -> 失败则 pinv 回退
    S: (B_g, m, m), B: (B_g, m, d)
    返回 X: (B_g, m, d)
    """
    device, dtype = S.device, S.dtype
    # 强制对称
    S = 0.5 * (S + S.transpose(-1, -2))

    # 按规模设置初始 eps（与 trace 成比例）
    tr = torch.clamp(S.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True), min=1e-6)  # (B_g,1)
    eps = (base_eps * tr).view(-1, 1, 1).to(dtype=dtype, device=device)

    I = torch.eye(S.size(-1), device=device, dtype=dtype).unsqueeze(0).expand_as(S)

    for _ in range(max_tries):
        S_reg = S + eps * I
        L, info = torch.linalg.cholesky_ex(S_reg)  # (B_g,m,m), info==0 表示成功
        ok = (info == 0)
        if ok.all():
            # 解 L L^T X = B
            X = torch.cholesky_solve(B, L)  # (B_g, m, d)
            return X
        eps = eps * 10.0  # 增大 eps 再试

    # 仍失败：伪逆回退
    S_pinv = torch.linalg.pinv(S + eps * I, hermitian=True)
    return S_pinv @ B


class LSTMKF(nn.Module):
    """Kalman 反向传播展开（按可见维度子选择进行更新），含数值稳健处理。"""
    def __init__(self, f_net, q_net, r_net, fdiag_net=None, jitter=1e-4, p0_scale=1.0,
                 min_q=1e-5, min_r=1e-4):
        super().__init__()
        self.f_net = f_net
        self.q_net = q_net
        self.r_net = r_net
        self.fdiag_net = fdiag_net
        self.jitter = jitter
        self.p0_scale = p0_scale
        self.min_q = min_q
        self.min_r = min_r

    @staticmethod
    def _symmetrize(P):
        return 0.5 * (P + P.transpose(-1, -2))

    def forward(self, x_hist_xy, z_seq_xy, z_mask_full21=None, P0=None):
        """
        x_hist_xy : (B,H,14)
        z_seq_xy  : (B,T,14)
        z_mask_full21 : (B,T,21) with (x,y,mask)*7
        """
        B, T, D = z_seq_xy.shape
        device, dtype = z_seq_xy.device, z_seq_xy.dtype
        I_full = torch.eye(D, device=device, dtype=dtype).unsqueeze(0).expand(B, D, D)

        # 预测网络
        x_prior = self.f_net(x_hist_xy)                               # (B,T,14)
        q_diag  = torch.clamp(self.q_net(x_prior), min=self.min_q)    # (B,T,14)
        r_diag  = torch.clamp(self.r_net(z_seq_xy), min=self.min_r)   # (B,T,14)
        f_diag_all = self.fdiag_net(x_prior) if self.fdiag_net is not None else None

        # mask
        if z_mask_full21 is not None:
            mask7  = z_mask_full21[..., 2::3].float()
            mask14 = torch.repeat_interleave(mask7, 2, dim=-1)
        else:
            mask14 = torch.ones_like(z_seq_xy, dtype=torch.float32)

        # 初始协方差
        P_prev = (self.p0_scale * I_full).clone() if P0 is None else P0.clone()

        x_post_list, P_post_list = [], []

        for t in range(T):
            xp_t   = x_prior[:, t, :]
            Qt_t   = torch.diag_embed(q_diag[:, t, :])
            if f_diag_all is None:
                P_pr_t = P_prev + Qt_t
            else:
                F_t = torch.diag_embed(f_diag_all[:, t, :])
                P_pr_t = F_t @ P_prev @ F_t.transpose(1, 2) + Qt_t

            x_post_t = xp_t.clone()
            P_post_t = P_pr_t.clone()

            obs_mask_bt = (mask14[:, t, :] > 0.5)   # (B,D)
            m_counts    = obs_mask_bt.sum(dim=1)    # (B,)
            unique_m = torch.unique(m_counts)

            for m in unique_m.tolist():
                if m == 0:
                    continue
                sel = (m_counts == m)
                if not torch.any(sel):
                    continue

                P_prior_g = P_pr_t[sel]         # (B_g,D,D)
                xp_g      = xp_t[sel]           # (B_g,D)
                z_g       = z_seq_xy[sel, t, :] # (B_g,D)
                r_g       = r_diag[sel, t, :]   # (B_g,D)
                mask_g    = obs_mask_bt[sel]    # (B_g,D)

                B_g = P_prior_g.size(0)
                idx_mat = torch.stack([torch.where(mask_g[i])[0] for i in range(B_g)], dim=0)  # (B_g, m)

                # 选取子块
                P_ho  = torch.gather(P_prior_g, 2, idx_mat.unsqueeze(1).expand(B_g, D, m))     # (B_g,D,m)
                Prows = torch.gather(P_prior_g, 1, idx_mat.unsqueeze(-1).expand(B_g, m, D))     # (B_g,m,D)
                P_oo  = torch.gather(Prows,     2, idx_mat.unsqueeze(1).expand(B_g, m, m))     # (B_g,m,m)
                P_oo  = self._symmetrize(P_oo)  # 强制对称

                z_obs = torch.gather(z_g, 1, idx_mat)    # (B_g,m)
                x_obs = torch.gather(xp_g, 1, idx_mat)   # (B_g,m)
                r_obs = torch.gather(r_g, 1, idx_mat)    # (B_g,m)
                R_obs = torch.diag_embed(r_obs)          # (B_g,m,m)

                # 创新 & S
                innov = z_obs - x_obs                     # (B_g,m)
                I_m = torch.eye(m, device=device, dtype=dtype).unsqueeze(0)
                S = P_oo + R_obs + self.jitter * I_m      # (B_g,m,m)
                S = self._symmetrize(S)

                # 稳健求解 S X = P_ho^T
                X = _solve_spd_safe(S, P_ho.transpose(1, 2), base_eps=self.jitter, max_tries=5)  # (B_g,m,D)
                K = X.transpose(1, 2)  # (B_g,D,m)

                # 状态更新
                x_upd = xp_g + torch.bmm(K, innov.unsqueeze(-1)).squeeze(-1)

                # 协方差更新（K S K^T）
                KS   = torch.bmm(K, S)
                KSKT = torch.bmm(KS, K.transpose(1, 2))
                P_upd = self._symmetrize(P_prior_g - KSKT)

                x_post_t[sel] = x_upd
                P_post_t[sel] = P_upd

            x_post_list.append(x_post_t)
            P_post_list.append(P_post_t)
            P_prev = P_post_t

        x_post = torch.stack(x_post_list, dim=1)   # (B,T,14)
        P_post = torch.stack(P_post_list, dim=1)   # (B,T,14,14)
        return {"x_prior": x_prior, "x_post": x_post, "P_post": P_post, "q_diag": q_diag, "r_diag": r_diag}


# ---------------------------
# Loss & helpers
# ---------------------------

def build_mask14(mask21: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 (x, y, mask)*7 的 21 维掩码转换为:
      - mask7  : (B, T, 7)   每个槽位(车)一个可见标志
      - mask14 : (B, T, 14)  将每车的可见标志复制到该车的 (x, y) 两个坐标上
    """
    if mask21.size(-1) % 3 != 0:
        raise ValueError(f"mask21 的最后一维应为 21 (=7*3)，但得到 {mask21.size(-1)}")
    mask7 = (mask21[..., 2::3] > 0.5).to(mask21.dtype)        # (B,T,7)
    mask14 = torch.repeat_interleave(mask7, 2, dim=-1)         # (B,T,14)
    return mask14, mask7


def masked_time_avg_mse(
    pred: torch.Tensor,      # (B,T,14)
    gt: torch.Tensor,        # (B,T,14)
    mask14: torch.Tensor,    # (B,T,14)
    *,
    norm_dim: bool = True,
    norm_time: bool = True,
    reduction: str = "mean",
    eps: float = 1e-6
) -> torch.Tensor:
    """公平、数值稳定的掩码 MSE。"""
    if pred.shape != gt.shape or pred.shape != mask14.shape:
        raise ValueError(f"pred/gt/mask14 形状需一致，得到 {pred.shape}, {gt.shape}, {mask14.shape}")

    m = mask14.to(dtype=pred.dtype)
    diff = (pred - gt) * m
    sq = (diff ** 2).sum(dim=-1)  # (B,T)

    if norm_dim:
        valid_dim = m.sum(dim=-1).clamp_min(eps)
        sq = sq / valid_dim

    if norm_time:
        valid_t = (m.sum(dim=-1) > 0).to(pred.dtype)  # (B,T)
        denom = valid_t.sum(dim=-1).clamp_min(1.0)    # (B,)
        loss_b = (sq * valid_t).sum(dim=-1) / denom
    else:
        loss_b = sq.mean(dim=-1)

    if reduction == "mean":
        return loss_b.mean()
    elif reduction == "sum":
        return loss_b.sum()
    elif reduction == "none":
        return loss_b
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


# ---------------------------
# Visualization
# ---------------------------

def _car(ax, y, x, w=4.0, l=7.0, color='k', fc='white', label=None, lw=1.0, alpha=0.8):
    rect = Rectangle((y - l/2, x - w/2), l, w, edgecolor=color, facecolor=fc,
                     lw=lw, alpha=alpha, zorder=3); ax.add_patch(rect)
    if label:
        ax.text(y, x + w*0.7, label, ha='center', va='bottom',
                fontsize=7, color=color, weight='bold', zorder=4)


def _plot_with_break(ax, pts, jump_ft, style='-', lw=1.8, color='gray', alpha=0.95, mask=None, label=None):
    T = pts.shape[0]; seg = []; used_label = False
    for t in range(T):
        if mask is not None and mask[t] < 0.5:
            if len(seg) >= 2:
                ax.plot(np.array(seg)[:,0], np.array(seg)[:,1], style, lw=lw, color=color, alpha=alpha,
                        label=(label if not used_label else None)); used_label = True
            seg = []; continue
        if t > 0 and len(seg) > 0:
            prev = seg[-1]; dist = np.linalg.norm(pts[t] - prev)
            if dist > jump_ft:
                if len(seg) >= 2:
                    ax.plot(np.array(seg)[:,0], np.array(seg)[:,1], style, lw=lw, color=color, alpha=alpha,
                            label=(label if not used_label else None)); used_label = True
                seg = []
        seg.append(pts[t])
    if len(seg) >= 2:
        ax.plot(np.array(seg)[:,0], np.array(seg)[:,1], style, lw=lw, color=color, alpha=alpha,
                label=(label if not used_label else None))


def _smooth_segments(pts, mask=None, jump_ft=30.0, win=5):
    T = pts.shape[0]
    if win <= 1: return pts.copy()
    win = int(win);  win += (win % 2 == 0)
    def smooth_1d(arr):
        k = win // 2
        pad = np.pad(arr, (k, k), mode='edge')
        ker = np.ones(win, dtype=np.float32) / win
        return np.convolve(pad, ker, mode='valid')
    valid = np.ones(T, dtype=bool) if mask is None else mask.astype(bool)
    breaks = np.zeros(T, dtype=bool)
    for t in range(1, T):
        if valid[t] and valid[t-1]:
            if np.linalg.norm(pts[t] - pts[t-1]) > jump_ft:
                breaks[t] = True
    sm = pts.copy(); start = 0
    while start < T:
        while start < T and (not valid[start]): start += 1
        if start >= T: break
        end = start
        while end+1 < T and valid[end+1] and (not breaks[end+1]): end += 1
        if end - start + 1 >= 3:
            seg = pts[start:end+1]
            sm[start:end+1, 0] = smooth_1d(seg[:,0])
            sm[start:end+1, 1] = smooth_1d(seg[:,1])
        start = end + 1
    return sm


def pick_good_sample(x_hist_full21, y_fut_full21,
                     min_ev_ratio=0.80, min_sv_ratio=0.20, min_good_svs=2) -> int:
    with torch.no_grad():
        B, H, _ = x_hist_full21.shape
        _, F, _ = y_fut_full21.shape
        mask_hist7 = x_hist_full21[..., 2::3].float()
        mask_fut7  = y_fut_full21[..., 2::3].float()
        mask_all   = torch.cat([mask_hist7, mask_fut7], dim=1)
        ev_ratio   = mask_all[..., 0].mean(dim=1)
        sv_ratio   = mask_all[..., 1:].mean(dim=1)
        good_svs   = (sv_ratio >= min_sv_ratio).sum(dim=1)
        ok  = (ev_ratio >= min_ev_ratio) & (good_svs >= min_good_svs)
        idx = torch.where(ok)[0]
        if idx.numel() > 0: return int(idx[0].item())
        score = good_svs * 1.0 + sv_ratio.mean(dim=1) * 0.1 + ev_ratio * 0.1
        return int(torch.argmax(score).item())


def plot_anchor_scene(
    x_hist, y_fut, out, stats_json_path,
    sample=0, lane_width_ft=12.0, save_path=None,
    y_view_half=80, x_view_half=25,
    y_fut_full21=None, x_hist_full21=None,
    anchor_where="hist_last", smooth_win=5, jump_ft=30.0,
    meta_title=None
):
    # 1) 读标准差以还原英尺
    with open(stats_json_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    try:
        std_x = float(stats["Local_X"]["std"])
        std_y = float(stats["Local_Y"]["std"])
    except Exception:
        print("[WARN] stats_json 缺少 Local_X/Local_Y.std，回退为 1.0。")
        std_x, std_y = 1.0, 1.0

    B, H, D = x_hist.shape
    _, T, D2 = y_fut.shape
    assert D == D2 and D % 2 == 0
    N = D // 2
    names = ["E","F","B","LF","LB","RF","RB"] if N == 7 else [f"C{j}" for j in range(N)]

    to_xy = lambda t, T_: t.detach().cpu().numpy().reshape(T_, N, 2)
    xh = to_xy(x_hist[sample], H)
    zf = to_xy(y_fut[sample], T)
    xp = to_xy(out["x_prior"][sample], T)
    xo = to_xy(out["x_post"][sample], T)

    def to_feet(a):
        a = a.copy()
        a[...,0] *= std_x  # lateral
        a[...,1] *= std_y  # longitudinal
        return a
    xh, zf, xp, xo = map(to_feet, [xh, zf, xp, xo])

    def swap(a):
        b = np.empty_like(a)
        b[...,0] = a[...,1]  # y
        b[...,1] = a[...,0]  # x
        return b
    xh_s, zf_s, xp_s, xo_s = map(swap, [xh, zf, xp, xo])

    if anchor_where == "future_first":
        anchor_pts = zf_s[0]
        if y_fut_full21 is not None:
            fut_full = y_fut_full21[sample].detach().cpu().numpy().reshape(T, 7, 3)
            anchor_mask7 = fut_full[0, :, 2]
        else:
            anchor_mask7 = (np.linalg.norm(anchor_pts, axis=1) > 1e-6).astype(np.float32)
    else:
        anchor_pts = xh_s[-1]
        if x_hist_full21 is not None:
            hist_full = x_hist_full21[sample].detach().cpu().numpy().reshape(H, 7, 3)
            anchor_mask7 = hist_full[-1, :, 2]
        else:
            anchor_mask7 = (np.linalg.norm(anchor_pts, axis=1) > 1e-6).astype(np.float32)

    if x_hist_full21 is not None:
        hist_full = x_hist_full21[sample].detach().cpu().numpy().reshape(H, 7, 3)
        hist_mask14 = np.repeat(hist_full[..., 2], 2, axis=1)
    else:
        hist_mask14 = np.repeat((np.linalg.norm(xh_s, axis=2) > 1e-6).astype(np.float32), 2, axis=1)

    if y_fut_full21 is not None:
        fut_full = y_fut_full21[sample].detach().cpu().numpy().reshape(T, 7, 3)
        fut_mask14 = np.repeat(fut_full[..., 2], 2, axis=1)
    else:
        fut_mask14 = np.repeat((np.linalg.norm(zf_s, axis=2) > 1e-6).astype(np.float32), 2, axis=1)

    xh_s_sm = xh_s.copy(); zf_s_sm = zf_s.copy()
    if smooth_win and smooth_win > 1:
        for j in range(N):
            m_hist = hist_mask14[:, 2*j] * hist_mask14[:, 2*j+1]
            xh_s_sm[:, j, :] = _smooth_segments(xh_s[:, j, :], mask=m_hist, jump_ft=jump_ft, win=smooth_win)
        for j in range(N):
            m_fut  = fut_mask14[:, 2*j] * fut_mask14[:, 2*j+1]
            zf_s_sm[:, j, :] = _smooth_segments(zf_s[:, j, :],  mask=m_fut,  jump_ft=jump_ft, win=smooth_win)

    fig, ax = plt.subplots(figsize=(9.2, 3.2), dpi=160, constrained_layout=True)
    if meta_title is not None and "veh" in meta_title and "frame" in meta_title:
        ax.set_title(f"Vehicle_ID={meta_title['veh']}  |  Frame_ID={meta_title['frame']}")
    else:
        ax.set_title("Anchor = selected-frame EV at (0,0)")
    ax.set_xlabel("y (feet)"); ax.set_ylabel("x (feet)")

    w = lane_width_ft
    for x_off in (-1.5*w, -0.5*w, +0.5*w, +1.5*w):
        ax.plot([-y_view_half, y_view_half], [x_off, x_off], '-', color='#BDBDBD', lw=1.2, zorder=1)

    _car(ax, 0.0, 0.0, w=4, l=7, color='k', fc='yellow', label='E')

    for j in range(1, N):
        if anchor_mask7[j] >= 0.5:
            y_anchor, x_anchor = anchor_pts[j, 0], anchor_pts[j, 1]
            _car(ax, y_anchor, x_anchor, w=4, l=7, color='gray', fc='white', label=names[j])

    ev_hist_mask = hist_mask14[:, 0] * hist_mask14[:, 1]
    ev_fut_mask  = fut_mask14[:,  0] * fut_mask14[:,  1]
    _plot_with_break(ax, xh_s_sm[:, 0, :], jump_ft, style='-',  color='black', lw=2.0, alpha=1.0, mask=ev_hist_mask, label='EV history')
    _plot_with_break(ax, zf_s_sm[:, 0, :], jump_ft, style='-',  color='green', lw=2.0, alpha=1.0, mask=ev_fut_mask,  label='EV future GT')
    _plot_with_break(ax, xo_s[:,    0, :], jump_ft, style='--', color='blue',  lw=2.0, alpha=1.0, mask=ev_fut_mask,  label='EV future Pred')

    for j in range(1, N):
        m_hist = hist_mask14[:, 2*j] * hist_mask14[:, 2*j+1]
        m_fut  = fut_mask14[:, 2*j] * fut_mask14[:, 2*j+1]
        _plot_with_break(ax, xh_s_sm[:, j, :], jump_ft, style='-',  color='gray', lw=1.2, alpha=0.9, mask=m_hist,
                         label=('SV history' if j == 1 else None))
        _plot_with_break(ax, zf_s_sm[:, j, :], jump_ft, style='--', color='gray', lw=1.2, alpha=0.9, mask=m_fut,
                         label=('SV future' if j == 1 else None))

    ax.set_xlim(-y_view_half, y_view_half)
    ax.set_ylim(-x_view_half, x_view_half)
    ax.grid(True, alpha=0.3, zorder=0)

    handles, labels = ax.get_legend_handles_labels()
    if len(labels) == 0:
        dummy = [
            Line2D([0],[0], color='black', lw=2.0, linestyle='-',  label='EV history'),
            Line2D([0],[0], color='green', lw=2.0, linestyle='-',  label='EV future GT'),
            Line2D([0],[0], color='blue',  lw=2.0, linestyle='--', label='EV future Pred'),
            Line2D([0],[0], color='gray',  lw=1.2, linestyle='-',  label='SV history'),
            Line2D([0],[0], color='gray',  lw=1.2, linestyle='--', label='SV future'),
        ]
        ax.legend(handles=dummy, loc='upper left', fontsize=9, frameon=True)
    else:
        ax.legend(loc='upper left', fontsize=9, frameon=True)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print("Saved figure to:", save_path)
        plt.close(fig)
    else:
        plt.show(); plt.close(fig)


# ---------------------------
# Train / Val / Test split helpers
# ---------------------------

def split_indices_by_vehicle(df, val_split, test_split, seed):
    rng = np.random.default_rng(seed)
    if "Vehicle_ID" not in df.columns:
        raise KeyError("vehicle 切分需要 CSV 含 Vehicle_ID 列")
    vids = df["Vehicle_ID"].unique()
    rng.shuffle(vids)
    n_test = max(1, int(len(vids) * test_split))
    n_val  = max(1, int(len(vids) * val_split))
    test_vids = set(vids[:n_test])
    val_vids  = set(vids[n_test:n_test+n_val])
    test_idx = df.index[df["Vehicle_ID"].isin(test_vids)].tolist()
    val_idx  = df.index[df["Vehicle_ID"].isin(val_vids)].tolist()
    all_idx  = set(df.index.tolist())
    train_idx = sorted(all_idx - set(test_idx) - set(val_idx))
    return train_idx, val_idx, test_idx


def split_indices_by_time(df, val_split, test_split):
    if "Vehicle_ID" not in df.columns or "Start_Frame" not in df.columns:
        raise KeyError("time 切分需要 CSV 含 Vehicle_ID 与 Start_Frame 列")
    lower = max(0.0, 1.0 - test_split - val_split)
    upper = max(0.0, 1.0 - test_split)
    tr_all, va_all, te_all = [], [], []
    for _, g in df.groupby("Vehicle_ID"):
        q_low = g["Start_Frame"].quantile(lower)
        q_up  = g["Start_Frame"].quantile(upper)
        te = g.index[g["Start_Frame"] >= q_up].tolist()
        va = g.index[(g["Start_Frame"] >= q_low) & (g["Start_Frame"] < q_up)].tolist()
        tr = g.index[g["Start_Frame"] < q_low].tolist()
        tr_all.extend(tr); va_all.extend(va); te_all.extend(te)
    return sorted(tr_all), sorted(va_all), sorted(te_all)


def split_indices_random(n, val_split, test_split, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(n * test_split))
    n_val  = max(1, int(n * val_split))
    test_idx = perm[:n_test].tolist()
    val_idx  = perm[n_test:n_test+n_val].tolist()
    train_idx= perm[n_test+n_val:].tolist()
    return train_idx, val_idx, test_idx


# ---------------------------
# Train / Eval loops
# ---------------------------

def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def get_device(arg: str):
    if arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def build_model(args, device):
    f_net = TransitionLSTM(
        input_dim=14, state_dim=14,
        hidden_dim=args.f_hidden, num_layers=args.f_layers,
        future_len=args.future_len, dropout=args.dropout
    )
    q_net = LSTM_Q_Noise(input_dim=14, q_hidden=args.q_hidden, q_layer=1, state_dim=14, dropout=args.dropout)
    r_net = LSTM_R_Noise(input_dim=14, r_hidden=args.r_hidden, r_layer=1, state_dim=14, dropout=args.dropout)
    fdiag_net = None if args.no_fdiag else LSTM_F_Diag(input_dim=14, hidden=args.fdiag_hidden, layers=1, state_dim=14, alpha=0.3, dropout=args.dropout)
    model = LSTMKF(f_net, q_net, r_net,
                   fdiag_net=fdiag_net,
                   jitter=args.jitter,
                   p0_scale=args.p0_scale,
                   min_q=args.min_q,
                   min_r=args.min_r)

    model = model.to(device)
    if args.compile:
        try:
            model = torch.compile(model)
            print("[INFO] torch.compile 已启用")
        except Exception as e:
            print(f"[WARN] torch.compile 失败：{e}")
    if args.dp and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"[INFO] DataParallel 启用：{torch.cuda.device_count()} GPUs")
    return model


def train_one_epoch(model, loader, optimizer, device, scaler, args, writer=None, epoch_idx=0):
    is_cuda = (device.type == "cuda")
    model.train()
    total_loss = 0.0; n_samples = 0
    total_post = 0.0; total_prior = 0.0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        x_hist_xy, y_fut_xy, y_fut_full21, x_hist_full21, vids, starts = batch
        x_hist_xy    = x_hist_xy.to(device, non_blocking=is_cuda).float()
        y_fut_xy     = y_fut_xy.to(device, non_blocking=is_cuda).float()
        y_fut_full21 = y_fut_full21.to(device, non_blocking=is_cuda).float()

        with autocast("cuda", enabled=(is_cuda and args.amp)):
            out = model(x_hist_xy, y_fut_xy, z_mask_full21=y_fut_full21, P0=None)
            mask14, _ = build_mask14(y_fut_full21)
            loss_post  = masked_time_avg_mse(out["x_post"],  y_fut_xy, mask14, norm_dim=True, norm_time=True)
            loss_prior = masked_time_avg_mse(out["x_prior"], y_fut_xy, mask14, norm_dim=True, norm_time=True)
            loss = loss_post + args.lambda_prior * loss_prior
            loss = loss / max(1, args.accum_steps)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % args.accum_steps == 0:
            if args.grad_clip > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters() if not isinstance(model, nn.DataParallel) else model.module.parameters(),
                    args.grad_clip
                )
            if scaler.is_enabled():
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = x_hist_xy.size(0)
        total_loss += loss.item() * bs * max(1, args.accum_steps)
        total_post += loss_post.item() * bs
        total_prior+= loss_prior.item() * bs
        n_samples  += bs

    avg = total_loss / max(n_samples, 1)
    avg_post = total_post / max(n_samples, 1)
    avg_prior= total_prior/ max(n_samples, 1)
    if writer:
        writer.add_scalar("train/loss", avg, epoch_idx)
        writer.add_scalar("train/loss_post", avg_post, epoch_idx)
        writer.add_scalar("train/loss_prior", avg_prior, epoch_idx)
    return avg


@torch.no_grad()
def evaluate_epoch(model, loader, device, args, writer=None, epoch_idx=0, tag_prefix="val"):
    is_cuda = (device.type == "cuda")
    model.eval()
    total_loss = 0.0; n_samples = 0
    total_post = 0.0; total_prior = 0.0

    for x_hist_xy, y_fut_xy, y_fut_full21, x_hist_full21, vids, starts in loader:
        x_hist_xy    = x_hist_xy.to(device, non_blocking=is_cuda).float()
        y_fut_xy     = y_fut_xy.to(device, non_blocking=is_cuda).float()
        y_fut_full21 = y_fut_full21.to(device, non_blocking=is_cuda).float()

        out = model(x_hist_xy, y_fut_xy, z_mask_full21=y_fut_full21, P0=None)
        mask14, _ = build_mask14(y_fut_full21)
        loss_post  = masked_time_avg_mse(out["x_post"],  y_fut_xy, mask14, norm_dim=True, norm_time=True)
        loss_prior = masked_time_avg_mse(out["x_prior"], y_fut_xy, mask14, norm_dim=True, norm_time=True)
        loss = loss_post + args.lambda_prior * loss_prior

        bs = x_hist_xy.size(0)
        total_loss += loss.item() * bs; n_samples += bs
        total_post += loss_post.item() * bs
        total_prior+= loss_prior.item() * bs

    avg = total_loss / max(n_samples, 1)
    avg_post = total_post / max(n_samples, 1)
    avg_prior= total_prior/ max(n_samples, 1)
    if writer:
        writer.add_scalar(f"{tag_prefix}/loss", avg, epoch_idx)
        writer.add_scalar(f"{tag_prefix}/loss_post", avg_post, epoch_idx)
        writer.add_scalar(f"{tag_prefix}/loss_prior", avg_prior, epoch_idx)
    return avg


# ---------------------------
# Main
# ---------------------------

def main():
    p = argparse.ArgumentParser("Train LSTM-KF (Euclidean loss Eq.21) with TB logging")
    p.add_argument("--csv", type=str, required=True, help="path to ngsim_processed_samples.csv")
    p.add_argument("--stats-json", type=str, required=True, help="path to ngsim_stats.json (for feet plotting)")
    p.add_argument("--hist-len", type=int, default=25)
    p.add_argument("--future-len", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--test-split", type=float, default=0.1, help="预留测试集比例（训练期间不使用）")
    p.add_argument("--split-by", type=str, default="vehicle", choices=["vehicle", "time", "random"],
                   help="vehicle=按 Vehicle_ID，time=每车末尾片段做test，random=随机样本")
    p.add_argument("--save-splits", action="store_true", help="保存索引到 out/splits.json 便于复现")
    p.add_argument("--splits-json", type=str, default="", help="从已有 splits.json 复用切分（优先于 --split-by/*split）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true", help="严格可复现（可能更慢）")
    p.add_argument("--out", type=str, default="./outputs/lstmkf_euclid")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--amp", action="store_true", help="enable mixed precision on CUDA")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-prior", type=float, default=0.8, help="λ in Eq.(21), weight on prior term")
    # model sizes
    p.add_argument("--f-hidden", type=int, default=64)
    p.add_argument("--f-layers", type=int, default=2)
    p.add_argument("--q-hidden", type=int, default=64)
    p.add_argument("--r-hidden", type=int, default=32)
    p.add_argument("--fdiag-hidden", type=int, default=64)
    p.add_argument("--no-fdiag", action="store_true", help="禁用F对角LSTM（用 I + Q ）")
    # resume / eval-only / eval-test
    p.add_argument("--resume", type=str, default="", help="加载已训练权重 (best.pt) 后再训练或仅评估")
    p.add_argument("--eval-only", action="store_true", help="只评估(不训练)——需配合 --resume")
    p.add_argument("--eval-test", action="store_true", help="训练或加载后，额外在测试集上评估并出图")
    # GPU 利用与稳定性
    p.add_argument("--accum-steps", type=int, default=1, help="梯度累积步数，用于等效更大batch")
    p.add_argument("--dp", action="store_true", help="启用 DataParallel 多卡（简易版）")
    p.add_argument("--compile", action="store_true", help="尝试 torch.compile")
    p.add_argument("--jitter", type=float, default=1e-3, help="KF S 矩阵正则（对角加载规模）")
    p.add_argument("--p0-scale", type=float, default=1.0, help="初始协方差比例 P0 = scale * I")
    p.add_argument("--min-q", type=float, default=1e-5, help="Q 对角元素下界")
    p.add_argument("--min-r", type=float, default=1e-4, help="R 对角元素下界（观测更保守）")

    args = p.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # dataset
    full = ProcessedNGSIMDataset(args.csv, args.hist_len, args.future_len)
    assert len(full) > 0, "Dataset is empty. Did you generate ngsim_processed_samples.csv?"
    df = full.df

    # compute or load splits
    if args.splits_json and os.path.isfile(args.splits_json):
        with open(args.splits_json, "r", encoding="utf-8") as f:
            sp = json.load(f)
        train_idx, val_idx, test_idx = sp["train_idx"], sp["val_idx"], sp["test_idx"]
        print(f"[INFO] Loaded splits from {args.splits_json} (split_by={sp.get('split_by','n/a')})")
    else:
        try:
            if args.split_by == "vehicle":
                train_idx, val_idx, test_idx = split_indices_by_vehicle(df, args.val_split, args.test_split, args.seed)
            elif args.split_by == "time":
                train_idx, val_idx, test_idx = split_indices_by_time(df, args.val_split, args.test_split)
            else:
                train_idx, val_idx, test_idx = split_indices_random(len(df), args.val_split, args.test_split, args.seed)
        except KeyError as e:
            print(f"[WARN] {e}，回退到随机切分。")
            train_idx, val_idx, test_idx = split_indices_random(len(df), args.val_split, args.test_split, args.seed)

    assert len(set(train_idx) & set(val_idx))  == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx)  & set(test_idx))  == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)

    if args.save_splits and not args.splits_json:
        with open(os.path.join(args.out, "splits.json"), "w", encoding="utf-8") as f:
            json.dump({
                "split_by": args.split_by,
                "val_split": args.val_split,
                "test_split": args.test_split,
                "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
            }, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved splits to {os.path.join(args.out, 'splits.json')}")

    # loaders
    train_set = Subset(full, train_idx)
    val_set   = Subset(full, val_idx)
    test_set  = Subset(full, test_idx)

    pin = (device.type == "cuda")
    pw  = (args.workers > 0)
    dl_kwargs = dict(pin_memory=pin, persistent_workers=pw)
    if pw and args.prefetch_factor is not None and args.prefetch_factor > 0:
        dl_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=False, **dl_kwargs)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, drop_last=False, **dl_kwargs)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, drop_last=False, **dl_kwargs)

    # model & optim
    model = build_model(args, device)
    if args.resume and os.path.isfile(args.resume):
        sd = torch.load(args.resume, map_location=device)
        if isinstance(model, nn.DataParallel):
            model_to_load = model.module
        else:
            model_to_load = model
        if isinstance(sd, dict) and "state_dict" not in sd and "model_state" not in sd:
            model_to_load.load_state_dict(sd)
        else:
            model_to_load.load_state_dict(sd if isinstance(sd, dict) and "state_dict" not in sd else sd.get("model_state", sd))
        print(f"[INFO] Loaded weights from {args.resume}")

    optimizer = torch.optim.Adam(
        model.parameters() if not isinstance(model, nn.DataParallel) else model.module.parameters(),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    # TB
    writer = SummaryWriter(os.path.join(args.out, "runs"))
    with open(os.path.join(args.out, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    best_val = float("inf")

    # ---- train or eval-only ----
    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, args, writer, epoch)
            val_loss   = evaluate_epoch(model, val_loader, device, args, writer, epoch, tag_prefix="val")
            print(f"[Epoch {epoch:03d}/{args.epochs}] train={train_loss:.6f} | val={val_loss:.6f}")

            # save last
            to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                "epoch": epoch,
                "model_state": to_save,
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
            }, os.path.join(args.out, "last.pth"))

            if val_loss < best_val:
                best_val = val_loss
                torch.save(to_save, os.path.join(args.out, "best.pt"))
                print(f"[INFO] New best checkpoint saved (val={best_val:.6f})")
    else:
        # 仅评估，不训练
        val_loss = evaluate_epoch(model, val_loader, device, args, writer=None, epoch_idx=0, tag_prefix="val")
        print(f"[EVAL-ONLY] val/loss = {val_loss:.6f}")

    writer.close()

    # ---------------------
    # Final visualization on VAL
    # ---------------------
    if len(val_set) > 0:
        model.eval()
        try:
            x_hist_xy, y_fut_xy, y_fut_full21, x_hist_full21, vids, starts = next(iter(val_loader))
            x_hist_xy    = x_hist_xy.to(device).float()
            y_fut_xy     = y_fut_xy.to(device).float()
            y_fut_full21 = y_fut_full21.to(device).float()
            out = model(x_hist_xy, y_fut_xy, z_mask_full21=y_fut_full21, P0=None)

            idx = pick_good_sample(x_hist_full21, y_fut_full21)
            x1  = x_hist_xy[idx:idx+1].cpu()
            y1  = y_fut_xy[idx:idx+1].cpu()
            o1  = {k: (v[idx:idx+1].detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
            y21 = y_fut_full21[idx:idx+1].cpu()
            x21 = x_hist_full21[idx:idx+1].cpu()
            meta = {"veh": int(vids[idx].item()), "frame": int(starts[idx].item())}
            save_png = os.path.join(args.out, "final_plot.png")
            plot_anchor_scene(
                x1, y1, o1,
                stats_json_path=args.stats_json,
                sample=0,
                lane_width_ft=12.0,
                save_path=save_png,
                y_view_half=80,
                x_view_half=25,
                y_fut_full21=y21,
                x_hist_full21=x21,
                anchor_where="hist_last",
                smooth_win=5,
                jump_ft=30.0,
                meta_title=meta
            )
        except StopIteration:
            print("[WARN] val_loader is empty; skip final visualization.")

    # ---------------------
    # (Optional) Evaluate & visualize on TEST
    # ---------------------
    if args.eval_test and len(test_set) > 0:
        test_loss = evaluate_epoch(model, test_loader, device, args, writer=None, epoch_idx=0, tag_prefix="test")
        print(f"[TEST] test/loss = {test_loss:.6f}")
        try:
            x_hist_xy, y_fut_xy, y_fut_full21, x_hist_full21, vids, starts = next(iter(test_loader))
            x_hist_xy    = x_hist_xy.to(device).float()
            y_fut_xy     = y_fut_xy.to(device).float()
            y_fut_full21 = y_fut_full21.to(device).float()
            out = model(x_hist_xy, y_fut_xy, z_mask_full21=y_fut_full21, P0=None)

            idx = pick_good_sample(x_hist_full21, y_fut_full21)
            x1  = x_hist_xy[idx:idx+1].cpu()
            y1  = y_fut_xy[idx:idx+1].cpu()
            o1  = {k: (v[idx:idx+1].detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
            y21 = y_fut_full21[idx:idx+1].cpu()
            x21 = x_hist_full21[idx:idx+1].cpu()
            meta = {"veh": int(vids[idx].item()), "frame": int(starts[idx].item())}
            save_png = os.path.join(args.out, "final_plot_test.png")
            plot_anchor_scene(
                x1, y1, o1,
                stats_json_path=args.stats_json,
                sample=0,
                lane_width_ft=12.0,
                save_path=save_png,
                y_view_half=80,
                x_view_half=25,
                y_fut_full21=y21,
                x_hist_full21=x21,
                anchor_where="hist_last",
                smooth_win=5,
                jump_ft=30.0,
                meta_title=meta
            )
        except StopIteration:
            print("[WARN] test_loader is empty; skip test visualization.")


if __name__ == "__main__":
    main()
