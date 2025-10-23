# -*- coding: utf-8 -*-
"""
LSTM-KF（物理先验：纵向VT + 横向单车道中心）
- 修复点：
  1) E_lon/E_lat 广播维度错误 → 使用 coef.unsqueeze(-2)
  2) pick_good_sample() 运行在 CPU，避免 CPU/GPU 混用导致的 cat 报错
  3) 车道中心英尺→数据单位使用 Local_X（横向）的 mean/std
  4) 可视化在无显示环境下强制使用 Agg 后端，确保保存 PNG

用法示例：
REPO="$(git rev-parse --show-toplevel)"; cd "$REPO"
conda activate lstm-kf
OUT="$REPO/outputs/lstmkf_vt_64000"; LOG="$OUT/train_vt.log"; mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=1
nohup python -u lstmkf_train_vt.py \
  --csv "$REPO/data/ngsim_processed_samples_64000.csv" \
  --stats-json "$REPO/data/ngsim_stats.json" \
  --hist-len 25 --future-len 25 \
  --batch-size 64 --epochs 10 \
  --lr 3e-4 \
  --f-hidden 64 --q-hidden 64 --r-hidden 32 --f-layers 2 \
  --device cuda \
  --out "$OUT" \
  --split-by vehicle --val-split 0.1 --test-split 0.1 \
  --save-splits \
  --dt 0.1 \
  --lane-centers-feet=-12,0,12 \
  --viz-n 6 --viz-split val \
  > "$LOG" 2>&1 &
"""

import os
import json
import argparse
import random
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# 强制无显示后端，保证 nohup/服务器环境能存图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from torch.amp import GradScaler, autocast

# ---------------------------
# 数据集工具
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
# Q/R 网络
# ---------------------------

class LSTM_Q_Noise(nn.Module):
    def __init__(self, input_dim, q_hidden, q_layer, state_dim, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, q_hidden, q_layer, batch_first=True, dropout=dropout)
        self.hidden2state = nn.Linear(q_hidden, state_dim)
        self.post_linear_dropout = nn.Dropout(p=dropout)
    def forward(self, x_seq):
        out, _ = self.lstm(x_seq)
        q_lin = self.hidden2state(out)
        q_lin = self.post_linear_dropout(q_lin)
        return F.softplus(q_lin) + 1e-6

class LSTM_R_Noise(nn.Module):
    def __init__(self, input_dim, r_hidden, r_layer, state_dim, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, r_hidden, r_layer, batch_first=True, dropout=dropout)
        self.hidden2state = nn.Linear(r_hidden, state_dim)
        self.post_linear_dropout = nn.Dropout(p=dropout)
    def forward(self, z_xy):
        out, _ = self.lstm(z_xy)
        r_lin = self.hidden2state(out)
        r_lin = self.post_linear_dropout(r_lin)
        return F.softplus(r_lin) + 1e-6

# ---------------------------
# 车道选择（规则法：最近中心）
# ---------------------------

def choose_lane_idx_nearest(x_hist_xy: torch.Tensor, lane_centers: torch.Tensor, ev_x_idx: int):
    """
    x_hist_xy:   (B,H,14)
    lane_centers:(B,3) / (1,3) / (3,)
    return: lane_idx (B,) ∈ {0,1,2}
    """
    x_last = x_hist_xy[:, -1, ev_x_idx]  # (B,)  横向位置
    if lane_centers.dim() == 1:
        lane_centers = lane_centers.unsqueeze(0).expand(x_last.size(0), -1)  # (B,3)
    diff = torch.abs(x_last.unsqueeze(1) - lane_centers)  # (B,3)
    lane_idx = torch.argmin(diff, dim=1)                  # (B,)
    return lane_idx

def pick_lane_center(lane_centers: torch.Tensor, lane_idx: torch.Tensor):
    """
    lane_centers: (B,3) or (1,3) or (3,)
    lane_idx:     (B,)
    return: p_ref (B,1)
    """
    if lane_centers.dim() == 1:
        lane_centers = lane_centers.unsqueeze(0).expand(lane_idx.size(0), -1)
    p_ref = torch.gather(lane_centers, 1, lane_idx.view(-1,1))  # (B,1)
    return p_ref

# ---------------------------
# Policy 头：输出 K 与 v_ref（每段常数）
# ---------------------------

class PolicyParamHead_Simple(nn.Module):
    """
    输出：
      - K_lon_VT: (B,2)  → [k_v, k_a]（对 [v,a] 的反馈乘以 S → 3维）
      - K_lat   : (B,3)  → [k_p, k_v, k_a]
      - v_ref   : (B,1)
    """
    def __init__(self, enc_hidden, num_layers, k_scale=2.0, hidden_mlp=256):
        super().__init__()
        self.k_scale = k_scale
        out_dim = 2 + 3 + 1
        self.mlp = nn.Sequential(
            nn.Linear(enc_hidden * num_layers, hidden_mlp),
            nn.ReLU(),
            nn.Linear(hidden_mlp, out_dim)
        )
    def forward(self, h_n):
        B = h_n.shape[1]
        feat = h_n.transpose(0,1).reshape(B, -1)   # (B, enc_hidden*num_layers)
        raw  = self.mlp(feat)
        i=0
        K_lon_VT = torch.tanh(raw[:, i:i+2]) * self.k_scale; i+=2
        K_lat    = torch.tanh(raw[:, i:i+3]) * self.k_scale; i+=3
        v_ref    = raw[:, i:i+1]
        return {"K_lon_VT":K_lon_VT, "K_lat":K_lat, "v_ref":v_ref}

# ---------------------------
# 构造 F/E（纵向VT + 横向单车道中心）
# ---------------------------

def build_FE_vt_single_lane(params, T_s, p_ref_T, device, dtype, B, T):
    # 只在 batch/time 维复制；最后两维保持 3x3 / 3x1
    A_base = torch.tensor([[1, T_s, 0.5*T_s*T_s],
                           [0, 1,   T_s],
                           [0, 0,   1]], device=device, dtype=dtype)        # (3,3)
    Bm_base = torch.tensor([[0],[0.5*T_s*T_s],[T_s]], device=device, dtype=dtype)  # (3,1)
    A  = A_base.view(1,1,3,3).expand(B, T, 3, 3).clone().contiguous()      # (B,T,3,3)
    Bm = Bm_base.view(1,1,3,1).expand(B, T, 3, 1).clone().contiguous()     # (B,T,3,1)

    S  = torch.tensor([[0,1,0],[0,0,1]], device=device, dtype=dtype)  # (2,3)

    def tile_T(x, T):
        return x.unsqueeze(1).expand(x.size(0), T, x.size(-1)).contiguous()

    # 每段常数沿时间复制
    K_vt  = tile_T(params["K_lon_VT"], T)  # (B,T,2)
    K_lat = tile_T(params["K_lat"],    T)  # (B,T,3)
    vref  = tile_T(params["v_ref"],    T)  # (B,T,1)

    # 纵向
    KvS  = torch.einsum('bti,ij->btj', K_vt, S)              # (B,T,3)
    F_lon = A - torch.matmul(Bm, KvS.unsqueeze(-2))          # (B,T,3,3)
    coef_lon = torch.sum(K_vt * torch.cat([vref, torch.zeros_like(vref)], dim=-1),
                         dim=-1, keepdim=True)               # (B,T,1)
    E_lon = Bm * coef_lon.unsqueeze(-2)                      # (B,T,3,1)

    # 横向
    F_lat = A - torch.matmul(Bm, K_lat.unsqueeze(-2))        # (B,T,3,3)
    zeros = torch.zeros_like(p_ref_T)
    coef_lat = torch.sum(K_lat * torch.cat([p_ref_T, zeros, zeros], dim=-1),
                         dim=-1, keepdim=True)               # (B,T,1)
    E_lat = Bm * coef_lat.unsqueeze(-2)                      # (B,T,3,1)

    return F_lon, E_lon, F_lat, E_lat




# ---------------------------
# KF 数值稳健线性代数
# ---------------------------

def _solve_spd_safe(S, B, base_eps=1e-6, max_tries=5):
    """
    期望 S 为对称正定，但可能数值不佳。
    策略：对称化 -> 按迹规模化对角加载 -> Cholesky -> 自适应增大 -> 失败则 pinv 回退
    S: (B_g, m, m), B: (B_g, m, d)
    返回 X: (B_g, m, d)
    """
    device, dtype = S.device, S.dtype
    S = 0.5 * (S + S.transpose(-1, -2))
    tr = torch.clamp(S.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True), min=1e-6)  # (B_g,1)
    eps = (base_eps * tr).view(-1, 1, 1).to(dtype=dtype, device=device)
    I = torch.eye(S.size(-1), device=device, dtype=dtype).unsqueeze(0).expand_as(S)
    for _ in range(max_tries):
        S_reg = S + eps * I
        L, info = torch.linalg.cholesky_ex(S_reg)
        ok = (info == 0)
        if ok.all():
            X = torch.cholesky_solve(B, L)
            return X
        eps = eps * 10.0
    S_pinv = torch.linalg.pinv(S + eps * I, hermitian=True)
    return S_pinv @ B

# ---------------------------
# 主模型：单 KF + 物理先验（VT + 单车道）+ LSTM 学 K
# ---------------------------

class LSTMKF_PhysicsVT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, future_len,
                 q_net, r_net, policy_head,
                 jitter=1e-3, p0_scale=1.0, ev_xy_index=(0,1),
                 min_q=1e-5, min_r=1e-4, dropout=0.0):
        super().__init__()
        self.future_len = future_len
        self.q_net, self.r_net = q_net, r_net
        self.jitter, self.p0_scale = jitter, p0_scale
        self.ev_x_idx, self.ev_y_idx = ev_xy_index  # 跨脚本保持 (x,y) 顺序
        self.min_q, self.min_r = min_q, min_r
        # 避免 num_layers=1 时 dropout 警告
        real_dropout = 0.0 if num_layers == 1 else dropout
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=real_dropout)
        self.policy_head = policy_head

    @staticmethod
    def _symmetrize(P):
        return 0.5 * (P + P.transpose(-1, -2))

    @staticmethod
    def _cv_prior(x_hist_xy, T):
        """其它 12 维用恒速外推（差分求速度）。"""
        B, H, D = x_hist_xy.shape
        last = x_hist_xy[:, -1, :]
        vel  = (x_hist_xy[:, -1, :] - x_hist_xy[:, -2, :]) if H>=2 else torch.zeros_like(last)
        pri, cur = [], last
        for _ in range(T):
            cur = cur + vel
            pri.append(cur.unsqueeze(1))
        return torch.cat(pri, 1)  # (B,T,D)

    def forward(self, x_hist_xy, z_seq_xy, z_mask_full21=None, P0=None, env=None, T_s=0.1):
        """
        x_hist_xy : (B,H,14)
        z_seq_xy  : (B,T,14)
        z_mask_full21 : (B,T,21) with (x,y,mask)*7
        env: {"lane_centers": (3,) or (1,3) or (B,3)}  与 x（横向）同单位（数据单位）
        """
        assert env is not None and "lane_centers" in env, "env['lane_centers'] 必须提供（数据单位）"
        B, T, D = z_seq_xy.shape
        device, dtype = z_seq_xy.device, z_seq_xy.dtype

        # 1) 编码 + 预测 K 与 v_ref（每段常数）
        _, (h_n, c_n) = self.encoder(x_hist_xy)
        params = self.policy_head(h_n)  # {"K_lon_VT","K_lat","v_ref"}

        # 2) 选车道中心（最近中心），整段固定（基于横向 x）
        lane_centers = env["lane_centers"].to(device=device, dtype=dtype)  # (3,) or (1,3) or (B,3)
        lane_idx = choose_lane_idx_nearest(x_hist_xy, lane_centers, self.ev_x_idx)  # (B,)
        p_ref    = pick_lane_center(lane_centers, lane_idx)                          # (B,1)
        p_ref_T  = p_ref.unsqueeze(1).expand(B, T, 1).contiguous()                   # (B,T,1)

        # 3) 构造 F/E（VT + 单车道）
        F_lon, E_lon, F_lat, E_lat = build_FE_vt_single_lane(params, T_s, p_ref_T, device, dtype, B, T)

        # ---- 4) EV 6维递推（纯函数式 & 无就地改动） ----
        # 4.1 先用恒速外推得到基础先验
        x_prior_base = self._cv_prior(x_hist_xy, T)  # (B,T,14)

        # 4.2 EV 初值 [p,v,a]（纵、横），v/a=0
        p0_lon = x_hist_xy[:, -1, self.ev_x_idx:self.ev_x_idx+1]  # (B,1)
        p0_lat = x_hist_xy[:, -1, self.ev_y_idx:self.ev_y_idx+1]  # (B,1)
        xlon = torch.cat([p0_lon, torch.zeros_like(p0_lon), torch.zeros_like(p0_lon)], dim=-1)  # (B,3)
        xlat = torch.cat([p0_lat, torch.zeros_like(p0_lat), torch.zeros_like(p0_lat)], dim=-1)  # (B,3)

        ev_lon_list, ev_lat_list = [], []
        for t in range(T):
            Fl_t = F_lon[:, t].clone()    # (B,3,3)  克隆掉视图
            El_t = E_lon[:, t].clone()    # (B,3,1)
            Ft_t = F_lat[:, t].clone()
            Et_t = E_lat[:, t].clone()

            xlon = (Fl_t @ xlon.unsqueeze(-1) + El_t).squeeze(-1)  # (B,3)
            xlat = (Ft_t @ xlat.unsqueeze(-1) + Et_t).squeeze(-1)  # (B,3)

            ev_lon_list.append(xlon)
            ev_lat_list.append(xlat)

        ev_lon_traj = torch.stack(ev_lon_list, dim=1)     # (B,T,3)
        ev_lat_traj = torch.stack(ev_lat_list, dim=1)     # (B,T,3)
        ev_x_traj   = ev_lon_traj[..., 0]                 # (B,T)
        ev_y_traj   = ev_lat_traj[..., 0]                 # (B,T)

        # 4.3 不对切片做就地赋值，改为“通道重组”构造新张量
        # 拆 14 个通道 → 替换 EV 的 x/y → 再 stack 回来
        chans = list(torch.unbind(x_prior_base, dim=2))   # 14 个 [(B,T), ...]
        chans[self.ev_x_idx] = ev_x_traj
        chans[self.ev_y_idx] = ev_y_traj
        x_prior = torch.stack(chans, dim=2).contiguous()  # (B,T,14)



        # 5) KF 更新（数值稳健）
        q_diag  = torch.clamp(self.q_net(x_prior), min=self.min_q)     # (B,T,14)
        r_diag  = torch.clamp(self.r_net(z_seq_xy), min=self.min_r)    # (B,T,14)

        if z_mask_full21 is not None:
            mask7  = z_mask_full21[..., 2::3].float()
            mask14 = torch.repeat_interleave(mask7, 2, dim=-1)
        else:
            mask14 = torch.ones_like(z_seq_xy, dtype=torch.float32)

        I_full = torch.eye(D, device=device, dtype=dtype).unsqueeze(0).expand(B, D, D)
        P_prev = (self.p0_scale * I_full).clone() if P0 is None else P0.clone()

        x_post_list, P_post_list = [], []

        for t in range(T):
            xp_t   = x_prior[:, t, :]
            Qt_t   = torch.diag_embed(q_diag[:, t, :])
            P_pr_t = P_prev + Qt_t  # 简洁版本（不引入 Fdiag）

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

                Ddim = D
                P_ho  = torch.gather(P_prior_g, 2, idx_mat.unsqueeze(1).expand(B_g, Ddim, m))
                Prows = torch.gather(P_prior_g, 1, idx_mat.unsqueeze(-1).expand(B_g, m, Ddim))
                P_oo  = torch.gather(Prows,     2, idx_mat.unsqueeze(1).expand(B_g, m, m))
                P_oo  = self._symmetrize(P_oo)

                z_obs = torch.gather(z_g, 1, idx_mat)
                x_obs = torch.gather(xp_g, 1, idx_mat)
                r_obs = torch.gather(r_g, 1, idx_mat)
                R_obs = torch.diag_embed(r_obs)

                innov = z_obs - x_obs
                I_m = torch.eye(m, device=device, dtype=dtype).unsqueeze(0)
                S = self._symmetrize(P_oo + R_obs + self.jitter * I_m)

                # 稳健求解 S X = P_ho^T
                X = _solve_spd_safe(S, P_ho.transpose(1, 2), base_eps=self.jitter, max_tries=5)  # (B_g,m,D)
                K = X.transpose(1, 2)  # (B_g,D,m)

                x_upd = xp_g + torch.bmm(K, innov.unsqueeze(-1)).squeeze(-1)

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
        return {
            "x_prior": x_prior, "x_post": x_post, "P_post": P_post,
            "q_diag": q_diag, "r_diag": r_diag,
            "K_lon_VT": params["K_lon_VT"],     # (B,2)
            "K_lat":    params["K_lat"],        # (B,3)
            "v_ref":    params["v_ref"],        # (B,1)
            "lane_idx": lane_idx,               # (B,)
            "p_ref":    p_ref                   # (B,1)
        }

# ---------------------------
# 损失 / 掩码
# ---------------------------

def build_mask14(mask21: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if mask21.size(-1) % 3 != 0:
        raise ValueError(f"mask21 的最后一维应为 21 (=7*3)，但得到 {mask21.size(-1)}")
    mask7 = (mask21[..., 2::3] > 0.5).to(mask21.dtype)        # (B,T,7)
    mask14 = torch.repeat_interleave(mask7, 2, dim=-1)         # (B,T,14)
    return mask14, mask7

def masked_time_avg_mse(
    pred: torch.Tensor, gt: torch.Tensor, mask14: torch.Tensor,
    *, norm_dim: bool = True, norm_time: bool = True,
    reduction: str = "mean", eps: float = 1e-6
) -> torch.Tensor:
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
# 可视化
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

def pick_good_sample(x_hist_full21: torch.Tensor, y_fut_full21: torch.Tensor,
                     min_ev_ratio=0.80, min_sv_ratio=0.20, min_good_svs=2) -> int:
    """
    注意：这里假定传入 CPU 张量，避免 device 不一致造成 cat 报错。
    """
    with torch.no_grad():
        # 全部转 CPU，确保安全
        if x_hist_full21.device.type != 'cpu':
            x_hist_full21 = x_hist_full21.cpu()
        if y_fut_full21.device.type != 'cpu':
            y_fut_full21 = y_fut_full21.cpu()

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
    anchor_where="hist_last", smooth_win=5, jump_break_ft=30.0,
    meta_title=None
):
    # 读取 std（feet 可视化；缺失则回退为 1）
    with open(stats_json_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    std_x = float(stats.get("Local_X", {}).get("std", 1.0))
    std_y = float(stats.get("Local_Y", {}).get("std", 1.0))

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
        a[...,0] *= std_x  # lateral (x)
        a[...,1] *= std_y  # longitudinal (y)
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
            xh_s_sm[:, j, :] = _smooth_segments(xh_s[:, j, :], mask=m_hist, jump_ft=jump_break_ft, win=smooth_win)
        for j in range(N):
            m_fut  = fut_mask14[:, 2*j] * fut_mask14[:, 2*j+1]
            zf_s_sm[:, j, :] = _smooth_segments(zf_s[:, j, :],  mask=m_fut,  jump_ft=jump_break_ft, win=smooth_win)

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
            _car(ax, y_anchor, x_anchor, w=4, l=7, color='gray', fc='white', label=['F','B','LF','LB','RF','RB'][j-1] if N==7 else f"C{j}")

    ev_hist_mask = hist_mask14[:, 0] * hist_mask14[:, 1]
    ev_fut_mask  = fut_mask14[:,  0] * fut_mask14[:,  1]
    _plot_with_break(ax, xh_s_sm[:, 0, :], jump_break_ft, style='-',  color='black', lw=2.0, alpha=1.0, mask=ev_hist_mask, label='EV history')
    _plot_with_break(ax, zf_s_sm[:, 0, :], jump_break_ft, style='-',  color='green', lw=2.0, alpha=1.0, mask=ev_fut_mask,  label='EV future GT')
    _plot_with_break(ax, xo_s[:,    0, :], jump_break_ft, style='--', color='blue',  lw=2.0, alpha=1.0, mask=ev_fut_mask,  label='EV future Pred')

    for j in range(1, N):
        m_hist = hist_mask14[:, 2*j] * hist_mask14[:, 2*j+1]
        m_fut  = fut_mask14[:, 2*j] * fut_mask14[:, 2*j+1]
        _plot_with_break(ax, xh_s_sm[:, j, :], jump_break_ft, style='-',  color='gray', lw=1.2, alpha=0.9, mask=m_hist,
                         label=('SV history' if j == 1 else None))
        _plot_with_break(ax, zf_s_sm[:, j, :], jump_break_ft, style='--', color='gray', lw=1.2, alpha=0.9, mask=m_fut,
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
# 可复现多样本可视化（可选）
# ---------------------------

@torch.no_grad()
def run_repeated_visualizations(
    model, loader, device, args, lane_centers_tensor,
    split_name="val", out_dir="./outputs", viz_n=6,
    strategy="good", seed=123, indices_str=""
):
    os.makedirs(os.path.join(out_dir, "viz"), exist_ok=True)
    rng = np.random.default_rng(seed)

    viz_indices = None
    if strategy == "indices" and indices_str:
        viz_indices = [int(x) for x in indices_str.split(",") if len(x.strip())>0]

    all_batches = list(loader)
    total_samples = sum(b[0].shape[0] for b in all_batches)

    def get_by_global_index(idx_global):
        cnt = 0
        for (x_hist_xy, y_fut_xy, y_fut_full21, x_hist_full21, vids, starts) in all_batches:
            bs = x_hist_xy.shape[0]
            if idx_global < cnt + bs:
                loc = idx_global - cnt
                return (x_hist_xy[loc:loc+1], y_fut_xy[loc:loc+1],
                        y_fut_full21[loc:loc+1], x_hist_full21[loc:loc+1],
                        vids[loc:loc+1], starts[loc:loc+1])
            cnt += bs
        raise IndexError("idx_global out of range")

    manifest = {
        "weights": args.resume if args.resume else "this_run",
        "split": split_name, "viz_n": viz_n,
        "strategy": strategy, "seed": int(seed),
        "indices": [],
        "lane_centers_data_units": lane_centers_tensor.cpu().tolist(),
        "dt": args.dt,
    }

    made = 0
    sample_cursor = 0

    while made < viz_n:
        if viz_indices is not None:
            idx_global = viz_indices[made % len(viz_indices)]
        elif strategy == "good":
            (x_hist_xy, y_fut_xy, y_fut_full21, x_hist_full21, vids, starts) = all_batches[sample_cursor % len(all_batches)]
            idx_in_batch = pick_good_sample(x_hist_full21, y_fut_full21)  # CPU 采样
            idx_global = sum(b[0].shape[0] for b in all_batches[:(sample_cursor % len(all_batches))]) + idx_in_batch
            sample_cursor += 1
        else:
            idx_global = int(rng.integers(0, total_samples))

        (xh, yf, yf21, xh21, vid, stf) = get_by_global_index(idx_global)

        # 先采样，再搬到 device
        env = {"lane_centers": lane_centers_tensor.to(device)}
        x_hist_xy    = xh.to(device).float()
        y_fut_xy     = yf.to(device).float()
        y_fut_full21 = yf21.to(device).float()

        out = model(x_hist_xy, y_fut_xy, z_mask_full21=y_fut_full21, P0=None, env=env, T_s=args.dt)

        save_png = os.path.join(out_dir, "viz", f"{split_name}_{made:02d}.png")
        plot_anchor_scene(
            x_hist_xy.cpu(), y_fut_xy.cpu(),
            {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k,v in out.items()},
            stats_json_path=args.stats_json,
            sample=0, lane_width_ft=12.0, save_path=save_png,
            y_view_half=80, x_view_half=25,
            y_fut_full21=y_fut_full21.cpu(),
            x_hist_full21=xh21.cpu(),
            anchor_where="hist_last", smooth_win=5, jump_break_ft=30.0,
            meta_title={"veh": int(vid.item()), "frame": int(stf.item())}
        )

        npz_path = os.path.join(out_dir, "viz", f"{split_name}_{made:02d}.npz")
        np.savez_compressed(
            npz_path,
            x_prior = out["x_prior"].cpu().numpy(),
            x_post  = out["x_post"].cpu().numpy(),
            q_diag  = out["q_diag"].cpu().numpy(),
            r_diag  = out["r_diag"].cpu().numpy(),
            y_gt    = y_fut_xy.cpu().numpy(),
            mask21  = y_fut_full21.cpu().numpy(),
            K_lon_VT= out["K_lon_VT"].cpu().numpy(),
            K_lat   = out["K_lat"].cpu().numpy(),
            v_ref   = out["v_ref"].cpu().numpy(),
            lane_idx= out["lane_idx"].cpu().numpy(),
            p_ref   = out["p_ref"].cpu().numpy(),
        )

        manifest["indices"].append(int(idx_global))
        print(f"[VIZ] saved: {save_png}")
        made += 1

    with open(os.path.join(out_dir, "viz", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[VIZ] manifest saved: {os.path.join(out_dir, 'viz', 'manifest.json')}")

# ---------------------------
# 切分与训练循环
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

# ---------------------------
# 车道中心配置（把 feet 转为数据单位，使用 Local_X）
# ---------------------------

def load_stats(stats_json: str):
    """返回 (mean_x, std_x, mean_y, std_y)。缺失则回退为 0/1。"""
    with open(stats_json, "r", encoding="utf-8") as f:
        s = json.load(f)
    mean_x = float(s.get("Local_X", {}).get("mean", 0.0))
    std_x  = float(s.get("Local_X", {}).get("std",  1.0))
    mean_y = float(s.get("Local_Y", {}).get("mean", 0.0))
    std_y  = float(s.get("Local_Y", {}).get("std",  1.0))
    return mean_x, std_x, mean_y, std_y

def parse_csv_list(s: str) -> Optional[List[float]]:
    if not s or s.strip()=="":
        return None
    return [float(x) for x in s.strip().split(",")]

def make_lane_centers_data_units(args, stats_json: str) -> torch.Tensor:
    """
    返回 (3,) 的 lane_centers（数据单位，与 x/横向一致）。
    优先使用 --lane-centers-data（直接数据单位）。
    否则使用 --lane-centers-feet，并用 Local_X 的 mean/std 将 feet → 数据单位：
        data = (feet - mean_x) / std_x
    """
    c_data = parse_csv_list(args.lane_centers_data)
    if c_data is not None:
        return torch.tensor(c_data, dtype=torch.float32)

    c_feet = parse_csv_list(args.lane_centers_feet)
    if c_feet is None:
        c_feet = [-12.0, 0.0, 12.0]
    try:
        mean_x, std_x, _, _ = load_stats(stats_json)
        if std_x <= 1e-6:
            std_x = 1.0
        c_data = [(v - mean_x)/std_x for v in c_feet]
        return torch.tensor(c_data, dtype=torch.float32)
    except Exception as e:
        print(f"[WARN] 读取 stats_json 失败或缺项：{e}。lane_centers_feet 将除以 std_x 或直接使用。")
        try:
            std_x = load_stats(stats_json)[1]
            if std_x <= 1e-6: std_x = 1.0
            c_data = [v/std_x for v in c_feet]
            return torch.tensor(c_data, dtype=torch.float32)
        except Exception:
            return torch.tensor(c_feet, dtype=torch.float32)

# ---------------------------
# 构建模型
# ---------------------------

def build_model(args, device):
    policy_head = PolicyParamHead_Simple(enc_hidden=args.f_hidden, num_layers=args.f_layers, k_scale=2.0, hidden_mlp=256)
    q_net = LSTM_Q_Noise(input_dim=14, q_hidden=args.q_hidden, q_layer=1, state_dim=14, dropout=args.dropout)
    r_net = LSTM_R_Noise(input_dim=14, r_hidden=args.r_hidden, r_layer=1, state_dim=14, dropout=args.dropout)

    model = LSTMKF_PhysicsVT(
        input_dim=14, hidden_dim=args.f_hidden, num_layers=args.f_layers,
        future_len=args.future_len,
        q_net=q_net, r_net=r_net, policy_head=policy_head,
        jitter=args.jitter, p0_scale=args.p0_scale,
        ev_xy_index=(args.ev_x_idx, args.ev_y_idx),
        min_q=args.min_q, min_r=args.min_r,
        dropout=args.dropout
    )
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

# ---------------------------
# 训练 / 评估
# ---------------------------

def train_one_epoch(model, loader, optimizer, device, scaler, args, writer=None, epoch_idx=0, lane_centers_tensor=None):
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

        env = {"lane_centers": lane_centers_tensor.to(device)}
        with autocast("cuda", enabled=(is_cuda and args.amp)):
            out = model(x_hist_xy, y_fut_xy, z_mask_full21=y_fut_full21, P0=None, env=env, T_s=args.dt)
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
def evaluate_epoch(model, loader, device, args, writer=None, epoch_idx=0, tag_prefix="val", lane_centers_tensor=None):
    is_cuda = (device.type == "cuda")
    model.eval()
    total_loss = 0.0; n_samples = 0
    total_post = 0.0; total_prior = 0.0

    for x_hist_xy, y_fut_xy, y_fut_full21, x_hist_full21, vids, starts in loader:
        x_hist_xy    = x_hist_xy.to(device, non_blocking=is_cuda).float()
        y_fut_xy     = y_fut_xy.to(device, non_blocking=is_cuda).float()
        y_fut_full21 = y_fut_full21.to(device, non_blocking=is_cuda).float()

        env = {"lane_centers": lane_centers_tensor.to(device)}
        out = model(x_hist_xy, y_fut_xy, z_mask_full21=y_fut_full21, P0=None, env=env, T_s=args.dt)
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
    p = argparse.ArgumentParser("Train LSTM-KF (Physics prior: VT + single lane)")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--stats-json", type=str, required=True)
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
    p.add_argument("--test-split", type=float, default=0.1)
    p.add_argument("--split-by", type=str, default="vehicle", choices=["vehicle", "time", "random"])
    p.add_argument("--save-splits", action="store_true")
    p.add_argument("--splits-json", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--out", type=str, default="./outputs/lstmkf_physics_vt")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-prior", type=float, default=0.8)

    # 模型规模
    p.add_argument("--f-hidden", type=int, default=128)
    p.add_argument("--f-layers", type=int, default=1)
    p.add_argument("--q-hidden", type=int, default=64)
    p.add_argument("--r-hidden", type=int, default=32)

    # KF 稳定性
    p.add_argument("--jitter", type=float, default=1e-3)
    p.add_argument("--p0-scale", type=float, default=1.0)
    p.add_argument("--min-q", type=float, default=1e-5)
    p.add_argument("--min-r", type=float, default=1e-4)

    # 先验相关
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--ev-x-idx", type=int, default=0, help="EV 的 x 在 14 维中的索引（横向）")
    p.add_argument("--ev-y-idx", type=int, default=1, help="EV 的 y 在 14 维中的索引（纵向）")
    p.add_argument("--lane-centers-data", type=str, default="")
    p.add_argument("--lane-centers-feet", type=str, default="-12,0,12")

    # 训练控制
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--dp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-test", action="store_true")

    # 可复现多样本可视化
    p.add_argument("--viz-n", type=int, default=0)
    p.add_argument("--viz-split", type=str, default="val", choices=["val","test"])
    p.add_argument("--viz-strategy", type=str, default="good", choices=["good","random","indices"])
    p.add_argument("--viz-indices", type=str, default="")
    p.add_argument("--viz-seed", type=int, default=123)

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

    # 模型 & 优化器
    model = build_model(args, device)

    # lane centers（数据单位，基于 Local_X）
    lane_centers_data = make_lane_centers_data_units(args, args.stats_json)  # (3,)
    print(f"[INFO] lane_centers (data units, Local_X) = {lane_centers_data.tolist()}")
    lane_centers_tensor = lane_centers_data  # forward 时放入 env

    if args.resume and os.path.isfile(args.resume):
        sd = torch.load(args.resume, map_location=device)
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
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

    # TensorBoard
    writer = SummaryWriter(os.path.join(args.out, "runs"))
    with open(os.path.join(args.out, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    best_val = float("inf")

    # ---- train or eval-only ----
    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, args, writer, epoch,
                                         lane_centers_tensor=lane_centers_tensor)
            val_loss   = evaluate_epoch(model, val_loader, device, args, writer, epoch, tag_prefix="val",
                                        lane_centers_tensor=lane_centers_tensor)
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
        val_loss = evaluate_epoch(model, val_loader, device, args, writer=None, epoch_idx=0, tag_prefix="val",
                                  lane_centers_tensor=lane_centers_tensor)
        print(f"[EVAL-ONLY] val/loss = {val_loss:.6f}")

    writer.close()

    # -------- 多次可视化（可复现）--------
    if args.viz_n > 0:
        which_loader = val_loader if args.viz_split == "val" else test_loader
        split_name   = args.viz_split
        model.eval()
        lane_centers_tensor = make_lane_centers_data_units(args, args.stats_json)

        run_repeated_visualizations(
            model=model,
            loader=which_loader,
            device=device,
            args=args,
            lane_centers_tensor=lane_centers_tensor,
            split_name=split_name,
            out_dir=args.out,
            viz_n=args.viz_n,
            strategy=args.viz_strategy,
            seed=args.viz_seed,
            indices_str=args.viz_indices
        )

    # ---------------------
    # Final single visualization on VAL
    # ---------------------
    if len(val_set) > 0:
        model.eval()
        try:
            # 先从 loader 取 CPU 张量
            x_hist_xy_cpu, y_fut_xy_cpu, y_fut_full21_cpu, x_hist_full21_cpu, vids, starts = next(iter(val_loader))
            # 先在 CPU 上挑样，避免 device 不一致
            idx = pick_good_sample(x_hist_full21_cpu, y_fut_full21_cpu)

            # 再搬到 device 做一次完整前向
            env = {"lane_centers": lane_centers_tensor.to(device)}
            x_hist_xy    = x_hist_xy_cpu.to(device).float()
            y_fut_xy     = y_fut_xy_cpu.to(device).float()
            y_fut_full21 = y_fut_full21_cpu.to(device).float()

            out = model(x_hist_xy, y_fut_xy, z_mask_full21=y_fut_full21, P0=None, env=env, T_s=args.dt)

            x1  = x_hist_xy_cpu[idx:idx+1]
            y1  = y_fut_xy_cpu[idx:idx+1]
            o1  = {k: (v[idx:idx+1].detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
            y21 = y_fut_full21_cpu[idx:idx+1]
            x21 = x_hist_full21_cpu[idx:idx+1]
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
                jump_break_ft=30.0,
                meta_title=meta
            )
        except StopIteration:
            print("[WARN] val_loader is empty; skip final visualization.")

    # ---------------------
    # (Optional) Evaluate & visualize on TEST
    # ---------------------
    if args.eval_test and len(test_set) > 0:
        test_loss = evaluate_epoch(model, test_loader, device, args, writer=None, epoch_idx=0, tag_prefix="test",
                                   lane_centers_tensor=lane_centers_tensor)
        print(f"[TEST] test/loss = {test_loss:.6f}")
        try:
            x_hist_xy_cpu, y_fut_xy_cpu, y_fut_full21_cpu, x_hist_full21_cpu, vids, starts = next(iter(test_loader))
            idx = pick_good_sample(x_hist_full21_cpu, y_fut_full21_cpu)

            env = {"lane_centers": lane_centers_tensor.to(device)}
            x_hist_xy    = x_hist_xy_cpu.to(device).float()
            y_fut_xy     = y_fut_xy_cpu.to(device).float()
            y_fut_full21 = y_fut_full21_cpu.to(device).float()
            out = model(x_hist_xy, y_fut_xy, z_mask_full21=y_fut_full21, P0=None, env=env, T_s=args.dt)

            x1  = x_hist_xy_cpu[idx:idx+1]
            y1  = y_fut_xy_cpu[idx:idx+1]
            o1  = {k: (v[idx:idx+1].detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
            y21 = y_fut_full21_cpu[idx:idx+1]
            x21 = x_hist_full21_cpu[idx:idx+1]
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
                jump_break_ft=30.0,
                meta_title=meta
            )
        except StopIteration:
            print("[WARN] test_loader is empty; skip test visualization.")

if __name__ == "__main__":
    main()
