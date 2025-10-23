# ngsim_preprocess.py
import os
import re
import json
from pathlib import Path
import subprocess

from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np

def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    try:
        p = subprocess.check_output(["git","-C", str(here),"rev-parse","--show-toplevel"], text=True).strip()
        return Path(p)
    except Exception:
        env = os.getenv("REPO_ROOT")
        if env:
            return Path(env)
        for parent in [here] + list(here.parents):
            if (parent / ".git").exists():
                return parent
        raise RuntimeError("Cannot locate repo root")
    
# ============== Configuration(repo-relative paths) ==============
ROOT = repo_root()
DATA = ROOT / "data"


CSV_BASENAME = "Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data_20250918.csv"
CSV_PATH   = DATA / CSV_BASENAME
OUT_CSV    = DATA / "ngsim_processed_samples.csv"
STATS_JSON = DATA / "ngsim_stats.json"

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
assert CSV_PATH.exists(), f"CSV not found: {CSV_PATH}"

HIST_LEN   = 25
FUTURE_LEN = 25

# Longitudinal window in feet (per NGSIM, longitudinal = Local_Y / Raw_Y)
MAX_LONG_DIST: float = 60.0  # 英尺（相对“当前帧自车”的距离窗）

# Optional lateral window in feet.
LATERAL_MAX: Optional[float] = 16     # None=关闭横向窗

MAX_VEHICLES_PER_RUN = 0  # 本次运行最多处理多少辆（便于调试）
FLUSH_EVERY = 50000   # 累积到 5万条样本就落盘清空，可按机器内存调大/调小

# 仅主线车道：6/7/8 为辅道/匝道
MAIN_LANES = {1, 2, 3, 4, 5}


# ============== Utilities ==============
def _to_float(x) -> float:
    """Convert DataFrame/Series/scalar to a float scalar."""
    if isinstance(x, pd.DataFrame):
        x = x.iloc[0, 0]
    elif isinstance(x, pd.Series):
        x = x.iloc[0]
    return float(x)


def _normalize_token(s: str) -> str:
    """把字符串标准化为仅含小写字母数字，便于鲁棒匹配."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def filter_by_location_us101_i80(df: pd.DataFrame) -> pd.DataFrame:
    """
    只从 Location 列筛选 US-101 与 I-80。
    - 对 Location 做小写、去空白/换行、去标点的归一化；
    - 保留包含 'us101' 或 'i80' 的行；
    - 若没有 Location 列或筛不到，抛错或给出清晰提示。
    """
    if "Location" not in df.columns:
        raise KeyError("The 'Location' column was not found in the CSV. Please verify the file format.")

    loc_raw = df["Location"].fillna("").astype(str)

    # 归一化：小写、合并空白、去首尾空白
    loc_norm_space = (
        loc_raw.str.lower()
               .str.replace(r"\s+", " ", regex=True)
               .str.strip()
    )
    # 压缩到仅字母数字，便于鲁棒匹配（'us-101' / 'US 101' -> 'us101'）
    loc_compact = loc_norm_space.str.replace(r"[^a-z0-9]+", "", regex=True)

    mask = loc_compact.str.contains("us101") | loc_compact.str.contains("i80")

    kept = df.loc[mask].copy()
    total = len(df)
    kept_n = len(kept)
    print(f"[INFO] Location filter: US-101/I-80 — kept {kept_n:,} / {total:,} rows.")
    if kept_n == 0:
        raise RuntimeError("No rows matched US-101/I-80 in the 'Location' column. Please check the 'Location' values in the CSV.")

    return kept


# ============== SVs selection (vectorized, NGSIM coordinate convention) ==============
def _neighbor_lr_main_only(lane0: int, lanes_present: np.ndarray):
    """只考虑主线 1..5 的左右相邻车道；lane0 不在主线则返回 (None, None)。"""
    if lane0 not in MAIN_LANES:
        return None, None
    s = set(int(x) for x in lanes_present.tolist())
    left  = lane0 - 1 if (lane0 - 1) in s and (lane0 - 1) in MAIN_LANES else None
    right = lane0 + 1 if (lane0 + 1) in s and (lane0 + 1) in MAIN_LANES else None
    return left, right


def find_six_svs_fast(frame_df: pd.DataFrame,
                      ego_row: pd.Series,
                      max_long_dist: float,
                      lateral_max: Optional[float] = LATERAL_MAX):
    """
    在同一帧内，基于 NGSIM 坐标定义：
      - Raw_X = 横向 (lateral)
      - Raw_Y = 纵向 (longitudinal)
    只在主线 1..5 中寻找邻居（6/7/8 视为“无”）。
    返回顺序：[同道前, 同道后, 左前, 左后, 右前, 右后]；未找到返回 None。
    """
    ego_id = int(ego_row["Vehicle_ID"])

    ego_lon = float(ego_row["Raw_Y"])  # longitudinal
    ego_lat = float(ego_row["Raw_X"])  # lateral
    lane0   = int(ego_row["Lane_ID"])

    # EV 不在主线：直接返回全 None（后续会跳样本或 mask=0）
    if lane0 not in MAIN_LANES:
        return [None] * 6

    others = frame_df[frame_df["Vehicle_ID"] != ego_id]
    if others.empty:
        return [None] * 6

    # 仅主线 1..5 的候选
    pool = others[others["Lane_ID"].isin(MAIN_LANES)].copy()
    if pool.empty:
        return [None] * 6

    # 相对位移（英尺）
    dlon = pool["Raw_Y"].to_numpy(dtype=float) - ego_lon
    dlat = pool["Raw_X"].to_numpy(dtype=float) - ego_lat

    lanes = pool["Lane_ID"].to_numpy(dtype=int)
    vids  = pool["Vehicle_ID"].to_numpy(dtype=int)

    # 只考虑当帧存在的主线车道
    lanes_in_frame = pool["Lane_ID"].dropna().astype(int).unique()
    lane_left, lane_right = _neighbor_lr_main_only(lane0, lanes_in_frame)

    # 纵向窗
    front_mask = (dlon > 0) & (dlon <= max_long_dist)
    back_mask  = (dlon < 0) & (-dlon <= max_long_dist)
    if lateral_max is not None:
        lat_mask = (np.abs(dlat) <= float(lateral_max))
        front_mask &= lat_mask
        back_mask  &= lat_mask

    def pick(mask):
        """从候选中选 ‘纵向最近，其次横向最近’；无候选返回 None。"""
        if not np.any(mask):
            return None
        order = np.lexsort((np.abs(dlat[mask]), np.abs(dlon[mask])))  # 先 |dlon| 再 |dlat|
        return int(vids[mask][order][0])

    # 同道前/后
    same = (lanes == lane0)
    vid_front = pick(same & front_mask)
    vid_back  = pick(same & back_mask)

    # 左道前/后（限定 1..5 且当帧存在）
    vid_lf = vid_lb = None
    if lane_left is not None:
        left = (lanes == lane_left)
        vid_lf = pick(left & front_mask)
        vid_lb = pick(left & back_mask)

    # 右道前/后（限定 1..5 且当帧存在）
    vid_rf = vid_rb = None
    if lane_right is not None:
        right = (lanes == lane_right)
        vid_rf = pick(right & front_mask)
        vid_rb = pick(right & back_mask)

    return [vid_front, vid_back, vid_lf, vid_lb, vid_rf, vid_rb]


# ============== Pack relative vectors (anchor = CURRENT FRAME EV), 21 dims/frame ==============
def pack_relative_vector_masked_anchor(frame_df_t: pd.DataFrame,
                                       ego_row_t: pd.Series,
                                       fixed_ids: List[Optional[int]],
                                       anchor_ego_row: pd.Series,
                                       max_long_dist: float,
                                       lateral_max: Optional[float] = LATERAL_MAX) -> np.ndarray:
    """
    以 anchor_ego_row（当前帧自车）为原点(0,0)，打包“时间 t”这一帧的 7 槽位：
      槽位顺序：E, F, B, LF, LB, RF, RB
      每个槽位为 (rel_x, rel_y, mask)：
        - rel_x/rel_y：基于标准化后的 Local_X/Local_Y 的相对坐标（目标 - anchor）
        - mask：1.0 表示该槽位在时间 t 有有效目标（存在且相对 anchor 的 |Raw_Y|<=窗，且可选 |Raw_X|<=横向窗）；否则 0.0
      返回：np.float32，形状 (21,)
    """
    # Anchor (CURRENT FRAME EV): 标准化坐标用于学习空间；Raw_* 英尺用于几何判断
    ax = float(anchor_ego_row["Local_X"])  # standardized lateral
    ay = float(anchor_ego_row["Local_Y"])  # standardized longitudinal
    arx = float(anchor_ego_row["Raw_X"])   # raw lateral (feet)
    ary = float(anchor_ego_row["Raw_Y"])   # raw longitudinal (feet)

    # EV 槽位
    ex_t = float(ego_row_t["Local_X"])
    ey_t = float(ego_row_t["Local_Y"])
    rel_e_x = ex_t - ax
    rel_e_y = ey_t - ay
    out = [rel_e_x, rel_e_y, 1.0]  # EV 一直存在；仅锚点帧是 (0,0,1)

    # 以 Vehicle_ID 为索引的快捷查找
    frame_df_indexed = frame_df_t if frame_df_t.index.name == "Vehicle_ID" \
                        else frame_df_t.set_index("Vehicle_ID", drop=False)

    # 六邻居（锚点帧锁定 ID）：逐帧转成相对锚点坐标
    for vid in fixed_ids:
        if vid is None or vid not in frame_df_indexed.index:
            out.extend([0.0, 0.0, 0.0])
            continue

        nr = frame_df_indexed.loc[vid]
        if isinstance(nr, pd.DataFrame):
            nr = nr.iloc[0]

        # 与锚点的几何窗判断
        dlon_raw_anchor = float(nr["Raw_Y"]) - ary
        if abs(dlon_raw_anchor) > max_long_dist:
            out.extend([0.0, 0.0, 0.0])
            continue

        if lateral_max is not None:
            dlat_raw_anchor = float(nr["Raw_X"]) - arx
            if abs(dlat_raw_anchor) > lateral_max:
                out.extend([0.0, 0.0, 0.0])
                continue

        # 学习空间的相对位移
        rel_x = float(nr["Local_X"]) - ax
        rel_y = float(nr["Local_Y"]) - ay
        out.extend([rel_x, rel_y, 1.0])

    return np.array(out, dtype=np.float32)  # (21,)


# ============== Standardization (consistent across runs) ==============
def load_or_fit_stats(df: pd.DataFrame, stats_json: str) -> Dict[str, Dict[str, float]]:
    """
    对 Local_X(横向)/Local_Y(纵向) 做全局 z-score，参数写入/读取 stats_json，保证多次运行一致。
    Raw_X/Raw_Y 始终保留原始尺度用于几何判断。
    """
    if os.path.exists(stats_json):
        with open(stats_json, "r", encoding="utf-8") as f:
            stats = json.load(f)
        for col in ["Local_X", "Local_Y"]:
            mean = stats[col]["mean"]; std = stats[col]["std"] or 1.0
            df[col] = (df[col] - mean) / std
        return stats
    else:
        stats = {}
        for col in ["Local_X", "Local_Y"]:
            mean = float(df[col].mean())
            std  = float(df[col].std()) or 1.0
            stats[col] = {"mean": mean, "std": std}
            df[col] = (df[col] - mean) / std
        with open(stats_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        return stats


# ============== Read processed vehicle IDs (resume-friendly) ==============
def read_processed_vehicle_ids(out_csv: str):
    if not os.path.exists(out_csv):
        return set()
    try:
        small = pd.read_csv(out_csv, usecols=["Vehicle_ID"])
        return set(small["Vehicle_ID"].unique().tolist())
    except Exception:
        return set()




# ============== Main ==============
if __name__ == "__main__":

    print("ROOT =", ROOT)
    print("DATA =", DATA)
    print("CSV_PATH exists? ->", CSV_PATH.exists())

    print("Loading raw data...")
    df_all = pd.read_csv(CSV_PATH)

    # -------- 仅保留 Location 列中含 US-101 / I-80 的行 --------
    df = filter_by_location_us101_i80(df_all)
    if len(df) == 0:
        raise RuntimeError("Location 过滤后没有任何行。请检查 CSV 中 Location 列的值。")

    # 备份原始英尺坐标用于几何判断（保持 NGSIM 惯例）：
    # Raw_X = lateral, Raw_Y = longitudinal
    df["Raw_X"] = df["Local_X"].astype(float)
    df["Raw_Y"] = df["Local_Y"].astype(float)

    # 标准化 Local_X/Local_Y（学习空间）
    _ = load_or_fit_stats(df, STATS_JSON)

    # 按帧建索引；去掉同帧重复 Vehicle_ID
    frame_index: Dict[int, pd.DataFrame] = {}
    for fid, g in df.groupby("Frame_ID"):
        g = g.drop_duplicates(subset=["Vehicle_ID"], keep="last")
        frame_index[int(fid)] = g

    # 若输出 CSV 不存在则写表头
    if not os.path.exists(OUT_CSV):
        pd.DataFrame(columns=[
            "Vehicle_ID", "Start_Frame", "hist_len", "fut_len",
            "x_hist_flat", "y_fut_flat"
        ]).to_csv(OUT_CSV, index=False)

    processed_ids = read_processed_vehicle_ids(OUT_CSV)
    print(f"Already processed vehicles: {len(processed_ids)}")

    rows = []
    newly = 0

    # 遍历车辆（本次运行限量，便于调试）
    for veh_id, traj in df.groupby("Vehicle_ID"):
        if veh_id in processed_ids:
            continue
        if MAX_VEHICLES_PER_RUN and newly >= MAX_VEHICLES_PER_RUN:
            break


        # 时间排序 + 足够长才处理
        traj = traj.sort_values("Frame_ID").reset_index(drop=True)
        if len(traj) < HIST_LEN + FUTURE_LEN:
            continue

        # 滑窗：i 为“当前/锚点帧”，同时是未来片段的第一帧
        for i in range(HIST_LEN, len(traj) - FUTURE_LEN):
            anchor = traj.iloc[i]

            # 仅主线样本：锚点不在 1..5 则跳过
            lane_anchor = int(anchor["Lane_ID"])
            if lane_anchor not in MAIN_LANES:
                continue

            frame_anchor = frame_index[int(anchor["Frame_ID"])]

            # 锁定锚点帧的邻居 ID（只在主线 1..5 内）
            fixed_ids = find_six_svs_fast(frame_anchor, anchor, MAX_LONG_DIST, LATERAL_MAX)

            # 历史序列（每帧相对同一锚点）
            hist_seq = []
            for j in range(i - HIST_LEN, i):
                ego_t = traj.iloc[j]
                frame_t = frame_index[int(ego_t["Frame_ID"])]
                hist_seq.append(
                    pack_relative_vector_masked_anchor(frame_t, ego_t, fixed_ids, anchor,
                                                       MAX_LONG_DIST, LATERAL_MAX)
                )
            hist_seq = np.array(hist_seq, dtype=np.float32)  # (H,21)

            # 未来序列（每帧相对同一锚点）
            fut_seq = []
            for k in range(i, i + FUTURE_LEN):
                ego_t = traj.iloc[k]
                frame_t = frame_index[int(ego_t["Frame_ID"])]
                fut_seq.append(
                    pack_relative_vector_masked_anchor(frame_t, ego_t, fixed_ids, anchor,
                                                       MAX_LONG_DIST, LATERAL_MAX)
                )
            fut_seq = np.array(fut_seq, dtype=np.float32)    # (F,21)

            # 断言：未来序列第 1 帧即锚点帧，EV 槽位应为 (0,0,1)
            ev0 = fut_seq[0, :3]  # (x, y, mask)
            assert np.allclose(ev0[:2], 0.0, atol=1e-6) and abs(ev0[2] - 1.0) < 1e-6, (
                f"EV at anchor not (0,0,1): {ev0} (veh={int(veh_id)}, frame={int(anchor['Frame_ID'])})"
            )

            rows.append({
                "Vehicle_ID": int(veh_id),
                "Start_Frame": int(traj.iloc[i]["Frame_ID"]),     # anchor frame
                "hist_len": HIST_LEN,
                "fut_len": FUTURE_LEN,
                "x_hist_flat": " ".join(map(str, hist_seq.reshape(-1).tolist())),
                "y_fut_flat": " ".join(map(str, fut_seq.reshape(-1).tolist()))
            })

            if len(rows) >= FLUSH_EVERY:
                pd.DataFrame(rows).to_csv(OUT_CSV, mode="a", index=False, header=False)
                rows.clear()

        newly += 1

    # 追加写出
    if rows:
        pd.DataFrame(rows).to_csv(OUT_CSV, mode="a", index=False, header=False)

    print(f"Newly processed vehicles this run: {newly}, appended samples: {len(rows)} to {OUT_CSV}")
    print("Done.")
