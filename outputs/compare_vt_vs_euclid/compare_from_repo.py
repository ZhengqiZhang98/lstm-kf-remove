import os, json, argparse, importlib.util
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset

def load_module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_json(p): 
    with open(p, "r", encoding="utf-8") as f: 
        return json.load(f)

def split_indices_by(df, mode, val_split, test_split, seed=42):
    import numpy as np
    rng = np.random.default_rng(seed)
    if mode=="vehicle" and "Vehicle_ID" in df.columns:
        vids = df["Vehicle_ID"].unique(); rng.shuffle(vids)
        n_test = max(1, int(len(vids)*test_split)); n_val = max(1, int(len(vids)*val_split))
        test_vids=set(vids[:n_test]); val_vids=set(vids[n_test:n_test+n_val])
        test_idx=df.index[df["Vehicle_ID"].isin(test_vids)].tolist()
        val_idx =df.index[df["Vehicle_ID"].isin(val_vids)].tolist()
        all_idx=set(df.index.tolist())
        train_idx=sorted(all_idx - set(test_idx) - set(val_idx))
        return train_idx, val_idx, test_idx
    # fallback: random
    perm=rng.permutation(len(df))
    n_test=max(1,int(len(df)*test_split)); n_val=max(1,int(len(df)*val_split))
    test_idx=perm[:n_test].tolist(); val_idx=perm[n_test:n_test+n_val].tolist()
    train_idx=perm[n_test+n_val:].tolist()
    return train_idx,val_idx,test_idx

def rmse_all_dim_time_sample(y, yhat, m):
    eps = 1e-12
    diff2 = (y - yhat)**2 * m
    # overall
    overall = torch.sqrt(diff2.sum() / m.sum().clamp_min(eps)).item()
    # per-dim
    pd = torch.sqrt(diff2.sum(dim=(0,1)) / m.sum(dim=(0,1)).clamp_min(eps))
    # per-time
    pt = torch.sqrt(diff2.sum(dim=(0,2)) / m.sum(dim=(0,2)).clamp_min(eps))
    # per-sample
    ps = torch.sqrt(diff2.sum(dim=(1,2)) / m.sum(dim=(1,2)).clamp_min(eps))
    return overall, pd.cpu().numpy(), pt.cpu().numpy(), ps.cpu().numpy()

def ade_fde_ev_only(y, yhat, m14):
    eps=1e-12
    ev_y  = y[..., :2]; ev_p = yhat[..., :2]; ev_m = m14[..., :2]
    vis = (ev_m[...,0]>0.5) & (ev_m[...,1]>0.5)
    dist = torch.linalg.norm(ev_y - ev_p, dim=-1)
    ade = (dist*vis).sum()/vis.sum().clamp_min(eps)
    last = vis[:,-1]
    fde = (dist[:,-1]*last).sum()/last.sum().clamp_min(eps)
    return ade.item(), fde.item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--stats-json", required=True)
    ap.add_argument("--splits-json", default="")
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--test-split", type=float, default=0.1)
    ap.add_argument("--vt-py", required=True)
    ap.add_argument("--eu-py", required=True)
    ap.add_argument("--vt-ckpt", required=True)
    ap.add_argument("--vt-hparams", required=True)
    ap.add_argument("--eu-ckpt", required=True)
    ap.add_argument("--eu-hparams", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lane-centers-feet", default="-12,0,12")
    ap.add_argument("--lane-centers-data", default="")
    ap.add_argument("--use-prior", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if (args.device in ["auto","cuda"] and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device}")

    vtmod = load_module_from(args.vt_py, "vtmod")
    eumod = load_module_from(args.eu_py, "eumod")

    # 数据集（两边实现一致，随便用一个）
    full = vtmod.ProcessedNGSIMDataset(args.csv, hist_len=25, fut_len=25)
    df   = full.df

    # 切分
    if args.splits_json and os.path.isfile(args.splits_json):
        sp = load_json(args.splits_json)
        train_idx, val_idx, test_idx = sp["train_idx"], sp["val_idx"], sp["test_idx"]
        print(f"[INFO] 使用 splits.json：{args.splits_json}")
    else:
        train_idx, val_idx, test_idx = split_indices_by(df, "vehicle", args.val_split, args.test_split, seed=42)
        print("[INFO] 未提供 splits.json，已按 Vehicle_ID 重新切分")

    subsets = {"train": Subset(full, train_idx), "val": Subset(full, val_idx), "test": Subset(full, test_idx)}
    ds = subsets[args.split]
    print(f"[INFO] split={args.split}, N={len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # 读取超参
    vt_hp = load_json(args.vt_hparams)
    eu_hp = load_json(args.eu_hparams)

    # ---- 构建 VT 模型 ----
    f_hidden = int(vt_hp.get("f_hidden", vt_hp.get("enc_hidden", 64)))
    f_layers = int(vt_hp.get("f_layers", vt_hp.get("num_layers", 1)))
    q_hidden = int(vt_hp.get("q_hidden", 64))
    r_hidden = int(vt_hp.get("r_hidden", 32))
    dropout  = float(vt_hp.get("dropout", 0.0))
    q_net = vtmod.LSTM_Q_Noise(14, q_hidden, 1, 14, dropout=dropout)
    r_net = vtmod.LSTM_R_Noise(14, r_hidden, 1, 14, dropout=dropout)
    policy = vtmod.PolicyParamHead_Simple(enc_hidden=f_hidden, num_layers=f_layers, k_scale=2.0, hidden_mlp=256)
    vt_model = vtmod.LSTMKF_PhysicsVT(
        input_dim=14, hidden_dim=f_hidden, num_layers=f_layers, future_len=25,
        q_net=q_net, r_net=r_net, policy_head=policy,
        jitter=float(vt_hp.get("jitter",1e-3)), p0_scale=float(vt_hp.get("p0_scale",1.0)),
        ev_xy_index=(int(vt_hp.get("ev_x_idx",0)), int(vt_hp.get("ev_y_idx",1))),
        min_q=float(vt_hp.get("min_q",1e-5)), min_r=float(vt_hp.get("min_r",1e-4)),
        dropout=dropout
    ).to(device)

    # ---- 构建 EU 模型 ----
    f_hidden = int(eu_hp.get("f_hidden", 128))
    f_layers = int(eu_hp.get("f_layers", 1))
    q_hidden = int(eu_hp.get("q_hidden", 64))
    r_hidden = int(eu_hp.get("r_hidden", 64))
    fdiag_hidden = int(eu_hp.get("fdiag_hidden", 64))
    dropout = float(eu_hp.get("dropout", 0.0))
    f_net = eumod.TransitionLSTM(14, 14, f_hidden, f_layers, 25, dropout=dropout)
    q_net = eumod.LSTM_Q_Noise(14, q_hidden, 1, 14, dropout=dropout)
    r_net = eumod.LSTM_R_Noise(14, r_hidden, 1, 14, dropout=dropout)
    fdiag = eumod.LSTM_F_Diag(14, fdiag_hidden, 1, 14, alpha=0.3, dropout=dropout)
    eu_model = eumod.LSTMKF(f_net, q_net, r_net, fdiag_net=fdiag).to(device)

    # ---- 加载权重（兼容多种保存格式）----
    def load_ckpt(m, path):
        sd = torch.load(path, map_location=device)
        if isinstance(sd, dict) and "state_dict" not in sd and "model_state" not in sd:
            m.load_state_dict(sd, strict=False)
        else:
            state = sd if isinstance(sd, dict) and "state_dict" not in sd else sd.get("model_state", sd)
            m.load_state_dict(state, strict=False)
    load_ckpt(vt_model, args.vt_ckpt)
    load_ckpt(eu_model, args.eu_ckpt)
    vt_model.eval(); eu_model.eval()

    # ---- VT 车道中心（数据单位）----
    mean_x = float(json.load(open(args.stats_json))["Local_X"].get("mean",0.0))
    std_x  = float(json.load(open(args.stats_json))["Local_X"].get("std",1.0)) or 1.0
    if args.lane_centers_data.strip():
        lane = torch.tensor([float(x) for x in args.lane_centers_data.split(",")], dtype=torch.float32)
    else:
        feet = [float(x) for x in args.lane_centers_feet.split(",")]
        lane = torch.tensor([(v-mean_x)/std_x for v in feet], dtype=torch.float32)
    lane = lane.to(device)

    # ---- 评估累积 ----
    D=14; T=25; eps=1e-12
    sumdiff2_vt=sumdiff2_eu=0.0; sumw=0.0
    sdim_vt=torch.zeros(D); sdim_eu=torch.zeros(D); wdim=torch.zeros(D)
    stim_vt=torch.zeros(T); stim_eu=torch.zeros(T); wtim=torch.zeros(T)
    ps_vt=[]; ps_eu=[]
    ade_num_vt=ade_num_eu=fde_num_vt=fde_num_eu=0.0; ade_den=fde_den=0.0

    with torch.no_grad():
        for x_hist, y_fut, y21, x_hist_full21, vids, starts in loader:
            x_hist = x_hist.to(device).float()
            y_fut  = y_fut.to(device).float()
            y21    = y21.to(device).float()
            mask14, _ = vtmod.build_mask14(y21)
            out_vt = vt_model(x_hist, y_fut, z_mask_full21=y21, env={"lane_centers": lane}, T_s=float(vt_hp.get("dt",0.1)))
            out_eu = eu_model(x_hist, y_fut, z_mask_full21=y21)
            yhat_vt = out_vt["x_prior"] if args.use_prior else out_vt["x_post"]
            yhat_eu = out_eu["x_prior"] if args.use_prior else out_eu["x_post"]

            # overall
            d2_vt = ((yhat_vt-y_fut)**2*mask14).sum().double().item()
            d2_eu = ((yhat_eu-y_fut)**2*mask14).sum().double().item()
            w     = mask14.sum().double().item()
            sumdiff2_vt += d2_vt; sumdiff2_eu += d2_eu; sumw += max(w, eps)

            # per-dim
            s_vd = ((yhat_vt-y_fut)**2*mask14).sum(dim=(0,1)).cpu()
            s_ed = ((yhat_eu-y_fut)**2*mask14).sum(dim=(0,1)).cpu()
            wd   = mask14.sum(dim=(0,1)).cpu().clamp_min(eps)
            sdim_vt += s_vd; sdim_eu += s_ed; wdim += wd

            # per-time
            s_vt = ((yhat_vt-y_fut)**2*mask14).sum(dim=(0,2)).cpu()
            s_et = ((yhat_eu-y_fut)**2*mask14).sum(dim=(0,2)).cpu()
            wt   = mask14.sum(dim=(0,2)).cpu().clamp_min(eps)
            stim_vt += s_vt; stim_eu += s_et; wtim += wt

            # per-sample
            s_vs = ((yhat_vt-y_fut)**2*mask14).sum(dim=(1,2))
            s_es = ((yhat_eu-y_fut)**2*mask14).sum(dim=(1,2))
            ws   = mask14.sum(dim=(1,2)).clamp_min(eps)
            ps_vt += torch.sqrt(s_vs/ws).cpu().numpy().tolist()
            ps_eu += torch.sqrt(s_es/ws).cpu().numpy().tolist()

            # ADE/FDE (EV-only)
            ev_y=y_fut[...,:2]; ev_v=yhat_vt[...,:2]; ev_e=yhat_eu[...,:2]; ev_m=mask14[...,:2]
            vis = (ev_m[...,0]>0.5)&(ev_m[...,1]>0.5)
            dist_v=torch.linalg.norm(ev_y-ev_v,dim=-1); dist_e=torch.linalg.norm(ev_y-ev_e,dim=-1)
            ade_num_vt += (dist_v*vis).sum().double().item()
            ade_num_eu += (dist_e*vis).sum().double().item()
            ade_den    += vis.sum().double().item()
            last=vis[:,-1]
            fde_num_vt += (dist_v[:,-1]*last).sum().double().item()
            fde_num_eu += (dist_e[:,-1]*last).sum().double().item()
            fde_den    += last.sum().double().item()

    ov_vt = float(np.sqrt(sumdiff2_vt/max(sumw,eps)))
    ov_eu = float(np.sqrt(sumdiff2_eu/max(sumw,eps)))
    rel   = float((ov_eu-ov_vt)/(ov_vt+1e-12)*100.0)  # Euclid 相对 VT（负数 = 更好）

    pd_vt = (np.sqrt((sdim_vt.numpy())/np.clip(wdim.numpy(),eps,None))).tolist()
    pd_eu = (np.sqrt((sdim_eu.numpy())/np.clip(wdim.numpy(),eps,None))).tolist()
    pt_vt = (np.sqrt((stim_vt.numpy())/np.clip(wtim.numpy(),eps,None))).tolist()
    pt_eu = (np.sqrt((stim_eu.numpy())/np.clip(wtim.numpy(),eps,None))).tolist()

    ade_vt = ade_num_vt/max(ade_den,eps); fde_vt=fde_num_vt/max(fde_den,eps)
    ade_eu = ade_num_eu/max(ade_den,eps); fde_eu=fde_num_eu/max(fde_den,eps)

    print("\n==== Overall RMSE ====")
    print(f"VT     : {ov_vt:.6f}")
    print(f"Euclid : {ov_eu:.6f}")
    print(f"Relative (Euclid vs VT): {rel:.2f}%  (负数=Euclid更好)")

    print("\n==== ADE/FDE (EV-only) ====")
    print(f"VT     : ADE={ade_vt:.6f}, FDE={fde_vt:.6f}")
    print(f"Euclid : ADE={ade_eu:.6f}, FDE={fde_eu:.6f}")

    print("\n==== Per-dimension RMSE ====")
    for i,(a,b) in enumerate(zip(pd_vt,pd_eu)):
        print(f"Dim{i:02d}: VT={a:.6f} | EU={b:.6f}")

    print("\n==== Per-timestep RMSE ====")
    for t,(a,b) in enumerate(zip(pt_vt,pt_eu)):
        print(f"t={t:02d}: VT={a:.6f} | EU={b:.6f}")

    # 保存结果
    summary = {
      "overall_rmse":{"vt":ov_vt,"euclid":ov_eu,"relative_euclid_vs_vt_percent":rel},
      "ade_fde_ev_only":{"vt":{"ADE":ade_vt,"FDE":fde_vt},"euclid":{"ADE":ade_eu,"FDE":fde_eu}},
      "per_dimension_rmse":{"vt":pd_vt,"euclid":pd_eu},
      "per_timestep_rmse":{"vt":pt_vt,"euclid":pt_eu},
      "counts":{"N":len(ds),"T":25,"D":14}
    }
    with open(os.path.join(args.out,"summary.json"),"w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2,ensure_ascii=False)
    df = pd.DataFrame({"sample_id":np.arange(len(ps_vt)),"rmse_vt":ps_vt,"rmse_euclid":ps_eu})
    df["improve_%_euclid_vs_vt"]=(df["rmse_vt"]-df["rmse_euclid"])/(df["rmse_vt"]+1e-12)*100.0
    df.to_csv(os.path.join(args.out,"per_sample_metrics.csv"),index=False,encoding="utf-8")

    print(f"\n[OK] 已保存 {os.path.join(args.out,'summary.json')} 以及 per_sample_metrics.csv")

if __name__=="__main__":
    main()
