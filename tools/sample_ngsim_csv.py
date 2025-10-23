"""

REPO="/sq/home/ge23lac/lstm-kf"   # 若你已 export 过，可复用现有的

python "$REPO/tools/sample_ngsim_csv.py" \
  --in  "$REPO/data/ngsim_processed_samples.csv" \
  --out "$REPO/data/ngsim_processed_samples_128000.csv" \
  --n 128000 --seed 42
  """

import pandas as pd, numpy as np, argparse

p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--out", dest="out", required=True)
p.add_argument("--n", type=int, default=640)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

df = pd.read_csv(args.inp)
rng = np.random.default_rng(args.seed)
idx = rng.choice(len(df), size=min(args.n, len(df)), replace=False)
df_small = df.iloc[idx].reset_index(drop=True)
df_small.to_csv(args.out, index=False)
print(f"Saved {len(df_small)} rows to {args.out}")
