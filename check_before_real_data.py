"""
READ-ONLY CHECKS ONLY. Does not modify any project file or write any output
besides stdout. Run with: .\.venv\Scripts\python.exe check_before_real_data.py
(place this at the trust_ids_poc/ root, next to data/).
"""
import numpy as np
import pandas as pd

BENIGN = "data/nbaiot/Danmini_Doorbell/benign_traffic.csv"
MIRAI = "data/nbaiot/Danmini_Doorbell/mirai_attacks/scan.csv"
GAFGYT = "data/nbaiot/Danmini_Doorbell/gafgyt_attacks/combo.csv"


def check_file(path, label):
    df = pd.read_csv(path)
    n_cols = df.shape[1]
    n_rows = df.shape[0]
    n_nan = int(df.isna().sum().sum())
    n_inf = int(np.isinf(df.to_numpy(dtype=float)).sum())
    print(f"--- {label} ---")
    print(f"  rows={n_rows}  cols={n_cols}  NaNs={n_nan}  Infs={n_inf}")
    return df


print("=== A/B: NaN/Inf checks ===")
mirai_df = check_file(MIRAI, "mirai_attacks/scan.csv")
gafgyt_df = check_file(GAFGYT, "gafgyt_attacks/combo.csv")
benign_df = pd.read_csv(BENIGN)

print("\n=== Column consistency check (not explicitly requested but required before any of this means anything) ===")
same_cols_mirai = list(benign_df.columns) == list(mirai_df.columns)
same_cols_gafgyt = list(benign_df.columns) == list(gafgyt_df.columns)
print(f"  benign vs mirai columns identical: {same_cols_mirai}")
print(f"  benign vs gafgyt columns identical: {same_cols_gafgyt}")

print("\n=== C: Block-1 vs Block-10 standardized distance (benign file) ===")
n = len(benign_df)
block_size = n // 10
blocks = []
start = 0
for i in range(10):
    end = start + block_size if i < 9 else n
    blocks.append(benign_df.iloc[start:end])
    start = end

block_means = np.array([b.mean(axis=0).to_numpy(dtype=float) for b in blocks])
overall_std = benign_df.std(axis=0).to_numpy(dtype=float)
overall_std[overall_std == 0] = 1e-9  # avoid div by zero on constant columns

standardized_means = block_means / overall_std  # standardize by overall feature std, same as prior inspection
dist_1_10 = np.linalg.norm(standardized_means[0] - standardized_means[9])
print(f"  Block 1 size={len(blocks[0])}  Block 10 size={len(blocks[9])}")
print(f"  Standardized distance Block1 vs Block10: {dist_1_10:.2f}")

print("\nDone. Paste this entire output back.")
