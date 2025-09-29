# Drop-in script: set your file paths below and run this cell/script (no CLI needed).
# It will create HD95 distribution plots (histogram + KDE, and boxplot) and save them to pngs/.

import os
import math
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Point these to your CSVs (case, hd95_mm, note) — edit as needed
FILES = {
    "Plain": "hd95_plain_cv.csv",
    "CBAM": "hd95_cbam_cv.csv",
    "CrissCross": "hd95_cc_cv.csv",
    "SE": "hd95_se_cv.csv",
    "AG": "hd95_ag_cv.csv",  # uncomment if you have it
}

# 2) Output paths
os.makedirs("pngs", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_HIST = f"pngs/hd95_distribution_hist_{ts}.png"
OUT_BOX = f"pngs/hd95_distribution_box_{ts}.png"
OUT_STATS = f"pngs/hd95_summary_stats_{ts}.csv"

# 3) Load and combine
frames = []
for variant, path in FILES.items():
    if not os.path.isfile(path):
        print(f"[WARN] Missing file: {path} (skipping {variant})")
        continue
    df = pd.read_csv(path)
    if "hd95_mm" not in df.columns:
        raise ValueError(f"{path} must contain column 'hd95_mm'")
    df = df.copy()
    df["variant"] = variant
    # keep finite values only
    df = df[np.isfinite(df["hd95_mm"]) & ~df["hd95_mm"].isna()]
    frames.append(df[["variant", "hd95_mm"]])

if not frames:
    raise SystemExit("No valid input files found. Please update FILES with correct paths.")

data = pd.concat(frames, ignore_index=True)

# 4) Summary stats per variant
def summarize(group):
    arr = group["hd95_mm"].astype(float).values
    return pd.Series({
        "count": arr.size,
        "mean": np.mean(arr),
        "std": np.std(arr, ddof=0),
        "median": np.median(arr),
        "p10": np.quantile(arr, 0.10),
        "p90": np.quantile(arr, 0.90),
        "min": np.min(arr),
        "max": np.max(arr),
    })

stats = data.groupby("variant", as_index=False).apply(summarize)
stats.to_csv(OUT_STATS, index=False)
print(f"[INFO] Saved summary stats: {OUT_STATS}")
print(stats)

# 5) Plot — histogram + KDE per variant (overlayed)
sns.set(style="whitegrid", context="talk")
plt.figure(figsize=(12, 7))
for variant, df in data.groupby("variant"):
    sns.histplot(
        df["hd95_mm"], bins=40, stat="density", element="step", fill=False, label=f"{variant} (hist)"
    )
    sns.kdeplot(df["hd95_mm"], linewidth=2, label=f"{variant} (kde)")
plt.xlabel("HD95 (mm)")
plt.ylabel("Density")
plt.title("HD95 Distribution (per variant)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_HIST, dpi=200)
plt.close()
print(f"[INFO] Saved histogram: {OUT_HIST}")

# 6) Plot — boxplot per variant
plt.figure(figsize=(10, 6))
order = sorted(data["variant"].unique())
sns.boxplot(data=data, x="variant", y="hd95_mm", order=order, showfliers=True)
sns.stripplot(data=data, x="variant", y="hd95_mm", order=order, color="black", alpha=0.25, size=3)
plt.xlabel("Variant")
plt.ylabel("HD95 (mm)")
plt.title("HD95 Boxplot by Variant")
plt.tight_layout()
plt.savefig(OUT_BOX, dpi=200)
plt.close()
print(f"[INFO] Saved boxplot: {OUT_BOX}")