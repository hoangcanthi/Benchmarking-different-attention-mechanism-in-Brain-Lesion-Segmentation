import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Configure dataset roots (relative to project root)
RAW_ROOT = os.path.join("nnUNet_raw", "Dataset201_ATLAS2")
IMG_DIR = os.path.join(RAW_ROOT, "imagesTr")
LBL_DIR = os.path.join(RAW_ROOT, "labelsTr")

# Fold-3 validation prediction folders for each model variant
PRED_DIRS = {
    "Plain": os.path.join(
        "nnUNet_results", "Dataset201_ATLAS2",
        "nnUNetTrainer__nnUNetPlans__2d", "fold_3", "validation"
    ),
    "CBAM": os.path.join(
        "nnUNet_results", "Dataset201_ATLAS2",
        "nnUNetTrainer_CBAM_Contrastive__nnUNetPlans__2d", "fold_3", "validation"
    ),
    "CrissCross": os.path.join(
        "nnUNet_results", "Dataset201_ATLAS2",
        "nnUNetTrainer_CrissCross__nnUNetPlans__2d", "fold_3", "validation"
    ),
    "SE": os.path.join(
        "nnUNet_results", "Dataset201_ATLAS2",
        "nnUNetTrainer_SE__nnUNetPlans__2d", "fold_3", "validation"
    ),
    "AG": os.path.join(
        "nnUNet_results", "Dataset201_ATLAS2",
        "nnUNetTrainer_AttentionGate__nnUNetPlans__2d", "fold_3", "validation"
    ),
}

OUT_DIR = "pngs"
OUT_SUBDIR = os.path.join(OUT_DIR, "overlays_fold3")
FOREGROUND_LABEL = 1


def _normalize(img2d: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(img2d, (1, 99))
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        p1, p99 = float(np.min(img2d)), float(np.max(img2d))
    x = (img2d - p1) / (p99 - p1 + 1e-8)
    return np.clip(x, 0.0, 1.0)


def _best_slice(lbl3d: np.ndarray) -> int:
    areas = (lbl3d == FOREGROUND_LABEL).sum(axis=(0, 1))
    if areas.max() == 0:
        return lbl3d.shape[2] // 2
    return int(np.argmax(areas))


def _iter_valid_cases(limit: int):
    count = 0
    for f in sorted(os.listdir(LBL_DIR)):
        if not f.endswith(".nii.gz"):
            continue
        case = f[:-7]
        try:
            lbl = np.asanyarray(nib.load(os.path.join(LBL_DIR, f)).dataobj)
        except Exception:
            continue
        if not np.any(lbl == FOREGROUND_LABEL):
            continue
        ok = True
        for d in PRED_DIRS.values():
            if not os.path.isfile(os.path.join(d, case + ".nii.gz")):
                ok = False
                break
        if not ok:
            continue
        yield case
        count += 1
        if count >= limit:
            break


def render_overlay(case_id: str) -> str:
    img_path = os.path.join(IMG_DIR, f"{case_id}_0000.nii.gz")
    lbl_path = os.path.join(LBL_DIR, f"{case_id}.nii.gz")
    img_nii = nib.load(img_path)
    lbl_nii = nib.load(lbl_path)
    img = np.asanyarray(img_nii.dataobj).squeeze()  # (H,W,Z)
    lbl = np.asanyarray(lbl_nii.dataobj)

    z = _best_slice(lbl)
    base = _normalize(img[..., z])
    gt2d = (lbl[..., z] == FOREGROUND_LABEL).astype(np.uint8)

    preds2d = {}
    for name, d in PRED_DIRS.items():
        pr = np.asanyarray(nib.load(os.path.join(d, f"{case_id}.nii.gz")).dataobj)
        if pr.shape != lbl.shape:
            raise ValueError(f"Shape mismatch for {name}: {pr.shape} vs {lbl.shape}")
        preds2d[name] = (pr[..., z] == FOREGROUND_LABEL).astype(np.uint8)

    os.makedirs(OUT_SUBDIR, exist_ok=True)
    out_png = os.path.join(OUT_SUBDIR, f"overlay_fold3_grid_{case_id}.png")

    # Layout: top row (GT) spans 5 columns; bottom row shows 5 models
    variants = ["Plain", "CBAM", "CrissCross", "SE", "AG"]
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1, 1], hspace=0.15, wspace=0.05)

    # Top row: GT spanning all columns
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(base, cmap="gray", interpolation="nearest")
    if gt2d.any():
        ax_top.contour(gt2d, levels=[0.5], colors=["#00ff00"], linewidths=1.8)
    ax_top.set_axis_off()
    ax_top.set_title(f"Ground Truth overlay — {case_id}", fontsize=13)

    # Bottom row: predictions per variant
    for i, v in enumerate(variants):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(base, cmap="gray", interpolation="nearest")
        if gt2d.any():
            ax.contour(gt2d, levels=[0.5], colors=["#00ff00"], linewidths=1.2)
        pr2d = preds2d[v]
        if pr2d.any():
            ax.contour(pr2d, levels=[0.5], colors=["#ff4d4d"], linewidths=1.2, linestyles="--")
        ax.set_axis_off()
        ax.set_title(v, fontsize=11)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def main():
    saved = []
    for case in _iter_valid_cases(limit=100):
        try:
            # Skip if already exists
            out_candidate = os.path.join(OUT_SUBDIR, f"overlay_fold3_grid_{case}.png")
            if os.path.isfile(out_candidate):
                print(f"[SKIP] Exists: {out_candidate}")
                continue
            out = render_overlay(case)
            saved.append(out)
            print(f"Saved: {out}")
        except Exception as e:
            print(f"[WARN] Skipped {case}: {e}")
    if not saved:
        print("No overlays created. Ensure predictions for fold 3 exist for all models.")


if __name__ == "__main__":
    main()


