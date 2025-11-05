import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    matthews_corrcoef,
)


def _inverse_normal_cdf(p: float) -> float:
    """
    Approximate inverse CDF (quantile) of the standard normal distribution.
    Implementation based on Peter J. Acklam's rational approximation.
    """
    # Coefficients for central region
    a = [
        -3.969683028665376e01,
         2.209460984245205e02,
        -2.759285104469687e02,
         1.383577518672690e02,
        -3.066479806614716e01,
         2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
         1.615858368580409e02,
        -1.556989798598866e02,
         6.680131188771972e01,
        -1.328068155288572e01,
    ]
    # Coefficients for tails
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
         4.374664141464968e00,
         2.938163982698783e00,
    ]
    d = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e00,
         3.754408661907416e00,
    ]

    # Break-points
    plow = 0.02425
    phigh = 1 - plow

    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")

    if p < plow:
        q = np.sqrt(-2.0 * np.log(p))
        numerator = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        denominator = ((((d[0]*q + d[1])*q + d[2])*q + d[3]) * q + 1.0)
        return -numerator / denominator
    if phigh < p:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        numerator = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        denominator = ((((d[0]*q + d[1])*q + d[2])*q + d[3]) * q + 1.0)
        return numerator / denominator

    q = p - 0.5
    r = q * q
    numerator = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
    denominator = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4]) * r + 1.0)
    return numerator / denominator


def generate_synthetic_scores(
    num_positive: int = 20000,
    num_negative: int = 20000,
    target_auc: float = 0.80,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic binary labels and continuous scores with approximately the desired AUC.

    We sample scores for negatives and positives from two normal distributions with equal variance.
    If both class-conditional score distributions are Normal(μ, σ^2) with equal σ, the AUC equals
    Φ(Δ/√2), where Δ = (μ_pos - μ_neg)/σ and Φ is the standard normal CDF. To target a specific AUC,
    we set Δ = √2 · Φ^{-1}(AUC). For σ=1 and μ_neg=0 this yields μ_pos = Δ.
    """
    rng = np.random.default_rng(random_seed)

    # Convert target AUC to mean separation Δ under equal-variance normal model
    from math import sqrt

    delta = sqrt(2.0) * _inverse_normal_cdf(target_auc)
    mu_neg, mu_pos, sigma = 0.0, float(delta), 1.0

    scores_negative = rng.normal(loc=mu_neg, scale=sigma, size=num_negative)
    scores_positive = rng.normal(loc=mu_pos, scale=sigma, size=num_positive)

    labels = np.concatenate([np.zeros(num_negative, dtype=int), np.ones(num_positive, dtype=int)])
    scores = np.concatenate([scores_negative, scores_positive])

    # Shuffle pairs to avoid any ordering artifacts
    indices = rng.permutation(labels.shape[0])
    return labels[indices], scores[indices]


def plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc_value: float,
    save_path: str,
) -> None:
    """
    Plot ROC curve with AUC and shaded area.
    """
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc_value:.2f})")
    plt.fill_between(fpr, tpr, step="pre", alpha=0.25, color="#aec7e8")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#888888", lw=1, label="Chance")

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    dirpath = os.path.dirname(save_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pr(
    precision: np.ndarray,
    recall: np.ndarray,
    pr_auc_value: float,
    save_path: str,
) -> None:
    """
    Plot Precision-Recall curve with shaded area under the curve.
    """
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="#ff7f0e", lw=2, label=f"PR (AUC = {pr_auc_value:.2f})")
    plt.fill_between(recall, precision, step="pre", alpha=0.25, color="#ffbb78")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    dirpath = os.path.dirname(save_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def compute_threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    denom_dice = (2 * tp + fp + fn)
    dice = (2.0 * tp / denom_dice) if denom_dice > 0 else 0.0
    mcc = matthews_corrcoef(labels, predictions)
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "dice": float(dice),
        "mcc": float(mcc),
    }


def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    plt.figure(figsize=(5.5, 4.5))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
    plt.yticks(tick_marks, ["True 0", "True 1"])

    # Annotate cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    dirpath = os.path.dirname(save_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix_explainer(cm: np.ndarray, save_path: str) -> None:
    """
    Explanatory 2x2 confusion matrix with labeled cells (TP, FN, FP, TN)
    and semantic coloring: correct (TP/TN) darker, errors (FP/FN) lighter.
    """
    # Map cm to display order where top-left is TP
    # sklearn's cm = [[tn, fp], [fn, tp]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    labels_grid = [["TP", "FN"], ["FP", "TN"]]
    counts_grid = [[tp, fn], [fp, tn]]

    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    # Draw colored squares
    for i in range(2):  # rows: 0=Positive, 1=Negative (Actual)
        for j in range(2):  # cols: 0=Positive, 1=Negative (Predicted)
            is_correct = (i == j)
            face = "#2a6f9e" if is_correct else "#a6c8e3"
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face, edgecolor="white", linewidth=2.0))
            # Big label (TP/FN/FP/TN)
            ax.text(
                j + 0.5,
                i + 0.5,
                labels_grid[i][j],
                ha="center",
                va="center",
                fontsize=28,
                color="white",
                weight="bold",
            )

    # Axes labels and ticks
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)  # invert y to have Positive on top
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Positive", "Negative"], fontsize=12)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Positive", "Negative"], fontsize=12)
    ax.set_xlabel("Predicted Values", fontsize=14)
    ax.set_ylabel("Actual Values", fontsize=14)
    ax.set_title("Confusion Matrix", fontsize=16, pad=10)

    # Clean up spines and ticks
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    ax.set_aspect("equal")

    plt.tight_layout()
    dirpath = os.path.dirname(save_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    # 1) Daten generieren (AUC ≈ 0.80)
    labels, scores = generate_synthetic_scores(target_auc=0.80, random_seed=7)

    # 2) ROC und AUC berechnen
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc_value = auc(fpr, tpr)

    # 2b) PR und AUC berechnen
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc_value = auc(recall, precision)

    # 3) Output-Verzeichnis und Plots speichern
    output_dir = os.path.join(os.path.dirname(__file__), "demo_plots")
    os.makedirs(output_dir, exist_ok=True)
    output_plot = os.path.join(output_dir, "roc_auc_demo.png")
    plot_roc(fpr, tpr, roc_auc_value, output_plot)

    # 3b) PR-Plot erstellen und speichern
    output_pr_plot = os.path.join(output_dir, "pr_auc_demo.png")
    plot_pr(precision, recall, pr_auc_value, output_pr_plot)

    # 4) Optional: exemplarische Punkte entlang der Kurve (zum Erklären)
    def _closest_index(arr: np.ndarray, target: float) -> int:
        return int(np.argmin(np.abs(arr - target)))

    idx_fpr_10 = _closest_index(fpr, 0.10)
    idx_fpr_20 = _closest_index(fpr, 0.20)
    idx_fpr_30 = _closest_index(fpr, 0.30)

    print("\nROC/PR evaluation")
    print("------------------")
    print(f"ROC AUC: {roc_auc_value:.3f}")
    print(f"PR  AUC: {pr_auc_value:.3f}")
    print("\nExample ROC points (approx.):")
    for name, i in [("FPR≈0.10", idx_fpr_10), ("FPR≈0.20", idx_fpr_20), ("FPR≈0.30", idx_fpr_30)]:
        print(f"  {name}: FPR={fpr[i]:.3f}, TPR={tpr[i]:.3f}, threshold={thresholds[i]:.4f}")

    # Confusion matrix + DSC + MCC at a representative threshold (near FPR≈0.20)
    thr = float(thresholds[idx_fpr_20])
    m = compute_threshold_metrics(labels, scores, thr)
    cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
    cm_path = os.path.join(output_dir, "confusion_matrix_demo.png")
    plot_confusion_matrix(cm, cm_path)
    # Also create an explanatory version
    cm_expl_path = os.path.join(output_dir, "confusion_matrix_explained_demo.png")
    plot_confusion_matrix_explainer(cm, cm_expl_path)
    print("\nConfusion matrix at threshold (FPR≈0.20):")
    print(f"  threshold: {m['threshold']:.4f}")
    print("  [ [TN, FP],")
    print("    [FN, TP] ]")
    print(f"  [ [{m['tn']}, {m['fp']}],")
    print(f"    [{m['fn']}, {m['tp']}] ]")
    print(f"  Dice coefficient: {m['dice']:.3f}")
    print(f"  Matthews corrcoef: {m['mcc']:.3f}")
    print(f"  Confusion matrix plot saved to: {cm_path}")
    print(f"  Confusion matrix explainer saved to: {cm_expl_path}")

    print(f"\nROC plot saved to: {output_plot}")
    print(f"PR  plot saved to: {output_pr_plot}")


if __name__ == "__main__":
    main()


