"""
Visualization utilities for tracking metrics.
"""
import os
import matplotlib.pyplot as plt


def plot_hota_curves(hota_results: dict, out_dir: str, prefix: str = "tracker"):
    """
    Generate and save HOTA metric visualization plots.
    
    Args:
        hota_results: Dictionary from HOTAAccumulator.compute() containing:
                     - alphas: List of IoU thresholds
                     - HOTA_curve, DetA_curve, AssA_curve, LocA_curve
        out_dir: Directory to save plots
        prefix: Filename prefix (e.g., tracker name)
    """
    os.makedirs(out_dir, exist_ok=True)

    alphas = hota_results["alphas"]
    HOTA_c = hota_results["HOTA_curve"]
    DetA_c = hota_results["DetA_curve"]
    AssA_c = hota_results["AssA_curve"]
    LocA_c = hota_results["LocA_curve"]

    # 1) HOTA vs alpha
    plt.figure(figsize=(7, 5))
    plt.plot(alphas, HOTA_c, marker="o")
    plt.xlabel("IoU threshold α")
    plt.ylabel("HOTA")
    plt.title("HOTA(α)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_HOTA_curve.png"), dpi=200)
    plt.close()

    # 2) Detection & Association
    plt.figure(figsize=(7, 5))
    plt.plot(alphas, DetA_c, marker="o", label="DetA")
    plt.plot(alphas, AssA_c, marker="o", label="AssA")
    plt.xlabel("IoU threshold α")
    plt.ylabel("Score")
    plt.title("Detection & Association Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_DetA_AssA_curves.png"), dpi=200)
    plt.close()

    # 3) Localization accuracy
    plt.figure(figsize=(7, 5))
    plt.plot(alphas, LocA_c, marker="o")
    plt.xlabel("IoU threshold α")
    plt.ylabel("LocA (mean IoU)")
    plt.title("Localization Accuracy (LocA)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_LocA_curve.png"), dpi=200)
    plt.close()

    # 4) All curves together (overview)
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, HOTA_c, marker="o", label="HOTA")
    plt.plot(alphas, DetA_c, marker="o", label="DetA")
    plt.plot(alphas, AssA_c, marker="o", label="AssA")
    plt.plot(alphas, LocA_c, marker="o", label="LocA")
    plt.xlabel("IoU threshold α")
    plt.ylabel("Score")
    plt.title("HOTA Metrics Overview")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_ALL_curves.png"), dpi=200)
    plt.close()

    print(f"[OK] Plots saved to: {out_dir}")
