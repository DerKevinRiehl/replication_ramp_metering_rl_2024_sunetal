"""
Plot combined learning curves from all trained models.

Loads saved ``learning_curves.npz`` files from:
    2_data_produced/drl_models/ddpg_1step_td/
    2_data_produced/drl_models/ddpg_10step_td/
    2_data_produced/mpc_drl_models/mpc_drl_1step_td/
    2_data_produced/mpc_drl_models/mpc_drl_10step_td/

Produces a 2×3 grid (Demand 1/2 × Low/Medium/High noise) with all
available algorithms overlaid on each subplot.

Usage:
    python plot_all_learning_curves.py
    python plot_all_learning_curves.py --window 21
    python plot_all_learning_curves.py --out combined_learning_curves.png
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no plt.show hang)
import matplotlib.pyplot as plt

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

# ======================================================================
# Where to find each algorithm's saved learning curves
# ======================================================================
DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "2_data_produced",
)

FIG_DIR = os.path.join(PROJECT_ROOT, "3_data_visualization")

# (display_label, npz_path, key_prefix)
# key_prefix is the algo string used when saving:
#   e.g.  "DDPG 1-step TD__Demand 1__Low"
ALGO_SOURCES = [
    ("DRL 1-step TD",      os.path.join(DATA_DIR, "drl_models",     "ddpg_1step_td",     "learning_curves.npz"), "DDPG 1-step TD"),
    ("DRL 10-step TD",     os.path.join(DATA_DIR, "drl_models",     "ddpg_10step_td",    "learning_curves.npz"), "DDPG 10-step TD"),
    ("MPC-DRL 1-step TD",  os.path.join(DATA_DIR, "mpc_drl_models", "mpc_drl_1step_td",  "learning_curves.npz"), "MPC-DRL 1-step TD"),
    ("MPC-DRL 10-step TD", os.path.join(DATA_DIR, "mpc_drl_models", "mpc_drl_10step_td", "learning_curves.npz"), "MPC-DRL 10-step TD"),
]

DEMANDS = ["Demand 1", "Demand 2"]
NOISES  = ["Low", "Medium", "High"]
NOISE_LABELS = ["Low-level noise", "Medium-level noise", "High-level noise"]

ALGO_COLORS = {
    "MPC-DRL 10-step TD": "blue",
    "MPC-DRL 1-step TD": "red",
    "DRL 10-step TD": "orange",
    "DRL 1-step TD": "green",
}


def load_all_curves() -> dict:
    """
    Load all available learning-curve data.

    Returns
    -------
    results : dict
        Nested dict::

            {
                "DRL 1-step TD": {
                    ("Demand 1", "Low"): np.ndarray,
                    ...
                },
                ...
            }

        Only includes algorithms whose .npz files exist on disk.
    """
    results = {}

    for label, npz_path, key_prefix in ALGO_SOURCES:
        if not os.path.isfile(npz_path):
            print(f"[SKIP] {label}: {npz_path} not found")
            continue

        data = np.load(npz_path)
        scenarios = {}

        for demand in DEMANDS:
            for noise in NOISES:
                key = f"{key_prefix}__{demand}__{noise}"
                if key in data:
                    scenarios[(demand, noise)] = data[key]

        if scenarios:
            results[label] = scenarios
            print(f"[OK]   {label}: {len(scenarios)} scenario(s) loaded")
        else:
            print(f"[WARN] {label}: .npz exists but no matching keys found")
            print(f"       Available keys: {list(data.keys())}")

        data.close()

    return results


def plot_combined(
    results: dict,
    window: int = 21,
    save_path: str | None = None,
):
    """
    Plot a 2×3 grid of learning curves (Demand × Noise).

    All available algorithms are overlaid on each subplot.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    # Global max episodes for consistent x-axis
    global_max_ep = 0
    for scenarios in results.values():
        for rewards in scenarios.values():
            global_max_ep = max(global_max_ep, len(rewards))

    for row, demand in enumerate(DEMANDS):
        for col, (noise, nlabel) in enumerate(zip(NOISES, NOISE_LABELS)):
            ax = axes[row, col]

            for algo_name, scenarios in results.items():
                key = (demand, noise)
                if key not in scenarios:
                    continue

                rewards = np.asarray(scenarios[key], dtype=np.float64)
                n_ep = len(rewards)
                episodes = np.arange(n_ep)

                # Adaptive smoothing
                w = min(window, n_ep) if n_ep > 0 else 1
                if w < 2:
                    ax.plot(
                        episodes,
                        rewards,
                        label=algo_name,
                        color=ALGO_COLORS.get(algo_name),
                    )
                else:
                    kernel = np.ones(w) / w
                    smoothed = np.convolve(rewards, kernel, mode="same")
                    ax.plot(
                        episodes,
                        smoothed,
                        label=algo_name,
                        color=ALGO_COLORS.get(algo_name),
                    )

            ax.set_xlim(0, max(global_max_ep - 1, 1))
            ax.set_ylim(-3000, -1000)

            if row == 0:
                ax.set_title(nlabel)
            if row == 1:
                ax.set_xlabel("Episodes")
            if col == 0:
                ax.set_ylabel("Reward")

    # Row labels
    for row, demand in enumerate(DEMANDS):
        axes[row, -1].annotate(
            demand,
            xy=(1.05, 0.5),
            xycoords="axes fraction",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
            rotation=-90,
        )

    # Common legend
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center",
            ncol=min(len(labels), 4),
            frameon=False,
            fontsize=10,
        )

    fig.tight_layout(rect=[0, 0.06, 0.96, 1.0])

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")

    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot combined learning curves from all trained models"
    )
    parser.add_argument(
        "--window", type=int, default=21,
        help="Moving-average window for smoothing (default: 21)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
           help="Output file path. If relative, it is resolved from project root. "
               "Default: 3_data_visualization/combined_learning_curves.png",
    )
    args = parser.parse_args()

    # Load all available curves
    print("Loading learning curves...\n")
    results = load_all_curves()

    if not results:
        print("\nNo learning curves found. Train models first:")
        print("  python train_drl.py --n_step 1")
        print("  python train_drl.py --n_step 10")
        print("  python train_mpc_drl.py --n_step 1")
        print("  python train_mpc_drl.py --n_step 10")
        return

    # Output path
    if args.out is None:
        out_path = os.path.join(FIG_DIR, "combined_learning_curves.png")
    elif os.path.isabs(args.out):
        out_path = args.out
    else:
        out_path = os.path.join(PROJECT_ROOT, args.out)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Plot
    print(f"\nPlotting {len(results)} algorithm(s)...")
    plot_combined(results, window=args.window, save_path=out_path)
    print("Done.")


if __name__ == "__main__":
    main()
