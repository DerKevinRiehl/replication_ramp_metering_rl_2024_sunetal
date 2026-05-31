from config import TS, N_CTRL
import numpy as np
import matplotlib.pyplot as plt


def create_demand_1() -> np.ndarray:
    """ Mainstream + On-ramp scenario 1 """
    demand = np.zeros((N_CTRL, 2))
    
    # --- MAINSTREAM ---
    # Costant 3500 until 1.75h, then drops to 1050 until 2.2h, then remains 1000
    idx_1_75 = int(1.75 * 3600 / TS)
    idx_2_2 = int(2.2 * 3600 / TS)
    
    demand[:idx_1_75, 0] = 3500
    demand[idx_1_75:idx_2_2, 0] = np.linspace(3500, 1050, idx_2_2 - idx_1_75)
    demand[idx_2_2:, 0] = 1050
    
    # --- ON-RAMP 1 SCENARIO ---
    # Base 300  pick 1500 from 0.2h to 0.7h
    idx_0_1 = int(0.1 * 3600 / TS)
    idx_0_2 = int(0.2 * 3600 / TS)
    idx_0_7 = int(0.7 * 3600 / TS)
    idx_0_8 = int(0.8 * 3600 / TS)
    
    demand[:idx_0_1, 1] = 300
    demand[idx_0_1:idx_0_2, 1] = np.linspace(300, 1500, idx_0_2 - idx_0_1)
    demand[idx_0_2:idx_0_7, 1] = 1500
    demand[idx_0_7:idx_0_8, 1] = np.linspace(1500, 300, idx_0_8 - idx_0_7)
    demand[idx_0_8:, 1] = 300
    
    return demand

def create_demand_2() -> np.ndarray:
    """Mainstream + On-ramp scenario 2 """
    demand = np.zeros((N_CTRL, 2))
    
    # --- MAINSTREAM  ---
    idx_1_75 = int(1.75 * 3600 / TS)
    idx_2_2 = int(2.2 * 3600 / TS)
    demand[:idx_1_75, 0] = 3500
    demand[idx_1_75:idx_2_2, 0] = np.linspace(3500, 1050, idx_2_2 - idx_1_75)
    demand[idx_2_2:, 0] = 1050
    
    # --- ON-RAMP 2 SCENARIO ---
    # Base 400, pick 1300 from 0.2h to 0.7h
    idx_0_1 = int(0.1 * 3600 / TS)
    idx_0_2 = int(0.2 * 3600 / TS)
    idx_0_7 = int(0.7 * 3600 / TS)
    idx_0_8 = int(0.8 * 3600 / TS)
    
    demand[:idx_0_1, 1] = 400
    demand[idx_0_1:idx_0_2, 1] = np.linspace(400, 1300, idx_0_2 - idx_0_1)
    demand[idx_0_2:idx_0_7, 1] = 1300
    demand[idx_0_7:idx_0_8, 1] = np.linspace(1300, 400, idx_0_8 - idx_0_7)
    demand[idx_0_8:, 1] = 400
    
    return demand


# ------------------------------------------------------------------
# Learning-curve visualisation
# ------------------------------------------------------------------

def plot_learning_curves(training_results: dict, save_path: str | None = None,
                         window: int = 21):
    """
    Plot smoothed reward-vs-episode learning curves on a 2×3 grid.

    Parameters
    ----------
    training_results : dict
        Nested dict of the form::

            {
                "DDPG 10-step TD": {
                    ("Demand 1", "Low"):    [r0, r1, ...],
                    ("Demand 1", "Medium"): [...],
                    ...
                },
                "DDPG 1-step TD": { ... },
            }

        Each leaf is a list/array of per-episode rewards (one reward per
        episode).
    save_path : str or None
        If given, the figure is saved to this path.
    window : int
        Moving-average window size for smoothing (default 21).
        If the reward array is shorter than *window*, the window is
        automatically shrunk so a curve is always drawn.
    """
    demands = ["Demand 1", "Demand 2"]
    noises = ["Low", "Medium", "High"]
    noise_labels = ["Low-level noise", "Medium-level noise", "High-level noise"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    # First pass: find the global max episode count so all subplots
    # share the same x-axis range.
    global_max_ep = 0
    for scenarios in training_results.values():
        for rewards in scenarios.values():
            global_max_ep = max(global_max_ep, len(rewards))

    for row, demand in enumerate(demands):
        for col, (noise, nlabel) in enumerate(zip(noises, noise_labels)):
            ax = axes[row, col]
            for algo_name, scenarios in training_results.items():
                key = (demand, noise)
                if key not in scenarios:
                    continue
                rewards = np.asarray(scenarios[key], dtype=np.float64)
                n_ep = len(rewards)

                # Episode indices (0-based)
                episodes = np.arange(n_ep)

                # Adaptive smoothing: shrink window if data is short
                w = min(window, n_ep) if n_ep > 0 else 1
                if w < 2:
                    # Too short to smooth – just plot raw
                    ax.plot(episodes, rewards, label=algo_name)
                else:
                    kernel = np.ones(w) / w
                    smoothed = np.convolve(rewards, kernel, mode="same")
                    ax.plot(episodes, smoothed, label=algo_name)

            ax.set_xlim(0, max(global_max_ep - 1, 1))
            if row == 0:
                ax.set_title(nlabel)
            if row == 1:
                ax.set_xlabel("Episodes")
            if col == 0:
                ax.set_ylabel("Reward")

    # Row labels on the far right
    for row, demand in enumerate(demands):
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

    # Single common legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=len(labels), frameon=False, fontsize=10)

    fig.tight_layout(rect=[0, 0.05, 0.96, 1.0])

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Learning-curve figure saved to {save_path}")

    plt.show()
    return fig


def print_result(experiment_results: dict, algo_name: str):
    print("\n" + "=" * 80)
    print(f"{algo_name} EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(experiment_results.to_string(index=False, float_format="%.2f"))
    print("=" * 80 + "\n")
