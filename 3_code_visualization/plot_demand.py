"""
Plot demand profiles to replicate the demand figure from the paper.

Mainstream (blue), On-Ramp Demand 1 (red), On-Ramp Demand 2 (orange).
Y-axis: 0–4000 veh/h, X-axis: 0–2.5 h.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "1_code_produce"))

import numpy as np
import matplotlib.pyplot as plt
from config import TS, N_CTRL
from util import create_demand_1, create_demand_2


def main():
    demand1 = create_demand_1()  # shape (N_CTRL, 2): [mainstream, on-ramp]
    demand2 = create_demand_2()  # shape (N_CTRL, 2): [mainstream, on-ramp]

    # Time axis in hours
    time_h = np.arange(N_CTRL) * TS / 3600.0

    # Mainstream is the same for both scenarios
    mainstream = demand1[:, 0]
    onramp_1 = demand1[:, 1]
    onramp_2 = demand2[:, 1]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(time_h, mainstream, color="blue",   linewidth=2, label="Mainstream")
    ax.plot(time_h, onramp_1,   color="red",    linewidth=2, label="On-Ramp Demand 1")
    ax.plot(time_h, onramp_2,   color="orange", linewidth=2, label="On-Ramp Demand 2")

    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 4000)
    ax.set_xlabel("Time (h)", fontsize=12)
    ax.set_ylabel("Demand (veh/h)", fontsize=12)
    ax.set_title("Traffic Demand Profiles", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # Save to the visualization output directory
    out_dir = os.path.join(os.path.dirname(__file__), "..", "3_data_visualization")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "demand_profiles.pdf")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved to {save_path}")

    # Also save PNG
    save_path_png = os.path.join(out_dir, "demand_profiles.png")
    fig.savefig(save_path_png, dpi=200, bbox_inches="tight")
    print(f"Figure saved to {save_path_png}")

    plt.show()


if __name__ == "__main__":
    main()
