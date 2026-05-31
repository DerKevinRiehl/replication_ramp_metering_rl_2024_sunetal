"""
Replication of Table V (Sun et al., 2024) - PMPC excluded.
Runs 10 independent seeds for stochastic demand scenarios to evaluate controllers.
Outputs two tables: one using BEST models, one using FINAL models.
"""

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from metanet.metanet_env import MetanetEnv
from config import METANET_PARAMS, GUSSIAN_NOISE, N_CTRL, NC_HF, M_STEP_HF, N_MULTI_START
from util import create_demand_1, create_demand_2

from controllers.no_control import NoControlController
from controllers.mpc_controller import MPCController
from controllers.ddpg_controller import DDPGController
from controllers.mpc_drl_controller import MPCDRLController

# 10 hardcoded seeds for reproducibility
SEEDS = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900]

def simulate_300s_tracking(env, controller, desc):
    """
    Simulation loop that aggregates computational time precisely 
    every 300s (30 steps of 10s) to match the paper's definition.
    """
    obs, _ = env.reset()
    controller.reset()

    solve_times_300s = []
    current_window_time = 0.0

    for t in tqdm(range(int(N_CTRL)), desc=desc, leave=False):
        t0 = time.perf_counter()
        action = controller.get_action(obs, t)
        dt = time.perf_counter() - t0
        
        current_window_time += dt

        # End of 300s window (t=29, 59, 89...)
        if (t + 1) % 30 == 0:
            solve_times_300s.append(current_window_time)
            current_window_time = 0.0

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    return {
        "tts_all": env.ep_tts,
        "twt_all": env.ep_twt,
        "min_speed": env.ep_min_speed,
        "queue_violation": env.ep_queue_violation,
        "Mean computation time [s]": np.mean(solve_times_300s) if solve_times_300s else 0.0,
        "Max computation time [s]": np.max(solve_times_300s) if solve_times_300s else 0.0,
    }

def run_10_seeds(controller_cls, controller_kwargs, env_params, demand_data, noise_cfg, label):
    """Runs 10 stochastic simulations and aggregates the metrics."""
    results = []
    for seed in SEEDS:
        env = MetanetEnv(
            metanet_param=env_params, 
            demand=demand_data, 
            noise_cfg=noise_cfg, 
            seed=seed
        )
        
        # Instantiate controller dynamically
        ctrl = controller_cls(env=env, demand=demand_data, **controller_kwargs) if controller_cls not in [NoControlController, DDPGController] else \
               controller_cls(v_free=env_params["v_free_1"]) if controller_cls is NoControlController else \
               controller_cls(demand=demand_data, **controller_kwargs)
        
        metrics = simulate_300s_tracking(env, ctrl, desc=f"  -> {label} [Seed {seed}]")
        results.append(metrics)

    # Average metrics, take absolute max for Max Computation Time
    return {
        "Total time spent [veh·h]": np.mean([r["tts_all"] for r in results]),
        "Total waiting time [veh·h]": np.mean([r["twt_all"] for r in results]),
        "Minimum speed [km/h]": np.mean([r["min_speed"] for r in results]),
        "Constraint violation [%]": np.mean([r["queue_violation"] for r in results]),
        "Mean computation time [s]": np.mean([r["Mean computation time [s]"] for r in results]),
        "Max computation time [s]": np.max([r["Max computation time [s]"] for r in results]),
    }

def build_table_v(model_type):
    """Evaluates all controllers and constructs Table V for a specific model_type ('best' or 'final')."""
    print(f"\n{'='*60}\nGENERATING TABLE V ({model_type.upper()} MODELS)\n{'='*60}")
    
    env_params = METANET_PARAMS["real"]
    mpc_params = METANET_PARAMS["estimated"]
    
    demands = {"Demand 1": create_demand_1(), "Demand 2": create_demand_2()}
    noise_levels = ["Low", "Medium", "High"]
    
    # Model Directories
    drl_10_dir = os.path.join("..", "2_data_produced", "drl_models", "ddpg_10step_td")
    mpc_drl_10_dir = os.path.join("..", "2_data_produced", "mpc_drl_models", "mpc_drl_10step_td")
    
    table_rows = []
    
    scenario_idx = 1
    for demand_name, demand_data in demands.items():
        for noise_name in noise_levels:
            scenario_name = f"Scenario {scenario_idx}"
            print(f"\n--- {scenario_name} ({demand_name}, {noise_name} Noise) ---")
            
            noise_cfg = GUSSIAN_NOISE[noise_name]
            tag = f"{demand_name}_{noise_name}"

            # 1. No Control
            res_nc = run_10_seeds(NoControlController, {}, env_params, demand_data, noise_cfg, "No Control")
            
            # 2. Standalone MPC
            res_mpc = run_10_seeds(MPCController, {"mpc_params": mpc_params, "w_tts": 1.0, "w_du": 0.4, "num_starts": N_MULTI_START, "verbose": False}, 
                                   env_params, demand_data, noise_cfg, "MPC")
            
            # 3. Standalone DRL (10-step TD)
            res_drl = run_10_seeds(DDPGController, {"n_step": 10, "training": False, "model_dir": drl_10_dir, "model_tag": tag, "model_type": model_type}, 
                                   env_params, demand_data, noise_cfg, "DRL 10-step")
            
            # 4. HF MPC
            res_hfmpc = run_10_seeds(MPCController, {"mpc_params": mpc_params, "Nc": NC_HF, "M_step": M_STEP_HF, "w_tts": 1.0, "w_du": 0.4, "num_starts": N_MULTI_START, "verbose": False}, 
                                     env_params, demand_data, noise_cfg, "HF-MPC")
            
            # 5. MPC-DRL Framework (10-step TD)
            res_mpc_drl = run_10_seeds(MPCDRLController, {"n_step": 10, "model_dir": mpc_drl_10_dir, "model_tag": tag, "model_type": model_type, "mpc_params": mpc_params, "w_tts": 1.0, "w_du": 0.4, "verbose": False}, 
                                       env_params, demand_data, noise_cfg, "MPC-DRL")

            # Append metrics as rows
            metrics = ["Total time spent [veh·h]", "Total waiting time [veh·h]", "Minimum speed [km/h]", 
                       "Constraint violation [%]", "Mean computation time [s]", "Max computation time [s]"]
            
            for m in metrics:
                table_rows.append({
                    "Scenarios": scenario_name if m == metrics[0] else "",
                    "Performance": m,
                    "No control": res_nc[m] if "computation" not in m else "-",
                    "Standalone MPC": res_mpc[m],
                    "Standalone DRL": res_drl[m],
                    "HF MPC": res_hfmpc[m],
                    "MPC-DRL framework": res_mpc_drl[m]
                })
            scenario_idx += 1

    df = pd.DataFrame(table_rows)
    print(f"\n{df.to_string(index=False)}")
    
    out_path = f"../2_data_produced/results/table_v_{model_type}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {model_type.upper()} table to {out_path}")

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs("../2_data_produced/results", exist_ok=True)
    
    build_table_v(model_type="best")
    build_table_v(model_type="final")