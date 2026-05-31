"""
Replication of Table V (Sun et al., 2024) - with Standalone ALINEA baseline.
Runs 10 independent seeds for stochastic demand scenarios to evaluate controllers.
Outputs two tables: one using BEST models, one using FINAL models.
"""

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

from metanet.metanet_env import MetanetEnv
from config import METANET_PARAMS, GUSSIAN_NOISE, N_CTRL, NC_HF, M_STEP_HF, N_MULTI_START
from util import create_demand_1, create_demand_2

from controllers.no_control import NoControlController
from controllers.alinea_controller import AlineaController
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
        if controller_cls is NoControlController:
            ctrl = controller_cls(v_free=env_params["v_free_1"])
        elif controller_cls is AlineaController:
            ctrl = controller_cls(
                v_free=env_params["v_free_1"],
                rho_crit=env_params["rho_crit_1"],
            )
        elif controller_cls is DDPGController:
            ctrl = controller_cls(demand=demand_data, **controller_kwargs)
        else:
            # MPC-based controllers need env + demand
            ctrl = controller_cls(env=env, demand=demand_data, **controller_kwargs)
        
        metrics = simulate_300s_tracking(env, ctrl, desc=f"  -> {label} [Seed {seed}]")
        results.append(metrics)

    # Return raw arrays of 10 runs to allow statistical testing
    return {
        "Total time spent [veh·h]": [r["tts_all"] for r in results],
        "Total waiting time [veh·h]": [r["twt_all"] for r in results],
        "Minimum speed [km/h]": [r["min_speed"] for r in results],
        "Constraint violation [%]": [r["queue_violation"] for r in results],
        "Mean computation time [s]": [r["Mean computation time [s]"] for r in results],
        "Max computation time [s]": [r["Max computation time [s]"] for r in results],
    }

def build_table_v(model_type):
    """Evaluates all controllers and constructs Table V for a specific model_type ('best' or 'final')."""
    print(f"\n{'='*60}\nGENERATING TABLE V WITH ALINEA ({model_type.upper()} MODELS)\n{'='*60}")
    
    env_params = METANET_PARAMS["real"]
    mpc_params = METANET_PARAMS["estimated"]
    
    demands = {"Demand 1": create_demand_1(), "Demand 2": create_demand_2()}
    noise_levels = ["Low", "Medium", "High"]
    
    # Model Directories
    drl_10_dir = os.path.join("..", "2_data_produced", "drl_models", "ddpg_10step_td")
    mpc_drl_10_dir = os.path.join("..", "2_data_produced", "mpc_drl_models", "mpc_drl_10step_td")
    
    table_rows = []
    table_rows_stats = []
    
    scenario_idx = 1
    for demand_name, demand_data in demands.items():
        for noise_name in noise_levels:
            scenario_name = f"Scenario {scenario_idx}"
            print(f"\n--- {scenario_name} ({demand_name}, {noise_name} Noise) ---")
            
            noise_cfg = GUSSIAN_NOISE[noise_name]
            tag = f"{demand_name}_{noise_name}"

            # 1. No Control
            res_nc = run_10_seeds(NoControlController, {}, env_params, demand_data, noise_cfg, "No Control")
            
            # 2. Standalone ALINEA
            res_alinea = run_10_seeds(AlineaController, {}, env_params, demand_data, noise_cfg, "ALINEA")
            
            # 3. Standalone MPC
            res_mpc = run_10_seeds(MPCController, {"mpc_params": mpc_params, "w_tts": 1.0, "w_du": 0.4, "num_starts": N_MULTI_START, "verbose": False}, 
                                   env_params, demand_data, noise_cfg, "MPC")
            
            # 4. Standalone DRL (10-step TD)
            res_drl = run_10_seeds(DDPGController, {"n_step": 10, "training": False, "model_dir": drl_10_dir, "model_tag": tag, "model_type": model_type}, 
                                   env_params, demand_data, noise_cfg, "DRL 10-step")
            
            # 5. HF MPC
            res_hfmpc = run_10_seeds(MPCController, {"mpc_params": mpc_params, "Nc": NC_HF, "M_step": M_STEP_HF, "w_tts": 1.0, "w_du": 0.4, "num_starts": N_MULTI_START, "verbose": False}, 
                                     env_params, demand_data, noise_cfg, "HF-MPC")
            
            # 6. MPC-DRL Framework (10-step TD)
            res_mpc_drl = run_10_seeds(MPCDRLController, {"n_step": 10, "model_dir": mpc_drl_10_dir, "model_tag": tag, "model_type": model_type, "mpc_params": mpc_params, "w_tts": 1.0, "w_du": 0.4, "verbose": False}, 
                                       env_params, demand_data, noise_cfg, "MPC-DRL")

            # Append metrics as rows
            metrics = ["Total time spent [veh·h]", "Total waiting time [veh·h]", "Minimum speed [km/h]", 
                       "Constraint violation [%]", "Mean computation time [s]", "Max computation time [s]"]
            
            def format_stat(res_baseline, m):
                if "computation" in m and res_baseline == res_nc:
                    return "-"
                
                arr_base = res_baseline[m]
                arr_target = res_mpc_drl[m]
                
                mean_val = np.mean(arr_base)
                # If values are identical or constant, std will be zero, catch potential warnings
                std_val = np.std(arr_base, ddof=1) if len(arr_base) > 1 else 0.0
                
                if res_baseline is res_mpc_drl:
                    return f"{mean_val:.2f} \u00b1 {std_val:.2f}"
                else:
                    if np.allclose(arr_base, arr_target):
                        pval = 1.0
                    else:
                        _, pval = stats.ttest_rel(arr_base, arr_target)
                    return f"{mean_val:.2f} \u00b1 {std_val:.2f} (p={pval:.3f})"

            for m in metrics:
                # Calculate aggregated table values exactly as before
                nc_val = np.mean(res_nc[m]) if "computation" not in m else "-"
                
                align_val = np.mean(res_alinea[m])
                mpc_val = np.mean(res_mpc[m])
                drl_val = np.mean(res_drl[m])
                hf_mpc_val = np.mean(res_hfmpc[m])
                mpc_drl_val = np.mean(res_mpc_drl[m])
                
                # Exception for max computation time
                if m == "Max computation time [s]":
                    align_val = np.max(res_alinea[m])
                    mpc_val = np.max(res_mpc[m])
                    drl_val = np.max(res_drl[m])
                    hf_mpc_val = np.max(res_hfmpc[m])
                    mpc_drl_val = np.max(res_mpc_drl[m])

                table_rows.append({
                    "Scenarios": scenario_name if m == metrics[0] else "",
                    "Performance": m,
                    "No control": nc_val,
                    "Standalone ALINEA": align_val,
                    "Standalone MPC": mpc_val,
                    "Standalone DRL": drl_val,
                    "HF MPC": hf_mpc_val,
                    "MPC-DRL framework": mpc_drl_val
                })
                
                table_rows_stats.append({
                    "Scenarios": scenario_name if m == metrics[0] else "",
                    "Performance": m,
                    "No control": format_stat(res_nc, m),
                    "Standalone ALINEA": format_stat(res_alinea, m),
                    "Standalone MPC": format_stat(res_mpc, m),
                    "Standalone DRL": format_stat(res_drl, m),
                    "HF MPC": format_stat(res_hfmpc, m),
                    "MPC-DRL framework": format_stat(res_mpc_drl, m)
                })
                
            scenario_idx += 1

    # Standard "As Is" table
    df = pd.DataFrame(table_rows)
    print(f"\n{df.to_string(index=False)}")
    out_path = f"../2_data_produced/results/table_v_with_alinea_{model_type}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {model_type.upper()} table to {out_path}")

    # Statistical table
    df_stats = pd.DataFrame(table_rows_stats)
    print(f"\n{df_stats.to_string(index=False)}")
    out_path_stats = f"../2_data_produced/results/table_v_with_alinea_stats_{model_type}.csv"
    df_stats.to_csv(out_path_stats, index=False)
    print(f"\nSaved {model_type.upper()} explicit statistical table to {out_path_stats}")

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs("../2_data_produced/results", exist_ok=True)
    
    build_table_v(model_type="best")
    #build_table_v(model_type="final")
