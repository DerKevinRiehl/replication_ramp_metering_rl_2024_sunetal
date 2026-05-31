"""
Training script for the Combined MPC-DRL Framework (Algorithm 1).

Implements the hierarchical triple-nested loop from Sun et al. (2024):
    Outer : MPC control loop      (T_c = 300 s, 30 steps per episode)
    Middle: DRL control loop      (T_d =  60 s,  5 steps per MPC step)
    Inner : Simulation loop       (T_s =  10 s,  6 steps per DRL step)

Usage examples:
    python train_mpc_drl.py --n_step 10                         # 10-step TD, all 6 scenarios
    python train_mpc_drl.py --n_step 1                          # 1-step TD, all 6 scenarios
    python train_mpc_drl.py --n_step 10 --episodes 5 --scenario "Demand 1" --noise Low

Models saved to  2_data_produced/mpc_drl_models/<label>/<scenario>_<noise>.pt
Learning curves  2_data_produced/mpc_drl_models/<label>/learning_curves.npz
"""

import argparse
import os
import sys
import time
import numpy as np
from tqdm import tqdm

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metanet.metanet_env import MetanetEnv
from controllers.mpc_controller import MPCController
from controllers.ddpg_mpc_agent import DDPGMPCAgent
from config import (
    METANET_PARAMS, GUSSIAN_NOISE,
    TS, TC, TD, T,
    NP, NC, M_STEP,
    NUM_EPISODES, SEED,
)
from util import create_demand_1, create_demand_2

# ======================================================================
# Timing multipliers (paper §III-A)
# ======================================================================
M1 = int(TC / TD)       # 5   — DRL steps per MPC step
M2 = int(TD / TS)       # 6   — simulation steps per DRL step
M  = int(TC / TS)       # 30  — simulation steps per MPC step

N_MPC_STEPS  = int(T / TC)   # 30  — MPC loops per episode
N_DRL_STEPS  = int(T / TD)   # 150 — DRL loops per episode
N_SIM_STEPS  = int(T / TS)   # 900 — simulation loops per episode


# ======================================================================
# Training one scenario
# ======================================================================

def train_one_scenario(
    demand_name: str,
    demand_data: np.ndarray,
    noise_name: str,
    n_step: int,
    num_episodes: int,
    save_dir: str,
    device: str = "cpu",
) -> list[float]:
    """
    Train the MPC-DRL combined framework on a single (demand, noise) scenario.

    Implements Algorithm 1 from the paper exactly:
        for episode in 1..M:
            for k_c in 0..29:                    # MPC outer loop
                solve MPC → u_b
                for k_d in k_c*m1..(k_c+1)*m1-1: # DRL inner loop
                    select u_rl, combine u_c = sat(u_b + u_rl)
                    for k_s in k_d*m2..(k_d+1)*m2-1:  # sim steps
                        env.step(u_c)
                    store transition, train DDPG

    Returns
    -------
    episode_rewards : list[float]
        Total reward per episode (for learning curve).
    """
    # ---- Environment (always uses "real" parameters) ----
    env_params = METANET_PARAMS["real"]
    noise_cfg = GUSSIAN_NOISE[noise_name]

    env = MetanetEnv(
        metanet_param=env_params,
        demand=demand_data,
        noise_cfg=noise_cfg,
        seed=SEED+53,
    )

    # ---- MPC controller (uses "estimated" parameters for model mismatch) ----
    mpc_params = METANET_PARAMS["estimated"]
    mpc = MPCController(
        env=env,
        demand=demand_data,
        mpc_params=mpc_params,
        Np=NP,
        Nc=NC,
        M_step=M_STEP,
        w_tts=1.0,
        w_du=0.4,
        num_starts=1,           # single start for training speed
        verbose=False,
    )

    # ---- DRL agent ----
    agent = DDPGMPCAgent(
        demand=demand_data,
        estimation_step=n_step,
        device=device,
    )

    episode_rewards = []
    best_reward = -np.inf

    label = f"MPC-DRL {n_step}-step TD | {demand_name} | {noise_name}"
    print(f"\n{'='*70}")
    print(f"Training: {label}")
    print(f"  Episodes: {num_episodes} | n-step TD: {n_step}")
    print(f"  MPC steps/ep: {N_MPC_STEPS} | DRL steps/ep: {N_DRL_STEPS} "
          f"| Sim steps/ep: {N_SIM_STEPS}")
    print(f"{'='*70}")

    for ep in tqdm(range(num_episodes), desc=label):
        # ---- Episode reset ----
        obs, _ = env.reset()
        mpc.reset()
        agent.reset()

        ep_reward = 0.0
        ep_critic_loss = 0.0
        n_updates = 0

        # ==============================================================
        # Algorithm 1, Line 6: MPC outer loop  k_c = 0 .. N_MPC_STEPS-1
        # ==============================================================
        for k_c in range(N_MPC_STEPS):
            # Current simulation step at MPC decision point
            sim_step_mpc = k_c * M                             # 0, 30, 60, ...

            # ----------------------------------------------------------
            # Line 7-9: Observe state, solve MPC, obtain u_b(k_c)
            # ----------------------------------------------------------
            u_b = mpc.get_action(obs, sim_step_mpc)            # shape (3,)

            # ==========================================================
            # Line 10: DRL inner loop  k_d = k_c*m1 .. (k_c+1)*m1 - 1
            # ==========================================================
            for k_d_local in range(M1):
                k_d = k_c * M1 + k_d_local                    # global DRL step
                sim_step_drl = k_d * M2                        # sim step at DRL decision

                # ------------------------------------------------------
                # Line 11: Receive state  x_rl(k_d)
                # ------------------------------------------------------
                state = agent.build_state(obs, sim_step_drl, u_b)

                # ------------------------------------------------------
                # Line 12: Select action  u_rl(k_d) = π_θ(x_rl) + w_n
                # ------------------------------------------------------
                u_rl = agent.select_action(state, add_noise=True)

                # ------------------------------------------------------
                # Line 13: Saturation  u_c = sat(u_rl + u_b)
                # ------------------------------------------------------
                u_c = agent.saturate(u_b, u_rl)

                # Update previous action for next state augmentation
                agent.update_prev_action(u_c)

                # ------------------------------------------------------
                # Lines 14-16: Execute u_c for m2 sim steps
                # ------------------------------------------------------
                step_reward = 0.0
                done = False
                for k_s in range(M2):
                    obs, r, terminated, truncated, info = env.step(u_c)
                    step_reward += r
                    if terminated or truncated:
                        done = True
                        break

                ep_reward += step_reward

                # ------------------------------------------------------
                # Line 17: Observe reward r_t(k_d) and next state x_rl(k_d+1)
                # ------------------------------------------------------
                next_sim_step = (k_d + 1) * M2
                next_state = agent.build_state(obs, next_sim_step, u_b)

                # ------------------------------------------------------
                # Line 18: Store transition in R
                # ------------------------------------------------------
                agent.store_transition(
                    state, u_rl, step_reward, next_state, done
                )

                # ------------------------------------------------------
                # Lines 19-22: Sample mini-batch, update critic, actor,
                #              and target networks
                # ------------------------------------------------------
                losses = agent.train_step()
                if losses["critic_loss"] > 0:
                    ep_critic_loss += losses["critic_loss"]
                    n_updates += 1

                if done:
                    break   # episode terminated early

            if done:
                break       # propagate break out of MPC loop

        # ---- End of episode bookkeeping ----
        episode_rewards.append(ep_reward)

        # Save best model
        if ep_reward > best_reward:
            best_reward = ep_reward
            model_path = os.path.join(
                save_dir, f"{demand_name}_{noise_name}_best.pt"
            )
            agent.save(model_path)

        # Periodic logging
        if (ep + 1) % 100 == 0 or ep == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_closs = ep_critic_loss / max(n_updates, 1)
            tqdm.write(
                f"  Ep {ep+1:4d}/{num_episodes} | "
                f"Reward: {ep_reward:.1f} | "
                f"Avg(100): {avg_reward:.1f} | "
                f"CriticL: {avg_closs:.4f} | "
                f"Buffer: {agent.buffer.size} | "
                f"Noise σ: {agent._noise_std:.4f}"
            )

    # Save final model
    final_path = os.path.join(
        save_dir, f"{demand_name}_{noise_name}_final.pt"
    )
    agent.save(final_path)
    print(f"  Final model saved to {final_path}")

    return episode_rewards


# ======================================================================
# Main entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Combined MPC-DRL Framework (Algorithm 1)"
    )
    parser.add_argument(
        "--n_step", type=int, default=10, choices=[1, 10],
        help="TD estimation steps (1 or 10, default: 10)",
    )
    parser.add_argument(
        "--episodes", type=int, default=NUM_EPISODES,
        help=f"Number of training episodes (default: {NUM_EPISODES})",
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        help="Specific demand scenario ('Demand 1' or 'Demand 2')",
    )
    parser.add_argument(
        "--noise", type=str, default=None,
        help="Specific noise level ('Low', 'Medium', 'High')",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device (cpu or cuda)",
    )
    args = parser.parse_args()

    # Output directory
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "2_data_produced", "mpc_drl_models",
        f"mpc_drl_{args.n_step}step_td",
    )
    os.makedirs(base_dir, exist_ok=True)

    # Demand scenarios
    demands = {
        "Demand 1": create_demand_1(),
        "Demand 2": create_demand_2(),
    }
    noise_levels = ["Low", "Medium", "High"]

    # Filter if specific scenario / noise requested
    if args.scenario:
        demands = {k: v for k, v in demands.items() if k == args.scenario}
    if args.noise:
        noise_levels = [n for n in noise_levels if n == args.noise]

    # Training results for learning curves
    training_results = {}
    algo_label = f"MPC-DRL {args.n_step}-step TD"
    training_results[algo_label] = {}

    t0 = time.time()
    
    for demand_name, demand_data in demands.items():
        for noise_name in noise_levels:
            ep_rewards = train_one_scenario(
                demand_name=demand_name,
                demand_data=demand_data,
                noise_name=noise_name,
                n_step=args.n_step,
                num_episodes=args.episodes,
                save_dir=base_dir,
                device=args.device,
            )
            training_results[algo_label][(demand_name, noise_name)] = ep_rewards

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Total training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")

    # Save learning curves (merge with existing .npz to support incremental training)
    curves_path = os.path.join(base_dir, "learning_curves.npz")
    flat = {}
    if os.path.isfile(curves_path):
        existing = np.load(curves_path)
        flat.update({k: existing[k] for k in existing.files})
        existing.close()
        print(f"  Merged with existing curves from {curves_path}")
    for algo, scenarios in training_results.items():
        for (d, n), rewards in scenarios.items():
            key = f"{algo}__{d}__{n}"
            flat[key] = np.array(rewards)
    np.savez(curves_path, **flat)
    print(f"Learning curves saved to {curves_path}")

    # Plot learning curves
    from util import plot_learning_curves
    fig_path = os.path.join(base_dir, "learning_curves.png")
    plot_learning_curves(training_results, save_path=fig_path)


if __name__ == "__main__":
    main()
