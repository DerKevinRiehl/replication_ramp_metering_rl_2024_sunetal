"""
Training script for standalone DDPG controller.

Usage:
    python train_drl.py --n_step 1           # 1-step TD, all 6 scenarios
    python train_drl.py --n_step 10          # 10-step TD, all 6 scenarios
    python train_drl.py --n_step 1 --episodes 5 --scenario "Demand 1" --noise Low

Models are saved to  2_data_produced/drl_models/<label>/<scenario>_<noise>.pt
Learning curves to   2_data_produced/drl_models/<label>/learning_curves.npz
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
from controllers.ddpg_controller import DDPGController
from config import (
    METANET_PARAMS, GUSSIAN_NOISE,
    DDPG_STEP, STEPS_PER_EP, NUM_EPISODES, SEED,
)
from util import create_demand_1, create_demand_2


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
    Train a standalone DDPG agent on a single (demand, noise) scenario.

    Returns
    -------
    episode_rewards : list[float]
        Total reward per episode (for learning curve).
    """
    env_params = METANET_PARAMS["real"]
    noise_cfg = GUSSIAN_NOISE[noise_name]

    env = MetanetEnv(
        metanet_param=env_params,
        demand=demand_data,
        noise_cfg=noise_cfg,
        seed=SEED,
    )

    agent = DDPGController(
        demand=demand_data,
        n_step=n_step,
        training=True,
        device=device,
    )

    episode_rewards = []
    best_reward = -np.inf
    ddpg_step = int(DDPG_STEP)
    steps_per_ep = int(STEPS_PER_EP)

    label = f"DDPG {n_step}-step TD | {demand_name} | {noise_name}"
    print(f"\n{'='*70}")
    print(f"Training: {label}")
    print(f"  Episodes: {num_episodes} | Buffer n-step: {n_step}")
    print(f"{'='*70}")

    for ep in tqdm(range(num_episodes), desc=label):
        obs, _ = env.reset()
        agent.reset()

        ep_reward = 0.0
        ep_critic_loss = 0.0
        n_updates = 0

        for drl_step in range(steps_per_ep):
            sim_step = drl_step * ddpg_step

            # Build state and get action
            state = agent.build_state(obs, sim_step)
            action = agent.get_action(obs, sim_step)
            agent.update_prev_action(action)

            # Execute action for DDPG_STEP sim steps, accumulating reward
            step_reward = 0.0
            done = False
            for sub in range(ddpg_step):
                obs, r, terminated, truncated, info = env.step(action)
                step_reward += r
                if terminated or truncated:
                    done = True
                    break

            ep_reward += step_reward

            # Build next state
            next_sim_step = (drl_step + 1) * ddpg_step
            next_state = agent.build_state(obs, next_sim_step)

            # Store transition
            agent.store_transition(state, action, step_reward, next_state, done)

            # Train
            losses = agent.train_step()
            if losses["critic_loss"] > 0:
                ep_critic_loss += losses["critic_loss"]
                n_updates += 1

            if done:
                break

        episode_rewards.append(ep_reward)

        # Save best model
        if ep_reward > best_reward:
            best_reward = ep_reward
            model_path = os.path.join(save_dir, f"{demand_name}_{noise_name}_best.pt")
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
                f"Buffer: {agent.buffer.size}"
            )

    # Save final model
    final_path = os.path.join(save_dir, f"{demand_name}_{noise_name}_final.pt")
    agent.save(final_path)
    print(f"  Final model saved to {final_path}")

    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Train standalone DDPG controller")
    parser.add_argument("--n_step", type=int, default=1, choices=[1, 10],
                        help="TD steps (1 or 10)")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES,
                        help=f"Number of training episodes (default: {NUM_EPISODES})")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Specific demand scenario ('Demand 1' or 'Demand 2')")
    parser.add_argument("--noise", type=str, default=None,
                        help="Specific noise level ('Low', 'Medium', 'High')")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device (cpu or cuda)")
    args = parser.parse_args()

    # Output directory
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "2_data_produced", "drl_models",
        f"ddpg_{args.n_step}step_td",
    )
    os.makedirs(base_dir, exist_ok=True)

    # Demand scenarios
    demands = {
        "Demand 1": create_demand_1(),
        "Demand 2": create_demand_2(),
    }
    noise_levels = ["Low", "Medium", "High"]

    # Filter if specific scenario/noise requested
    if args.scenario:
        demands = {k: v for k, v in demands.items() if k == args.scenario}
    if args.noise:
        noise_levels = [n for n in noise_levels if n == args.noise]

    # Training results for learning curves
    training_results = {}
    label = f"DDPG {args.n_step}-step TD"
    training_results[label] = {}

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
            training_results[label][(demand_name, noise_name)] = ep_rewards

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
