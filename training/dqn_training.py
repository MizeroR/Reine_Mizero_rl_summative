"""
DQN Training — Digital Wellbeing Coach Environment
Runs 10 hyperparameter configurations and saves results + best model.

Usage:
    python training/dqn_training.py
"""

import os
import json
import time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import WellbeingEnv

MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "dqn")
RESULTS_FILE = os.path.join(MODELS_DIR, "dqn_results.json")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 10 Hyperparameter configurations ──────────────────────────────────────────
HP_CONFIGS = [
    # (run_id, learning_rate, batch_size, gamma, exploration_fraction, target_update_interval)
    {"run": 1,  "lr": 1e-3,  "batch": 64,  "gamma": 0.99, "exp_frac": 0.20, "target_update": 500},
    {"run": 2,  "lr": 5e-4,  "batch": 64,  "gamma": 0.99, "exp_frac": 0.15, "target_update": 500},
    {"run": 3,  "lr": 1e-4,  "batch": 128, "gamma": 0.99, "exp_frac": 0.20, "target_update": 1000},
    {"run": 4,  "lr": 1e-3,  "batch": 32,  "gamma": 0.95, "exp_frac": 0.25, "target_update": 500},
    {"run": 5,  "lr": 5e-4,  "batch": 128, "gamma": 0.99, "exp_frac": 0.10, "target_update": 250},
    {"run": 6,  "lr": 2e-4,  "batch": 64,  "gamma": 0.97, "exp_frac": 0.30, "target_update": 750},
    {"run": 7,  "lr": 1e-3,  "batch": 64,  "gamma": 0.95, "exp_frac": 0.15, "target_update": 1000},
    {"run": 8,  "lr": 3e-4,  "batch": 256, "gamma": 0.99, "exp_frac": 0.20, "target_update": 500},
    {"run": 9,  "lr": 1e-4,  "batch": 32,  "gamma": 0.90, "exp_frac": 0.35, "target_update": 250},
    {"run": 10, "lr": 5e-4,  "batch": 64,  "gamma": 0.97, "exp_frac": 0.20, "target_update": 750},
]

TOTAL_TIMESTEPS = 80_000
EVAL_EPISODES   = 20


def train_dqn(hp: dict, seed: int = 42) -> dict:
    """Train a single DQN run and return evaluation metrics."""
    env      = WellbeingEnv()
    eval_env = WellbeingEnv()

    model = DQN(
        policy                = "MlpPolicy",
        env                   = env,
        learning_rate         = hp["lr"],
        batch_size            = hp["batch"],
        gamma                 = hp["gamma"],
        exploration_fraction  = hp["exp_frac"],
        target_update_interval= hp["target_update"],
        verbose               = 0,
        seed                  = seed,
        device                = "auto",
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    elapsed = time.time() - t0

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True
    )

    env.close()
    eval_env.close()

    return {
        **hp,
        "mean_reward": round(mean_reward, 4),
        "std_reward":  round(std_reward,  4),
        "train_time_s": round(elapsed, 2),
    }


def run_all():
    results = []
    best_reward = -np.inf
    best_run    = None
    best_model  = None

    print("=" * 60)
    print("DQN Hyperparameter Search — 10 runs")
    print("=" * 60)

    for hp in HP_CONFIGS:
        print(f"\n[Run {hp['run']:>2}/10]  lr={hp['lr']:.0e}  "
              f"batch={hp['batch']:>3}  gamma={hp['gamma']}  "
              f"exp_frac={hp['exp_frac']}", flush=True)

        result = train_dqn(hp, seed=42 + hp["run"])
        results.append(result)

        print(f"  → mean_reward={result['mean_reward']:>8.3f}  "
              f"std={result['std_reward']:.3f}  "
              f"time={result['train_time_s']:.1f}s")

        if result["mean_reward"] > best_reward:
            best_reward = result["mean_reward"]
            best_run    = hp["run"]

            # Re-train on best config to save
            env = WellbeingEnv()
            best_model = DQN(
                policy                = "MlpPolicy",
                env                   = env,
                learning_rate         = hp["lr"],
                batch_size            = hp["batch"],
                gamma                 = hp["gamma"],
                exploration_fraction  = hp["exp_frac"],
                target_update_interval= hp["target_update"],
                verbose               = 0,
                seed                  = 42,
                device                = "auto",
            )
            best_model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
            env.close()

    # Save best model
    best_path = os.path.join(MODELS_DIR, "dqn_best")
    best_model.save(best_path)
    print(f"\n✓ Best model (run {best_run}, reward={best_reward:.3f}) saved → {best_path}.zip")

    # Save results table
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results table saved → {RESULTS_FILE}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Run':>4}  {'LR':>8}  {'Batch':>5}  {'Gamma':>5}  "
          f"{'Mean Reward':>11}  {'Std':>6}  {'Time(s)':>7}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["mean_reward"], reverse=True):
        marker = " ◄ BEST" if r["run"] == best_run else ""
        print(f"{r['run']:>4}  {r['lr']:>8.0e}  {r['batch']:>5}  "
              f"{r['gamma']:>5}  {r['mean_reward']:>11.3f}  "
              f"{r['std_reward']:>6.3f}  {r['train_time_s']:>7.1f}{marker}")

    return results, best_run


if __name__ == "__main__":
    run_all()
