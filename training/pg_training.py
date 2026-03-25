"""
Policy Gradient Training — Digital Wellbeing Coach Environment
Algorithms: PPO, A2C (via Stable Baselines3) + REINFORCE (manual PyTorch)

Each algorithm runs 10 hyperparameter configurations.
Best models are saved under models/pg/.

Usage:
    python training/pg_training.py
    python training/pg_training.py --algo ppo
    python training/pg_training.py --algo a2c
    python training/pg_training.py --algo reinforce
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import WellbeingEnv

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "pg")
os.makedirs(MODELS_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 80_000
EVAL_EPISODES   = 20

# ── HP grids ──────────────────────────────────────────────────────────────────

PPO_CONFIGS = [
    {"run": 1,  "lr": 3e-4,  "n_steps": 512,  "batch": 64,  "gamma": 0.99, "ent_coef": 0.01, "clip": 0.2},
    {"run": 2,  "lr": 1e-4,  "n_steps": 1024, "batch": 64,  "gamma": 0.99, "ent_coef": 0.01, "clip": 0.2},
    {"run": 3,  "lr": 5e-4,  "n_steps": 512,  "batch": 128, "gamma": 0.99, "ent_coef": 0.02, "clip": 0.2},
    {"run": 4,  "lr": 3e-4,  "n_steps": 256,  "batch": 32,  "gamma": 0.95, "ent_coef": 0.00, "clip": 0.1},
    {"run": 5,  "lr": 2e-4,  "n_steps": 1024, "batch": 128, "gamma": 0.99, "ent_coef": 0.01, "clip": 0.3},
    {"run": 6,  "lr": 3e-4,  "n_steps": 2048, "batch": 64,  "gamma": 0.99, "ent_coef": 0.01, "clip": 0.2},
    {"run": 7,  "lr": 1e-3,  "n_steps": 512,  "batch": 64,  "gamma": 0.97, "ent_coef": 0.005,"clip": 0.2},
    {"run": 8,  "lr": 3e-4,  "n_steps": 512,  "batch": 64,  "gamma": 0.99, "ent_coef": 0.01, "clip": 0.2},
    {"run": 9,  "lr": 1e-4,  "n_steps": 256,  "batch": 32,  "gamma": 0.90, "ent_coef": 0.02, "clip": 0.1},
    {"run": 10, "lr": 5e-4,  "n_steps": 1024, "batch": 64,  "gamma": 0.97, "ent_coef": 0.01, "clip": 0.2},
]

A2C_CONFIGS = [
    {"run": 1,  "lr": 7e-4,  "n_steps": 5,   "gamma": 0.99, "ent_coef": 0.01, "vf_coef": 0.5},
    {"run": 2,  "lr": 3e-4,  "n_steps": 5,   "gamma": 0.99, "ent_coef": 0.01, "vf_coef": 0.5},
    {"run": 3,  "lr": 1e-3,  "n_steps": 10,  "gamma": 0.99, "ent_coef": 0.02, "vf_coef": 0.5},
    {"run": 4,  "lr": 5e-4,  "n_steps": 20,  "gamma": 0.95, "ent_coef": 0.00, "vf_coef": 0.25},
    {"run": 5,  "lr": 7e-4,  "n_steps": 5,   "gamma": 0.99, "ent_coef": 0.01, "vf_coef": 1.0},
    {"run": 6,  "lr": 2e-4,  "n_steps": 10,  "gamma": 0.97, "ent_coef": 0.01, "vf_coef": 0.5},
    {"run": 7,  "lr": 1e-3,  "n_steps": 5,   "gamma": 0.95, "ent_coef": 0.005,"vf_coef": 0.5},
    {"run": 8,  "lr": 4e-4,  "n_steps": 15,  "gamma": 0.99, "ent_coef": 0.01, "vf_coef": 0.5},
    {"run": 9,  "lr": 7e-4,  "n_steps": 30,  "gamma": 0.90, "ent_coef": 0.02, "vf_coef": 0.75},
    {"run": 10, "lr": 5e-4,  "n_steps": 5,   "gamma": 0.97, "ent_coef": 0.01, "vf_coef": 0.5},
]

REINFORCE_CONFIGS = [
    {"run": 1,  "lr": 1e-3,  "gamma": 0.99, "hidden": 128, "episodes": 1200},
    {"run": 2,  "lr": 5e-4,  "gamma": 0.99, "hidden": 64,  "episodes": 1200},
    {"run": 3,  "lr": 2e-3,  "gamma": 0.99, "hidden": 128, "episodes": 1000},
    {"run": 4,  "lr": 1e-3,  "gamma": 0.95, "hidden": 256, "episodes": 1200},
    {"run": 5,  "lr": 3e-4,  "gamma": 0.99, "hidden": 128, "episodes": 1500},
    {"run": 6,  "lr": 1e-3,  "gamma": 0.97, "hidden": 64,  "episodes": 1000},
    {"run": 7,  "lr": 2e-3,  "gamma": 0.95, "hidden": 128, "episodes": 800},
    {"run": 8,  "lr": 5e-4,  "gamma": 0.99, "hidden": 256, "episodes": 1200},
    {"run": 9,  "lr": 1e-4,  "gamma": 0.90, "hidden": 64,  "episodes": 1500},
    {"run": 10, "lr": 1e-3,  "gamma": 0.97, "hidden": 128, "episodes": 1000},
]


# ── PPO ────────────────────────────────────────────────────────────────────────

def train_ppo(hp: dict, seed: int = 42) -> dict:
    env      = WellbeingEnv()
    eval_env = WellbeingEnv()

    model = PPO(
        policy      = "MlpPolicy",
        env         = env,
        learning_rate = hp["lr"],
        n_steps     = hp["n_steps"],
        batch_size  = hp["batch"],
        gamma       = hp["gamma"],
        ent_coef    = hp["ent_coef"],
        clip_range  = hp["clip"],
        verbose     = 0,
        seed        = seed,
        device      = "auto",
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    elapsed = time.time() - t0

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True)
    env.close(); eval_env.close()

    return {**hp, "mean_reward": round(mean_r, 4), "std_reward": round(std_r, 4),
            "train_time_s": round(elapsed, 2)}


# ── A2C ────────────────────────────────────────────────────────────────────────

def train_a2c(hp: dict, seed: int = 42) -> dict:
    env      = WellbeingEnv()
    eval_env = WellbeingEnv()

    model = A2C(
        policy      = "MlpPolicy",
        env         = env,
        learning_rate = hp["lr"],
        n_steps     = hp["n_steps"],
        gamma       = hp["gamma"],
        ent_coef    = hp["ent_coef"],
        vf_coef     = hp["vf_coef"],
        verbose     = 0,
        seed        = seed,
        device      = "auto",
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    elapsed = time.time() - t0

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True)
    env.close(); eval_env.close()

    return {**hp, "mean_reward": round(mean_r, 4), "std_reward": round(std_r, 4),
            "train_time_s": round(elapsed, 2)}


# ── REINFORCE (manual PyTorch) ─────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Compute discounted returns G_t for each timestep."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    # Normalise for training stability
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def train_reinforce(hp: dict, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = WellbeingEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy    = PolicyNetwork(obs_dim, act_dim, hp["hidden"])
    optimizer = optim.Adam(policy.parameters(), lr=hp["lr"])

    episode_rewards = []

    t0 = time.time()

    for ep in range(hp["episodes"]):
        obs, _ = env.reset(seed=seed + ep)
        log_probs: list[torch.Tensor] = []
        rewards:   list[float]        = []
        done = False

        while not done:
            obs_t  = torch.tensor(obs, dtype=torch.float32)
            probs  = policy(obs_t)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))

            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        episode_rewards.append(sum(rewards))

        returns   = compute_returns(rewards, hp["gamma"])
        log_probs = torch.stack(log_probs)
        loss      = -(log_probs * returns).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

    elapsed = time.time() - t0

    # Evaluation
    eval_rewards = []
    for _ in range(EVAL_EPISODES):
        obs, _ = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            with torch.no_grad():
                probs  = policy(torch.tensor(obs, dtype=torch.float32))
                action = probs.argmax().item()
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            done = term or trunc
        eval_rewards.append(ep_r)

    env.close()

    mean_r = float(np.mean(eval_rewards))
    std_r  = float(np.std(eval_rewards))

    return {
        **hp,
        "mean_reward":  round(mean_r, 4),
        "std_reward":   round(std_r,  4),
        "train_time_s": round(elapsed, 2),
    }


def save_reinforce_policy(hp: dict, path: str, seed: int = 42):
    """Re-train REINFORCE with best HP and save state dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env     = WellbeingEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy    = PolicyNetwork(obs_dim, act_dim, hp["hidden"])
    optimizer = optim.Adam(policy.parameters(), lr=hp["lr"])

    for ep in range(hp["episodes"]):
        obs, _ = env.reset(seed=seed + ep)
        log_probs: list[torch.Tensor] = []
        rewards:   list[float]        = []
        done = False
        while not done:
            obs_t  = torch.tensor(obs, dtype=torch.float32)
            probs  = policy(obs_t)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        returns   = compute_returns(rewards, hp["gamma"])
        log_probs = torch.stack(log_probs)
        loss      = -(log_probs * returns).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

    env.close()
    torch.save({"policy_state": policy.state_dict(), "hp": hp}, path)


# ── Generic runner ─────────────────────────────────────────────────────────────

def _run_grid(algo: str, configs: list[dict],
              train_fn, header_fields: list[str]) -> tuple[list[dict], int]:
    results = []
    best_reward = -np.inf
    best_run    = None

    print("=" * 70)
    print(f"{algo.upper()} Hyperparameter Search — {len(configs)} runs")
    print("=" * 70)

    for hp in configs:
        fields = "  ".join(f"{k}={v}" for k, v in hp.items()
                           if k in header_fields)
        print(f"\n[Run {hp['run']:>2}/{len(configs)}]  {fields}", flush=True)

        result = train_fn(hp, seed=42 + hp["run"])
        results.append(result)

        print(f"  → mean_reward={result['mean_reward']:>8.3f}  "
              f"std={result['std_reward']:.3f}  "
              f"time={result['train_time_s']:.1f}s")

        if result["mean_reward"] > best_reward:
            best_reward = result["mean_reward"]
            best_run    = hp["run"]

    return results, best_run


def _print_table(results: list[dict], best_run: int, cols: list[str]):
    header = f"{'Run':>4}  " + "  ".join(f"{c:>10}" for c in cols) + \
             f"  {'Mean Reward':>11}  {'Std':>6}  {'Time(s)':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x["mean_reward"], reverse=True):
        row = f"{r['run']:>4}  " + "  ".join(f"{r[c]:>10}" for c in cols) + \
              f"  {r['mean_reward']:>11.3f}  {r['std_reward']:>6.3f}  {r['train_time_s']:>7.1f}"
        if r["run"] == best_run:
            row += "  ◄ BEST"
        print(row)


# ── Entry points ───────────────────────────────────────────────────────────────

def run_ppo():
    results, best_run = _run_grid(
        "PPO", PPO_CONFIGS, train_ppo,
        ["lr", "n_steps", "batch", "gamma", "clip"]
    )
    with open(os.path.join(MODELS_DIR, "ppo_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _print_table(results, best_run, ["lr", "n_steps", "batch", "gamma"])

    # Save best PPO model
    best_hp = next(c for c in PPO_CONFIGS if c["run"] == best_run)
    env = WellbeingEnv()
    m = PPO("MlpPolicy", env, learning_rate=best_hp["lr"],
            n_steps=best_hp["n_steps"], batch_size=best_hp["batch"],
            gamma=best_hp["gamma"], ent_coef=best_hp["ent_coef"],
            clip_range=best_hp["clip"], verbose=0, seed=42, device="auto")
    m.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    env.close()
    path = os.path.join(MODELS_DIR, "ppo_best")
    m.save(path)
    print(f"\n✓ PPO best model (run {best_run}) saved → {path}.zip")
    return results, best_run


def run_a2c():
    results, best_run = _run_grid(
        "A2C", A2C_CONFIGS, train_a2c,
        ["lr", "n_steps", "gamma", "ent_coef", "vf_coef"]
    )
    with open(os.path.join(MODELS_DIR, "a2c_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _print_table(results, best_run, ["lr", "n_steps", "gamma", "vf_coef"])

    # Save best A2C model
    best_hp = next(c for c in A2C_CONFIGS if c["run"] == best_run)
    env = WellbeingEnv()
    m = A2C("MlpPolicy", env, learning_rate=best_hp["lr"],
            n_steps=best_hp["n_steps"], gamma=best_hp["gamma"],
            ent_coef=best_hp["ent_coef"], vf_coef=best_hp["vf_coef"],
            verbose=0, seed=42, device="auto")
    m.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    env.close()
    path = os.path.join(MODELS_DIR, "a2c_best")
    m.save(path)
    print(f"\n✓ A2C best model (run {best_run}) saved → {path}.zip")
    return results, best_run


def run_reinforce():
    results, best_run = _run_grid(
        "REINFORCE", REINFORCE_CONFIGS, train_reinforce,
        ["lr", "gamma", "hidden", "episodes"]
    )
    with open(os.path.join(MODELS_DIR, "reinforce_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _print_table(results, best_run, ["lr", "gamma", "hidden", "episodes"])

    # Save best REINFORCE policy
    best_hp = next(c for c in REINFORCE_CONFIGS if c["run"] == best_run)
    path = os.path.join(MODELS_DIR, "reinforce_best.pt")
    save_reinforce_policy(best_hp, path, seed=42)
    print(f"\n✓ REINFORCE best model (run {best_run}) saved → {path}")
    return results, best_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "a2c", "reinforce", "all"],
                        default="all")
    args = parser.parse_args()

    if args.algo in ("ppo",  "all"): run_ppo()
    if args.algo in ("a2c",  "all"): run_a2c()
    if args.algo in ("reinforce", "all"): run_reinforce()
