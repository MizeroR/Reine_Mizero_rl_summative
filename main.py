"""
main.py — Digital Wellbeing Coach RL Demo
Loads the best-performing saved model and runs one full episode
with the Pygame GUI open and verbose terminal output.

Usage:
    python main.py                      # auto-selects best model across all algos
    python main.py --algo dqn
    python main.py --algo ppo
    python main.py --algo a2c
    python main.py --algo reinforce
    python main.py --no-gui             # terminal only (headless)
"""

import os
import sys
import json
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from environment.custom_env import WellbeingEnv, ACTION_NAMES

MODELS_DIR = os.path.join(ROOT, "models")

# ANSI colours for terminal output
_R  = "\033[91m"
_G  = "\033[92m"
_Y  = "\033[93m"
_B  = "\033[94m"
_C  = "\033[96m"
_W  = "\033[97m"
_DIM= "\033[2m"
_RST= "\033[0m"


# ── Model loaders ──────────────────────────────────────────────────────────────

def _load_sb3(algo: str):
    """Load a Stable Baselines3 model (DQN / PPO / A2C)."""
    from stable_baselines3 import DQN, PPO, A2C
    cls_map = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
    subdir = "dqn" if algo == "dqn" else "pg"
    path = os.path.join(MODELS_DIR, subdir, f"{algo}_best.zip")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}\n"
                                f"Run training/{algo}_training.py first.")
    cls = cls_map[algo]
    return cls.load(path)


def _load_reinforce():
    """Load the REINFORCE PyTorch policy."""
    import torch
    from training.pg_training import PolicyNetwork
    path = os.path.join(MODELS_DIR, "pg", "reinforce_best.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}\n"
                                f"Run python training/pg_training.py --algo reinforce first.")
    checkpoint = torch.load(path, map_location="cpu")
    hp     = checkpoint["hp"]
    policy = PolicyNetwork(obs_dim=7, act_dim=6, hidden=hp["hidden"])
    policy.load_state_dict(checkpoint["policy_state"])
    policy.eval()
    return policy


def _predict_reinforce(policy, obs: np.ndarray) -> int:
    import torch
    with torch.no_grad():
        probs  = policy(torch.tensor(obs, dtype=torch.float32))
        return int(probs.argmax().item())


# ── Auto-select best algorithm ─────────────────────────────────────────────────

def _best_algo() -> str:
    """Read saved results JSONs and return the algo with highest mean reward."""
    candidates = {
        "dqn":      os.path.join(MODELS_DIR, "dqn",  "dqn_results.json"),
        "ppo":      os.path.join(MODELS_DIR, "pg",   "ppo_results.json"),
        "a2c":      os.path.join(MODELS_DIR, "pg",   "a2c_results.json"),
        "reinforce":os.path.join(MODELS_DIR, "pg",   "reinforce_results.json"),
    }
    best_algo   = None
    best_reward = -np.inf

    for algo, path in candidates.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            runs = json.load(f)
        top = max(runs, key=lambda r: r["mean_reward"])
        if top["mean_reward"] > best_reward:
            best_reward = top["mean_reward"]
            best_algo   = algo

    if best_algo is None:
        raise RuntimeError(
            "No trained models found.\n"
            "Run: python training/dqn_training.py\n"
            "     python training/pg_training.py"
        )

    return best_algo


# ── Terminal helpers ───────────────────────────────────────────────────────────

def _sas_bar(sas: float, width: int = 30) -> str:
    filled = int(width * sas / 48)
    bar    = "█" * filled + "░" * (width - filled)
    if sas < 12:
        col = _G
    elif sas < 24:
        col = _Y
    elif sas < 36:
        col = "\033[33m"
    else:
        col = _R
    return f"{col}[{bar}]{_RST} {sas:.1f}/48"


def _print_step(step: int, state: np.ndarray, action: int,
                reward: float, complied: bool, info: dict):
    screen, unlocks, sleep, sas, social, prod, tod = state

    complied_str = f"{_G}✓ complied{_RST}" if complied else f"{_R}✗ ignored{_RST}"
    reward_str   = (f"{_G}+{reward:.2f}{_RST}" if reward >= 0
                    else f"{_R}{reward:.2f}{_RST}")

    print(f"\n{_C}{'─'*62}{_RST}")
    print(f"  {_W}Step {step:>2}/30{_RST}   "
          f"Action: {_B}{ACTION_NAMES[action]}{_RST}   "
          f"Reward: {reward_str}   {complied_str}")
    print(f"  SAS-SV  : {_sas_bar(sas)}")
    print(f"  Screen  : {screen:.1f}h   "
          f"Unlocks: {int(unlocks)}   "
          f"Sleep: {sleep:.1f}/10   "
          f"Hour: {int(tod):02d}:00")
    print(f"  Social: {social*100:.0f}%   "
          f"Productivity: {prod*100:.0f}%   "
          f"Entertainment: {max(0, 1-social-prod)*100:.0f}%")


def _print_summary(history: list[dict], total_reward: float, success: bool):
    print(f"\n{_C}{'═'*62}{_RST}")
    print(f"  {_W}Episode Summary{_RST}")
    print(f"{'─'*62}")
    print(f"  Steps taken      : {len(history)}")
    print(f"  Total reward     : {_G if total_reward >= 0 else _R}"
          f"{total_reward:.3f}{_RST}")
    print(f"  Outcome          : {'🟢 LOW RISK ACHIEVED' if success else '🔴 Episode timed out'}")

    action_counts = {}
    for h in history:
        a = h["action"]
        action_counts[a] = action_counts.get(a, 0) + 1

    print(f"\n  {'Action':<30} {'Count':>5}  {'%':>5}")
    print(f"  {'─'*30}  {'─'*5}  {'─'*5}")
    for a, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / len(history)
        print(f"  {ACTION_NAMES[a]:<30} {cnt:>5}  {pct:>4.0f}%")

    compliances = [h["complied"] for h in history if h["action"] != 0]
    if compliances:
        rate = 100 * sum(compliances) / len(compliances)
        print(f"\n  Compliance rate  : {rate:.0f}%")

    sas_start = history[0]["sas"]  if history else 0
    sas_end   = history[-1]["sas"] if history else 0
    print(f"  SAS-SV start→end : {sas_start:.1f} → {sas_end:.1f}  "
          f"(Δ {sas_end - sas_start:+.1f})")
    print(f"{_C}{'═'*62}{_RST}\n")


# ── Main runner ────────────────────────────────────────────────────────────────

def run(algo: str | None = None, gui: bool = True):
    if algo is None:
        print(f"{_DIM}Scanning saved models to find best algorithm...{_RST}")
        algo = _best_algo()

    print(f"\n{_W}╔══════════════════════════════════════════════════════╗")
    print(f"║   Digital Wellbeing Coach — RL Demo                  ║")
    print(f"║   Algorithm : {algo.upper():<38}║")
    print(f"╚══════════════════════════════════════════════════════╝{_RST}\n")

    # Load model
    is_reinforce = (algo == "reinforce")
    if is_reinforce:
        model = _load_reinforce()
    else:
        model = _load_sb3(algo)

    render_mode = "human" if gui else None
    env = WellbeingEnv(render_mode=render_mode)
    obs, info = env.reset(seed=0)

    print(f"  Starting state   : SAS-SV={info['sas_score']:.1f}  "
          f"(high-risk, target < 12)\n")

    total_reward = 0.0
    done         = False
    success      = False

    while not done:
        if is_reinforce:
            action = _predict_reinforce(model, obs)
        else:
            action, _ = model.predict(obs, deterministic=True)
            action    = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done          = terminated or truncated

        _print_step(
            info["step"], obs, action, reward,
            info["complied"], info
        )

        if terminated:
            success = True

    _print_summary(env.history, total_reward, success)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the best RL agent on the Wellbeing env")
    parser.add_argument("--algo",   choices=["dqn", "ppo", "a2c", "reinforce"],
                        default=None, help="Force a specific algorithm")
    parser.add_argument("--no-gui", action="store_true",
                        help="Disable Pygame window (headless mode)")
    args = parser.parse_args()

    run(algo=args.algo, gui=not args.no_gui)
