"""
random_agent.py — Digital Wellbeing Coach Environment
Static demonstration: agent takes purely random actions (no trained model).

Purpose:
    Shows the Pygame visualization and environment dynamics without any
    training involved. Every action is sampled uniformly at random from
    the action space, demonstrating that the environment, observation space,
    action space, and rendering pipeline all work correctly.

Usage:
    python random_agent.py              # with Pygame GUI
    python random_agent.py --no-gui     # terminal only
    python random_agent.py --episodes 3 # run multiple episodes
"""

import os
import sys
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from environment.custom_env import WellbeingEnv, ACTION_NAMES

# ANSI colours
_G  = "\033[92m"
_R  = "\033[91m"
_Y  = "\033[93m"
_C  = "\033[96m"
_W  = "\033[97m"
_B  = "\033[94m"
_RST= "\033[0m"
_DIM= "\033[2m"


def _sas_bar(sas: float, width: int = 28) -> str:
    filled = int(width * sas / 48)
    bar    = "█" * filled + "░" * (width - filled)
    col    = _G if sas < 24 else (_Y if sas < 36 else _R)
    return f"{col}[{bar}]{_RST} {sas:.1f}/48"


def run_random_episode(episode: int, render_mode: str | None, seed: int):
    env = WellbeingEnv(render_mode=render_mode)
    obs, info = env.reset(seed=seed)

    print(f"\n{_W}{'═'*62}{_RST}")
    print(f"  {_C}Episode {episode}  —  Random Agent (no model){_RST}")
    print(f"  Starting SAS-SV : {info['sas_score']:.1f}  (target < 24)")
    print(f"{'─'*62}")

    total_reward = 0.0
    done         = False
    action_counts = {}

    while not done:
        # ── Purely random action ───────────────────────────────────────────
        action = env.action_space.sample()
        action_counts[action] = action_counts.get(action, 0) + 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        screen, unlocks, sleep, sas, social, prod, tod = obs
        complied     = info["complied"]
        step         = info["step"]
        action_name  = ACTION_NAMES[action]

        complied_str = f"{_G}✓{_RST}" if complied else f"{_R}✗{_RST}"
        reward_str   = f"{_G}+{reward:.2f}{_RST}" if reward >= 0 else f"{_R}{reward:.2f}{_RST}"

        print(f"  Step {step:>2}/30  {_B}{action_name:<30}{_RST} "
              f"{complied_str}  reward={reward_str}")
        print(f"         SAS: {_sas_bar(sas)}  "
              f"Screen: {screen:.1f}h  Sleep: {sleep:.1f}/10")

        if terminated:
            print(f"\n  {_G}✓ MILD RISK ACHIEVED at step {step}!{_RST}")
        elif truncated:
            print(f"\n  {_R}✗ Episode timed out (30 steps){_RST}")

        # Slow down so the GUI is watchable
        if render_mode == "human":
            time.sleep(0.4)

    # ── Episode summary ────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  Total reward : {_W}{total_reward:.3f}{_RST}")
    print(f"  SAS change   : {obs[3]:.1f}  (started {info.get('prev_sas', '?')})")

    print(f"\n  {'Action':<30} {'Count':>5}  {'%':>5}")
    print(f"  {'─'*30}  {'─'*5}  {'─'*5}")
    for a, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / info["step"]
        print(f"  {ACTION_NAMES[a]:<30} {cnt:>5}  {pct:>4.0f}%")

    env.close()
    return total_reward


def main():
    parser = argparse.ArgumentParser(
        description="Random agent demo — no model, purely random actions"
    )
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run (default: 1)")
    parser.add_argument("--no-gui",   action="store_true",
                        help="Disable Pygame window")
    parser.add_argument("--seed",     type=int, default=0,
                        help="Base random seed")
    args = parser.parse_args()

    render_mode = None if args.no_gui else "human"

    print(f"\n{_W}╔══════════════════════════════════════════════════════╗")
    print(f"║   Digital Wellbeing Coach — Random Agent Demo        ║")
    print(f"║   Actions chosen uniformly at random  (no model)    ║")
    print(f"╚══════════════════════════════════════════════════════╝{_RST}")
    print(f"{_DIM}  Observation space : 7 continuous features{_RST}")
    print(f"{_DIM}  Action space      : 6 discrete interventions{_RST}")
    print(f"{_DIM}  Terminal condition : SAS-SV < 24 or 30 steps{_RST}")

    rewards = []
    for ep in range(1, args.episodes + 1):
        r = run_random_episode(ep, render_mode, seed=args.seed + ep)
        rewards.append(r)

    if args.episodes > 1:
        print(f"\n{_C}{'═'*62}{_RST}")
        print(f"  {_W}Summary across {args.episodes} random episodes{_RST}")
        print(f"  Mean reward : {np.mean(rewards):.3f}")
        print(f"  Std reward  : {np.std(rewards):.3f}")
        print(f"  Min / Max   : {np.min(rewards):.3f} / {np.max(rewards):.3f}")
        print(f"{_C}{'═'*62}{_RST}\n")


if __name__ == "__main__":
    main()
