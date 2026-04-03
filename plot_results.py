"""
plot_results.py — Digital Wellbeing Coach RL
Generates all report figures from saved results JSON files.

Figures produced (saved to plots/):
    1. cumulative_rewards.png   — mean reward per run, DQN / PPO / REINFORCE (subplots)
    2. convergence.png          — best-run reward ranked across all algorithms
    3. dqn_analysis.png         — DQN: effect of gamma, exploration fraction, batch size
    4. pg_entropy.png           — PPO entropy coefficient effect + REINFORCE gamma effect
    5. hyperparameter_heatmap.png — LR vs gamma reward heatmap (DQN & PPO)
    6. generalization.png       — reward std comparison (stability / generalization)

Usage:
    python plot_results.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT      = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load results ───────────────────────────────────────────────────────────────

def load(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)

DQN_R = load(os.path.join(ROOT, "models", "dqn", "dqn_results.json"))
PPO_R = load(os.path.join(ROOT, "models", "pg",  "ppo_results.json"))
RF_R  = load(os.path.join(ROOT, "models", "pg",  "reinforce_results.json"))

ALGO_DATA = {
    "DQN":       (DQN_R, "#4A90D9"),
    "PPO":       (PPO_R, "#E67E22"),
    "REINFORCE": (RF_R,  "#9B59B6"),
}

STYLE = {
    "figure.facecolor": "#0F0F1A",
    "axes.facecolor":   "#1A1A2E",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#CCCCEE",
    "axes.titlecolor":  "#FFFFFF",
    "xtick.color":      "#AAAACC",
    "ytick.color":      "#AAAACC",
    "grid.color":       "#333355",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "text.color":       "#DDDDFF",
    "legend.facecolor": "#1A1A2E",
    "legend.edgecolor": "#444466",
}
plt.rcParams.update(STYLE)


def _save(name: str):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=plt.rcParams["figure.facecolor"])
    plt.close()
    print(f"  ✓ Saved → {path}")


# ── 1. Cumulative Rewards — 3 algorithms in subplots ─────────────────────────

def plot_cumulative_rewards():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Mean Reward per Hyperparameter Run — All Algorithms",
                 fontsize=14, fontweight="bold", y=1.03)

    for ax, (algo, (data, color)) in zip(axes.flat, ALGO_DATA.items()):
        runs    = [d["run"]         for d in data]
        means   = [d["mean_reward"] for d in data]
        stds    = [d["std_reward"]  for d in data]
        best_r  = max(means)
        best_i  = means.index(best_r)

        bars = ax.bar(runs, means, color=color, alpha=0.75, zorder=3,
                      label=algo, edgecolor="white", linewidth=0.4)
        ax.errorbar(runs, means, yerr=stds, fmt="none",
                    ecolor="white", elinewidth=1, capsize=3, alpha=0.6)

        # Highlight best bar
        bars[best_i].set_edgecolor("white")
        bars[best_i].set_linewidth(2)
        ax.annotate(f"★ {best_r:.1f}",
                    xy=(runs[best_i], best_r),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8, color="white", fontweight="bold")

        ax.set_title(algo, fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("Run #", fontsize=9)
        ax.set_ylabel("Mean Reward", fontsize=9)
        ax.set_xticks(runs)
        ax.set_ylim(0, max(means) * 1.25)
        ax.grid(axis="y", zorder=0)

    plt.tight_layout()
    _save("cumulative_rewards.png")


# ── 2. Convergence — best reward per algorithm ranked ─────────────────────────

def plot_convergence():
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Best Mean Reward — Algorithm Comparison", fontsize=13,
                 fontweight="bold")

    algos  = list(ALGO_DATA.keys())
    colors = [v[1] for v in ALGO_DATA.values()]
    bests  = [max(d["mean_reward"] for d in v[0]) for v in ALGO_DATA.values()]
    stds   = [next(d["std_reward"]  for d in v[0]
                   if d["mean_reward"] == max(dd["mean_reward"] for dd in v[0]))
              for v in ALGO_DATA.values()]

    sorted_data = sorted(zip(bests, stds, algos, colors), reverse=True)
    bests_s, stds_s, algos_s, colors_s = zip(*sorted_data)

    bars = ax.barh(algos_s, bests_s, color=colors_s, alpha=0.8,
                   edgecolor="white", linewidth=0.5, height=0.5)
    ax.errorbar(bests_s, algos_s, xerr=stds_s, fmt="none",
                ecolor="white", elinewidth=1.5, capsize=5)

    for bar, val in zip(bars, bests_s):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, color="white",
                fontweight="bold")

    ax.set_xlabel("Mean Reward (20 eval episodes)", fontsize=10)
    ax.set_xlim(0, max(bests_s) * 1.18)
    ax.grid(axis="x", zorder=0)
    ax.invert_yaxis()
    plt.tight_layout()
    _save("convergence.png")


# ── 3. DQN Analysis — gamma, exploration fraction, batch ──────────────────────

def plot_dqn_analysis():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("DQN Hyperparameter Analysis", fontsize=13, fontweight="bold")
    color = ALGO_DATA["DQN"][1]

    # Gamma vs mean reward
    gammas = sorted(set(d["gamma"] for d in DQN_R))
    g_means = [np.mean([d["mean_reward"] for d in DQN_R if d["gamma"] == g])
               for g in gammas]
    axes[0].bar([str(g) for g in gammas], g_means, color=color, alpha=0.8,
                edgecolor="white", linewidth=0.4)
    axes[0].set_title("Gamma vs Mean Reward", fontsize=11)
    axes[0].set_xlabel("Gamma (discount factor)")
    axes[0].set_ylabel("Mean Reward")
    axes[0].grid(axis="y")

    # Exploration fraction vs mean reward
    exps = sorted(set(d["exp_frac"] for d in DQN_R))
    e_means = [np.mean([d["mean_reward"] for d in DQN_R if d["exp_frac"] == e])
               for e in exps]
    axes[1].bar([str(e) for e in exps], e_means, color=color, alpha=0.8,
                edgecolor="white", linewidth=0.4)
    axes[1].set_title("Exploration Fraction vs Mean Reward", fontsize=11)
    axes[1].set_xlabel("Exploration Fraction")
    axes[1].set_ylabel("Mean Reward")
    axes[1].grid(axis="y")

    # Batch size vs mean reward
    batches = sorted(set(d["batch"] for d in DQN_R))
    b_means = [np.mean([d["mean_reward"] for d in DQN_R if d["batch"] == b])
               for b in batches]
    axes[2].bar([str(b) for b in batches], b_means, color=color, alpha=0.8,
                edgecolor="white", linewidth=0.4)
    axes[2].set_title("Batch Size vs Mean Reward", fontsize=11)
    axes[2].set_xlabel("Batch Size")
    axes[2].set_ylabel("Mean Reward")
    axes[2].grid(axis="y")

    plt.tight_layout()
    _save("dqn_analysis.png")


# ── 4. PG Entropy — effect of entropy coefficient (PPO & REINFORCE) ──────────

def plot_pg_entropy():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Policy Gradient — Entropy & Gamma Effect", fontsize=13,
                 fontweight="bold")

    # PPO — entropy coefficient effect
    ax = axes[0]
    color = ALGO_DATA["PPO"][1]
    ents  = sorted(set(d["ent_coef"] for d in PPO_R))
    means = [np.mean([d["mean_reward"] for d in PPO_R if d["ent_coef"] == e]) for e in ents]
    stds  = [np.std( [d["mean_reward"] for d in PPO_R if d["ent_coef"] == e]) for e in ents]
    ax.plot([str(e) for e in ents], means, marker="o", color=color,
            linewidth=2, markersize=8, label="Mean Reward")
    ax.fill_between(range(len(ents)),
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=color, alpha=0.2, label="±1 std")
    ax.set_xticks(range(len(ents)))
    ax.set_xticklabels([str(e) for e in ents])
    ax.set_title("PPO — Entropy Coefficient Effect", fontsize=11)
    ax.set_xlabel("Entropy Coefficient (ent_coef)")
    ax.set_ylabel("Mean Reward")
    ax.legend(fontsize=9)
    ax.grid()

    # REINFORCE — gamma effect
    ax = axes[1]
    color = ALGO_DATA["REINFORCE"][1]
    gammas = sorted(set(d["gamma"] for d in RF_R))
    means  = [np.mean([d["mean_reward"] for d in RF_R if d["gamma"] == g]) for g in gammas]
    stds   = [np.std( [d["mean_reward"] for d in RF_R if d["gamma"] == g]) for g in gammas]
    ax.plot([str(g) for g in gammas], means, marker="o", color=color,
            linewidth=2, markersize=8, label="Mean Reward")
    ax.fill_between(range(len(gammas)),
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=color, alpha=0.2, label="±1 std")
    ax.set_xticks(range(len(gammas)))
    ax.set_xticklabels([str(g) for g in gammas])
    ax.set_title("REINFORCE — Gamma (Discount Factor) Effect", fontsize=11)
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Mean Reward")
    ax.legend(fontsize=9)
    ax.grid()

    plt.tight_layout()
    _save("pg_entropy.png")


# ── 5. Hyperparameter Heatmap — LR vs Gamma ───────────────────────────────────

def plot_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Learning Rate × Gamma — Mean Reward Heatmap", fontsize=13,
                 fontweight="bold")

    for ax, (algo, data, _) in zip(axes, [
        ("DQN", DQN_R, None),
        ("PPO", PPO_R, None),
    ]):
        lrs    = sorted(set(d["lr"]    for d in data))
        gammas = sorted(set(d["gamma"] for d in data))

        matrix = np.full((len(gammas), len(lrs)), np.nan)
        for d in data:
            gi = gammas.index(d["gamma"])
            li = lrs.index(d["lr"])
            if np.isnan(matrix[gi, li]):
                matrix[gi, li] = d["mean_reward"]
            else:
                matrix[gi, li] = max(matrix[gi, li], d["mean_reward"])

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                       vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
        plt.colorbar(im, ax=ax, label="Mean Reward")
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], rotation=45, fontsize=8)
        ax.set_yticks(range(len(gammas)))
        ax.set_yticklabels([str(g) for g in gammas])
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Gamma")
        ax.set_title(f"{algo}", fontsize=11, fontweight="bold")

        for gi in range(len(gammas)):
            for li in range(len(lrs)):
                if not np.isnan(matrix[gi, li]):
                    ax.text(li, gi, f"{matrix[gi,li]:.0f}", ha="center",
                            va="center", fontsize=7, color="black", fontweight="bold")

    plt.tight_layout()
    _save("hyperparameter_heatmap.png")


# ── 6. Generalization — std comparison across algorithms ──────────────────────

def plot_generalization():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Stability & Generalization Analysis", fontsize=13,
                 fontweight="bold")

    # Left: reward std per algo (lower = more stable)
    algos  = list(ALGO_DATA.keys())
    colors = [v[1] for v in ALGO_DATA.values()]
    best_stds = [
        next(d["std_reward"] for d in v[0]
             if d["mean_reward"] == max(dd["mean_reward"] for dd in v[0]))
        for v in ALGO_DATA.values()
    ]
    axes[0].bar(algos, best_stds, color=colors, alpha=0.8,
                edgecolor="white", linewidth=0.5)
    axes[0].set_title("Reward Std — Best Run per Algorithm\n(lower = more stable)",
                      fontsize=10)
    axes[0].set_ylabel("Std Reward")
    axes[0].grid(axis="y")
    for i, (a, s) in enumerate(zip(algos, best_stds)):
        axes[0].text(i, s + 0.1, f"{s:.2f}", ha="center", fontsize=9,
                     color="white", fontweight="bold")

    # Right: reward range (max - min across runs) per algo
    ranges = [
        max(d["mean_reward"] for d in v[0]) - min(d["mean_reward"] for d in v[0])
        for v in ALGO_DATA.values()
    ]
    axes[1].bar(algos, ranges, color=colors, alpha=0.8,
                edgecolor="white", linewidth=0.5)
    axes[1].set_title("Reward Range Across HP Runs\n(lower = less sensitive to tuning)",
                      fontsize=10)
    axes[1].set_ylabel("Max − Min Mean Reward")
    axes[1].grid(axis="y")
    for i, (a, r) in enumerate(zip(algos, ranges)):
        axes[1].text(i, r + 0.1, f"{r:.2f}", ha="center", fontsize=9,
                     color="white", fontweight="bold")

    plt.tight_layout()
    _save("generalization.png")


# ── Run all ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nGenerating report figures...")
    plot_cumulative_rewards()
    plot_convergence()
    plot_dqn_analysis()
    plot_pg_entropy()
    plot_heatmap()
    plot_generalization()
    print(f"\nAll figures saved to → plots/")
    print("Include these in your report under the Discussion & Analysis section.")
