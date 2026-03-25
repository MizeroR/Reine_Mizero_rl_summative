# Digital Wellbeing Coach — RL Summative Assignment

**Student:** Reine Mizero
**Repository:** `Reine_Mizero_rl_summative`

---

## Project Overview

This project implements a custom Reinforcement Learning environment modelling a **smartphone addiction intervention system**. An RL agent acts as a digital wellbeing coach that observes a simulated user's behavioural state and recommends interventions to reduce their addiction risk score (SAS-SV).

This is a standalone RL assignment. The domain theme is shared with the capstone project but the RL pipeline is independent.

---

## Environment Description

**File:** `environment/custom_env.py`
**Class:** `WellbeingEnv` (inherits from `gymnasium.Env`)

### Observation Space
7 continuous features (`gym.spaces.Box`):

| Index | Feature | Range | Description |
|---|---|---|---|
| 0 | `screen_time` | [0, 16] | Daily screen time in hours |
| 1 | `unlocks` | [0, 200] | Number of phone unlocks per day |
| 2 | `sleep_quality` | [0, 10] | Sleep quality score |
| 3 | `sas_score` | [0, 48] | SAS-SV smartphone addiction risk score |
| 4 | `social_ratio` | [0, 1] | Proportion of time on social media |
| 5 | `productivity_ratio` | [0, 1] | Proportion of time on productivity apps |
| 6 | `time_of_day` | [0, 23] | Current hour |

### Action Space
6 discrete actions (`gym.spaces.Discrete(6)`):

| Action | Name | Compliance Probability |
|---|---|---|
| 0 | Do Nothing | 100% |
| 1 | Reduce Screen Time (−30 min) | 65% |
| 2 | Block Social Media | 55% |
| 3 | Send Mindfulness Prompt | 75% |
| 4 | Enforce a Break | 50% |
| 5 | Enable Focus Mode | 70% |

### Reward Function
- **+1.5 × ΔSAS** when SAS-SV decreases (risk reduction rewarded)
- **+1.0 × ΔSAS** when SAS-SV increases (penalises risk growth)
- **+1.0** compliance bonus when user follows the intervention
- **−0.5** non-compliance penalty when user ignores the intervention
- **+10.0** terminal bonus for reaching low-risk status (SAS-SV < 12)
- **−0.3** mild penalty for aggressive interventions when risk is already low

### Dynamics
Each step the user state evolves based on:
- Whether the intervention was complied with (stochastic per action)
- Natural upward drift in SAS-SV (+0.3 to +1.0 per step, simulating habitual behaviour)
- Secondary effects on sleep, unlocks, and app-usage ratios

### Terminal Conditions
- **Success:** SAS-SV score drops below 12 (low-risk status achieved)
- **Truncation:** Episode reaches 30 timesteps

### Start State
Random high-risk user: SAS-SV ∈ [30, 48], screen time ∈ [6, 12]h, sleep quality ∈ [2, 6]

---

## Project Structure

```
Reine_Mizero_rl_summative/
├── environment/
│   ├── custom_env.py        # Custom Gymnasium environment
│   └── rendering.py         # Pygame 2D dashboard visualizer
├── training/
│   ├── dqn_training.py      # DQN — 10 HP runs using Stable Baselines3
│   └── pg_training.py       # PPO, A2C, REINFORCE — 10 HP runs each
├── models/
│   ├── dqn/                 # dqn_best.zip + dqn_results.json
│   └── pg/                  # ppo_best.zip, reinforce_best.pt + result JSONs
├── main.py                  # Runs best-performing model with GUI + terminal output
├── requirements.txt
└── README.md
```

---

## Algorithms

| Algorithm | Type | Implementation |
|---|---|---|
| DQN | Value-Based | Stable Baselines3 |
| PPO | Policy Gradient (on-policy) | Stable Baselines3 |
| A2C | Actor-Critic (on-policy) | Stable Baselines3 |
| REINFORCE | Policy Gradient (Monte Carlo) | Manual PyTorch implementation |

> **Note:** REINFORCE is not available in Stable Baselines3 and was implemented manually using a `PolicyNetwork` (2-layer MLP) trained with Monte Carlo returns and gradient clipping.

---

## Hyperparameter Results

All algorithms trained for **80,000 timesteps** (SB3 algorithms) or **800–1,500 episodes** (REINFORCE) on the same `WellbeingEnv`. Evaluation uses **20 episodes** with a deterministic policy.

### DQN — 10 Hyperparameter Runs

| Run | LR | Batch | Gamma | Exp Frac | Target Update | Mean Reward | Std | Time (s) |
|---|---|---|---|---|---|---|---|---|
| **9** ★ | 1e-4 | 32 | 0.90 | 0.35 | 250 | **39.994** | 10.650 | 173.1 |
| 10 | 5e-4 | 64 | 0.97 | 0.20 | 750 | 38.613 | 10.839 | 191.0 |
| 4 | 1e-3 | 32 | 0.95 | 0.25 | 500 | 38.572 | 11.546 | 182.9 |
| 7 | 1e-3 | 64 | 0.95 | 0.15 | 1000 | 37.105 | 6.844 | 188.0 |
| 6 | 2e-4 | 64 | 0.97 | 0.30 | 750 | 36.929 | 7.711 | 184.3 |
| 3 | 1e-4 | 128 | 0.99 | 0.20 | 1000 | 34.896 | 11.493 | 273.5 |
| 2 | 5e-4 | 64 | 0.99 | 0.15 | 500 | 30.942 | 6.937 | 196.8 |
| 1 | 1e-3 | 64 | 0.99 | 0.20 | 500 | 30.650 | 10.438 | 170.9 |
| 5 | 5e-4 | 128 | 0.99 | 0.10 | 250 | 15.233 | 10.508 | 227.8 |
| 8 | 3e-4 | 256 | 0.99 | 0.20 | 500 | 3.811 | 6.720 | 180.1 |

### PPO — 10 Hyperparameter Runs

| Run | LR | n_steps | Batch | Gamma | Ent Coef | Clip | Mean Reward | Std | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| **1** ★ | 3e-4 | 512 | 64 | 0.99 | 0.01 | 0.2 | **44.687** | 8.327 | 258.9 |
| 8 | 3e-4 | 512 | 64 | 0.99 | 0.01 | 0.2 | 43.287 | 10.289 | 236.1 |
| 9 | 1e-4 | 256 | 32 | 0.90 | 0.02 | 0.1 | 42.119 | 9.783 | 343.5 |
| 6 | 3e-4 | 2048 | 64 | 0.99 | 0.01 | 0.2 | 42.064 | 9.983 | 255.9 |
| 3 | 5e-4 | 512 | 128 | 0.99 | 0.02 | 0.2 | 41.761 | 8.769 | 219.1 |
| 7 | 1e-3 | 512 | 64 | 0.97 | 0.005 | 0.2 | 41.341 | 7.417 | 243.4 |
| 10 | 5e-4 | 1024 | 64 | 0.97 | 0.01 | 0.2 | 39.836 | 9.712 | 173.6 |
| 2 | 1e-4 | 1024 | 64 | 0.99 | 0.01 | 0.2 | 38.833 | 8.813 | 347.1 |
| 5 | 2e-4 | 1024 | 128 | 0.99 | 0.01 | 0.3 | 37.714 | 11.799 | 248.3 |
| 4 | 3e-4 | 256 | 32 | 0.95 | 0.00 | 0.1 | 36.971 | 12.685 | 373.1 |

### REINFORCE — 10 Hyperparameter Runs

| Run | LR | Gamma | Hidden | Episodes | Mean Reward | Std | Time (s) |
|---|---|---|---|---|---|---|---|
| **7** ★ | 2e-3 | 0.95 | 128 | 800 | **43.070** | 9.316 | 57.5 |
| 9 | 1e-4 | 0.90 | 64 | 1500 | 42.906 | 10.224 | 118.7 |
| 3 | 2e-3 | 0.99 | 128 | 1000 | 42.810 | 9.324 | 77.7 |
| 2 | 5e-4 | 0.99 | 64 | 1200 | 42.796 | 9.727 | 135.2 |
| 10 | 1e-3 | 0.97 | 128 | 1000 | 41.863 | 10.390 | 97.0 |
| 5 | 3e-4 | 0.99 | 128 | 1500 | 37.874 | 11.113 | 127.4 |
| 8 | 5e-4 | 0.99 | 256 | 1200 | 34.290 | 7.266 | 115.6 |
| 1 | 1e-3 | 0.99 | 128 | 1200 | 32.245 | 9.115 | 116.1 |
| 6 | 1e-3 | 0.97 | 64 | 1000 | 30.547 | 5.936 | 63.8 |
| 4 | 1e-3 | 0.95 | 256 | 1200 | 30.029 | 4.506 | 113.9 |

---

## Overall Comparison

| Algorithm | Best Mean Reward | Std | Best Run Config |
|---|---|---|---|
| **PPO** ★ | **44.687** | **8.327** | lr=3e-4, n_steps=512, clip=0.2 |
| REINFORCE | 43.070 | 9.316 | lr=2e-3, gamma=0.95, hidden=128 |
| DQN | 39.994 | 10.650 | lr=1e-4, batch=32, gamma=0.90 |
| A2C | — | — | Not trained (time constraint) |

**Winner: PPO** — highest mean reward (44.687) and lowest standard deviation (8.327), indicating both peak performance and consistency across evaluation episodes.

---

## Analysis

### Why PPO performed best
PPO's clipping mechanism (clip=0.2) prevents large destructive policy updates, making it stable on this environment where a single bad policy change can cascade across the 30-step episode. Its n_steps=512 setting also provides a good balance between data freshness and update quality.

### Why REINFORCE was competitive despite its simplicity
The environment's short episodes (max 30 steps) reduce the high-variance problem that typically hurts REINFORCE. With a small MDP, Monte Carlo returns are less noisy, allowing REINFORCE to perform surprisingly close to PPO. Notably, REINFORCE also trained significantly faster (57s vs 259s for the best run).

### Why DQN scored lowest
DQN's best configuration used a lower gamma (0.90) and high exploration fraction (0.35), suggesting the agent needed significant randomness to avoid poor local optima. The Q-network had difficulty learning precise value estimates given the stochastic compliance dynamics.

### Observed agent behaviour (demo episode)
The PPO agent demonstrated **action collapse** — using "Reduce Screen Time" 87% of the time. While SAS-SV dropped from 41.5 to 32.9 (−9.2), the agent did not reach the low-risk terminal condition within 30 steps. This reflects a suboptimal but partially learned policy: the agent correctly identifies that reducing screen time lowers risk, but does not diversify across social media blocking or mindfulness prompts which have stronger SAS-SV delta when complied with.

---

## Setup & Usage

### Install dependencies
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Train models
```bash
# Train DQN (10 hyperparameter runs)
python training/dqn_training.py

# Train PPO, A2C, REINFORCE
python training/pg_training.py --algo ppo
python training/pg_training.py --algo a2c
python training/pg_training.py --algo reinforce

# Or all at once
python training/pg_training.py
```

### Run demo
```bash
# Auto-selects best algorithm, opens Pygame GUI
python main.py

# Force a specific algorithm
python main.py --algo ppo

# Headless (terminal only, no Pygame window)
python main.py --no-gui
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `gymnasium` | ≥0.29.0 | RL environment base class |
| `stable-baselines3` | ≥2.2.0 | DQN, PPO, A2C implementations |
| `torch` | ≥2.1.0 | REINFORCE policy network |
| `pygame` | ≥2.5.0 | Live dashboard visualisation |
| `numpy` | ≥1.26.0 | Numerical operations |

> Requires **Python 3.12**. PyTorch does not yet support Python 3.14.
