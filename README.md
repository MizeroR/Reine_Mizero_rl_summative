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

All algorithms trained for **80,000 timesteps** (SB3 algorithms) or **800–1,500 episodes** (REINFORCE) on the same `WellbeingEnv`. Evaluation uses **20 episodes** with a deterministic policy. Terminal condition: SAS-SV < 24 (mild-risk status).

### DQN — 10 Hyperparameter Runs

| Run | LR | Batch | Gamma | Exp Frac | Target Update | Mean Reward | Std | Time (s) |
|---|---|---|---|---|---|---|---|---|
| **9** ★ | 1e-4 | 32 | 0.90 | 0.35 | 250 | **57.053** | 9.160 | 166.3 |
| 4 | 1e-3 | 32 | 0.95 | 0.25 | 500 | 55.998 | 6.658 | 192.1 |
| 6 | 2e-4 | 64 | 0.97 | 0.30 | 750 | 55.882 | 9.633 | 182.8 |
| 3 | 1e-4 | 128 | 0.99 | 0.20 | 1000 | 51.830 | 6.075 | 194.8 |
| 7 | 1e-3 | 64 | 0.95 | 0.15 | 1000 | 50.835 | 7.679 | 190.2 |
| 10 | 5e-4 | 64 | 0.97 | 0.20 | 750 | 50.045 | 9.042 | 146.0 |
| 5 | 5e-4 | 128 | 0.99 | 0.10 | 250 | 49.580 | 6.686 | 188.5 |
| 8 | 3e-4 | 256 | 0.99 | 0.20 | 500 | 49.014 | 8.954 | 188.0 |
| 1 | 1e-3 | 64 | 0.99 | 0.20 | 500 | 47.088 | 7.254 | 205.3 |
| 2 | 5e-4 | 64 | 0.99 | 0.15 | 500 | 46.767 | 6.707 | 177.4 |

### PPO — 10 Hyperparameter Runs

| Run | LR | n_steps | Batch | Gamma | Ent Coef | Clip | Mean Reward | Std | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| **10** ★ | 5e-4 | 1024 | 64 | 0.97 | 0.01 | 0.2 | **59.648** | 7.118 | 208.2 |
| 2 | 1e-4 | 1024 | 64 | 0.99 | 0.01 | 0.2 | 59.640 | 8.340 | 242.0 |
| 4 | 3e-4 | 256 | 32 | 0.95 | 0.00 | 0.1 | 56.782 | 9.087 | 408.9 |
| 7 | 1e-3 | 512 | 64 | 0.97 | 0.005 | 0.2 | 55.810 | 8.474 | 253.4 |
| 9 | 1e-4 | 256 | 32 | 0.90 | 0.02 | 0.1 | 55.348 | 9.982 | 325.1 |
| 5 | 2e-4 | 1024 | 128 | 0.99 | 0.01 | 0.3 | 54.717 | 6.492 | 225.0 |
| 3 | 5e-4 | 512 | 128 | 0.99 | 0.02 | 0.2 | 53.009 | 10.017 | 198.3 |
| 6 | 3e-4 | 2048 | 64 | 0.99 | 0.01 | 0.2 | 52.419 | 7.846 | 252.3 |
| 8 | 3e-4 | 512 | 64 | 0.99 | 0.01 | 0.2 | 51.965 | 8.289 | 248.8 |
| 1 | 3e-4 | 512 | 64 | 0.99 | 0.01 | 0.2 | 49.109 | 8.378 | 280.1 |

### A2C — 10 Hyperparameter Runs

| Run | LR | n_steps | Gamma | Ent Coef | VF Coef | Mean Reward | Std | Time (s) |
|---|---|---|---|---|---|---|---|---|
| **9** ★ | 7e-4 | 30 | 0.90 | 0.02 | 0.75 | **58.647** | 9.666 | 163.4 |
| 8 | 4e-4 | 15 | 0.99 | 0.01 | 0.50 | 58.322 | 8.881 | 187.5 |
| 7 | 1e-3 | 5 | 0.95 | 0.005 | 0.50 | 58.205 | 9.135 | 257.7 |
| 2 | 3e-4 | 5 | 0.99 | 0.01 | 0.50 | 58.050 | 7.856 | 249.4 |
| 6 | 2e-4 | 10 | 0.97 | 0.01 | 0.50 | 57.202 | 9.145 | 205.6 |
| 4 | 5e-4 | 20 | 0.95 | 0.00 | 0.25 | 56.176 | 11.410 | 213.3 |
| 10 | 5e-4 | 5 | 0.97 | 0.01 | 0.50 | 56.080 | 9.606 | 244.8 |
| 1 | 7e-4 | 5 | 0.99 | 0.01 | 0.50 | 55.620 | 9.339 | 285.4 |
| 3 | 1e-3 | 10 | 0.99 | 0.02 | 0.50 | 54.883 | 8.364 | 215.0 |
| 5 | 7e-4 | 5 | 0.99 | 0.01 | 1.00 | 53.869 | 9.867 | 282.1 |

### REINFORCE — 10 Hyperparameter Runs

| Run | LR | Gamma | Hidden | Episodes | Mean Reward | Std | Time (s) |
|---|---|---|---|---|---|---|---|
| **10** ★ | 1e-3 | 0.97 | 128 | 1000 | **59.977** | 8.626 | 123.7 |
| 6 | 1e-3 | 0.97 | 64 | 1000 | 59.343 | 6.315 | 82.9 |
| 4 | 1e-3 | 0.95 | 256 | 1200 | 59.274 | 7.820 | 149.5 |
| 9 | 1e-4 | 0.90 | 64 | 1500 | 58.951 | 8.537 | 150.5 |
| 3 | 2e-3 | 0.99 | 128 | 1000 | 58.014 | 10.621 | 103.2 |
| 7 | 2e-3 | 0.95 | 128 | 800 | 57.429 | 8.393 | 101.7 |
| 2 | 5e-4 | 0.99 | 64 | 1200 | 55.605 | 6.453 | 84.8 |
| 1 | 1e-3 | 0.99 | 128 | 1200 | 50.139 | 9.937 | 155.5 |
| 8 | 5e-4 | 0.99 | 256 | 1200 | 49.627 | 7.103 | 208.5 |
| 5 | 3e-4 | 0.99 | 128 | 1500 | 46.556 | 7.252 | 200.4 |

---

## Overall Comparison

| Algorithm | Best Mean Reward | Std | Best Run Config |
|---|---|---|---|
| **REINFORCE** ★ | **59.977** | 8.626 | lr=1e-3, gamma=0.97, hidden=128, episodes=1000 |
| PPO | 59.648 | **7.118** | lr=5e-4, n_steps=1024, clip=0.2 |
| A2C | 58.647 | 9.666 | lr=7e-4, n_steps=30, gamma=0.90 |
| DQN | 57.053 | 9.160 | lr=1e-4, batch=32, gamma=0.90 |

**Winner: REINFORCE** — highest mean reward (59.977). PPO is the most consistent with the lowest standard deviation (7.118), making it the most reliable choice in production.

---

## Analysis

### Why REINFORCE won
With the corrected environment (reduced natural drift, terminal condition at SAS < 24), episodes are short enough that Monte Carlo returns have low variance. This is exactly the setting where REINFORCE excels — it sees complete episodes and updates cleanly without the complexity of bootstrapping. The winning config (lr=1e-3, gamma=0.97, hidden=128) uses a moderate discount that slightly de-emphasises distant steps, which suits a 30-step episode well.

### Why PPO is the most consistent
PPO's clipping mechanism (clip=0.2) prevents large destructive policy updates. It achieved the lowest std (7.118) across evaluation episodes, meaning it performs reliably regardless of the initial user state. In a real deployment scenario PPO would be the safer choice despite REINFORCE's marginally higher peak reward.

### Why A2C and DQN trailed
A2C updates very frequently (n_steps=5–30) which introduces noise on this stochastic environment. Its best run used n_steps=30 which is actually one full episode — essentially behaving like REINFORCE, explaining why it scored close to the top. DQN struggled because its Q-network must model the precise value of each state under stochastic compliance, which is harder than learning a direct policy. Notably DQN's best config again used low gamma (0.90) and high exploration (0.35), consistent with the first training run.

### Observed agent behaviour (demo episode)
The REINFORCE agent used "Send Mindfulness Prompt" 100% of the time, reducing SAS-SV from 40.5 to 22.7 (Δ −17.8) and terminating just before the 30-step limit. This is another case of action collapse — the agent found one action that reliably reduces SAS via sleep improvement and committed to it. A future improvement would be adding an entropy bonus to the reward to encourage action diversity across social media blocking and focus mode interventions.

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
