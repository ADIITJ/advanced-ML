# Assignment 4 - PPO from Scratch

This assignment implements **Proximal Policy Optimization (PPO)** from scratch and evaluates it on a Gymnasium task (`CartPole-v1`), comparing:

- PPO with **clipped objective**
- PPO with **unclipped objective**

## Project Structure

- `ppo_from_scratch.py`: Full implementation (actor-critic, rollouts, return/advantage computation, PPO updates, plotting).
- `requirements.txt`: Dependencies.
- `report.md`: Assignment report (method, experiments, findings).
- `outputs/`: Generated plots and result summary after running experiments.

## Key PPO Components Implemented

1. **Policy network (Actor)**
2. **Value network (Critic)**
3. **Rollout generation**
4. **Reward handling and return estimation**
5. **Advantage calculation (GAE)**
6. **Objective functions**
   - Clipped objective
   - Unclipped objective

## Setup

```bash
pip install -r requirements.txt
```

## Run Experiments

```bash
python ppo_from_scratch.py --env CartPole-v1 --timesteps 40000 --output-dir outputs
```

## Tuned Run (Near-Optimal)

```bash
python ppo_from_scratch.py --env CartPole-v1 --timesteps 120000 --rollout-steps 2048 --epochs 10 --minibatches 8 --lr 0.00025 --gamma 0.99 --gae-lambda 0.95 --clip-coef 0.2 --ent-coef 0.0 --vf-coef 0.5 --max-grad-norm 0.5 --hidden-size 64 --eval-episodes 50 --output-dir outputs_tuned_120k
```

## Render Environment During Training

```bash
python ppo_from_scratch.py --env CartPole-v1 --timesteps 20000 --render --output-dir outputs
```

Notes:
- Rendering uses `render_mode="human"` and opens a visualization window.
- Rendering slows training; use lower timesteps if needed.

## Outputs

After execution, the script saves:

- `outputs/ppo_clipped_vs_unclipped.png`
- `outputs/results_summary.txt`

The summary file now includes deterministic evaluation mean and standard deviation for both variants.

Use these artifacts in your report.
