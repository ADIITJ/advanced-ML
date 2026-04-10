import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    env_id: str = "CartPole-v1"
    total_timesteps: int = 40_000
    rollout_steps: int = 1024
    update_epochs: int = 10
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    seed: int = 42
    hidden_size: int = 64
    render: bool = False


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64):
        super().__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, act_dim),
        )

        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        logits = self.policy_net(obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_net(obs).squeeze(-1)
        return action, logprob, entropy, value

    def get_value(self, obs: torch.Tensor):
        return self.value_net(obs).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        logits = self.policy_net(obs)
        if deterministic:
            return torch.argmax(logits, dim=-1)
        dist = Categorical(logits=logits)
        return dist.sample()


class RolloutBuffer:
    def __init__(self, rollout_steps: int, obs_dim: int):
        self.obs = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps,), dtype=np.int64)
        self.logprobs = np.zeros((rollout_steps,), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps,), dtype=np.float32)
        self.dones = np.zeros((rollout_steps,), dtype=np.float32)
        self.values = np.zeros((rollout_steps,), dtype=np.float32)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y


def collect_rollout(
    env,
    model: ActorCritic,
    buffer: RolloutBuffer,
    next_obs: np.ndarray,
    next_done: float,
    device: torch.device,
    render: bool,
) -> Tuple[np.ndarray, float, List[float]]:
    episode_returns = []
    ep_return = 0.0

    for step in range(len(buffer.rewards)):
        if render:
            env.render()

        buffer.obs[step] = next_obs
        buffer.dones[step] = next_done

        obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, logprob, _, value = model.get_action_and_value(obs_tensor)

        action_np = action.item()
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        buffer.actions[step] = action_np
        buffer.logprobs[step] = logprob.item()
        buffer.values[step] = value.item()
        buffer.rewards[step] = reward

        ep_return += reward
        next_done = float(done)

        if done:
            episode_returns.append(ep_return)
            ep_return = 0.0
            next_obs, _ = env.reset()

    return next_obs, next_done, episode_returns


def compute_returns_and_advantages(
    buffer: RolloutBuffer,
    model: ActorCritic,
    next_obs: np.ndarray,
    next_done: float,
    device: torch.device,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        next_value = model.get_value(
            torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        ).item()

    advantages = np.zeros_like(buffer.rewards, dtype=np.float32)
    lastgaelam = 0.0

    for t in reversed(range(len(buffer.rewards))):
        if t == len(buffer.rewards) - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - buffer.dones[t + 1]
            nextvalues = buffer.values[t + 1]

        delta = buffer.rewards[t] + gamma * nextvalues * nextnonterminal - buffer.values[t]
        advantages[t] = lastgaelam = (
            delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        )

    returns = advantages + buffer.values
    return returns, advantages


def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    returns: np.ndarray,
    advantages: np.ndarray,
    config: PPOConfig,
    device: torch.device,
    clipped_objective: bool,
) -> Dict[str, float]:
    b_obs = torch.tensor(buffer.obs, dtype=torch.float32, device=device)
    b_actions = torch.tensor(buffer.actions, dtype=torch.int64, device=device)
    b_logprobs = torch.tensor(buffer.logprobs, dtype=torch.float32, device=device)
    b_returns = torch.tensor(returns, dtype=torch.float32, device=device)
    b_advantages = torch.tensor(advantages, dtype=torch.float32, device=device)

    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

    batch_size = len(buffer.rewards)
    minibatch_size = batch_size // config.num_minibatches
    clipfracs = []

    policy_loss_value = 0.0
    value_loss_value = 0.0
    entropy_loss_value = 0.0

    indices = np.arange(batch_size)

    for _ in range(config.update_epochs):
        np.random.shuffle(indices)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_idx = indices[start:end]

            _, newlogprob, entropy, newvalue = model.get_action_and_value(
                b_obs[mb_idx], b_actions[mb_idx]
            )

            logratio = newlogprob - b_logprobs[mb_idx]
            ratio = logratio.exp()

            with torch.no_grad():
                clipfrac = ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                clipfracs.append(clipfrac)

            if clipped_objective:
                pg_loss1 = -b_advantages[mb_idx] * ratio
                pg_loss2 = -b_advantages[mb_idx] * torch.clamp(
                    ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
            else:
                policy_loss = (-b_advantages[mb_idx] * ratio).mean()

            value_loss = 0.5 * ((newvalue - b_returns[mb_idx]) ** 2).mean()
            entropy_loss = entropy.mean()

            loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            policy_loss_value = policy_loss.item()
            value_loss_value = value_loss.item()
            entropy_loss_value = entropy_loss.item()

    with torch.no_grad():
        value_pred = model.get_value(b_obs).cpu().numpy()
    ev = explained_variance(value_pred, b_returns.cpu().numpy())

    return {
        "policy_loss": policy_loss_value,
        "value_loss": value_loss_value,
        "entropy": entropy_loss_value,
        "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        "explained_variance": float(ev),
    }


def train_ppo(config: PPOConfig, clipped_objective: bool = True):
    set_seed(config.seed)

    render_mode = "human" if config.render else None
    env = gym.make(config.env_id, render_mode=render_mode)

    obs, _ = env.reset(seed=config.seed)
    env.action_space.seed(config.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(obs_dim, act_dim, config.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

    buffer = RolloutBuffer(config.rollout_steps, obs_dim)

    next_obs = obs
    next_done = 0.0

    global_step = 0
    episode_returns: List[float] = []
    timestep_points: List[int] = []
    rolling_avg_returns: List[float] = []

    while global_step < config.total_timesteps:
        next_obs, next_done, returns_from_rollout = collect_rollout(
            env,
            model,
            buffer,
            next_obs,
            next_done,
            device,
            config.render,
        )
        global_step += config.rollout_steps

        if returns_from_rollout:
            episode_returns.extend(returns_from_rollout)

        returns, advantages = compute_returns_and_advantages(
            buffer,
            model,
            next_obs,
            next_done,
            device,
            config.gamma,
            config.gae_lambda,
        )

        stats = ppo_update(
            model,
            optimizer,
            buffer,
            returns,
            advantages,
            config,
            device,
            clipped_objective,
        )

        avg_return = (
            float(np.mean(episode_returns[-20:])) if episode_returns else 0.0
        )
        timestep_points.append(global_step)
        rolling_avg_returns.append(avg_return)

        variant = "CLIPPED" if clipped_objective else "UNCLIPPED"
        print(
            f"[{variant}] steps={global_step} avg_return(20)={avg_return:.2f} "
            f"policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f} "
            f"entropy={stats['entropy']:.4f} clipfrac={stats['clipfrac']:.4f} "
            f"explained_var={stats['explained_variance']:.4f}"
        )

    env.close()

    return {
        "model": model,
        "timestep_points": timestep_points,
        "rolling_avg_returns": rolling_avg_returns,
        "episode_returns": episode_returns,
    }


def evaluate_policy(
    model: ActorCritic,
    env_id: str,
    episodes: int = 20,
    seed: int = 123,
    deterministic: bool = True,
) -> Dict[str, float]:
    env = gym.make(env_id)
    device = next(model.parameters()).device
    returns: List[float] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = model.act(obs_tensor, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            ep_return += reward

        returns.append(ep_return)

    env.close()
    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
    }


def render_trained_agent(model: ActorCritic, env_id: str, episodes: int = 3, seed: int = 123):
    env = gym.make(env_id, render_mode="human")
    obs, _ = env.reset(seed=seed)

    device = next(model.parameters()).device

    for _ in range(episodes):
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(obs_tensor)
            obs, _, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

        obs, _ = env.reset()

    env.close()


def record_policy_video(
    model: ActorCritic,
    env_id: str,
    video_dir: str,
    name_prefix: str,
    episodes: int = 2,
    seed: int = 123,
):
    os.makedirs(video_dir, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        name_prefix=name_prefix,
        episode_trigger=lambda ep_idx: ep_idx < episodes,
    )

    device = next(model.parameters()).device
    obs, _ = env.reset(seed=seed)

    for _ in range(episodes):
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(obs_tensor)
            obs, _, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
        obs, _ = env.reset()

    env.close()


def plot_results(
    clipped_history: Dict[str, List[float]],
    unclipped_history: Dict[str, List[float]],
    output_path: str,
):
    plt.figure(figsize=(10, 6))
    plt.plot(
        clipped_history["timestep_points"],
        clipped_history["rolling_avg_returns"],
        label="PPO clipped objective",
        linewidth=2,
    )
    plt.plot(
        unclipped_history["timestep_points"],
        unclipped_history["rolling_avg_returns"],
        label="PPO unclipped objective",
        linewidth=2,
        linestyle="--",
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Rolling average return (last 20 episodes)")
    plt.title("PPO: Clipped vs Unclipped Objective on CartPole-v1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_comparison(config: PPOConfig, output_dir: str, eval_episodes: int = 30):
    os.makedirs(output_dir, exist_ok=True)

    print("\nTraining with clipped objective...\n")
    clipped_history = train_ppo(config, clipped_objective=True)

    print("\nTraining with unclipped objective...\n")
    config_unclipped = PPOConfig(**{**config.__dict__, "seed": config.seed + 1})
    unclipped_history = train_ppo(config_unclipped, clipped_objective=False)

    clipped_eval = evaluate_policy(
        clipped_history["model"],
        config.env_id,
        episodes=eval_episodes,
        seed=config.seed + 10_000,
        deterministic=True,
    )
    unclipped_eval = evaluate_policy(
        unclipped_history["model"],
        config.env_id,
        episodes=eval_episodes,
        seed=config.seed + 20_000,
        deterministic=True,
    )

    video_dir = os.path.join(output_dir, "videos")
    record_policy_video(
        clipped_history["model"],
        config.env_id,
        video_dir,
        name_prefix="clipped_policy",
        episodes=2,
        seed=config.seed,
    )
    record_policy_video(
        unclipped_history["model"],
        config.env_id,
        video_dir,
        name_prefix="unclipped_policy",
        episodes=2,
        seed=config.seed + 1,
    )

    plot_path = os.path.join(output_dir, "ppo_clipped_vs_unclipped.png")
    plot_results(clipped_history, unclipped_history, plot_path)

    clipped_final = (
        float(np.mean(clipped_history["episode_returns"][-20:]))
        if clipped_history["episode_returns"]
        else 0.0
    )
    unclipped_final = (
        float(np.mean(unclipped_history["episode_returns"][-20:]))
        if unclipped_history["episode_returns"]
        else 0.0
    )

    summary = {
        "clipped_final_avg_return": clipped_final,
        "unclipped_final_avg_return": unclipped_final,
        "clipped_eval_mean": clipped_eval["mean"],
        "clipped_eval_std": clipped_eval["std"],
        "unclipped_eval_mean": unclipped_eval["mean"],
        "unclipped_eval_std": unclipped_eval["std"],
        "plot_path": plot_path,
        "video_dir": video_dir,
    }

    summary_path = os.path.join(output_dir, "results_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PPO Comparison Summary\n")
        f.write(f"Environment: {config.env_id}\n")
        f.write(f"Total timesteps per run: {config.total_timesteps}\n")
        f.write(f"Clipped final avg return (last 20 episodes): {clipped_final:.2f}\n")
        f.write(f"Unclipped final avg return (last 20 episodes): {unclipped_final:.2f}\n")
        f.write(
            f"Clipped deterministic eval mean±std ({eval_episodes} eps): "
            f"{clipped_eval['mean']:.2f} ± {clipped_eval['std']:.2f}\n"
        )
        f.write(
            f"Unclipped deterministic eval mean±std ({eval_episodes} eps): "
            f"{unclipped_eval['mean']:.2f} ± {unclipped_eval['std']:.2f}\n"
        )
        f.write(f"Performance plot: {plot_path}\n")
        f.write(f"Policy rollout videos folder: {video_dir}\n")
        f.write("Hyperparameters:\n")
        f.write(f"  learning_rate={config.learning_rate}\n")
        f.write(f"  rollout_steps={config.rollout_steps}\n")
        f.write(f"  update_epochs={config.update_epochs}\n")
        f.write(f"  num_minibatches={config.num_minibatches}\n")
        f.write(f"  clip_coef={config.clip_coef}\n")
        f.write(f"  ent_coef={config.ent_coef}\n")
        f.write(f"  vf_coef={config.vf_coef}\n")
        f.write(f"  gamma={config.gamma}\n")
        f.write(f"  gae_lambda={config.gae_lambda}\n")
        f.write(f"  max_grad_norm={config.max_grad_norm}\n")
        f.write(f"  hidden_size={config.hidden_size}\n")

    print("\nComparison complete")
    print(f"Clipped final avg return:   {clipped_final:.2f}")
    print(f"Unclipped final avg return: {unclipped_final:.2f}")
    print(
        f"Clipped deterministic eval mean±std ({eval_episodes} eps): "
        f"{clipped_eval['mean']:.2f} ± {clipped_eval['std']:.2f}"
    )
    print(
        f"Unclipped deterministic eval mean±std ({eval_episodes} eps): "
        f"{unclipped_eval['mean']:.2f} ± {unclipped_eval['std']:.2f}"
    )
    print(f"Saved plot to: {plot_path}")
    print(f"Saved rollout videos to: {video_dir}")
    print(f"Saved summary to: {summary_path}")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="PPO from scratch: clipped vs unclipped")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium env id")
    parser.add_argument("--timesteps", type=int, default=40_000, help="Total timesteps per run")
    parser.add_argument("--rollout-steps", type=int, default=1024, help="Steps per rollout")
    parser.add_argument("--epochs", type=int, default=10, help="Update epochs per rollout")
    parser.add_argument("--minibatches", type=int, default=8, help="Number of minibatches")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Grad clipping norm")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--eval-episodes", type=int, default=30, help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    config = PPOConfig(
        env_id=args.env,
        total_timesteps=args.timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.epochs,
        num_minibatches=args.minibatches,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        hidden_size=args.hidden_size,
        seed=args.seed,
        render=args.render,
    )

    run_comparison(config, args.output_dir, eval_episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
