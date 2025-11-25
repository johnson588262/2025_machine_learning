import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from final import RoboTaxiEnv, MultiAgentWrapper, ACTION_MEANING

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, obs):
        # obs: [batch, obs_dim]
        return self.net(obs)


class Critic(nn.Module):
    """
    中央 critic：吃進所有 agents 的 obs flatten 後的 global state
    """
    def __init__(self, global_state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        # state: [batch, global_state_dim]
        return self.net(state).squeeze(-1)


class RolloutBuffer:
    def __init__(self, T, n_agents, obs_dim, global_state_dim):
        self.T = T
        self.n_agents = n_agents

        self.obs = torch.zeros(T, n_agents, obs_dim, dtype=torch.float32, device=device)
        self.global_state = torch.zeros(T, global_state_dim, dtype=torch.float32, device=device)
        self.actions = torch.zeros(T, n_agents, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(T, n_agents, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(T, n_agents, dtype=torch.float32, device=device)
        self.dones = torch.zeros(T, dtype=torch.float32, device=device)
        self.values = torch.zeros(T, n_agents, dtype=torch.float32, device=device)
        self.ptr = 0

    def add(self, obs, global_state, actions, logprobs, rewards, done, values):
        t = self.ptr
        self.obs[t] = obs
        self.global_state[t] = global_state
        self.actions[t] = actions
        self.logprobs[t] = logprobs
        self.rewards[t] = rewards
        self.dones[t] = float(done)
        self.values[t] = values
        self.ptr += 1

    def is_full(self):
        return self.ptr >= self.T


def compute_advantages(buffer, actor, critic, gamma=0.99, lam=0.95):
    """
    GAE-Lambda: 逐 agent 算 advantage
    """
    T = buffer.T
    n_agents = buffer.n_agents

    with torch.no_grad():
        # 最後一個 state 的 V(s_T)
        last_values = critic(buffer.global_state[-1].unsqueeze(0)).item()

    advantages = torch.zeros(T, n_agents, device=device)
    returns = torch.zeros(T, n_agents, device=device)

    gae = torch.zeros(n_agents, device=device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_values = last_values
            next_nonterminal = 1.0 - buffer.dones[t]
        else:
            next_values = critic(buffer.global_state[t + 1].unsqueeze(0)).item()
            next_nonterminal = 1.0 - buffer.dones[t + 1]

        V_t = buffer.values[t].mean().item()
        r_t = buffer.rewards[t]  # [n_agents]

        delta = r_t + gamma * next_values * next_nonterminal - V_t
        gae = delta + gamma * lam * next_nonterminal * gae

        advantages[t] = gae
        returns[t] = advantages[t] + V_t

    return advantages, returns


def ppo_update(
    actor,
    critic,
    buffer,
    advantages,
    returns,
    actor_optimizer,
    critic_optimizer,
    clip_eps=0.2,
    epochs=4,
    batch_size=256,
    vf_coef=0.5,
    ent_coef=0.03,
):
    T, n_agents = buffer.T, buffer.n_agents

    # 展平 time 和 agent 維度
    obs = buffer.obs.reshape(T * n_agents, -1)
    actions = buffer.actions.reshape(T * n_agents)
    old_logprobs = buffer.logprobs.reshape(T * n_agents)
    adv = advantages.reshape(T * n_agents)
    ret = returns.reshape(T * n_agents)

    # normalize advantage
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    global_state = buffer.global_state  # [T, global_dim]

    n_samples = T * n_agents
    idxs = np.arange(n_samples)

    for _ in range(epochs):
        np.random.shuffle(idxs)
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_idx = idxs[start:end]

            batch_obs = obs[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_logprobs = old_logprobs[batch_idx]
            batch_adv = adv[batch_idx]
            batch_ret = ret[batch_idx]

            logits = actor(batch_obs)
            dist = torch.distributions.Categorical(logits=logits)
            new_logprobs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logprobs - batch_old_logprobs)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_adv
            actor_loss = -torch.min(surr1, surr2).mean() - ent_coef * entropy

            # critic loss：用 global_state 訓練 V(s)
            V_s = critic(global_state).unsqueeze(1).expand(-1, n_agents)
            V_s = V_s.reshape(T * n_agents)[batch_idx]
            value_loss = vf_coef * (batch_ret - V_s).pow(2).mean()

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            (actor_loss + value_loss).backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            actor_optimizer.step()
            critic_optimizer.step()


def main():
    base_env = RoboTaxiEnv(seed=0)
    env = MultiAgentWrapper(base_env)

    obs_dict = env.reset()
    agent_ids = env.agents
    n_agents = len(agent_ids)

    # 確定 obs_dim / act_dim
    sample_obs = next(iter(obs_dict.values()))
    obs_dim = sample_obs.shape[0]
    act_dim = 7  # 0~6

    global_state_dim = obs_dim * n_agents

    actor = Actor(obs_dim, act_dim).to(device)
    critic = Critic(global_state_dim).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)

    rollout_horizon = 256
    max_episodes = 500

    for ep in range(1, max_episodes + 1):
        buffer = RolloutBuffer(rollout_horizon, n_agents, obs_dim, global_state_dim)

        obs_dict = env.reset()
        ep_reward_sum = 0.0

        for t in range(rollout_horizon):
            # obs_dict -> tensor [n_agents, obs_dim]
            obs_list = [obs_dict[aid] for aid in agent_ids]
            obs = torch.tensor(np.stack(obs_list, axis=0), dtype=torch.float32, device=device)

            global_state = obs.flatten().unsqueeze(0)  # [1, global_state_dim]

            with torch.no_grad():
                logits = actor(obs)  # [n_agents, act_dim]
                dist = torch.distributions.Categorical(logits=logits)
                actions_t = dist.sample()
                logprobs_t = dist.log_prob(actions_t)

                V = critic(global_state)  # scalar
                values_t = torch.full((n_agents,), V.item(), device=device)

            # 丟進 env 的 multi-agent action_dict
            action_dict = {aid: int(actions_t[i].item()) for i, aid in enumerate(agent_ids)}
            next_obs_dict, reward_dict, done_dict, info = env.step(action_dict)

            rewards_t = torch.tensor(
                [reward_dict[aid] for aid in agent_ids],
                dtype=torch.float32,
                device=device,
            )

            # ⭐ Shared reward：用 max 讓整隊拿「最好的那台」當全隊 reward
            global_r = rewards_t.max()
            rewards_t = torch.full_like(rewards_t, global_r)

            done = any(done_dict.values())
            ep_reward_sum += rewards_t.mean().item()

            buffer.add(
                obs=obs,
                global_state=global_state.squeeze(0),
                actions=actions_t,
                logprobs=logprobs_t,
                rewards=rewards_t,
                done=done,
                values=values_t,
            )

            obs_dict = next_obs_dict

            if done or buffer.is_full():
                break

        # PPO 更新
        advantages, returns = compute_advantages(buffer, actor, critic)
        ppo_update(actor, critic, buffer, advantages, returns,
                   actor_optimizer, critic_optimizer)

        mean_r = ep_reward_sum / (buffer.ptr + 1)
        completed = base_env.episode_completed
        collisions = base_env.episode_collisions

        print(
            f"[Episode {ep}] mean reward (per step): {mean_r:.3f}, "
            f"steps: {buffer.ptr} | completed: {completed}, collisions: {collisions}"
        )

        if ep % 50 == 0:
            print("ASCII render at episode", ep)
            base_env.render_ascii()


if __name__ == "__main__":
    main()
