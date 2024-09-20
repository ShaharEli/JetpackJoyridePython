# ppo_agent.py

import torch
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from main import JetpackJoyrideEnv
import json


class PPOMemory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.done = []
        self.values = []  # Store state values for advantage estimation


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),  # Increased layer size
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        # Actor's layer
        self.actor = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1),
        )
        # Critic's layer
        self.critic = nn.Linear(256, 1)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        shared = self.shared(state)
        probs = self.actor(shared)
        dist = Categorical(probs)
        action = dist.sample()
        value = self.critic(shared)
        return action.item(), dist.log_prob(action), dist.entropy(), value

    def evaluate(self, states, actions):
        shared = self.shared(states)
        probs = self.actor(shared)
        dist = Categorical(probs)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        values = self.critic(shared)
        return action_logprobs, values.squeeze(), dist_entropy


def train():
    # Hyperparameters
    env_name = "JetpackJoyride-v0"
    render = False
    log_interval = 20  # Print avg reward in the interval
    max_episodes = 100000  # Max training episodes
    max_timesteps = 1500  # Max timesteps in one episode

    update_timestep = 2000  # Update policy every n timesteps
    K_epochs = 10  # Update policy for K epochs
    eps_clip = 0.2  # Clip parameter for PPO
    gamma = 0.99  # Discount factor
    gae_lambda = 0.95  # GAE lambda

    lr = 0.0001  # Learning rate
    betas = (0.9, 0.999)
    scores = []

    # Creating environment
    env = JetpackJoyrideEnv()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    memory = PPOMemory()
    ppo = ActorCritic(state_dim, action_dim).to("cpu")
    optimizer = optim.Adam(ppo.parameters(), lr=lr, betas=betas)
    writer = SummaryWriter()

    # Training loop
    running_reward = 0
    avg_length = 0
    timestep = 0

    for i_episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        state = torch.FloatTensor(state).to("cpu")
        for t in range(max_timesteps):
            timestep += 1

            # Running policy:
            action, log_prob, entropy, value = ppo.act(state)
            state_, reward, done, _, info = env.step(action)
            state_ = torch.FloatTensor(state_).to("cpu")

            # Saving data in memory
            memory.states.append(state)
            memory.actions.append(torch.tensor(action))
            memory.log_probs.append(log_prob)
            memory.rewards.append(reward)
            memory.done.append(done)
            memory.values.append(value.item())

            state = state_
            running_reward += reward

            if render:
                env.render()
            if done:
                scores.append(env.game.distance)
                break

            # Update if it's time
            if timestep % update_timestep == 0:
                # Compute advantages using GAE
                advantages = []
                returns = []
                gae = 0
                next_value = ppo.critic(ppo.shared(state)).item()
                for i in reversed(range(len(memory.rewards))):
                    delta = (
                        memory.rewards[i]
                        + gamma * (next_value if not memory.done[i] else 0)
                        - memory.values[i]
                    )
                    gae = delta + gamma * gae_lambda * (1 - memory.done[i]) * gae
                    advantages.insert(0, gae)
                    next_value = memory.values[i]
                    returns.insert(0, gae + memory.values[i])

                advantages = torch.tensor(advantages, dtype=torch.float32).to("cpu")
                returns = torch.tensor(returns, dtype=torch.float32).to("cpu")

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Convert list to tensor
                old_states = torch.stack(memory.states).to("cpu").detach()
                old_actions = torch.stack(memory.actions).to("cpu").detach()
                old_log_probs = torch.stack(memory.log_probs).to("cpu").detach()

                # Optimize policy for K epochs:
                for _ in range(K_epochs):
                    # Evaluations
                    logprobs, state_values, dist_entropy = ppo.evaluate(
                        old_states, old_actions
                    )

                    # Finding the ratio (pi_theta / pi_theta__old):
                    ratios = torch.exp(logprobs - old_log_probs)

                    # Surrogate loss
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

                    # Compute loss
                    loss = (
                        -torch.min(surr1, surr2).mean()
                        + 0.5 * nn.MSELoss()(state_values, returns)
                        - 0.01 * dist_entropy.mean()
                    )

                    # Take gradient step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Clear memory
                memory.clear()
                timestep = 0

        avg_length += t

        # Logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print(
                "Episode {} \t Avg length: {} \t Avg reward: {} \t Avg distance: {}".format(
                    i_episode, avg_length, running_reward, sum(scores) / len(scores)
                )
            )
            writer.add_scalar("Average Reward", running_reward, i_episode)
            writer.add_scalar("Distance", env.game.distance, i_episode)
            running_reward = 0
            avg_length = 0
            if i_episode % 1000 == 0:
                torch.save(ppo.state_dict(), "ppo_model.pth")
                with open("scores.json", "w") as f:
                    json.dump(scores, f)

    env.close()
    writer.close()


if __name__ == "__main__":
    train()
