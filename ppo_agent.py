# ppo_agent.py

import torch
import gym
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from main import JetpackJoyrideEnv
import json


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.done = []

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.done = []


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
        )
        # Actor's layer
        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )
        # Critic's layer
        self.critic = nn.Sequential(nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        shared = self.shared(state)
        probs = self.actor(shared)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def evaluate(self, state, action):
        shared = self.shared(state)
        probs = self.actor(shared)
        dist = Categorical(probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        value = self.critic(shared)
        return action_logprobs, torch.squeeze(value), dist_entropy


def train():
    # Hyperparameters
    env_name = "JetpackJoyride-v0"
    render = False
    solved_reward = 5000  # Stop training if avg_reward > solved_reward
    log_interval = 20  # Print avg reward in the interval
    max_episodes = 100000  # Max training episodes
    max_timesteps = 1000  # Max timesteps in one episode

    update_timestep = 2000  # Update policy every n timesteps
    action_std = 0.5  # Constant std for action distribution (Multivariate Normal)
    K_epochs = 4  # Update policy for K epochs
    eps_clip = 0.2  # Clip parameter for PPO
    gamma = 0.99  # Discount factor

    lr = 0.0003  # Parameters for Adam optimizer
    betas = (0.9, 0.999)
    scores = []

    random_seed = None

    # Creating environment
    env = JetpackJoyrideEnv()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    memory = PPOMemory()
    ppo = ActorCritic(state_dim, action_dim).to("cpu")
    optimizer = optim.Adam(ppo.parameters(), lr=lr, betas=betas)
    writer = SummaryWriter()

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    # Training loop
    running_reward = 0
    avg_length = 0
    timestep = 0

    for i_episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        state = torch.FloatTensor(state).to("cpu")
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            action, log_prob, _ = ppo.act(state)
            state_, reward, done, _, info = env.step(action)
            # Saving data in memory
            memory.states.append(state)
            memory.actions.append(torch.tensor(action))
            memory.log_probs.append(log_prob)
            memory.rewards.append(reward)
            memory.done.append(done)

            # Update if its time
            if timestep % update_timestep == 0:
                # Monte Carlo estimate of returns
                rewards = []
                discounted_reward = 0
                for reward, is_terminal in zip(
                    reversed(memory.rewards), reversed(memory.done)
                ):
                    if is_terminal:
                        discounted_reward = 0
                    discounted_reward = reward + (gamma * discounted_reward)
                    rewards.insert(0, discounted_reward)

                # Normalizing the rewards:
                rewards = torch.tensor(rewards, dtype=torch.float32).to("cpu")
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

                # Convert list to tensor
                old_states = torch.squeeze(
                    torch.stack(memory.states).to("cpu"), 1
                ).detach()
                old_actions = torch.squeeze(
                    torch.stack(memory.actions).to("cpu")
                ).detach()
                old_log_probs = torch.squeeze(torch.stack(memory.log_probs)).detach()

                # Optimize policy for K epochs:
                for _ in range(K_epochs):
                    # Evaluations
                    logprobs, state_values, dist_entropy = ppo.evaluate(
                        old_states, old_actions
                    )

                    # Finding the ratio (pi_theta / pi_theta__old):
                    ratios = torch.exp(logprobs - old_log_probs.detach())

                    # Finding Surrogate Loss:
                    advantages = rewards - state_values.detach()
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

                    # Final loss of clipped objective PPO:
                    loss = (
                        -torch.min(surr1, surr2)
                        + 0.5 * nn.MSELoss()(state_values, rewards)
                        - 0.01 * dist_entropy
                    )

                    # Take gradient step
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()

                # Clear memory
                memory.clear()
                timestep = 0

            state = torch.FloatTensor(state_).to("cpu")
            running_reward += reward

            if render:
                env.render()
            if done:
                scores.append(env.game.distance)
                break

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
