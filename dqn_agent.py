# dqn_agent.py

import torch
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from main import JetpackJoyrideEnv
import json

# Hyperparameters
env_name = "JetpackJoyride-v0"
render = False
log_interval = 20
max_episodes = 100000
max_timesteps = 200000

batch_size = 64
gamma = 0.99  # Discount factor
lr = 1e-4  # Learning rate
memory_size = 100000  # Replay buffer size
target_update = 1000  # Update target network every n steps
epsilon_start = 1.0  # Starting value of epsilon
epsilon_end = 0.01  # Minimum value of epsilon
epsilon_decay = 375000  # Decay rate of epsilon

random_seed = None

# Set random seed for reproducibility
if random_seed is not None:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

# Experience tuple
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Q-Network
class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def train():
    env = JetpackJoyrideEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim).to("cpu")
    target_net = DQN(state_dim, action_dim).to("cpu")
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(memory_size)
    writer = SummaryWriter()

    steps_done = 0
    epsilon = epsilon_start
    scores = []

    for i_episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to("cpu")
        total_reward = 0

        for t in range(max_timesteps):
            steps_done += 1

            # Epsilon-greedy action selection
            sample = random.random()
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(
                -1.0 * steps_done / epsilon_decay
            )
            if sample > epsilon:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = q_values.max(1)[1].item()
            else:
                action = env.action_space.sample()

            # Take action
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to("cpu")

            # Store transition in replay memory
            memory.push(
                state,
                torch.tensor([[action]]),
                torch.tensor([[reward]]),
                next_state,
                torch.tensor([[done]], dtype=torch.float),
            )

            state = next_state

            # Perform optimization
            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))

                batch_state = torch.cat(batch.state)
                batch_action = torch.cat(batch.action)
                batch_reward = torch.cat(batch.reward)
                batch_next_state = torch.cat(batch.next_state)
                batch_done = torch.cat(batch.done)

                # Compute current Q values
                current_q_values = policy_net(batch_state).gather(1, batch_action)

                # Compute next Q values using target network
                with torch.no_grad():
                    max_next_q_values = (
                        target_net(batch_next_state).max(1)[0].unsqueeze(1)
                    )
                    target_q_values = batch_reward + (
                        gamma * max_next_q_values * (1 - batch_done)
                    )

                # Compute loss
                loss = nn.MSELoss()(current_q_values, target_q_values)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update target network
                if steps_done % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if render:
                env.render()
            if done:
                scores.append(env.game.distance)
                break

        # Logging
        if i_episode % log_interval == 0:
            avg_score = sum(scores[-log_interval:]) / log_interval
            print(
                f"Episode {i_episode} \t Average Score: {avg_score:.2f} \t Epsilon: {epsilon:.4f}"
            )
            writer.add_scalar("Average Score", avg_score, i_episode)
            writer.add_scalar("Epsilon", epsilon, i_episode)
            torch.save(policy_net.state_dict(), "dqn_model.pth")
            with open("scores.json", "w") as f:
                json.dump(scores, f)

    env.close()
    writer.close()


if __name__ == "__main__":
    train()
