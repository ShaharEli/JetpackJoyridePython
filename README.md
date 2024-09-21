# DQN Agent for Jetpack Joyride

## Description
This project implements a Deep Q-Network (DQN) agent to play the Jetpack Joyride game using PyTorch and OpenAI Gym. The agent is trained using experience replay and a target network to stabilize training.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/jetpack-joyride-dqn.git
    cd jetpack-joyride-dqn
    ```
2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. To train the DQN agent, run:
    ```sh
    python dqn_agent.py
    ```
2. You can adjust the hyperparameters in `dqn_agent.py` to fine-tune the training process.
3. Change render to True in dqn_agent.py to see the agent play the game.
4. Run 
    ```sh
    tensorboard --logdir=runs
    ```
    to visualize the training progress in TensorBoard.
## Hyperparameters
The following hyperparameters are defined in `dqn_agent.py`:

- `env_name`: Name of the environment (`JetpackJoyride-v0`)
- `render`: Whether to render the environment during training (`False`)
- `log_interval`: Interval for logging training progress (`20`)
- `max_episodes`: Maximum number of episodes for training (`100000`)
- `max_timesteps`: Maximum number of timesteps per episode (`200000`)
- `batch_size`: Size of the mini-batch for training (`64`)
- `gamma`: Discount factor (`0.99`)
- `lr`: Learning rate (`1e-4`)
- `memory_size`: Size of the replay buffer (`100000`)
- `target_update`: Frequency of updating the target network (`1000` steps)
- `epsilon_start`: Initial value of epsilon for epsilon-greedy policy (`1.0`)
- `epsilon_end`: Minimum value of epsilon (`0.03`)
- `epsilon_decay`: Decay rate of epsilon (`375000` steps)
- `random_seed`: Seed for random number generators (`None`)

