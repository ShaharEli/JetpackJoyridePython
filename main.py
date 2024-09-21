import random
import gym
from gym import spaces
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import logging

# Constants
WIDTH = 1000
HEIGHT = 600
FPS = 60
INIT_Y = HEIGHT - 130
GRAVITY = 0.4

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode([WIDTH, HEIGHT])
surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
pygame.display.set_caption("Jetpack Joyride Remake in Python!")
font = pygame.font.Font("freesansbold.ttf", 32)
timer = pygame.time.Clock()

# Set up logging
logging.basicConfig(
    filename="jetpack_joyride_ppo.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),  # Increased layer size
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),  # Increased layer size
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        return self.network(state)


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)


class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.00005
        )
        self.memory = PPOMemory()
        self.gamma = 0.95
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.epochs = 15
        self.actors = []
        self.scores = []

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)
        return action.item(), log_prob.item(), value.item()

    def update(self):
        states = torch.FloatTensor(self.memory.states)
        actions = torch.LongTensor(self.memory.actions)
        rewards = torch.FloatTensor(self.memory.rewards)
        values = torch.FloatTensor(self.memory.values)
        log_probs = torch.FloatTensor(self.memory.log_probs)
        dones = torch.FloatTensor(self.memory.dones)

        # Compute advantages
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae_lam = (
                delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae_lam
            )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(self.epochs):
            # Compute actor loss
            new_action_probs = self.actor(states)
            new_dist = Categorical(new_action_probs)
            new_log_probs = new_dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()

            # Compute critic loss
            new_values = self.critic(states).squeeze()
            returns = advantages + values
            critic_loss = nn.MSELoss()(new_values, returns)

            # Compute total loss
            loss = actor_loss + 0.5 * critic_loss

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory.clear()

    def add_score(self, score):
        self.actors.append(self.actor.state_dict())
        self.scores.append(score)

    def save(self, filename):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "actors": self.actors,
                "scores": self.scores,
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.actors = checkpoint["actors"]
        self.scores = checkpoint["scores"]


class Player:
    def __init__(self):
        self.y = INIT_Y
        self.booster = False
        self.y_velocity = 0
        self.counter = 0
        self.dead = False

        # Load player sprites
        self.player_fly_surface = pygame.image.load(
            "assets/sprites/skins/PlayerFly_" + "Blue" + ".png"
        ).convert_alpha()
        self.player_fly_surface = pygame.transform.scale(
            self.player_fly_surface, [64, 68]
        )

        self.player_dead_surface = pygame.image.load(
            "assets/sprites/skins/PlayerDead_" + "Blue" + ".png"
        ).convert_alpha()
        self.player_dead_surface = pygame.transform.scale(
            self.player_dead_surface, [82, 74]
        )

        self.player_surface = self.player_fly_surface  # Default player sprite
        self.player_rect = self.player_surface.get_rect(center=(120, INIT_Y))
        self.collision_rect = pygame.Rect(
            self.player_rect.x + 10, self.player_rect.y + 10, 44, 58
        )  # Adjust to make it slightly smaller than the visible sprite

        # Fly particle sprite
        self.fly_particle_surface = pygame.image.load(
            "assets/sprites/Bullet.png"
        ).convert_alpha()
        self.fly_particle_surface = pygame.transform.smoothscale(
            self.fly_particle_surface,
            (
                int(self.fly_particle_surface.get_width() * 2.2),
                int(self.fly_particle_surface.get_height() * 2.2),
            ),
        )

    def get_collision_rect(self):
        return self.collision_rect  # Return the smaller collision rectangle

    def update(self, gravity, colliding):
        # Apply gravity or booster effect
        if self.dead:
            return
        if self.booster:
            self.y_velocity -= gravity
        else:
            self.y_velocity += gravity

        # Prevent the player from moving out of bounds
        if self.y + self.y_velocity > INIT_Y:
            self.y = INIT_Y
            self.y_velocity = 0
        elif self.y + self.y_velocity < 0:
            self.y = 0
            self.y_velocity = 0
        else:
            self.y += self.y_velocity

        # Update player rectangle for proper position and animation
        self.counter = (self.counter + 1) % 40
        self.player_rect = self.player_surface.get_rect(center=(120, self.y + 25))

        # Adjust the player collision rectangle to be smaller and more accurate
        self.player_rect = pygame.Rect(
            self.player_rect.x + 10, self.player_rect.y + 25, 44, 58
        )  # Manually shrink the collision box to fit the visible part of the player
        self.collision_rect = pygame.Rect(
            self.player_rect.x + 10, self.player_rect.y + 15, 44, 50  # Tuned dimensions
        )

    def draw(self, screen):
        # Draw the player and booster particles
        play = screen.blit(self.player_surface, self.player_rect)
        if self.booster:
            for _ in range(4):
                x = random.randint(0, 10)
                y = random.randint(0, 10)
                screen.blit(
                    self.fly_particle_surface,
                    (self.player_rect.x - 10 + x, self.player_rect.y + 40 + y),
                )
        return play

    def get_collision_rect(self):
        return self.collision_rect  # Use this for collision detection


class Laser:
    def __init__(self):
        self.lasers = []
        self.new_laser = True
        # Load and scale the laser sprite
        self.laser_surface = pygame.image.load(
            "assets/sprites/Zapper1.png"
        ).convert_alpha()
        self.laser_surface = pygame.transform.scale2x(self.laser_surface)

        self.laser_surface2 = pygame.image.load(
            "assets/sprites/Zapper2.png"
        ).convert_alpha()
        self.laser_surface2 = pygame.transform.scale2x(self.laser_surface2)

        self.laser_surface3 = pygame.image.load(
            "assets/sprites/Zapper3.png"
        ).convert_alpha()
        self.laser_surface3 = pygame.transform.scale2x(self.laser_surface3)

        self.laser_surface4 = pygame.image.load(
            "assets/sprites/Zapper4.png"
        ).convert_alpha()
        self.laser_surface4 = pygame.transform.scale2x(self.laser_surface4)

    def generate_laser(self):
        # Generate a vertical laser with a random x-position and a random height
        offset = random.randint(10, 300)
        laser_x = WIDTH + offset  # The laser starts off-screen to the right
        laser_height = random.randint(100, 300)
        laser_y_top = random.randint(
            30, HEIGHT - laser_height - 30
        )  # Random y position, ensuring it fits within screen
        new_lase = [
            [laser_x, laser_y_top],
            [laser_x, laser_y_top + laser_height],
        ]  # Vertical laser
        self.lasers.append(new_lase)

    def update(self, game_speed):
        if self.new_laser:
            self.generate_laser()
            self.new_laser = False

        # Move lasers and remove off-screen ones
        for lase in self.lasers:
            lase[0][0] -= game_speed
            lase[1][0] -= game_speed
        self.lasers = [lase for lase in self.lasers if lase[1][0] > 0]

        if len(self.lasers) == 0 or self.lasers[-1][1][0] < WIDTH - 400:
            self.new_laser = True

    def draw(self, screen):
        # Draw each laser sprite at its corresponding coordinates
        for lase in self.lasers:

            if random.randint(0, 1) == 0:
                screen.blit(self.laser_surface, lase)
            else:
                screen.blit(self.laser_surface2, lase)

    def check_collision(self, player_rect):
        for lase in self.lasers:
            # Create a bounding box for the laser line
            laser_rect = pygame.Rect(
                lase[0][0] - 5, lase[0][1], 10, lase[1][1] - lase[0][1]
            )
            if player_rect.colliderect(laser_rect):
                return True  # Collision detected
        return False


class Rocket:
    def __init__(self, player):
        self.counter = 0
        self.active = False
        self.delay = 0
        self.coords = [WIDTH, HEIGHT / 2]
        self.player = player  # Each rocket is tied to a player
        self.rocket_surface = pygame.image.load(
            "assets/sprites/Rocket.png"
        ).convert_alpha()
        self.rocket_surface = pygame.transform.smoothscale(
            self.rocket_surface,
            (
                int(self.rocket_surface.get_width() * 0.75),
                int(self.rocket_surface.get_height() * 0.75),
            ),
        )

        # Load and scale the warning sprite
        self.warning_surface = pygame.image.load(
            "assets/sprites/RocketWarning.png"
        ).convert_alpha()
        self.warning_surface = pygame.transform.smoothscale(
            self.warning_surface,
            (
                int(self.warning_surface.get_width() * 1.2),
                int(self.warning_surface.get_height() * 1.2),
            ),
        )

    def update(self, game_speed, colliding):

        if not self.active:
            self.counter += 1
        if self.counter > 180:
            self.counter = 0
            self.active = True
            self.delay = 0
            self.coords = [WIDTH, HEIGHT / 2]

        if self.active:
            if self.delay < 90:
                if self.coords[1] > self.player.y + 10:
                    self.coords[1] -= 3
                else:
                    self.coords[1] += 3
                self.delay += 1
            else:
                self.coords[0] -= 10 + game_speed
            if self.coords[0] < -50:
                self.active = False

    def draw(self, screen):
        if self.active:
            if self.delay < 90:
                # Draw warning indicator before the rocket enters the screen
                screen.blit(
                    self.warning_surface, (self.coords[0] - 60, self.coords[1] - 25)
                )
            else:
                # Draw the rocket sprite
                screen.blit(self.rocket_surface, (self.coords[0], self.coords[1] - 25))

    def check_collision(self, player_rect):
        if self.active:
            # Create a bounding box for the rocket sprite, slightly shrinking it
            rocket_rect = pygame.Rect(self.coords[0] - 30, self.coords[1] - 25, 50, 50)
            if player_rect.colliderect(rocket_rect):
                return True  # Collision detected
        return False


class UI:
    def __init__(self):
        self.distance = 0
        self.high_score, self.lifetime = self.load_player_info()
        self.bg_surface = pygame.image.load(
            "assets/sprites/BackdropMain.png"
        ).convert()  # convert the image to a pygame lightweight format
        self.bg_surface = pygame.transform.scale2x(self.bg_surface)  # double the size
        self.bg_x = 0  # Initial background position

    def load_player_info(self):
        with open("player_info.txt", "r") as file:
            read = file.readlines()
            high_score = int(read[0])
            lifetime = int(read[1])
        return high_score, lifetime

    def save_player_info(self):
        with open("player_info.txt", "w") as file:
            file.write(f"{int(self.high_score)}\n")
            file.write(f"{int(self.lifetime)}")

    def update(self, game_speed):
        self.distance += game_speed
        if self.distance > self.high_score:
            self.high_score = self.distance

    def draw_bg(self, screen):
        screen.blit(self.bg_surface, (self.bg_x, 0))  # First background
        screen.blit(
            self.bg_surface, (self.bg_x + self.bg_surface.get_width(), 0)
        )  # Second background

    def draw_score(self, screen):
        screen.blit(
            font.render(f"Distance: {int(self.distance)} m", True, "white"), (10, 10)
        )
        screen.blit(
            font.render(f"High Score: {int(self.high_score)} m", True, "white"),
            (10, 70),
        )

    def preview(self, generation, best):
        screen.blit(font.render(f"Generation: {generation}", True, "white"), (10, 10))
        screen.blit(
            font.render(f"Best from this generation: {best}", True, "white"),
            (10, 70),
        )
        screen.blit(
            font.render(f"Distance: {int(self.distance)} m", True, "white"), (10, 130)
        )


class Game:
    def __init__(self, seed=None):
        self.game_speed = 3
        self.player = Player()
        self.laser = Laser()
        self.rocket = Rocket(self.player)
        self.ui = UI()
        self.seed = seed
        self.state_dim = 12  # Updated state dimension
        self.action_dim = 2
        self.distance = 0

    def get_state(self):
        if self.laser.lasers:
            lasers = sorted(self.laser.lasers, key=lambda x: x[0][0])
            lasers = list(filter(lambda x: x[0][0] >= 100, lasers))
            if len(lasers) >= 2:
                laser1, laser2 = lasers[:2]
            else:
                laser1 = lasers[0]
                laser2 = laser1
        else:
            laser1 = [[WIDTH, 0], [WIDTH, HEIGHT]]
            laser2 = laser1

        laser1_x = laser1[0][0] / WIDTH
        laser1_y_top = laser1[0][1] / HEIGHT
        laser1_y_bottom = laser1[1][1] / HEIGHT
        laser2_x = laser2[0][0] / WIDTH
        laser2_y_top = laser2[0][1] / HEIGHT
        laser2_y_bottom = laser2[1][1] / HEIGHT

        missile_up = self.rocket.coords[1] / HEIGHT if self.rocket.active else -1
        missile_horizontal = self.rocket.coords[0] / WIDTH if self.rocket.active else -1

        player_y = self.player.y / HEIGHT
        player_y_velocity = self.player.y_velocity / 10.0  # Assuming max speed of 10
        game_speed = self.game_speed / 10.0  # Assuming max game_speed of 10
        missile_active = 1.0 if self.rocket.active else 0.0

        return torch.tensor(
            [
                player_y,
                player_y_velocity,
                game_speed,
                laser1_x,
                laser1_y_top,
                laser1_y_bottom,
                laser2_x,
                laser2_y_top,
                laser2_y_bottom,
                missile_up,
                missile_horizontal,
                missile_active,
            ],
            dtype=torch.float32,
        )

    def compute_reward(
        self,
        collision,
        distance_to_obstacle,
        obstacle_y_start,
        obstacle_y_end,
        missile_distance,
        missile_y,
    ):
        reward = 1.0  # Base reward for survival
        player_y = self.player.y

        # Calculate floor and ceiling proximity
        floor_proximity = (
            HEIGHT - self.player.player_surface.get_height()
        ) - player_y  # Distance to floor
        ceiling_proximity = player_y  # Distance to ceiling
        is_close_to_boundary = floor_proximity < 70 or ceiling_proximity < 70

        if collision:
            reward = -10.0  # Penalty for collision
            return reward

        # Penalty for being too close to boundaries
        if is_close_to_boundary:
            reward -= 1.0

        # Obstacle avoidance logic
        if distance_to_obstacle is not None:
            if obstacle_y_start is not None and obstacle_y_end is not None:
                player_bottom = player_y + self.player.player_rect.height
                if player_bottom >= obstacle_y_start and player_y <= obstacle_y_end:
                    # Player is within the vertical range of the obstacle
                    reward -= 2.0  # Penalty for being in front of the obstacle

                    # Determine the direction to avoid the obstacle
                    obstacle_center = (obstacle_y_start + obstacle_y_end) / 2
                    avoiding_direction = (
                        -1 if player_y < obstacle_center else 1
                    )  # Move away from obstacle center
                    if hasattr(self, "prev_y"):
                        y_change = player_y - self.prev_y
                    else:
                        y_change = 0

                    if y_change * avoiding_direction > 0:
                        reward += 2.0  # Reward for moving in the correct direction to avoid obstacle

                    self.infront_obstacle = True
                else:
                    if hasattr(self, "infront_obstacle") and self.infront_obstacle:
                        reward += 5.0  # Reward for successfully avoiding the obstacle
                        self.infront_obstacle = False

                self.prev_y = player_y

        # Missile avoidance logic
        if self.rocket.active and missile_distance is not None:
            missile_horizontal_distance = missile_distance
            missile_vertical_distance = abs(player_y - missile_y)

            if missile_horizontal_distance < 200:
                # Missile is close horizontally
                reward -= 2.0  # Penalty for being close to missile

                # Determine the direction to avoid the missile
                avoiding_direction = (
                    -1 if player_y < missile_y else 1
                )  # Move away from missile

                y_change = (
                    player_y - self.prev_y_missile
                    if hasattr(self, "prev_y_missile")
                    else 0
                )

                if y_change * avoiding_direction > 0:
                    reward += 2.0  # Reward for moving in the correct direction to avoid missile

                self.close_to_missile = True
            else:
                if hasattr(self, "close_to_missile") and self.close_to_missile:
                    reward += 5.0  # Reward for successfully avoiding the missile
                    self.close_to_missile = False

            self.prev_y_missile = player_y

        return reward

    def get_distance_to_next_obstacle(self):
        player_x = 120  # Player's x-coordinate
        if self.laser.lasers:
            lasers = sorted(self.laser.lasers, key=lambda x: x[0][0])
            next_laser = next(
                (laser for laser in lasers if laser[0][0] > player_x), None
            )
            if next_laser:
                distance = next_laser[0][0] - player_x
                obstacle_y_start = next_laser[0][1]
                obstacle_y_end = next_laser[1][1]
                return distance, obstacle_y_start, obstacle_y_end
        return None, None, None

    def update(self, action):
        self.distance += self.game_speed
        self.laser.update(self.game_speed)
        self.player.booster = action == 1
        self.player.update(GRAVITY, (False, False))
        self.rocket.update(self.game_speed, self.player.y)
        # Check for collisions
        player_rect = self.player.player_rect
        if self.laser.check_collision(player_rect) or self.rocket.check_collision(
            player_rect
        ):
            self.player.dead = True

    def reset(self):
        self.player = Player()
        self.laser = Laser()
        self.rocket = Rocket(self.player)
        self.game_speed = 3
        self.ui.distance = 0
        self.previous_distance_to_obstacle = None
        self.prev_y = 0
        self.prev_y_missile = None
        self.infront_obstacle = False
        self.close_to_missile = False

    def draw(self, screen):
        screen.fill("black")
        self.ui.draw_bg(screen)
        self.laser.draw(screen)
        self.player.draw(screen)
        self.rocket.draw(screen)
        self.ui.draw_score(screen)
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            timer.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            done = self.update()
            self.draw()

            if done:
                self.best_reward = max(self.best_reward, self.ui.distance)

        pygame.quit()

    def save_agent(self):
        self.agent.save("ppo_agent.pth")
        logging.info("Agent saved")

    def load_agent(self):
        try:
            self.agent.load("ppo_agent.pth")
            print("Loading agent")
            logging.info("Agent loaded")
        except FileNotFoundError:
            logging.info("No saved agent found. Starting with a new agent.")


class JetpackJoyrideEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode=None):
        super(JetpackJoyrideEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0: booster off, 1: booster on
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        # Initialize game
        self.render_mode = render_mode
        self.game = Game()
        self.player = self.game.player

        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode([WIDTH, HEIGHT])
            pygame.display.set_caption("Jetpack Joyride Remake in Python!")
        else:
            self.screen = None  # No rendering

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the game to the initial state
        if np.random.rand() < 0.005:
            print("Resetting game, previous distance: ", self.game.distance)
        self.game = Game()
        self.player = self.game.player
        obs = self.game.get_state().numpy()
        return obs, {}

    def step(self, action):
        # Apply action
        self.game.update(action)

        # Get observation
        obs = self.game.get_state().numpy()

        # Compute reward
        reward = 1  # Reward for staying alive

        # Check if done
        done = self.player.dead

        if done:
            reward -= 100  # Penalty for dying

        # Additional info (optional)
        info = {"distance": self.game.distance}

        if self.render_mode == "human":
            self.render()

        return obs, reward, done, False, info

    def render(self):
        # Draw the game
        self.game.draw(self.screen)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

    def get_distance(self):
        return self.game.distance
