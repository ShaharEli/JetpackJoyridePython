import random
import pygame
import torch

from Population import Population

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


class Player:
    def __init__(self):
        self.y = INIT_Y
        self.booster = False
        self.y_velocity = 0
        self.counter = 0
        self.dead = False

    def update(self, gravity, colliding):
        # Apply gravity or booster effect
        if self.dead:
            return
        if self.booster:
            self.y_velocity -= gravity
        else:
            self.y_velocity += gravity

        # Prevent the player from moving out of bounds
        if self.y + self.y_velocity > INIT_Y:  # Prevent passing the floor
            self.y = INIT_Y
            self.y_velocity = 0
        elif self.y + self.y_velocity < 0:  # Prevent passing the ceiling
            self.y = 0
            self.y_velocity = 0
        else:
            self.y += self.y_velocity

        # Update counter for animation purposes
        self.counter = (self.counter + 1) % 40

    def draw(self):
        play = pygame.rect.Rect((120, self.y + 10), (25, 60))
        if self.dead:
            return None
        if self.y < INIT_Y:
            if self.booster:
                pygame.draw.ellipse(screen, "red", [100, self.y + 50, 20, 30])
                pygame.draw.ellipse(screen, "orange", [105, self.y + 50, 10, 30])
                pygame.draw.ellipse(screen, "yellow", [110, self.y + 50, 5, 30])
            pygame.draw.rect(screen, "yellow", [128, self.y + 60, 10, 20], 0, 3)
            pygame.draw.rect(screen, "orange", [130, self.y + 60, 10, 20], 0, 3)
        else:
            if self.counter < 10:
                pygame.draw.line(
                    screen, "yellow", (128, self.y + 60), (140, self.y + 80), 10
                )
                pygame.draw.line(
                    screen, "orange", (130, self.y + 60), (120, self.y + 80), 10
                )
            elif 10 <= self.counter < 20:
                pygame.draw.rect(screen, "yellow", [128, self.y + 60, 10, 20], 0, 3)
                pygame.draw.rect(screen, "orange", [130, self.y + 60, 10, 20], 0, 3)
            elif 20 <= self.counter < 30:
                pygame.draw.line(
                    screen, "yellow", (128, self.y + 60), (120, self.y + 80), 10
                )
                pygame.draw.line(
                    screen, "orange", (130, self.y + 60), (140, self.y + 80), 10
                )
            else:
                pygame.draw.rect(screen, "yellow", [128, self.y + 60, 10, 20], 0, 3)
                pygame.draw.rect(screen, "orange", [130, self.y + 60, 10, 20], 0, 3)
        pygame.draw.rect(screen, "white", [100, self.y + 20, 20, 30], 0, 5)
        pygame.draw.ellipse(screen, "orange", [120, self.y + 20, 30, 50])
        pygame.draw.circle(screen, "orange", (135, self.y + 15), 10)
        pygame.draw.circle(screen, "black", (138, self.y + 12), 3)
        return play


class Laser:
    def __init__(self):
        self.lasers = []
        self.new_laser = True

    def generate_laser(self):
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

        # Move lasers and remove off-screen lasers
        for lase in self.lasers:
            lase[0][0] -= game_speed
            lase[1][0] -= game_speed
        self.lasers = [lase for lase in self.lasers if lase[1][0] > 0]

        if len(self.lasers) == 0 or self.lasers[-1][0][0] < WIDTH - 400:
            self.new_laser = True

    def draw(self):
        for lase in self.lasers:
            # Draw vertical lasers
            pygame.draw.line(
                screen, "yellow", (lase[0][0], lase[0][1]), (lase[1][0], lase[1][1]), 10
            )
            pygame.draw.circle(screen, "yellow", (lase[0][0], lase[0][1]), 12)
            pygame.draw.circle(screen, "yellow", (lase[1][0], lase[1][1]), 12)

    def check_collision(self, player_rect):
        for lase in self.lasers:
            # Create a bounding box for the vertical laser line
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

    def update(self, game_speed):
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

    def draw(self):
        if self.active:
            if self.delay < 90:
                pygame.draw.rect(
                    screen,
                    "dark red",
                    [self.coords[0] - 60, self.coords[1] - 25, 50, 50],
                    0,
                    5,
                )
                screen.blit(
                    font.render("!", True, "black"),
                    (self.coords[0] - 40, self.coords[1] - 20),
                )
            else:
                pygame.draw.rect(
                    screen, "red", [self.coords[0], self.coords[1] - 10, 50, 20], 0, 5
                )
                pygame.draw.ellipse(
                    screen,
                    "orange",
                    [self.coords[0] + 50, self.coords[1] - 10, 50, 20],
                    7,
                )

    def check_collision(self, player_rect):
        if self.active:
            # Create a bounding box for the rocket
            rocket_rect = pygame.Rect(self.coords[0] - 60, self.coords[1] - 25, 50, 50)
            if player_rect.colliderect(rocket_rect):
                return True  # Collision detected
        return False


class UI:
    def __init__(self):
        self.distance = 0
        self.high_score, self.lifetime = self.load_player_info()

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

    def draw(self):
        screen.blit(
            font.render(f"Distance: {int(self.distance)} m", True, "white"), (10, 10)
        )
        screen.blit(
            font.render(f"High Score: {int(self.high_score)} m", True, "white"),
            (10, 70),
        )


class Game:
    def __init__(self, population_size=10, warm_up_generations_before_rockets=0):
        self.game_speed = 3
        self.step = 0
        self.warm_up_generations_before_rockets = warm_up_generations_before_rockets
        self.population = Population(
            size=population_size,
            creature_args={
                "input_size": 8,  # y pos, velocity, laser1 up/down, laser2 up/down, missile up
                "output_size": 2,  # Boost or not
                "hidden_size": 64,
            },
        )
        self.population.load("population.pth")
        self.players = [Player() for _ in range(population_size)]
        self.laser = Laser()
        self.rockets = [
            Rocket(self.players[i]) for i in range(population_size)
        ]  # One rocket per player
        self.distance = 0
        self.ui = UI()

    def run(self):
        running = True
        self.ui.load_player_info()
        while running:
            timer.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update game objects
            self.update()

            # Draw everything
            self.draw()

            # Check if all players are dead
            if all(player.dead for player in self.players):
                self.ui.save_player_info()
                self.end_generation()
                self.step += 1

    def update(self):
        self.distance += self.game_speed
        self.laser.update(self.game_speed)
        self.ui.update(self.game_speed)
        for i, player in enumerate(self.players):
            if not player.dead:
                state = self.get_state(player)
                action = self.population.creatures[i].act(state)
                player.booster = action == 1
                player.update(GRAVITY, (False, False))
                if self.step > self.warm_up_generations_before_rockets:
                    self.rockets[i].update(self.game_speed)
                # Check for collisions
                player_rect = player.draw()
                if self.laser.check_collision(player_rect) or self.rockets[
                    i
                ].check_collision(player_rect):
                    player.dead = True
                    self.population.creatures[i].set_fitness(self.distance)

    def draw(self):
        screen.fill("black")
        # draw distance and high score
        self.ui.draw()
        self.laser.draw()
        for i, player in enumerate(self.players):
            if not player.dead:
                player.draw()
                self.rockets[i].draw()
        pygame.display.flip()

    def get_state(self, player):
        # Get the closest lasers and missile position relative to the player
        if self.laser.lasers:
            lasers = sorted(self.laser.lasers, key=lambda x: x[0][0])
            lasers = list(filter(lambda x: x[0][0] >= 100, lasers))
            if len(lasers) > 1:
                laser1, laser2 = lasers[:2]
            else:
                laser1 = laser2 = lasers[0]
        else:
            laser1 = [[WIDTH, 0], [WIDTH, HEIGHT]]
            laser2 = laser1

        missile_up = (
            self.rockets[self.players.index(player)].coords[1]
            if self.rockets[self.players.index(player)].active
            else HEIGHT / 2
        )

        return torch.tensor(
            [
                player.y,
                player.y_velocity,
                self.game_speed,
                laser1[0][1],  # closest laser up y
                laser1[1][1],  # closest laser down y
                laser2[0][1],  # second closest laser up y
                laser2[1][1],  # second closest laser down y
                missile_up,
            ],
            dtype=torch.float32,
        )

    def end_generation(self):
        # Evolve the population based on fitness
        self.population.evolve()
        self.population.save("population.pth")
        # Reset for the next generation
        self.distance = 0
        self.players = [Player() for _ in range(len(self.players))]
        self.laser = Laser()
        self.rockets = [Rocket(self.players[i]) for i in range(len(self.players))]
        self.ui.load_player_info()
        self.ui.distance = 0
        self.game_speed = 3


if __name__ == "__main__":
    game = Game(population_size=150)
    game.run()
    pygame.quit()
