import random
import numpy as np
import pygame
import torch
import imageio
import json
from Population import Population

# Constants
WIDTH = 1000
HEIGHT = 600
FPS = 60
INIT_Y = HEIGHT - 130
GRAVITY = 0.4
VIDEO_STEPS_JUMP = 10
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

    def draw(self):
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

    def draw(self):
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

    def update(self, game_speed, player_y):

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
            rocket_rect = pygame.Rect(self.coords[0] - 60, self.coords[1] - 25, 50, 50)
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

    def draw_bg(self):
        screen.blit(self.bg_surface, (self.bg_x, 0))  # First background
        screen.blit(
            self.bg_surface, (self.bg_x + self.bg_surface.get_width(), 0)
        )  # Second background

    def draw_score(self):
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
    def __init__(
        self,
        population_size=10,
        warm_up_generations_before_rockets=0,
        is_in_preview=False,
        seed=None,
    ):
        self.game_speed = 3
        self.step = 0
        self.warm_up_generations_before_rockets = warm_up_generations_before_rockets
        self.population = Population(
            size=population_size,
            creature_args={
                "input_size": 12,  # y pos, velocity, laser1 up/down, laser2 up/down, missile up
                "output_size": 2,  # Boost or not
                "hidden_size": 64,
            },
            is_in_preview=is_in_preview,
        )
        self.is_in_preview = is_in_preview
        self.population.load("population.pth")
        self.players = [
            Player() for _ in range(population_size if not is_in_preview else 1)
        ]
        self.laser = Laser()
        self.rockets = [
            Rocket(self.players[i])
            for i in range(population_size if not is_in_preview else 1)
        ]  # One rocket per player
        self.distance = 0
        self.ui = UI()
        self.seed = seed  # Store the seed
        self.scores = []
        with open("evolution_agent_without_rockets_scores.json", "r") as file:
            self.scores = json.load(file)

        self.frames = []
        self.recording = True

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
                self.scores.append(self.distance)
                self.ui.save_player_info()
                self.end_generation()
                if self.is_in_preview and self.population.preview_done():
                    running = False
                self.step += 1
                if self.recording:
                    # output_filename = f"game_recording_generation_without_rockets_{self.step}.mp4"
                    # imageio.mimsave(output_filename, self.frames, fps=FPS)
                    # self.recording = False
                    self.frames = []
                    with open('evolution_agent_without_rockets_scores.json', 'w') as f:
                        json.dump(self.scores, f)

                if self.step % VIDEO_STEPS_JUMP == 0:
                    self.recording = True

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
                # if self.step >= self.warm_up_generations_before_rockets:
                #     self.rockets[i].update(self.game_speed, player.y)
                # # Check for collisions
                player_rect = player.draw()
                if self.laser.check_collision(player_rect) or self.rockets[
                    i
                ].check_collision(player_rect):
                    player.dead = True
                    self.population.creatures[i].set_fitness(self.distance)

    def draw(self):
        screen.fill("black")
        # draw distance and high score
        self.ui.draw_bg()
        self.laser.draw()

        for i, player in enumerate(self.players):
            if not player.dead:
                player.draw()
                self.rockets[i].draw()
        if self.is_in_preview:
            self.ui.preview(
                self.population.get_preview_generation(),
                self.population.get_preview_generation_best(),
            )
        else:
            self.ui.draw_score()
        pygame.display.flip()
        # if self.recording:
        #     frame = pygame.surfarray.array3d(screen)  # Capture the screen
        #     self.frames.append(np.rot90(np.fliplr(frame)))

    def get_state(self, player):
        # Get the closest lasers and missile position relative to the player
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
        rocket = self.rockets[self.players.index(player)]
        missile_up = rocket.coords[
                         1] / HEIGHT if rocket.active else -1
        missile_horizontal = rocket.coords[
                                 0] / WIDTH if rocket.active else -1

        player_y = player.y / HEIGHT
        player_y_velocity = player.y_velocity / 10.0  # Assuming max speed of 10
        game_speed = self.game_speed / 10.0  # Assuming max game_speed of 10
        missile_active = 1.0 if rocket.active else 0.0

        return  torch.tensor([
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

    def end_generation(self):
        # Evolve the population based on fitness
        if self.is_in_preview:
            self.population.fetch_next_creature()
        else:
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
        print(len(self.population.best_fitness), "games")


if __name__ == "__main__":
    game = Game(population_size=150, is_in_preview=False, seed=42)
    game.run()
    pygame.quit()
