import random
import pygame

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
            screen.blit(self.laser_surface, (lase[0][0], lase[0][1]))

    def check_collision(self, player_rect):
        for lase in self.lasers:
            # Create a bounding box for the laser line
            laser_rect = pygame.Rect(
                lase[0][0] - 5, lase[0][1], 10, lase[1][1] - lase[0][1]
            )
            if player_rect.colliderect(laser_rect):
                return True  # Collision detected
        return False


class Game:
    def __init__(self):
        self.game_speed = 3
        self.player = Player()
        self.laser = Laser()
        self.rocket = Rocket(self.player)  # Add rocket instance tied to the player
        self.distance = 0
        self.scores = []  # Array to store scores for 750 games

    def run(self):
        for i in range(100):  # Run 750 games
            print(f"{i + 1}th / 100 game")
            self.distance = 0
            self.player.dead = False
            self.player.y = INIT_Y
            self.laser.lasers = []

            while not self.player.dead:
                timer.tick(FPS)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()

                # Randomly decide if booster is on or off
                self.player.booster = random.choice([True, False])

                # Update game objects
                self.update()

                # Draw everything
                self.draw()

            # Save the score after the player dies
            self.scores.append(self.distance)

        # Save the scores to a file
        with open("random_scores_with_score.json", "w") as f:
            import json

            json.dump(self.scores, f)

    def update(self):
        self.distance += self.game_speed
        self.laser.update(self.game_speed)
        self.rocket.update(self.game_speed)  # Update the rocket
        self.player.update(GRAVITY, (False, False))
        player_rect = self.player.get_collision_rect()

        # Check for collisions with both the laser and rocket
        if self.laser.check_collision(player_rect) or self.rocket.check_collision(
            player_rect
        ):
            self.player.dead = True

    def draw(self):
        screen.fill("black")
        self.laser.draw()
        self.rocket.draw()  # Draw the rocket
        self.player.draw()
        pygame.display.flip()


if __name__ == "__main__":
    game = Game()
    game.run()
    pygame.quit()
