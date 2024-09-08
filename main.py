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
        # Generate a vertical laser with a random x-position
        offset = random.randint(10, 300)
        laser_x = WIDTH + offset  # The laser starts off-screen to the right
        laser_y_top = random.randint(
            100, HEIGHT - 150
        )  # Random y position, ensuring it fits within the screen
        new_lase = [laser_x, laser_y_top]  # Laser's top-left corner
        self.lasers.append(new_lase)

    def update(self, game_speed):
        if self.new_laser:
            self.generate_laser()
            self.new_laser = False

        # Move lasers to the left
        for lase in self.lasers:
            lase[
                0
            ] -= game_speed  # Only the x-coordinate is updated since it's horizontal movement

        # Remove lasers that have moved off-screen
        self.lasers = [
            lase for lase in self.lasers if lase[0] > -self.laser_surface.get_width()
        ]

        # Trigger new laser generation when the last one moves a certain distance
        if len(self.lasers) == 0 or self.lasers[-1][0] < WIDTH - 400:
            self.new_laser = True

    def draw(self):
        # Draw each laser sprite at its corresponding coordinates
        for lase in self.lasers:
            # random laser
            available_lasers = [
                self.laser_surface,
                self.laser_surface3,
            ]
            random_laser = random.choice(available_lasers)
            screen.blit(random_laser, (lase[0], lase[1]))

    def check_collision(self, player_rect):
        for lase in self.lasers:
            # Create a bounding box for the laser sprite, slightly shrinking the rect
            laser_rect = pygame.Rect(
                lase[0] + 5,
                lase[1] + 5,
                self.laser_surface.get_width() - 10,
                self.laser_surface.get_height() - 10,
            )
            if player_rect.colliderect(laser_rect):
                return True  # Collision detected
        return False


class Rocket:
    def __init__(self):
        self.counter = 0
        self.active = False
        self.delay = 0
        self.coords = [WIDTH, HEIGHT / 2]

        # Load and scale the rocket sprite
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
            self.coords = [WIDTH, random.randint(50, HEIGHT - 50)]  # Random Y position

        if self.active:
            if self.delay < 90:
                # Track the player's Y position with some delay
                if self.coords[1] > player_y + 10:
                    self.coords[1] -= 3
                else:
                    self.coords[1] += 3
                self.delay += 1
            else:
                # Move the rocket horizontally after delay
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
            rocket_rect = pygame.Rect(
                self.coords[0] + 5,
                self.coords[1] - 20,
                self.rocket_surface.get_width() - 10,
                self.rocket_surface.get_height() - 10,
            )
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


class Game:
    def __init__(self):
        self.game_speed = 3
        self.player = Player()
        self.laser = Laser()
        self.rocket = Rocket()
        self.ui = UI()
        self.pause = False
        self.restart_cmd = False
        self.lines = [0, WIDTH / 4, 2 * WIDTH / 4, 3 * WIDTH / 4]
        self.bg_color = (128, 128, 128)
        self.new_bg = 0

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.ui.save_player_info()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.pause = not self.pause
                if event.key == pygame.K_SPACE and not self.pause:
                    self.player.booster = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    self.player.booster = False
        return True

    def update(self):
        if not self.pause:
            self.ui.update(self.game_speed)
            self.player.update(GRAVITY, (False, False))
            self.laser.update(self.game_speed)
            self.rocket.update(self.game_speed, self.player.y)

    def draw(self):
        self.ui.draw_bg()

        self.laser.draw()
        self.rocket.draw()

        # Check for collisions with lasers and rocket
        player_rect = self.player.draw()
        if self.laser.check_collision(player_rect) or self.rocket.check_collision(
            player_rect
        ):
            self.player.dead = True  # Player dies on collision
            self.running = False
        self.ui.draw_score()
        pygame.display.flip()

    def run(self):
        self.running = True
        while self.running:
            self.running = self.handle_input()
            self.update()
            self.draw()
            timer.tick(FPS)


if __name__ == "__main__":
    game = Game()
    game.run()
    pygame.quit()
