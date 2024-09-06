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

    def update(self, gravity, colliding):
        # Apply gravity or booster effect
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
        laser_type = random.randint(0, 1)
        offset = random.randint(10, 300)
        if laser_type == 0:
            laser_width = random.randint(100, 300)
            laser_y = random.randint(100, HEIGHT - 100)
            new_lase = [
                [WIDTH + offset, laser_y],
                [WIDTH + offset + laser_width, laser_y],
            ]
        else:
            laser_height = random.randint(100, 300)
            laser_y = random.randint(100, HEIGHT - 400)
            new_lase = [
                [WIDTH + offset, laser_y],
                [WIDTH + offset, laser_y + laser_height],
            ]
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
        for lase in self.lasers:
            pygame.draw.line(
                screen, "yellow", (lase[0][0], lase[0][1]), (lase[1][0], lase[1][1]), 10
            )
            pygame.draw.circle(screen, "yellow", (lase[0][0], lase[0][1]), 12)
            pygame.draw.circle(screen, "yellow", (lase[1][0], lase[1][1]), 12)

    def check_collision(self, player_rect):
        for lase in self.lasers:
            # Create a bounding box for the laser line
            laser_rect = pygame.Rect(
                lase[0][0], lase[0][1] - 5, lase[1][0] - lase[0][0], 10
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
                if self.coords[1] > player_y + 10:
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
        screen.fill("black")
        self.laser.draw()
        self.rocket.draw()

        # Check for collisions with lasers and rocket
        player_rect = self.player.draw()
        if self.laser.check_collision(player_rect) or self.rocket.check_collision(
            player_rect
        ):
            self.player.dead = True  # Player dies on collision
            self.running = False
        self.ui.draw()
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
