import pygame, classes, random

pygame.init()
pygame.font.init()

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 700

GRAVITY = 0.7
JUMP_STRENGTH = -12
PIPE_SPEED = 3.5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

games_played = 1


class Flappy_Bird:
    def __init__(self):
        self.init_bird_x = 100
        self.init_bird_y = SCREEN_HEIGHT / 2 - 25

        self.bird = classes.Bird(self.init_bird_x, self.init_bird_y)

        self.run = True

        self.score = 0

        self.font_size = 100
        self.text_position = (SCREEN_WIDTH / 2 - 10, self.font_size)
        self.font = pygame.font.Font(None, self.font_size)

        top_pipe_height, bottom_pipe_height = self.randomize_pipe_height()

        self.init_top_pipe_x = SCREEN_WIDTH
        self.init_top_pipe_y = 0
        self.init_bottom_pipe_x = SCREEN_WIDTH
        self.init_bottom_pipe_y = SCREEN_HEIGHT - bottom_pipe_height

        self.pipe_arr = [
            [
                classes.Pipe(
                    self.init_top_pipe_x, self.init_top_pipe_y, top_pipe_height
                ),
                classes.Pipe(
                    self.init_bottom_pipe_x, self.init_bottom_pipe_y, bottom_pipe_height
                ),
            ]
        ]

        self.last_pipe_x = self.pipe_arr[0][0].x

    def randomize_pipe_height(self):
        height_1 = random.randint(50, SCREEN_HEIGHT - 250)
        height_2 = SCREEN_HEIGHT - height_1 - 200

        return [height_1, height_2]

    def event_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.run = False

                if event.key == pygame.K_SPACE:
                    self.bird.velocity = JUMP_STRENGTH

    def generate_pipes(self):
        if len(self.pipe_arr) < 4:
            top_pipe_height, bottom_pipe_height = self.randomize_pipe_height()

            gap_between_pipes = 400

            for pair in self.pipe_arr:
                for pipe in pair:
                    self.last_pipe_x = pipe.x

            self.pipe_arr.append(
                [
                    classes.Pipe(
                        self.last_pipe_x + gap_between_pipes, 0, top_pipe_height
                    ),
                    classes.Pipe(
                        self.last_pipe_x + gap_between_pipes,
                        SCREEN_HEIGHT - bottom_pipe_height,
                        bottom_pipe_height,
                    ),
                ]
            )

    def delete_pipes(self):
        first_pipe = self.pipe_arr[0][0]

        if first_pipe.x < -first_pipe.width:
            self.pipe_arr.pop(0)

    def update_bird(self):
        self.bird.draw(screen)
        self.bird.velocity += GRAVITY
        self.bird.y += self.bird.velocity

    def display_score(self):
        text_surface = self.font.render(str(self.score), True, BLACK)
        screen.blit(text_surface, self.text_position)

    def reset_game(self):
        global games_played
        games_played += 1

        self.bird = classes.Bird(self.init_bird_x, self.init_bird_y)

        self.score = 0

        top_pipe_height, bottom_pipe_height = self.randomize_pipe_height()

        self.init_top_pipe_x = SCREEN_WIDTH
        self.init_top_pipe_y = 0
        self.init_bottom_pipe_x = SCREEN_WIDTH
        self.init_bottom_pipe_y = SCREEN_HEIGHT - bottom_pipe_height

        self.pipe_arr = [
            [
                classes.Pipe(
                    self.init_top_pipe_x, self.init_top_pipe_y, top_pipe_height
                ),
                classes.Pipe(
                    self.init_bottom_pipe_x, self.init_bottom_pipe_y, bottom_pipe_height
                ),
            ]
        ]

        self.last_pipe_x = self.pipe_arr[0][0].x

    def game_loop(self):
        while self.run:
            screen.fill(WHITE)
            
            self.event_handler()
            self.generate_pipes()
            self.update_bird()
            self.delete_pipes()
            self.display_score()

            for i in self.pipe_arr:
                for pipe in i:
                    pipe.draw(screen)
                    pipe.x -= PIPE_SPEED

                    if pipe.collision(self.bird):
                        self.reset_game()

            if self.bird.y - self.bird.width > SCREEN_HEIGHT:
                self.reset_game()

            first_pipe = self.pipe_arr[0][0]

            if first_pipe.x < 0 and first_pipe.x > -4:
                self.score += 1

            pygame.display.flip()
            clock.tick(60)


flappy_bird = Flappy_Bird()
flappy_bird.game_loop()

print(games_played)

pygame.quit()
