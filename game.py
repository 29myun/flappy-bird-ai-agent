import pygame, classes, random, numpy as np

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

        self.done = False

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
                self.done = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.done = True

                if event.key == pygame.K_SPACE:
                    self.bird.velocity = JUMP_STRENGTH

    def generate_pipes(self):
        if len(self.pipe_arr) < 3:
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

    def update_bird(self):
        self.bird.velocity += GRAVITY
        self.bird.y += self.bird.velocity

    def display_score(self):
        text_surface = self.font.render(str(self.score), True, BLACK)
        screen.blit(text_surface, self.text_position)

    def get_state(self):
        # Example state: [bird_y, bird_velocity, pipe_x, pipe_y, ...]
        bird_y = self.bird.y
        bird_velocity = self.bird.velocity
        pipe_x = self.pipe_arr[0][0].x
        pipe_y = self.pipe_arr[0][0].height  # or .y depending on your Pipe class

        # Distance to next pipe, gap, etc.
        next_pipe_bottom_y = self.pipe_arr[0][1].y

        state = [
            bird_y,
            bird_velocity,
            pipe_x - self.bird.x,  # horizontal distance to next pipe
            pipe_y,  # top pipe height
            next_pipe_bottom_y,  # bottom pipe y
        ]

        return np.array(state, dtype=float)

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

    def play_step(self, action):
        reward = 0
        done = False

        if action == 1:
            self.bird.velocity = JUMP_STRENGTH

        screen.fill(WHITE)

        self.generate_pipes()
        self.update_bird()
        self.display_score()
        self.event_handler()


        for i in self.pipe_arr:
            for pipe in i:
                pipe.draw(screen)
                pipe.x -= PIPE_SPEED

                if pipe.collision(self.bird):
                    reward = -1
                    done = True
                    final_score = self.score
                    self.reset_game()
                    return reward, done, final_score

        if (
            self.bird.y + self.bird.width > SCREEN_HEIGHT
            or self.bird.y < 0
        ):
            reward = -1
            done = True
            final_score = self.score
            self.reset_game()
            return reward, done, final_score

        first_pipe = self.pipe_arr[0]

        if self.bird.x > first_pipe[0].x + first_pipe[0].width:
            self.pipe_arr.pop(0)
            self.score += 1
            reward = 1
        
        if self.bird.y > first_pipe[0].height and (self.bird.y + self.bird.height) < first_pipe[1].y:
            reward += 0.01
        else:
            reward -= 0.01

        self.bird.draw(screen)
        pygame.display.flip()
        clock.tick(60)

        # reward -= 0.001 DO THIS at 3rd training session

        return reward, self.done, self.score

game = Flappy_Bird()
def game_loop() -> None:
    while True:
        done = game.play_step(0)[1]
        
        if done:
            break

if __name__ == "__main__":
    # game_loop()
    pygame.quit()

