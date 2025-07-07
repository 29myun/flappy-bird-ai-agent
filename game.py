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
pygame.display.set_caption("Flappy Bird")
clock = pygame.time.Clock()

class Flappy_Bird:
    def __init__(self):
        # Bird
        self.init_bird_x = 100
        self.init_bird_y = SCREEN_HEIGHT / 2 - 25
        self.bird = classes.Bird(self.init_bird_x, self.init_bird_y)

        # Font
        self.font_size = 100
        self.text_position = (SCREEN_WIDTH / 2 - 10, self.font_size)
        self.font = pygame.font.Font(None, self.font_size)

        self.score = 0

        # Pipe
        self.vertical_gap_between_pipes = 250
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

        # Other
        self.done = False

    # Get random pipe size
    def randomize_pipe_height(self):
        pipe_min_height = 50
        height_1 = random.randint(pipe_min_height, SCREEN_HEIGHT - self.vertical_gap_between_pipes - pipe_min_height)
        height_2 = SCREEN_HEIGHT - height_1 - self.vertical_gap_between_pipes

        return [height_1, height_2]

    def event_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.done = True

                # if event.key == pygame.K_SPACE:
                #     self.bird.velocity = JUMP_STRENGTH

    def generate_pipes(self):
        # Number of pipe pairs cannot exceed 3
        if len(self.pipe_arr) < 3:
            top_pipe_height, bottom_pipe_height = self.randomize_pipe_height()

            horizontal_gap_between_pipes = 400

            for pair in self.pipe_arr:
                for pipe in pair:
                    self.last_pipe_x = pipe.x

            self.pipe_arr.append(
                [
                    classes.Pipe(
                        self.last_pipe_x + horizontal_gap_between_pipes, 0, top_pipe_height
                    ),
                    classes.Pipe(
                        self.last_pipe_x + horizontal_gap_between_pipes,
                        SCREEN_HEIGHT - bottom_pipe_height,
                        bottom_pipe_height,
                    ),
                ]
            )

    def update_bird(self):
        # y-direction only
        self.bird.velocity += GRAVITY
        self.bird.y += self.bird.velocity

    def display_score(self):
        text_surface = self.font.render(str(self.score), True, BLACK)
        screen.blit(text_surface, self.text_position)

    # A list of the game state for the agent
    def get_state(self):
        bird_y = self.bird.y / SCREEN_HEIGHT
        bird_velocity = self.bird.velocity / 20
        horizontal_distance_to_pipe = (self.pipe_arr[0][0].x - self.bird.x) / SCREEN_WIDTH
        distance_to_bot_of_top_pipe = (self.bird.y - self.pipe_arr[0][0].height) / SCREEN_HEIGHT 
        distance_to_top_of_bot_pipe = ((self.bird.y + self.bird.height) - self.pipe_arr[0][1].y) / SCREEN_HEIGHT

        state = [bird_y, bird_velocity, horizontal_distance_to_pipe, distance_to_bot_of_top_pipe, distance_to_top_of_bot_pipe]

        return np.array(state, dtype=float)

    # Re-initialize the game state
    def reset_game(self):        
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
        screen.fill(WHITE)
        self.bird.draw(screen)

        self.event_handler() 
        self.display_score()
        self.update_bird()
        self.generate_pipes()

        reward = 0
        done = False
        first_pipe_pair = self.pipe_arr[0]
        bird_center = self.bird.y + (self.bird.height / 2)
        gap_center = first_pipe_pair[0].height + (self.vertical_gap_between_pipes / 2)
        distance_y = abs(bird_center - gap_center)
        
        # Agent's control (one move)
        if action == 1:
            self.bird.velocity = JUMP_STRENGTH
            
        # Agent gains reward relative to how close the bird is to the center of the gap of a pipe pair
        reward += max(0, 1 - (distance_y / (self.vertical_gap_between_pipes / 2)))

        # Remove pipes, increment score, and gain reward when bird passes a pipe pair
        if self.bird.x > first_pipe_pair[0].x + first_pipe_pair[0].width:
            self.pipe_arr.pop(0)
            self.score += 1
            reward += 5

        for i in self.pipe_arr:
            for pipe in i:
                pipe.x -= PIPE_SPEED
                pipe.draw(screen)

                # Collision for bird and pipe; Gain negative reward and reset game
                if pipe.collision(self.bird):
                    reward = -5
                    done = True
                    final_score = self.score
                    self.reset_game()
                    return reward, done, final_score

        # Collision for bird to ceiling and ground; Gain negative reward and reset game
        if (
            self.bird.y + self.bird.height > SCREEN_HEIGHT or self.bird.y < 0
        ):
            reward = -5
            done = True
            final_score = self.score
            self.reset_game()
            return reward, done, final_score
        
        # Agent gains extra reward for each pipe it passes
        reward += 0.01 * self.score

        pygame.display.flip()
        clock.tick(60)

        return reward, self.done, self.score

# Human playtest
game = Flappy_Bird()
def game_loop():
    while True:
        done = game.play_step(0)[1]
        
        if done:
            break

if __name__ == "__main__":
    game_loop()
    pygame.quit()