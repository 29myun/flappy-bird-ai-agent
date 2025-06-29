import pygame    

class Pipe:
    def __init__(self, x, y, height):
        self.x = float(x)
        self.y = float(y)
        self.width = 75
        self.height = height

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 255, 0), ((self.x), (self.y), self.width, self.height))

    def collision(self, bird):
        bird_rect = pygame.Rect(int(bird.x), int(bird.y), bird.width, bird.height)
        pipe_rect = pygame.Rect(int(self.x), int(self.y), self.width, self.height)
        
        return bird_rect.colliderect(pipe_rect)


class Bird:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.width = 50
        self.height = 50
        self.velocity = 0

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), ((self.x), (self.y), self.width, self.height))
