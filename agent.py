<<<<<<< HEAD
=======
import numpy as np
>>>>>>> 4f851fdecdbd841d6f1dc5005a62e8ac3d7bd569
import torch
import random
from collections import deque
from game import Flappy_Bird
from model import Linear_QNet, QTrainer 
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
<<<<<<< HEAD
        self.epsilon = 80  # start with high exploration
        self.epsilon_min = 5  # minimum exploration
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(5, 256, 2)
        self.model.load()
=======
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(5, 256, 2)
>>>>>>> 4f851fdecdbd841d6f1dc5005a62e8ac3d7bd569
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        return game.get_state() 

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
<<<<<<< HEAD
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # Convert to tensors for batch processing
=======
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns a list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
>>>>>>> 4f851fdecdbd841d6f1dc5005a62e8ac3d7bd569
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
<<<<<<< HEAD
        # Decay epsilon but keep above minimum
        self.epsilon = max(self.epsilon_min, 80 - self.n_games)
=======
        # random moves: tradeoff (exploration -> when ai is inexperienced / exploitation -> as ai gets more advanced)
        self.epsilon = 80 - self.n_games
>>>>>>> 4f851fdecdbd841d6f1dc5005a62e8ac3d7bd569
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        return move 

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Flappy_Bird()
    
    while True:
<<<<<<< HEAD
        if agent.n_games == 50000:
=======
        if agent.n_games == 1000:
>>>>>>> 4f851fdecdbd841d6f1dc5005a62e8ac3d7bd569
            break

        # get old state
        state_old = agent.get_state(game)

        # get move
        action = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # remember
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset_game()
            agent.n_games += 1 
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f"Games Played: {agent.n_games}\nScore: {score}\nRecord: {record}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()