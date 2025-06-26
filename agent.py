import numpy as np
import torch
import random
from collections import deque
from game import Flappy_Bird, games_played

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = None # TODO
        self.trainer = None # TODO

    def get_state(self, game):
        game.get_state()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns a list of tuples
        else:
            mini_sample = self.memory


        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff (exploration -> when ai is inexperienced / exploitation -> as ai gets more advanced)
        self.epsilon = 80 - self.n_games

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Flappy_Bird()
    
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset_game()
            agent.n_games = game.games_played
            agent.train_long_memory()

            if score > record:
                reward = score

                print(f"Games Played: {agent.n_games}\nScore: {score}\nRecord: {record}")



if __name__ == '__main__':
    train()