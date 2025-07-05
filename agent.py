model_type = input("Enter in the model type: ")
train = str.lower(input("Is this model trained or untrained?: "))

import numpy as np
import torch
import random
import pygame
from collections import deque
from game import Flappy_Bird
from model import Linear_QNet, QTrainer 
import data

pygame.display.set_caption(f"Flappy Bird AI Training Model: {model_type}")

MAX_MEMORY = 100_000
BATCH_SIZE = 128
LR = 0.001
MAX_GAMES = 5000

class Agent:
    def __init__(self, model_type):
        self.model_type = model_type
        self.n_games = 0
        self.epsilon_max = 80
        self.epsilon_min = 1
        self.epsilon = self.epsilon_max
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.keep_exploring = False

        try:
            input_size, hidden_size, output_size = map(int, model_type.split("_"))
        except ValueError:
            raise ValueError(f"Invalid model_type format: '{model_type}'. Expected format: '5_128_2'")
        
        if 5 > input_size or 64 > hidden_size > 512 or not output_size  == 2:
            raise ValueError("Invalid layer sizes.")
        
        self.model = Linear_QNet(input_size, hidden_size, output_size)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        return game.get_state() 

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        # Convert to tensors for batch processing
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-0.01 * self.n_games) # faster decay when n_games is lower, slower decay when n_games is higher (epsilon greedy exploration)
        
        if random.randint(0, 200) < self.epsilon and self.keep_exploring: # after about 2200 games, the agent will randomize its move once every 10 cycles if this condition is met
            move = random.randint(0, 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        return move 

def untrained(agent, game):
    agent.model.load_untrained(model_type)

    record = 0

    agent.keep_exploring = True
    
    while True:
        if agent.n_games == MAX_GAMES:
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
            # train long memory
            game.reset_game()
            agent.n_games += 1 
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save_untrained(model_type)
                data.save(model_type, record, agent.n_games, "./untrained_models_data")

            print(f"Games Played: {agent.n_games}\nScore: {score}\nRecord: {record}")

def trained(agent, game):    
    agent.model.load_trained(model_type)
        
    record = data.load(model_type)["record"]
    total_games_played = data.load(model_type)["total_games_played"]
    
    agent.keep_exploring = False
    
    while True:
        if agent.n_games == MAX_GAMES:
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
            # train long memory
            game.reset_game()
            total_games_played += 1
            agent.n_games += 1 
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save_trained(model_type)
                data.save(model_type, record, total_games_played, "./saved_models_data")

            print(f"Games Played: {agent.n_games}\nTotal Games Played: {total_games_played}\nScore: {score}\nRecord: {record}")

if __name__ == '__main__':
    agent = Agent(model_type)
    game = Flappy_Bird()

    if train == "untrained":
        untrained(agent, game)
    elif train == "trained":
        trained(agent, game)
    else:
        raise RuntimeError("Invalid training option")

    print(f"Finished {train} session with model: {model_type}")
