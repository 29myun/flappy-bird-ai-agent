# Choose to pick the sizes of a 3-layered model
model_type = input("Enter in the model type: ")
# Choose to train an agent or make it play
train = str.lower(input("Is this model trained or untrained?: "))

import numpy as np
import torch
import random
import pygame
from collections import deque
from game import Flappy_Bird
from model import Linear_QNet, QTrainer 
import data

# Window title
pygame.display.set_caption(f"Flappy Bird AI Training Model: {model_type}")

MAX_MEMORY = 100_000
BATCH_SIZE = 128
LR = 0.001
MAX_GAMES = 1000
MAX_SCORE = 1000

class Agent:
    def __init__(self, model_type):
        self.model_type = model_type
        self.n_games = 0
        self.epsilon_max = 80 # Max exploration rate
        self.epsilon_min = 1 # Min exploration rate
        self.epsilon = self.epsilon_max # Exploration rate
        self.gamma = 0.9 #
        self.memory = deque(maxlen=MAX_MEMORY)
        self.keep_exploring = False

        # Error handling for model format and model layer sizes
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

    # Adds the game state to the memory deque
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Create a separate instance of self.memory and randomize all of the game states within it when the length of the memory deque > BATCH_SIZE (128)
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        
        # Keep the memory deque as it is when length of the memory deque is <= BATCH_SIZE (128)
        else:
            mini_sample = list(self.memory)

        # Put every game state into their own tuple
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        # Convert to tensors for batch processing
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-0.01 * self.n_games) # faster decay when n_games is lower, slower decay when n_games is higher (epsilon greedy exploration)
        
        if random.randint(0, 200) < self.epsilon and self.keep_exploring: 
            move = random.randint(0, 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        return move 

# Train loop
def untrained(agent, game):
    agent.model.load_untrained(model_type)

    record = 0

    agent.keep_exploring = True
    
    while agent.n_games < MAX_GAMES:

        # Get old game state
        state_old = agent.get_state(game)
        
        # Get move
        action = agent.get_action(state_old)

        # Perform move
        reward, done, score = game.play_step(action)
        
        # Get new game state
        state_new = agent.get_state(game)
        
        agent.train_short_memory(state_old, action, reward, state_new, done)
        
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            game.reset_game()
            agent.n_games += 1 
            agent.train_long_memory()

            # Save model when agent gets new high score
            if score > record:
                record = score
                agent.model.save_untrained(model_type)
                data.save(model_type, record, agent.n_games, "./untrained_models_data")

            print(f"Games Played: {agent.n_games}\nScore: {score}\nRecord: {record}")
        
# Play loop
def trained(agent, game):    
    agent.model.load_trained(model_type)
        
    record = data.load(model_type)["record"]
    total_games_played = data.load(model_type)["total_games_played"]
    
    agent.keep_exploring = False
    
    while agent.n_games < MAX_GAMES:

        state_old = agent.get_state(game)
        
        action = agent.get_action(state_old)

        reward, done, score = game.play_step(action)
        
        state_new = agent.get_state(game)       
        
        agent.train_short_memory(state_old, action, reward, state_new, done)

        agent.remember(state_old, action, reward, state_new, done)

        if done:
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
