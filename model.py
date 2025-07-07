import torch
import os
import sys
import numpy as np

nn = torch.nn
optim = torch.optim
F = torch.nn.functional  

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    # Save untrained model weights to a .pth file
    def save_untrained(self, model_type):
        file_name = model_type + "_model.pth"
        model_folder_path = './untrained_models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"Saved untrained model: {file_name}")

    # Save trained model weights to a .pth file
    def save_trained(self, model_type):
        file_name = model_type + "_model.pth"
        model_folder_path = './saved_models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"Saved trained model to: {file_path}")

    # Load weights from saved untrained model
    def load_untrained(self, model_type):
        file_name = model_type + "_model.pth"
        model_folder_path = './untrained_models'
        file_path = os.path.join(model_folder_path, file_name)
        
        if not os.path.exists(file_path):
            print("No model found, starting fresh")
        else:
            self.load_state_dict(torch.load(file_path))
            print(f"Loading untrained model from: {file_path}")

    # Load weights from saved trained model
    def load_trained(self, model_type):
        file_name = model_type + "_model.pth"
        model_folder_path = './saved_models'
        file_path = os.path.join(model_folder_path, file_name)
        
        if not os.path.exists(file_path):
            raise RuntimeError(f"Model {model_type} not found")
        else:
            self.load_state_dict(torch.load(file_path))
            print(f"Loading trained model from: {file_path}")

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma  # Discount factor: how much to prioritize future rewards
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert inputs to PyTorch tensors
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            # Expand dimensions of each tensor to include a batch of size 1
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Q-values predicted by the current model for current state
        prediction = self.model(state)

        # Clone predictions to use as targets, without gradients
        target = prediction.detach().clone()

        with torch.no_grad():  # No gradient tracking needed when computing target Q-values
            for i in range(len(done)):
                Q_new = reward[i]
                if not done[i]:
                    # Bellman equation: immediate reward + discounted max Q from next state
                    Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
                # Update only the Q-value for the taken action
                target[i][action[i].item()] = Q_new

        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(prediction, target)  # Compare predicted Q-values vs target Q-values
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update weights
