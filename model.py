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
    
    def save_untrained(self, model_type):
        file_name = model_type + "_model.pth"
        model_folder_path = './untrained_models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"Saved untrained model: {file_name}")
    
    def save_trained(self, model_type):
        file_name = model_type + "_model.pth"
        model_folder_path = './saved_models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"Saved trained model To: {file_path}")
    
    def load_untrained(self, model_type):
        file_name = model_type + "_model.pth"
        model_folder_path = './untrained_models'
        file_path = os.path.join(model_folder_path, file_name)
        
        if not os.path.exists(file_path):
            print("No model found, starting fresh")
        else:
            self.load_state_dict(torch.load(file_path))
            print(f"Loading untrained model from: {file_path}")
   
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
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        prediction = self.model(state)

        target = prediction.detach().clone()
        with torch.no_grad():
            for i in range(len(done)):
                Q_new = reward[i]
                if not done[i]:
                    Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
                target[i][action[i].item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(prediction, target)
        loss.backward()
        self.optimizer.step()