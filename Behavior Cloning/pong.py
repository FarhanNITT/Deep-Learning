import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

class CNNPolicyNet(nn.Module):

    def __init__(self, num_actions):

        super(CNNPolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(128*6*6, 128)                 # 4608 = 128 * 6 * 6              
        self.policy = nn.Linear(128, num_actions)
        self.value = nn.Linear(128, 1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        policy_logits = self.policy(x)
        value = self.value(x)    

        return policy_logits, value


    def predict(self, x, device):

        # x to tensor and float32
        x = torch.tensor(x, dtype=torch.float32).to(device)
        x = x.permute(0, 3, 1, 2)    # shape: (1, 4, 84, 84) 
        policy_logits, value = self.forward(x)
        return F.softmax(policy_logits, dim=-1), value  
    


def train(model, dataloader, optimizer, criterion_policy, criterion_value, device):

    epoch_loss = 0
    model.train()
    for batch in dataloader:
        obs, actions = batch
        obs= obs.float() / 255.0
        obs = obs.permute(0, 3, 1, 2)
        obs, actions = obs.to(device), actions.to(device)
        policy_logits, value = model(obs)
        
        # compute policy_loss (cross-entropy) and value_loss (MSE)
        policy_loss = criterion_policy(policy_logits, actions)
        value_loss = criterion_value(value, torch.zeros_like(value))    

        # compute total loss
        loss = policy_loss + value_loss

        epoch_loss += loss.item()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return epoch_loss / len(dataloader)

if __name__ == "__main__":

    device = torch.device("cpu")
    # Set the base path to your data folder
    data_path = "/home/sarthak_m/Deep Learning/Assgnmnt/Assignment4/pong_data"

# Paths to the .pt files
    observations_file = os.path.join(data_path, "pong_observations", "pong_observations.pt")
    actions_file = os.path.join(data_path, "pong_actions", "pong_actions.pt")

# Load the data using torch.load
    try:
        observations = torch.load(observations_file)
        actions = torch.load(actions_file)
    except Exception as e:
        print("Error loading .pt files:", e)

    dataset = TensorDataset(observations, actions)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_actions = 6
    model = CNNPolicyNet(num_actions).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)  
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    # train model
    epochs = 100
    for epoch in range(epochs):
        loss = train(model, dataloader, optimizer, criterion_policy, criterion_value, device)
        print(f"Loss: {loss}, Epoch {epoch+1}/{epochs} complete")

    # save model        
    model_save_path = os.path.join(data_path, "pong_cnn_model_test14.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
