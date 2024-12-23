import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.optim
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F


transform_feature= transforms.Compose([
    transforms.ToTensor(),             # transform the PIL image to a tensor
    transforms.Normalize((0.5,), (0.5,)),   # normalizing the data to 0.5 mean and 0.5 standard deviation
])

batch_size = 8
learning_rate = lr=0.001
momentum=0.9
num_epochs = 15

training_data = datasets.FashionMNIST(root = './data',transform=transform_feature,download=True,train=True)
testing_data = datasets.FashionMNIST(root='./data',transform=transform_feature,download=True,train=False)

training_DL = DataLoader(training_data,batch_size=batch_size,shuffle=True,num_workers=2)
testing_DL = DataLoader(testing_data,batch_size=batch_size,shuffle=False,num_workers=2)

classes= ('T-Shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot')


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # input is a gray scale image (1x28x28) - (6x24x24)
        self.pool = nn.MaxPool2d(2, 2)  # (6x24x24) - (6x12x12)
        self.conv2 = nn.Conv2d(6, 16, 5) # (6x12x12) - (16x8x8)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self,x):
         out = F.relu(self.pool(self.conv1(x))) # (1x28x28) - (6x12x12)
         out = F.relu(self.pool(self.conv2(out)))  #(6x12x12) - (16x4x4)

         out= torch.flatten(out,1)
         out = F.relu(self.fc1(out))
         out = F.relu(self.fc2(out))
         out = self.fc3(out)

         return out

model = CNN()
loss_fn = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum) # Stochastic gradient descent

def train(model, train_loader, loss_fn, optimizer, num_epochs):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        training_loss = 0.0
        for img, label in train_loader:
            optimizer.zero_grad()
            outputs = model(img)
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {training_loss/len(train_loader):.4f}")

def validate(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct_prediction = 0
    total = 0
    with torch.no_grad():  # No gradients needed for validation
        for images, labels in test_loader:
            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted classes
            total += labels.size(0)  # Count total samples
            correct_prediction += (predicted == labels).sum().item()  # Count correct predictions

    accuracy = 100 * correct_prediction / total
    print(f"Test Accuracy: {accuracy:.2f}%")

train(model, training_DL, loss_fn, optimizer, num_epochs)
PATH = './fashion_cnn.pth'
torch.save(model.state_dict(), PATH)
validate(model, testing_DL)
