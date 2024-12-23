import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
import os
import torch.optim as optim
import time
from tempfile import TemporaryDirectory
from torch.optim import lr_scheduler

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform_feature = transforms.Compose([
    transforms.ToTensor(),             # transform the PIL image to a tensor
    transforms.Normalize((0.5,), (0.5,)),   # normalizing the data to 0.5 mean and 0.5 standard deviation
])

batch_size = 8
learning_rate = lr=0.001
momentum = 0.9
num_epochs = 15

training_data = datasets.FashionMNIST(root='./data', transform=transform_feature, download=True, train=True)
testing_data = datasets.FashionMNIST(root='./data', transform=transform_feature, download=True, train=False)

training_DL = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
testing_DL = DataLoader(testing_data, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

dataset_sizes = {'train': len(training_data), 'test': len(testing_data)}
dataloaders = {'train': training_DL, 'test': testing_DL}

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        best_acc = 0.0

        for epoch in range(num_epochs):
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)  # Move to device

                    optimizer.zero_grad()

                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)  # getting batch loss
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # Load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

# Initialize the model
model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Adjust the final layer for the number of classes in FashionMNIST
model_ft.fc = nn.Linear(num_ftrs, 10)

# Move the model to the appropriate device
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=momentum)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Start training the model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
