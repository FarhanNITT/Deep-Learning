import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class FMNIST_DL(nn.Module):  

    def __init__(self):
        super(FMNIST_DL, self).__init__()  # Initialize the constructor of the base class
        
        self.flatten = nn.Flatten()         # Flatten the input
        self.layer1 = nn.Linear(28*28, 512)  # First layer
        self.layer2 = nn.Linear(512, 256)   # Second layer
        self.layer3 = nn.Linear(256, 128)   # Third layer 
        self.layer4 = nn.Linear(128, 64)    # Fourth layer 
        self.output_layer = nn.Linear(64, 10)  # Output layer for 10 classes

    def forward(self, x):
        out = self.flatten(x)  # Flatten the input
        out = self.layer1(out)  # Apply layer1
        out = nn.ReLU()(out)  # Correctly apply ReLU after layer1
        out = self.layer2(out)  # Apply layer2
        out = nn.ReLU()(out)  # Correctly apply ReLU after layer2
        out = self.layer3(out)  # Apply layer3
        out = nn.ReLU()(out)  # Correctly apply ReLU after layer3
        out = self.layer4(out)  # Apply layer4
        out = self.output_layer(out)  # Output layer (no activation here for CrossEntropyLoss)
        
        return out


transform_feature= transforms.Compose([
    transforms.ToTensor(),             # transform the PIL image to a tensor
])

# Define Hyperparameters 
num_epochs = 10
batch_size = 32
learning_rate = 1e-3


training_data= datasets.FashionMNIST(root='./data',transform=transform_feature,download=True,train=True)

# Use only 1000 samples
subset_indices = np.random.choice(len(training_data), 1000, replace=False)
train_subset = torch.utils.data.Subset(training_data, subset_indices)

train_loader= DataLoader(train_subset,batch_size=batch_size,shuffle=True)  # we shuffle the training data for better generalisation but not testing data


# recording history of model parameters

num_initializations = 2

all_param_list=[]
all_cost_list =[]


def collect_parameters_and_cost():

    # param.view(-1)  flattens the parameter to 1D and then we concatenate them
    params_vector = torch.cat([param.view(-1) for param in model.parameters()])
    return params_vector.detach().numpy()  # we detach the tensor from the computational graph and convert to numpy array


for run in range(num_initializations):  # Run SGD twice for different initializations


    model = FMNIST_DL()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)


    parameter_history = []
    cost_history = []
    
    for epoch in range(num_epochs):  
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Collect parameters and cost after each update
            parameter_history.append(collect_parameters_and_cost())
            cost_history.append(loss.item()) # Convert lists to numpy arrays for PCAhistory.append(loss.item())

    all_param_list.append(parameter_history)
    all_cost_list.append(cost_history)

# Convert lists to numpy arrays for PCA
all_reduced_params = []
all_losses_array = []

for params_list, losses_list in zip(all_param_list, all_cost_list):
    params_array = np.array(params_list)
    losses_array = np.array(losses_list)
    
    # Perform PCA to reduce dimensions
    pca = PCA(n_components=2)
    reduced_params = pca.fit_transform(params_array)
    
    all_reduced_params.append(reduced_params)
    all_losses_array.append(losses_array)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each initialization's trajectory
colors = ['b', 'r']  # Different colors for different initializations

for i in range(num_initializations):
    ax.scatter(all_reduced_params[i][:, 0], all_reduced_params[i][:, 1], all_losses_array[i], c=colors[i], marker='o', label=f'Initialization {i+1}')


# Create a grid of points in the 2-D PCA space

grid_size = 25  # mentioned in question

# We first concatenate all the first PCA values from all initializations and then find their min and max values to set the range fro the grid on the 2 PCA axes
pc1_range = np.linspace(np.min(np.concatenate([ap[:, 0] for ap in all_reduced_params])), 
                         np.max(np.concatenate([ap[:, 0] for ap in all_reduced_params])), 
                         grid_size)
pc2_range = np.linspace(np.min(np.concatenate([ap[:, 1] for ap in all_reduced_params])), 
                         np.max(np.concatenate([ap[:, 1] for ap in all_reduced_params])), 
                         grid_size)

# grid_pc1 and grid_pc2 are 2d arrays containing all the coordinates of points in the 2d pca space
grid_pc1, grid_pc2 = np.meshgrid(pc1_range, pc2_range)

# we flatten the 2-D grid arrays into 1-D arrays, making it easier to create a list of individual grid points.
grid_points = np.column_stack([grid_pc1.ravel(), grid_pc2.ravel()])

# So grid_points is a column matrix where each row in a point of form [pca1,pca2]

# We now map the 2-D grid points in PCA space back to the original parameter space to evaluate the neural network’s cross-entropy loss at those specific points. 

# PCA reduces the parameter space to just two dimensions, giving us a view of the primary directions (principal components) where the most variance occurs in the parameter configurations during training.

# The grid we create in this 2-D PCA space represents a sample of possible parameter configurations in the “most significant” directions, but it only exists in reduced form. To get actual parameter values and evaluate loss, we must map each grid point back to the original space.


# Evaluate cost values over the grid

grid_losses = np.zeros(grid_points.shape[0])

for idx, point in enumerate(grid_points):
    # Project back to the full parameter space
    full_params = pca.inverse_transform(point.reshape(1, -1))
    
    # Load these parameters into the NN
    with torch.no_grad():
        start = 0
        for param in model.parameters():
            end = start + param.numel()  # param.numel() gives the total number of elements in the current parameter tensor
            
            param.data.copy_(torch.tensor(full_params[0][start:end]).reshape(param.shape))

            # selects a slice of full_params  corresponding to the current parameter tensor, then reshape to match param shape
            # cop the value of these parameters to update the model with the new PCA values to compute the corresponding fce loss


            start = end

    # Compute the cross-entropy loss on the training set
    total_loss = 0
    with torch.no_grad():
        for images, labels in train_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    grid_losses[idx] = total_loss / len(train_loader)  # Average loss

grid_losses = grid_losses.reshape(grid_size, grid_size)
# Implementing a 3D plot that shows the parameter trajectory across two SGD initializations, along with a 3D surface of the loss landscape in the PCA-reduced space.

ax.plot_surface(grid_pc1, grid_pc2, grid_losses, alpha=0.5, cmap='viridis')

# Labels and title
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA2 Component 2')
ax.set_zlabel('Train Loss')
ax.set_title('Gradient Descent Trajectory and Loss Landscape')
ax.legend()

plt.show()








