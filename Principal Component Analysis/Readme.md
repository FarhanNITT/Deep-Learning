# Gradient Descent Visualization in FCNNs
This assignment visualizes the gradient descent trajectory for a fixed fully connected neural network (FCNN) architecture with at least two hidden layers. Using principal component analysis (PCA), the high-dimensional parameter space is reduced to two dimensions for plotting, with the cross-entropy loss represented on the vertical axis. The project highlights the optimization landscape explored during training using stochastic gradient descent (SGD).

### Features
1) **SGD Trajectories:** : Captures parameter trajectories (p) during two separate training runs on 1,000 samples of Fashion MNIST.
2) **Dimensionality Reduction:** Uses PCA to reduce the NN's high-dimensional parameter space to two principal components.
3) **3D Scatter Plot:** Visualizes SGD trajectories in 3D, with the first two PCA components as axes and the cross-entropy loss (fCE(p)) as the vertical axis.
4) **Optimization Landscape**: Creates a dense 25x25 grid in the PCA-reduced space and plots the loss surface using matplotlib's plot_surface.
5) **Combined Visualization:** Integrates the trajectory scatter plot with the optimization landscape surface plot.

### Steps to Run

1) **Train the FCNN:** Perform SGD on the network twice using the provided train.py script. Save parameter trajectories (p) and cross-entropy loss values (fCE(p)).
2) **PCA Analysis:** Use sklearn.decomposition.PCA to compute the first two principal components from collected p vectors.
3) **Visualization:** Project parameter trajectories into the 2D PCA space. Generate a 3D scatter plot of SGD paths. Compute and render the loss surface over a dense PCA-reduced grid.

### Output
The final visualization combines:

1) **SGD Trajectories:** Representing the movement of the network's parameters during optimization.
2) **Optimization Landscape:** Highlighting the loss function surface in the PCA-reduced space.

<p align="center">
  <img src="Crossentropy.jpg" alt="ff" width="500">
</p>
