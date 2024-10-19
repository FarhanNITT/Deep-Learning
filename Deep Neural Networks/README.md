# Back Propogation of Multi-layer Neural Network

In this assignment, we develop a multi-layer feed-forward neural network to classify images of fashion items from the Fashion MNIST dataset, which consists of 10 different classes. The network will process 28x28 pixel images, converting them into 784-dimensional vectors, and output a probability distribution over the classes. A key focus of this assignment is implementing the backpropagation algorithm from scratch to efficiently compute gradients for training.

The architecture of the neural network will follow these equations:

<p align="center">
  <img src="FFNN.jpg" alt="ff" width="300">
</p>

Our task is to minimize the unregularized cross-entropy cost function defined as:

<p align="center">
  <img src="Crossentropy.jpg" alt="ff" width="500">
</p>

### Key Components
1) **Backpropagation Implementation:** A primary focus of this assignment is to implement the backpropagation algorithm from scratch. This process will involve calculating the gradients of the loss function with respect to the weights and biases, allowing for effective weight updates during training.
2) **Hyperparameter Tuning:** We explore various hyperparameters and architectural choices that impact network performance, including:
   1) Number of hidden layers (3, 4, or 5)
   2) Number of units in each hidden layer (30, 40, or 50)
   3) Learning rate (ranging from 0.001 to 0.5)
   4) Mini-batch size (16, 32, 64, 128, or 256)
   5) Number of epochs
   6) L2 regularization strength (applied to weight matrices)
   7) Learning rate decay frequency and rate
   8) Data augmentation techniques
      
3) **Gradient Checking:** Implement gradient checking using the provided check_grad method to verify the correctness of your gradient computations. This involves comparing numerical estimates of gradients with your analytical derivatives.

### Tasks:

1) **Implement SGD:** Develop a stochastic gradient descent algorithm capable of training the multi-layer network, ensuring that backpropagation supports any number of hidden layers.
2) **Gradient Verification:** Include the results of gradient checking for a network configuration with 3 hidden layers and 64 neurons in each.
3) **Performance Tracking:** Report test accuracy and unregularized cross-entropy. Aim for a test accuracy of at least 88%.
4) **Weight Visualization:** Visualize the first layer of weights by reshaping them into 28x28 matrices and presenting them as images in a grid.
