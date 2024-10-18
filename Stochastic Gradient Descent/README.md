 # Training 2-Layer Linear Neural Networks with Stochastic Gradient Descent 

In this assignment, we develop an age regression model that takes a 48x48 pixel grayscale face image and predicts the age of the individual as a real number. The dataset is provided in four separate files for training and testing purposes. [Dataset](https://drive.google.com/drive/folders/159VCdPYo8FVZOmxLXbWJTNSFf9NnXHf3?usp=sharing)

Our model is a simple 2-layer linear neural network, described mathematically as 𝑦^ =g(x;w)=x.Tw + b, where w represents the weights and 𝑏 the bias. The goal is to minimize the mean squared error (MSE) cost function, which measures the difference between the predicted ages and the actual ages.

We implement stochastic gradient descent (SGD) for weight optimization, strictly using linear algebra operations in NumPy, without relying on external machine learning libraries.

**Key components of the assignment include:**
1) **Hyperparameter Optimization:** We fine-tune several hyperparameters, including mini-batch size, learning rate, and number of epochs. To ensure robust performance, we use a validation set created by reserving a portion of the training data.
2) **Systematic Search:** Employ a grid search approach to explore at least two values for each hyperparameter, utilizing nested loops for organization.
3) **Performance Evaluation:** After optimizing our model, evaluate it on a separate test set and report the final training and test MSE values. Additionally, include the training cost values from the last 10 iterations of gradient descent.

The model was trained using various combinations of learning rates, mini-batch sizes, and number of epochs, as outlined below:
1) Learning Rates Tested: 1e-5, 1e-4, 1e-3
2) Mini-Batch Sizes Tested: 32, 64, 128
3) Number of Epochs Tested: 50, 100, 150

Higher learning rates, such as 1e-1 and 1e0, led to overshooting in MSE values, causing instability in the form of "NaN" and "inf" values. Therefore, lower learning rates were chosen for further experimentation. The optimal hyperparameters were found to be:

Optimal Number of Epochs: 150
Optimal Learning Rate: 0.0001
Optimal Mini-Batch Size: 64
Training MSE: 0.7645
Testing MSE: 0.7691
