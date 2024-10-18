import numpy as np
import matplotlib.pyplot as plt

# Define the quadratic function
def quadratic_function(x1, x2, a1, a2, c1, c2):
    return a1 * (x1 - c1)**2 + a2 * (x2 - c2)**2

if __name__ == "__main__":
    
    data = np.loadtxt('gradient_descent_sequence.txt')
    x1 = data[:, 0]
    x2 = data[:, 1]
# Example constants (replace these with your actual constants or estimations)
    a1 = 2.0
    a2 = 0.1
    c1 = np.mean(x1)  # Center of the x1 values
    c2 = np.mean(x2)  # Center of the x2 values
    
    x1_grid, x2_grid = np.meshgrid(np.linspace(min(x1) - 1, max(x1) + 1, 200),
                               np.linspace(min(x2) - 1, max(x2) + 1, 200))
    z = quadratic_function(x1_grid, x2_grid, a1, a2, c1, c2)
    
    # Create the contour plot
    plt.figure(figsize=(10, 6))
    contour = plt.contour(x1_grid, x2_grid, z, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Function Value')

    # Superimpose the scatter plot of gradient descent points
    plt.scatter(x1, x2, c='red', marker='x', label='Gradient Descent Points')

    # Add labels and title
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour Plot with Gradient Descent Points')
    plt.legend()
    plt.grid(False)
    plt.axis('equal')
    plt.show()
  