#%%
import numpy as np

def prox_f(x, alpha):
    """Proximal operator for the function f."""
    # Example: f(x) = ||x||_1, so prox_f is soft-thresholding
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)
    #return np.array([3,x[1]])

def prox_g(x, alpha):
    """Proximal operator for the function g."""
    # Example: g(x) = 0.5 * ||x||^2, so prox_g is a shrinkage operator
    #return x / (1 + alpha)
    return np.array([x[0],1])

def grad_h(x):
    """Gradient of the function h."""
    # Example: h(x) = 0.5 * ||x - c||^2, gradient is (x - c)
    c = np.array([2.0, 2.0])  # Example center c
    return x - c

def three_operator_splitting(x0, prox_f, prox_g, grad_h, alpha, max_iter=1000, tol=1e-6):
    """Three-operator splitting algorithm for minimizing f + g + h."""
    x = x0
    history = [x0]
    z = prox_g(x,alpha)
    u = np.zeros_like(x)
    for _ in range(max_iter):
        x = prox_f(z - alpha * (u+grad_h(x)), alpha)
        z = prox_g(x + alpha * u, alpha)
        u += (x - z) / alpha
        history.append(x)

        # Check convergence
        if np.linalg.norm(x - history[-2]) < tol:
            break

    return x, np.array(history)

# Example usage
x0 = np.array([0.0, 0.0])  # Initial point
alpha = 0.1  # Step size

solution, iterates = three_operator_splitting(x0, prox_f, prox_g, grad_h, alpha, tol=1e-10)
print("Solution:", solution)

# %%
import matplotlib.pyplot as plt

plt.scatter(iterates[:, 0], iterates[:, 1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter plot of iterates')
plt.show()
# %%
