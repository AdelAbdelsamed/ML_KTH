import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def compute_P_matrix(data, targets, kernel_type = 'Linear'):
    """
    Args:
        data(list)          : Training Data.
        targets(ndarray)    : Target values of the training set.
        kernel_type (string): String indicating the type of the Kernel
            Current Kernel Types are 'Linear', 'RBF'
        ...
    """
    if data.shape[0] != np.size(targets):
        print('Size of data and targets must match!')
        exit()

    P = np.ndarray(shape=(N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = targets[i]*targets[j]*kernel(data[i], data[j], kernel_type)

    return P


def kernel(x1, x2, type = 'Linear', **kwargs):
    """
    Args:
        x1(ndarray)  : First data point.
        x2(ndarray)  : Second data point.
        type (string): String indicating the type of the Kernel
            Current Kernel Types are 'Linear', 'RBF'
        ...
    """
    types = ['Linear', 'RBF'] # List of all defined Kernel Types

    # Check if Kernel Type is valid
    if type not in 'Linear':
        print('Entered Kernel Type is Unknown!')
        exit()
    # Dimensions check
    if np.size(x1) != np.size(x2):
        print('Data points are not the same size!')
        exit()

    if type == "Linear":
        kappa = np.dot(x1, x2)
    elif type == 'Polynomial':
        p = kwargs['order'] 
        kappa = (np.dot(x1, x2) + 1)**p
    elif type == 'RBF':
        sigma = kwargs['sigma'] 
        kappa = np.exp(-0.5*np.dot(x1 - x2, x1 - x2)/(sigma**2))

    return kappa


def objective(alpha):
    """
    Args:
        alpha(ndarray): Lagrange Multiplier.
        ...
    """
    if np.size(alpha) != N:
        print('Dimensions of alpha must be consistent with No. of data points!')

    return 0.5 *alpha.T @ P @ alpha - np.sum(alpha)

def zerofun(alpha):
    """
    Args:
        alpha(ndarray)   : Lagrange Multiplier.
        ...
    """
    if np.size(alpha) != np.size(targets):
        print('Length of Lagrange Multiplier alpha and number of targets must match!')
        exit()

    return np.dot(alpha, targets)

def compute_threshold(nz_alpha, support_vec, support_targets, s, t_s, kernel_type):
    """
    Args:
        nz_alpha(ndarray)          : Lagrange Multiplier.
        support_vec(list)          : Support vectors.
        support_targets(ndarray)   : Support targets.
        s(ndarray)                 : Data point on the margin.
        t_s(scalar)                 : Classifier for s.
        ...
    """    
    return sum(nz_alpha[i] * support_targets[i] * kernel(s, support_vec[i], kernel_type) for i in range(nz_alpha.shape[0])) - t_s

def indicator_fcn(nz_alpha, support_vec, support_targets, s, b, kernel_type):
    """
    Args:
        nz_alpha(ndarray)          : Lagrange Multiplier.
        support_vec(ndarray)       : Support vectors.
        support_targets(ndarray)   : Support targets.
        s(ndarray)                 : New data point.
        b(ndarray)                 : Threshold value.
        ...
    """
    if np.size(nz_alpha) != np.size(support_targets):
        print('Non-zero Lagrange Multiplier alpha must be the same length as support targets!')
        exit()
    elif np.size(nz_alpha) != support_vec.shape[0]:
        print('Non-zero Lagrange Multiplier alpha must be the same length as no. of support vectors!')
        exit()
    elif not np.isscalar(b):
        print('Threshold value must be a scalar!')
        exit()

    return sum(nz_alpha[i] * support_targets[i] * kernel(s, support_vec[i,:], kernel_type) for i in range(support_vec.shape[0])) - b

# Generate random seed
#np.random.seed(100)

# Create Class A (Targets = 1)    
#classA = np.concatenate((0.2*np.random.randn(10,2) + np.array([[1.5, 0.5]]), 0.2+np.random.randn(10, 2) + np.array([[-1.5, 0.5]]) ))
classA = np.concatenate((0.2*np.random.randn(10,2) + np.array([[2.5, 2.5]]), 0.2+np.random.randn(10, 2) + np.array([[1.5, 1]]) ))
# Create Class B (Targets = -1)    
classB = 0.2*np.random.randn(20 , 2) + np.array([[0.0, -0.5]])
# Create Data point
inputs = np.concatenate((classA, classB))
targets = np.concatenate ((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
N = inputs.shape[0] # Number of rows (samples)
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]


# Plot the data sets 
plt.figure(1)
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.' )
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.' )
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('SVM Classification: Dataset')
plt.axis('equal') # Force same scale on both axes
plt.savefig('svmplot.pdf') # Save a copy in a file
#plt.show( ) # Show the plot on the screen

P = compute_P_matrix(inputs, targets, 'Linear')

# Solve the Lagrangian Dual Problem
start = np.zeros(N) # Initial guess
#C = 1 # Upper bound on alpha
B = [(0, None) for b in range(N)] # Set the bounds for the limits an alpha
XC = {'type': 'eq', 'fun':zerofun} # Set the equality constraints
ret = minimize(objective, start, bounds=B, constraints= XC)
if not ret['success']:
    print('Fail: Data is not linearly seperable! Optimization failed!')
    exit()
else:
    print('Success: Optimal alpha found!')
    alpha_opt = ret['x']

# Define support vectors 
sup_vec_idx = [idx for idx, value in enumerate(alpha_opt) if value > 1e-5]
nz_alpha = alpha_opt[sup_vec_idx] 
sup_vec = inputs[sup_vec_idx, :] 
sup_targets = targets[sup_vec_idx]

# Obtain the threshold
b = compute_threshold(nz_alpha, sup_vec, sup_targets, sup_vec[0], sup_targets[0], 'Linear')

# Plot the decision boundary
xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)
X, Y = np.meshgrid(xgrid, ygrid)
# Compute the indicator function for each point on the grid
grid = np.array([indicator_fcn(nz_alpha, sup_vec, sup_targets, np.array([x, y]), b, 'Linear') for x, y in zip(X.ravel(), Y.ravel())])
grid = grid.reshape(X.shape)

plt.figure(1)
plt.contour(X, Y, grid, [-1.0, 0.0, 1.0], colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
plt.title('SVM Decision Boundary')
plt.show()






    


