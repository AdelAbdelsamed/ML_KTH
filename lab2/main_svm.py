import numpy as np
import random
import matplotlib.pyplot as plt
from svm import svm
from svm import rbf

# Generate data points
np.random.seed(95)
# Create Class A (Targets = 1)    
classA = np.concatenate((0.2*np.random.randn(10,2) + np.array([[1.5, 0.5]]), 0.2*np.random.randn(10, 2) + np.array([[-1.5,0.5]]) ))
# Create Class B (Targets = -1)    
classB = 0.6*np.random.randn(20 , 2) + np.array([[-0.0, -1.0]])
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
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.', label ='Class A')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.', label = 'Class B' )
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('SVM Classification: Dataset')
plt.axis('equal') # Force same scale on both axes

# Specify SVM Classifier
svm1 = svm(inputs, targets, slack = True, C =None)
#svm1.set_kernel_type('RBF', **{'sigma': 2})
svm1.set_kernel_type('Linear')
svm1.compute_P_matrix()
svm1.solve_svm()

sup_vec = svm1.get_sup_vec()
print(svm1.get_nz_alpha())
sup_targets = svm1.get_sup_targets()

# Plot the decision boundary
xgrid = np.linspace(-6, 6)
ygrid = np.linspace(-6, 6)
X, Y = np.meshgrid(xgrid, ygrid)
# Compute the indicator function for each point on the grid
grid = np.array([svm1.indicator_fcn(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())])
grid = grid.reshape(X.shape)
plt.figure(1)
plt.grid()
plt.plot(sup_vec[:,0], sup_vec[:, 1],'k*', label = 'Support Vectors')
plt.legend()
plt.contour(X, Y, grid, [-1.0, 0.0, 1.0], colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
plt.title('SVM Decision Boundary')
plt.show()


### Plot influence of sigma on the RBF Kernel
# dist = np.linspace(-10,10)
# rbf1 = [rbf(dist[i], 1) for i in range(dist.shape[0])]
# rbf2 = [rbf(dist[i], 4) for i in range(dist.shape[0])]
# plt.figure()
# plt.plot(dist, rbf1, 'b', label = 'sigma = 1')
# plt.plot(dist, rbf2, 'g', label = 'sigma = 4')
# plt.xlabel('d12')
# plt.ylabel('RBF Kernel Output')
# plt.legend()
# plt.title('Influence of sigma on the RBF Kernel')
# plt.show()

