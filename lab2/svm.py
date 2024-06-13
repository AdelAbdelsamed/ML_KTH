import numpy as np
import math
from scipy.optimize import minimize



class svm:
    def __init__(self, data, targets, slack = False, C =1.0):
        """
        Initialize the Support Vector Machine (SVM) classifier.

        Args:
            kernel_type (str): The kernel function to use ('linear', 'poly', 'rbf', etc.).
            C (float): Regularization parameter (default is 1.0).
        """
        # Problem dependent variables
        self.kernel_type = None
        self.kernel_param = {}
        self.slack = slack
        self.C = C
        
        # Data dependent variables
        self.N = data.shape[0]
        self.data = data 
        self.targets = targets
        self.P = np.ndarray([self.N, self.N])

        # After solving the Dual Problem
        self.b = 0 # Threshold
        self.nz_alpha = np.ndarray([])
        self.sup_targets = np.ndarray([])
        self.sup_vec = np.ndarray([])

        print('SVM created successfully!')


    def set_kernel_type(self, kernel_type, **kwargs):
        """
        Set Kernel type
        Args:
            kernel_type (str): The kernel function to use ('linear', 'poly', 'rbf', etc.).
        """
        types = ['Linear', 'Polynomial', 'RBF'] # List of all defined Kernel Types
        # Check if Kernel Type is valid
        if kernel_type not in types:
            print('Entered Kernel Type is Unknown!')
            exit()
        self.kernel_type = kernel_type
        if kernel_type == 'Polynomial':
            self.kernel_param['p'] = kwargs['order'] 
        elif kernel_type == 'RBF':
            self.kernel_param['sigma'] = kwargs['sigma'] 
        print('Kernel type set to ' + kernel_type + ' successfully!')

    def kernel(self, x1, x2, **kwargs):
        """
        Compute the output of the kernel function
        Args:
            x1(ndarray)  : First data point.
            x2(ndarray)  : Second data point.
            **kwargs     : Optional parameters Linear 'order': p, RBF 'sigma': sigma
            ...
        """
        # Dimensions check
        if np.size(x1) != np.size(x2):
            print('Data points are not the same size!')
            exit()

        if self.kernel_type == "Linear":
            kappa = np.dot(x1, x2)
        elif self.kernel_type == 'Polynomial':
            kappa = (np.dot(x1, x2) + 1)**self.kernel_param['p']
        elif self.kernel_type == 'RBF':
            kappa = np.exp(-0.5*np.dot(x1 - x2, x1 - x2)/(2*self.kernel_param['sigma']**2))

        return kappa
        


    def compute_P_matrix(self):
        """
        Compute the 'P-matrix'
        Args:
            data(list)          : Training Data.
            targets(ndarray)    : Target values of the training set.
            kernel_type (string): String indicating the type of the Kernel
                Current Kernel Types are 'Linear', 'RBF'
            ...
        """
        if self.data.shape[0] != np.size(self.targets):
            print('Size of data and targets must match!')
            exit()

        P = np.ndarray(shape=(self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                P[i, j] = self.targets[i]*self.targets[j]*self.kernel(self.data[i,:], self.data[j, :])

        self.P = P
        print('P-matrix was computed successfully')
    
    def objective(self, alpha):
        """
        Args:
        alpha(ndarray): Lagrange Multiplier.
        ...
        """
        if np.size(alpha) != self.N:
            print('Dimensions of alpha must be consistent with No. of data points!')

        return 0.5 *alpha.T @ self.P @ alpha - np.sum(alpha)
    
    def zerofun(self, alpha):
        """
        Args:
            alpha(ndarray)   : Lagrange Multiplier.
            ...
        """
        if np.size(alpha) != np.size(self.targets):
            print('Length of Lagrange Multiplier alpha and number of targets must match!')
            exit()

        return np.dot(alpha, self.targets)
    
    def compute_threshold(self, s, t_s):
        """
        Compute threshold value 
        Args:
            s(ndarray)                 : Data point on the margin.
            t_s(scalar)                 : Classifier for s.
            ...
        """    
        self.b = sum(self.nz_alpha[i] * self.sup_targets[i] * self.kernel(s, self.sup_vec[i]) for i in range(self.nz_alpha.shape[0])) - t_s

    def indicator_fcn(self, s):
        """
            Indicator Function computation
        Args:
            nz_alpha(ndarray)          : Lagrange Multiplier.
            support_vec(ndarray)       : Support vectors.
            support_targets(ndarray)   : Support targets.
            s(ndarray)                 : New data point.
            b(ndarray)                 : Threshold value.
            ...
        """
        if np.size(self.nz_alpha) != np.size(self.sup_targets):
            print('Non-zero Lagrange Multiplier alpha must be the same length as support targets!')
            exit()
        elif np.size(self.nz_alpha) != self.sup_vec.shape[0]:
            print('Non-zero Lagrange Multiplier alpha must be the same length as no. of support vectors!')
            exit()
        elif not np.isscalar(self.b):
            print('Threshold value must be a scalar!')
            exit()

        return sum(self.nz_alpha[i] * self.sup_targets[i] * self.kernel(s, self.sup_vec[i,:]) for i in range(self.sup_vec.shape[0])) - self.b

    def solve_svm(self):
        
        # Check if kernel type is entered
        if self.kernel_type == None:
            print('No Kernel type entered!')
            exit()

        # Solve the Lagrangian Dual Problem
        start = np.zeros(self.N) # Initial guess
        
        # Slack activated? 
        if self.slack:
            ub = self.C
        else:
            ub = None
        
        B = [(0, ub) for b in range(self.N)] # Set the bounds for the limits an alpha
        XC = {'type': 'eq', 'fun':self.zerofun} # Set the equality constraints
        # Solve the problem
        ret = minimize(self.objective, start, bounds=B, constraints= XC)
        if not ret['success']:
            print('Optimization failed: Minimizer could not find an optimal value!')
            exit()
        else:
            print('Optimization succeeded: Optimal alpha found!')
            alpha_opt = ret['x']

        # Define support vectors 
        sup_vec_idx = [idx for idx, value in enumerate(alpha_opt) if value > 1e-5]
        self.nz_alpha = alpha_opt[sup_vec_idx] 
        self.sup_vec = self.data[sup_vec_idx, :] 
        self.sup_targets = self.targets[sup_vec_idx]

        if self.sup_vec.size == 0 :
            print('Optimization failed: No support vector found!')
            exit()

        # Obtain the threshold
        self.compute_threshold(self.sup_vec[0], self.sup_targets[0])

    def get_sup_vec(self):
        return self.sup_vec
    
    def get_sup_targets(self):
        return self.sup_targets
    
    def get_nz_alpha(self):
        return self.nz_alpha


def rbf(x1, sigma):
    return np.exp(-0.5*np.dot(x1, x1)/(2*sigma**2))
