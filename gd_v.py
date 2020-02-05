import numpy as np
from numpy.linalg import inv

'''
T_elements = 11
x = np.linspace(-10, 10, T_elements)

X = np.vstack((np.ones(T_elements), x))
'''

"""
Funtions: 

    np.random.randn: 
        Create an array of the given shape and populate it with random 
        samples from a uniform distribution over [0, 1).

    np.linalg.inv: 
        Compute the (multiplicative) inverse of a matrix.

    .dot:
        Dot product of two arrays.

    .T:
        The transposed array.

    Learning rate - Alpha: 
        Size of steps took in any direction.
    
    Cost Function:
        Cost of the model.
    
    Gradients:
        The direction of the steps.
    
    m:
        Number of observations
"""


# Linear data with random numbers Gaussian noise

bias = 1

X = 2 * np.random.randn(100, bias)
Y = 4 + 3 * X + np.random.rand(100, 1)

m = len(X)

theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)


def cost(x, y, theta):
    '''
    Calculates the cost for given X and Y. 

    theta:
        Vector.
    x:
        Row vector 
    y: 
        Vector
    '''

    m = len(y)

    pred = x.dot(theta)                                 # h(theta)
    f_cost = (1 / 2 * m) * np.sum(np.square(pred - y))

    return f_cost
