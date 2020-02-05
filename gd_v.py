import numpy as np
from numpy.linalg import inv

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


def cost(x, y, theta):
    '''
    Calculates the cost for given X and Y. 

    theta:
        Vector.
    x:
        Row vector 
    y: 
        Vector

    pred: 
        # h(theta), Nos da todas las hipotesis
    '''

    m = len(y)

    pred = x.dot(theta)
    f_cost = (1 / 2 * m) * np.sum(np.square(pred - y))

    return f_cost


def gradient_descent(x, y, theta, alpha, max_it):
    '''
    Returns the final theta vector.

    alpha: 
        Learning rate
    X:
        Matrix of X with added bias.
        dim :
            m,2
    Y: 
        Vector of Y

    theta:
        dim: 
            m,1 
    '''

    m = len(y)

    cost_history = np.zeros(max_it)
    #alm_theta = np.zeros((max_it, 2))

    for i in range(max_it):

        pred = np.dot(x, theta)

        theta = theta - alpha * ((1 / m) * (x.T.dot((pred - y))))
        #alm_theta[i, :] = theta.T
        cost_history[i] = cost(x, y, theta)

    return theta

# Linear data with random numbers Gaussian noise


bias = 1

X = 2 * np.random.randn(100, bias)
Y = 4 + 3 * X + np.random.rand(100, 1)

# Analytical way of Linear Regression
#theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

alpha = 0.01
max_it = 1000

X_vStack = np.c_[np.ones((len(X), 1)), X]

f, c = X_vStack.shape

theta = np.random.rand(c, 1)

print("Theta: ", gradient_descent(X_vStack, Y, theta, alpha, max_it))

'''
T_elements = 11
x = np.linspace(-10, 10, T_elements)

X = np.vstack((np.ones(T_elements), x))
'''
