from random import randint

alpha = 0.001      # Learning rate
m = 0  # Cantidad de elementos
theta = randint(0, 2)
tol = 0


def f(x):
    pass


def grad(alpha, theta, max_it, tol):

    while tol < max_it:

        prev = theta

        theta = theta - alpha * f(x)

        tol += tol

    return theta


"""
def gradient(alpha, theta, max):

    q_i = q_i - (alpha*(1/m)*sum(q_0 + q_1*x_1**(i)-y**(i))) / \
        (1/m*sum(q_0+q_1*x_1**(i)-y**(i)*x_1**(i)))
    pass
"""
