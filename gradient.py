from random import randint

alpha = 0.001      # Learning rate
m = 0  # Cantidad de elementos
x = 0  # First guess
theta = randint(0, 2)


def gradient(alpha, theta, max):

    q_i = q_i - (alpha*(1/m)*sum(q_0 + q_1*x_1**(i)-y**(i))) / \
        (1/m*sum(q_0+q_1*x_1**(i)-y**(i)*x_1**(i)))
    pass
