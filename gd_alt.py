from random import randint

alpha = 0.001           # Learning rate
m = 0                   # Cantidad de elementos
theta = randint(0, 2)
tol = 0
max_it = 1000


def f_prim(x):

    return 2 * (x + 5)  #


def grad(alpha, theta, max_it, tol):

    while tol < max_it:

        prev = theta

        theta = theta - alpha * (f_prim(prev))**(2)

        tol = tol + 1

    return theta


print("El minimo local ocurre en: ", grad(alpha, theta, max_it, tol))
