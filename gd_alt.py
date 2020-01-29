from random import randint

alpha = 0.001           # Learning rate
m = 0                   # Cantidad de elementos
theta = randint(0, 2)   # Theta
it = 0                 # Tolerancia
max_it = 1000           # Maximo de iteraciones


def f_prim(x):  # Funcion de prueba

    return 2 * (x + 5)  #


def grad(alpha, theta, max_it, it):

    while it < max_it:

        prev = theta

        theta = theta - alpha * (f_prim(prev))**(2)

        it = it + 1

    return theta


print("El minimo local ocurre en: ", grad(alpha, theta, max_it, it))
