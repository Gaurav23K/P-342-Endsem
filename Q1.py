import you_can
import math

h = 6626*10^(-37)
k = 1381*10^(-26)
c = 3*10^8

file = open('Q1_itr(Newton_Raphson).txt', 'w')


def func_a(x):
    y = (x-5) * (math.e)**x + 5           # Given equation
    return y


z = [1.8153627653382098e-16, 4.965114231761082]     #Value of roots taken from the output
i = 0
while i < 2:
    b = h*c/(k*z[i])            # Since lambda_m * T is equals to b
    print("The value of wein's constant(b) for x value {} is found to be: {}".format(z[i], b))
    i += 1

print(you_can.newton_raphson(-1.5, func_a, file))
print(you_can.newton_raphson(14, func_a, file))




""""
***********************************OUTPUT*********************************************

We have found 2 roots for the given equation :

(i) 1.8153627653382098e-16
(ii) 4.965114231761082



The value of wein's constant(b) for x value 1.8153627653382098e-16 is found to be: 5.819635364200504e+17
The value of wein's constant(b) for x value 4.965114231761082 is found to be: 21.27795828026266
Newton-Raphson method.
The root has been found to be at 1.5590025377120278e-11 and it required 6 iterations.
Newton-Raphson method.
The root has been found to be at 4.965114231761082 and it required 16 iterations.
"""





