import you_can
import math

file1 = open("data_Q3h_(0.01).txt", "w+")
file2 = open("data_Q3l_(0.01).txt", "w+")
def fun2(x, y, z):
    return -9.8

def fun1(x, y, z):
    return z

x0 = 0
y0 = 2
xn=5
n = 6
h = 0.01
zl0= 5
zh0=60
yn=45

you_can.shooting_method(x0, y0, zh0, zl0, xn, yn, h, fun1, fun2, file1, file2)

"""
************************************OUTPUT********************************************

yn =  45
zl0 = 33.09999999999985
yh yl = 2= 179.50000000000122 44.999999999999865
zh0 = 33.09999999999988
yh yl = 2= 44.99999999999996 44.999999999999865

Our final velocity is: 33.09999999999999 m/sec
"""
