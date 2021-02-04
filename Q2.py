import you_can
import math

def func(x):
    f = (4*math.sqrt(1/9.8)*1/math.sqrt(1 - (math.sin(math.pi/8))**2*(math.sin(x))**2))
    return f
N = 10
S1 = you_can.simpson(0, math.pi/2, func, N)
print("The T value has been found to be :", S1[0], "with an error of : ",S1[1])

""""
************************************OUTPUT*******************************************

The T value has been found to be : 2.087320017479594, with an error of :  0.7078127084107334

"""

