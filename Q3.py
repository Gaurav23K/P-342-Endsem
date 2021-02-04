import you_can
import math
red = open('esem_table.dat','r')
b=red.read()
b1 = [word.split('\t') for word in b.split('\n')]
x=[]
y=[]
for j in range(12):
    x.append(float(b1[j+2][0]))
    y.append(float(b1[j+2][-1]))
print("Equation 1")
a,b,r=you_can.lt_sq_fit(x,y)
print("The intercept",a)
print("The slope",b)
print("The pearson's coeff",r)

for l in range(len(y)):
    y[l]=math.log(y[l],math.e)
print("Equation 2")
a,b,r=you_can.lt_sq_fit(x,y)
print("The intercept",a)
print("The slope",b)
print("The pearson's coeff",r)
"""
**********************************OUTPUT***********************************************
Equation 1
The intercept 2.0291025641025655
The slope -0.4747086247086251
The pearson's coeff 0.9705318844905291
Equation 2
The intercept 0.7902775293458726
The slope -0.3955961745485569
The pearson's coeff 0.9982366554936271
"""