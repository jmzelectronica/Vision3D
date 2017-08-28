
#Tarea dos
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import time
x = np.array([17.3,19.3,19.5,19.7,22.9,23.1,26.4,26.8,27.6,28.1,28.2,28.7,29,29.6,
29.9,29.9,30.3,31.3,36,39.5,40.4,44.3,44.6,50.4,55.9])
y = np.array([71.7,48.3,88.3,75,91.7,100,73.3,65,75,88.3,68.3,96.7,76.7,78.3,
60,71.7,85,85,88.3,100,100,100,91.7,100,71.7])

xy= x*y
x2= x**2
y2=y*y
a=sum(x)
b=sum(y)
c=sum(xy)
d=sum(x2)
e=sum(y2)
n=25
promx =a/n
promy =b/n
print a,b,c,d

b0 = ((n*c)-(a*b))/((n*d)-(a*a))
b1 = ((d*b)-(a*c))/((n*d)-(a**2))

m = (a*b-n*c)/(a**2-n*d)
b = promy - m*promx


ygorrito = b1 + (b0*30)
print b0
print b1
print ygorrito

linea=[b1,105]
#plt.scatter(x,y)
#plt.plot(linea)
plt.plot(x,y,'o', label='Datos')
plt.plot(x,m*x+b, label='ajuste')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)
plt.show()

