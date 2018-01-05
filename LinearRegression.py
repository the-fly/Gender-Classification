from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

def get_random_data(hm,variance,step,correlation = False):
    val = 0
    ys = []
    for i in range(hm) :
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if(correlation and correlation == 'pos'):
            val+=step
        if(correlation and correlation == 'neg'):
            val-=step
    xs =[i for i in range(hm)]
    return np.array(xs,dtype=np.float64),np.array(ys,dtype = np.float64)

style.use('fivethirtyeight')
xs = np.array([1,2,3,4,5,6],dtype = np.float64)
ys = np.array([5,4,6,5,6,7],dtype = np.float64)

xs,ys = get_random_data(100,5,5,'pos')

def bestFitSlope(xs,ys):
    m = ( ((mean(xs)*mean(ys)) - mean(xs*ys))/
          ((mean(xs)**2) - mean(xs**2)))
    b = mean(ys)-m*mean(xs)
    return m,b
m,b = bestFitSlope(xs,ys)
print(m,b)

def sq_error(ys_orig,ys_line):
    return  sum((ys_line-ys_orig)**2)

def r_square(ys_orig,ys_line):
    ys_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_reg = sq_error(ys_orig,ys_line)
    squared_error_mean = sq_error(ys_orig,ys_mean_line)
    return 1-(squared_error_reg/squared_error_mean)
    
regression_line = [m*x+b for x in xs]

r_squared = r_square(ys,regression_line)
print(r_squared)

plt.scatter(xs,ys)
plt.plot(xs,regression_line)
plt.show()

 
