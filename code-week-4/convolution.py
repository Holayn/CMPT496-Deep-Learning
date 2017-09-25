#convolve
import numpy as np

h = [2,1,0]
x = [3,4,5]
 

y = np.convolve(x,h)
print(y)
