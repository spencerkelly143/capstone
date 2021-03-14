import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

#double check this is working right
#seems to go down when drift is 0

#change
def gmbrownian(step, time, drift, volatility, initValue):
    iter = int(time/step)
    price = np.zeros(iter)
    dist=np.random.standard_normal(size=iter)
    t = np.linspace(0, time, iter)
    br = np.cumsum(dist)*sqrt(step)
    price = initValue + (drift-0.5*(volatility**2))*t + volatility*br
    print("Ref Price array:")
    print(price)    
    return price
    