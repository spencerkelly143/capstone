import numpy as np
import matplotlib.pyplot as plt

def randWalk(N, drift, volatility, initValue):

    Z=np.random.normal(drift, volatility, N)
    price = np.cumsum(Z) + initValue
    print("Ref Price array:")
    print(price)

    #prevent price from going below negative.
    price = [0.01 if val <= 0 else val for val in price]

    # t = np.linspace(0,N,N)
    # plt.plot(t,price)
    # plt.show()

    return price
