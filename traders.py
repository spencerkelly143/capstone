import numpy.random as npr
import numpy as np

def traderDemand(Size):
    """
    buyfl/sellfl:  generates normally distributed array of floats. multiply by 100 to
                    get values around 100.

    buy/sell:    take absolute value of each value in buyfl and modulo 101 to ensure
                all value between 0 and 100. It also makes these values ints.
                This is a rough way fo doing this that causes the values to no
                longer be normally distributed, so this may be adjusted.
    """
    buyfl = npr.randn(int(Size+1))*40        #generates normally distributed array
    buy = np.array([int(abs(x)%101) for x in buyfl])
    sellfl = npr.randn(int(Size+1))*40
    sell = np.array([int(abs(x)%101) for x in sellfl])
    demand = {
        "buy": buy,
        "sell": sell,
    }
    return demand
