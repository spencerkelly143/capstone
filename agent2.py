from itertools import count
from decimal import Decimal
import numpy as np
import random

#Competitor 2: Persistent market maker selects a fixed εb and εa for all time steps.​
#In practice, it's code logic is very similar to type 1 competitors
class Agent2:
    _ids = count(0)

    def __init__(self, emax):
        self._id = next(self._ids)
        self.emax = emax
        self.profit = [0]
        self.inventory = [0]
        self.trades = [] #record of trade with volume at each timestep 
        #set random epsilons based on emax
        self.epsilon_bid = (random.uniform(0, self.emax))
        self.epsilon_ask = (random.uniform(0, self.emax))
        return

    def settle(self,sellOrder, bid, buyWinner, buyOrder, ask, sellWinner):
        #print(self._id)
        if self._id == buyWinner:
            self.inventory.append(self.inventory[-1] + buyOrder)
            self.profit.append(self.profit[-1] - buyOrder*bid)
            self.trades.append(buyOrder)#record trade
        if self._id == sellWinner:
            self.inventory.append(self.inventory[-1] - sellOrder)
            self.profit.append(self.profit[-1] + sellOrder*ask)
            self.trades.append(-1*sellOrder ) #record trade (negative means a sell)
        if(self._id != sellWinner and self._id != buyWinner):
            self.profit.append(self.profit[-1])
            self.trades.append(0) #record trade

    def quote(self, price, buyOrder, sellOrder):
        """
        gets the bid and ask spread,
        """
        bid, ask = self.bid_ask(price, buyOrder, sellOrder)
        return bid, ask

    def bid_ask(self, price, buyOrder, sellOrder):
        bid = round(Decimal(price*(1-self.epsilon_bid)),2)
        ask = round(Decimal(price* (1+ self.epsilon_ask)),2)
        return bid, ask

