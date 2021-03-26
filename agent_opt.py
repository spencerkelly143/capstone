# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:04:14 2021

@author: Willem Atack
"""


import numpy as np
from decimal import Decimal
import random

class Agent_opt():
    # Actions for nudging is broke, budge ratio instead

    #

    #states q = QLearner, c = competitor
    # [(qbid-cbid)/refprice][(qask-cask)/refprice][inventory]

    def __init__(self, configuration, initialQTable,numCompetitors):
       #config Q-learning params
        self._id = numCompetitors
        self.qLearningConfig = configuration
        self.spread = [] # 2d array -> bid,ask at timestep
        self.spreadRatios = [[configuration["init_epsilon_bid"],configuration["init_epsilon_ask"]]] # 2d array -> epsilon_bid,epsilon_ask at timestep
        self.profit=[0]
        self.inventory = [0]
        self.trades = [] # record of trade with volume at each timestep
        self.states = [] # OBSERVED States indexes (inventory,bidratio,askratio)
        self.actions = [] # 2d array -> action index, q value for that action at timestep
        self.rewards = [] # rewards array
        self.qTable = initialQTable #the agent will be re-created each episode, but the final qtable at the end of each episode should persist
        self.learningCurve = [0] #learned info at each timestep
        return
    
    


    def quote(self, price):
     
       
        newBid = price*(1-0.05)
        newAsk = price*(1+0.05)
       
        self.spread.append([newBid,newAsk])
        #return actual bid ask spread
        return self.spread[-1][0], self.spread[-1][1]
        #update new state in settle()

    def settle(self,sellOrder, bid, buyWinner, buyOrder, ask, sellWinner,price, lastPrice,emax):
        if self._id == buyWinner and self._id == sellWinner: #QL agent hold tightest bid/ask spread
            self.inventory.append(self.inventory[-1] + buyOrder - sellOrder)
            self.profit.append(self.profit[-1] + self.inventory[-2]*(price - lastPrice) + buyOrder*(price - bid) + sellOrder*(ask-price))
            self.trades.append(buyOrder - sellOrder)#record trade
        elif self._id == buyWinner:
            self.inventory.append(self.inventory[-1] + buyOrder)
            self.profit.append(self.profit[-1] + self.inventory[-2]*(price - lastPrice) + buyOrder*(price - bid))
            self.trades.append(buyOrder)#record trade
        elif self._id == sellWinner:
            self.inventory.append(self.inventory[-1] - sellOrder)
            self.profit.append(self.profit[-1] + self.inventory[-2]*(price - lastPrice) + sellOrder*(ask-price))
            self.trades.append(-1*sellOrder ) #record trade (negative means a sell)
        if(self._id != sellWinner and self._id != buyWinner):
            self.inventory.append(self.inventory[-1])
            self.profit.append(self.profit[-1] + self.inventory[-2]*(price - lastPrice))
            self.trades.append(0) #record trade
        return

        

