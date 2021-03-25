import numpy as np
from decimal import Decimal
import random
import h5py

class AgentGreedy():
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
        print("load Q Table")
        h5f = h5py.File('data.h5','r')
        self.qTable = h5f['dataset_1'][:]#the agent will be re-created each episode, but the final qtable at the end of each episode should persist
        h5f.close()
        self.learningCurve = [0] #learned info at each timestep
        return
    def selectStateIndex(self,inventory,bidRatio,askRatio):
        '''
        The right way to do this is probably to make a range dictionary csv
        and then import it to this file. I looked up some other ways and got lazy.
        Python also doesnt even have built in switch statements, crazy
        '''
        #inventory
        if(inventory <= -900):
            inventoryIndex =0
        elif(inventory >-900 and inventory <=-100):
            inventoryIndex=1
        elif(inventory >-100 and inventory <=-50):
            inventoryIndex=2
        elif(inventory >-50 and inventory <=0):
            inventoryIndex=3
        elif(inventory >0 and inventory <= 50 ):
            inventoryIndex =4
        elif(inventory >50 and inventory <=100):
            inventoryIndex=5
        elif(inventory >100 and inventory <900):
            inventoryIndex=6
        else:
            inventoryIndex=7 #inventory>150
        #bid ratio
        if(bidRatio <= -0.2):
            bidIndex =0
        elif(bidRatio >-0.2 and bidRatio <=-0.15):
            bidIndex=1
        elif(bidRatio >-0.15 and bidRatio <=-0.1):
            bidIndex=2
        elif(bidRatio >-0.1 and bidRatio <=-0.05):
            bidIndex=3
        elif(bidRatio >-0.05 and bidRatio <=0):
            bidIndex=4
        elif(bidRatio >0 and bidRatio <=0.05):
            bidIndex=5
        elif(bidRatio >0.05 and bidRatio <=0.1):
            bidIndex=6
        elif(bidRatio >0.1 and bidRatio <=0.15):
            bidIndex=7
        elif(bidRatio >0.15 and bidRatio <=0.2):
            bidIndex=8
        else:
            bidIndex=9
        #ask ratio
        if(askRatio <= -0.2):
            askIndex =0
        elif(askRatio >-0.2 and askRatio <=-0.15):
            askIndex=1
        elif(askRatio >-0.15 and askRatio <=-0.1):
            askIndex=2
        elif(askRatio >-0.1 and askRatio <=-0.05):
            askIndex=3
        elif(askRatio >-0.05 and askRatio <=0):
            askIndex=4
        elif(askRatio >0 and askRatio <=0.05):
            askIndex=5
        elif(askRatio >0.05 and askRatio <=0.1):
            askIndex=6
        elif(askRatio >0.1 and askRatio <=0.15):
            askIndex=7
        elif(askRatio >0.15 and askRatio <=0.2):
            askIndex=8
        else:
            askIndex=9
        stateIndex = [inventoryIndex, bidIndex, askIndex]
        return stateIndex
    def pickAction(self,stateIndex, restrict_bid, restrict_ask):
        '''
    returns index of best action according to Q tensor or a random action based
    on epsilon.

                        action        actionIndex

                    "increaseBidep" ->       0
                    "decreaseBidep" ->       1
                    "increaseAskep" ->       2
                    "decreaseAskep" ->       3
                    "increase both" ->       4
                    "inc bid, dec ask"->     5
                    "dec bid, inc ask" ->    6
                    "decrease both" ->       7
                    "do nothing"  ->         8
        '''
        actionIndex = 8 #default to doing nothing

        #pick optimal action based on index and q values
        qOptions = self.qTable[stateIndex[0]][stateIndex[1]][stateIndex[2]]
        bothOp = [1,3,7,8]
        askOp = [0,1,3,5,7,8]
        bidOp = [1,2,3,6,7,8]
        if(restrict_ask and restrict_bid):
            newOptions = qOptions[bothOp]
            actionIndex = newOptions.argmax() #chosen action

            if(actionIndex==0): actionIndex = actionIndex+1
            elif(actionIndex==1): actionIndex = actionIndex+2
            elif(actionIndex==2): actionIndex = actionIndex+5
            elif(actionIndex==3): actionIndex = actionIndex+5
        elif(restrict_ask):
            newOptions = qOptions[askOp]
            actionIndex = newOptions.argmax() #chosen action
            if(actionIndex==2): actionIndex = actionIndex+1
            elif(actionIndex==3):actionIndex = actionIndex+2
            elif(actionIndex==4): actionIndex = actionIndex+3
            elif(actionIndex==5): actionIndex = actionIndex+3
        elif(restrict_bid):
            newOptions = qOptions[bidOp]
            actionIndex = newOptions.argmax() + 1#chosen action

            if(actionIndex==4): actionIndex = actionIndex+2
            elif(actionIndex==5): actionIndex = actionIndex+2
            elif(actionIndex==6): actionIndex = actionIndex+2
        else:
            actionIndex = qOptions.argmax() #chosen action
        maxActionIndex = actionIndex
        actionValue = qOptions[actionIndex]
        maxActionValue = qOptions[maxActionIndex]
        self.actions.append([actionIndex,actionValue])
        return actionIndex, actionValue, maxActionIndex, maxActionValue


    def quote(self, price, competitorSpread,emax):
        #initial spread
        if(not self.spread):
            self.spread.append([price*(1-self.qLearningConfig["init_epsilon_bid"]), price*(1+self.qLearningConfig["init_epsilon_ask"])])

        #bid/ask = last times bid ask
        oldBid = self.spread[-1][0]
        oldAsk = self.spread[-1][1]

        #observable state for bid and ask ratios
        bidRatio = (float(competitorSpread["bid"])-oldBid)/price
        askRatio = (oldAsk-float(competitorSpread["ask"]))/price

        inventory = self.inventory[-1]
        stateIndex = self.selectStateIndex(inventory,bidRatio,askRatio)
        #update with current (pre trade) state

        epsilon_bid = self.spreadRatios[-1][0]
        epsilon_ask = self.spreadRatios[-1][1]

        if(epsilon_bid>=emax):
            restrict_bid = True
        else:
            restrict_bid = False
        if(epsilon_ask>=emax):
            restrict_ask = True
        else:
            restrict_ask = False

        self.states.append(stateIndex)
        actionIndex, actionValue, maxActionIndex, maxActionValue  = self.pickAction(stateIndex, restrict_bid, restrict_ask)

        #move bid/ask based on state and Q
        nudgeConstant = self.qLearningConfig["nudge"] # 0.002

        # update epsilon_bid and epsilon_ask
        # do action based on selected actionIndex
        # increase/decrease epsilons or do nothing
        if(actionIndex ==0):
            epsilon_bid = epsilon_bid + nudgeConstant
        elif(actionIndex==1):
            epsilon_bid = epsilon_bid - nudgeConstant
        elif(actionIndex==2):
            epsilon_ask = epsilon_ask + nudgeConstant
        elif(actionIndex==3):
            epsilon_ask = epsilon_ask - nudgeConstant
        elif(actionIndex ==4):
            epsilon_bid = epsilon_bid + nudgeConstant
            epsilon_ask = epsilon_ask + nudgeConstant
        elif(actionIndex==5):
            epsilon_bid = epsilon_bid + nudgeConstant
            epsilon_ask = epsilon_ask - nudgeConstant
        elif(actionIndex==6):
            epsilon_bid = epsilon_bid - nudgeConstant
            epsilon_ask = epsilon_ask + nudgeConstant
        elif(actionIndex==7):
            epsilon_bid = epsilon_bid - nudgeConstant
            epsilon_ask = epsilon_ask - nudgeConstant
        #if action is 8 ("do nothing") then epsilons are unchanged

        #sanity check, ensure non-negative epsilons
        if(epsilon_bid<0): epsilon_bid=0
        if(epsilon_ask<0): epsilon_ask=0
        self.spreadRatios.append([epsilon_bid,epsilon_ask])#add new bid/ask eps ratios to storage

        newBid = price*(1-epsilon_bid)
        newAsk = price*(1+epsilon_ask)

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
