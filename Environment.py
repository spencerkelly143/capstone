import numpy as np
import random
from traders import traderDemand

from brownian import gmbrownian


class Environment:
    def __init__(self,refPriceConfig):
        #init environment
        print("Create Environment")
        print(refPriceConfig)

        self.states = []
        # tightest bid/ask from last timestep, current ref price,
        self.refPriceConfig = refPriceConfig
        #init ref prices over time (ref price is independent of all agent actions and environment state.)
        self.refPrices = gmbrownian(
            refPriceConfig["step"],
            refPriceConfig["time"],
            refPriceConfig["drift"],
            refPriceConfig["volatility"],
            refPriceConfig["initValue"]
        )
        #init demand over time (currently simplified to random normally distributed demand)
        self.demand = traderDemand(refPriceConfig["time"]/refPriceConfig["step"])
        self.t = 0 #starting timestep

        #todo, keep track of all trades in environment
        return

    def getConfig(self):
        return self.refPriceConfig
    def getRefPrices(self):
        return self.refPrices
    def getCurrentRefPrice(self):
        if (self.t < len(self.refPrices)-1 ):
            return self.refPrices[self.t]
        return self.refPrices[-1]
    def getLastRefPrice(self):
        if (self.t < len(self.refPrices)-1 ):
            return self.refPrices[self.t-1]
        return self.refPrices[-1]
    def getDemand(self):
        #format: demand{"buy": [...],"sell": [...]}
        return self.demand
    def getCurrentTimeStep(self):
        return self.t
    def updateCurrentTimeStep(self):
        self.t = self.t+1
        return self.t
    def updateState(self, newstate):
        self.states.append(newstate)


    def getStates(self):
        return self.states
    def getCurrentState(self):
        if not self.states:
            return []
        return self.states[-1]
