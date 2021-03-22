import numpy as np
import time as TIME

from Environment import Environment
from agent import Agent
from agent2 import Agent2
from agentQ import AgentQ
import matplotlib.pyplot as plt

#simulation configuration
step = 0.25
time = 10000
steps = time/step
numCompetitors = 5
emax = 0.3 # updated to be more realistic/useful
refPriceConfig = {
    "step": step,
    "time": time,
    "drift": 0, #Test at 0 to be closer to math model
    "volatility": 0.01,
    "initValue": 20
}

#Qlearner configuration
qConfig = {
    "mu": 0.8, #exploration coefficient (%80 of time it is greedy) *change this
    "gamma": 0.999, #discount factor (should be ~1 due to high number of timesteps)
    "alpha": 0.2, #learning rate
    "nudge": 0.002, # nudge constant for epsilon_bid and epsilon_ask
    "init_epsilon_bid": 0.1,
    "init_epsilon_ask": 0.1,
    "max_inventory": 1000,
    "min_inventory": -1000,
}

#create environment
env = Environment(refPriceConfig)

#set initial tightest spread to simply be $1 outside initial ref price
#This is not the actual tighest spread, but the QL agent wont have access to the actual spread initialy anyway
env.updateState({
        "tightestSpread": {"bid": refPriceConfig["initValue"]-1, "ask": refPriceConfig["initValue"]+1},
        "refPrice": refPriceConfig["initValue"]
        })

#create and shape random Q tensor [inventory][bid][ask][actions]
qTable = np.random.rand(7200) #array of random floats between 0-1
qTable = np.reshape(qTable, (8,10,10,9))
#create Q learning agent
Qagent = AgentQ(qConfig,qTable,numCompetitors)
#create competitor agents
agents = [Agent(emax) for i in range(numCompetitors)]

#Run Simulation
start_time = TIME.time()

done = False
while(not done):
    #get current state variables
    currentTimeStep = env.getCurrentTimeStep()
    price = env.getCurrentRefPrice()
    lastPrice = env.getLastRefPrice()
    buyOrder = env.getDemand()["buy"][currentTimeStep]
    sellOrder = env.getDemand()["sell"][currentTimeStep]
    minInv =  qConfig["min_inventory"]
    maxInv =  qConfig["max_inventory"]

    #determine buy and sell winners
    #default to impossibly bad bid ask spread with winner being a non-agent (id 10)
    bids = [[-1,10]]
    asks = [[99999,10]]
    for i in range(numCompetitors):
        # check if it can make the buy
        # if yes, then add to bids array with [[bidprice, agentid]] 
        # check if it can make the sell 
        # if yes, then add to asks array with [[askprice, agentid]] 
        bid, ask = agents[i].quote(price, buyOrder, sellOrder)
        if(agents[i].inventory[-1] + buyOrder <= maxInv):
            bids.append([bid,i])
        if(agents[i].inventory[-1] - sellOrder >= minInv):
            asks.append([ask,i])
        
    #get bid/ask from qlearner (with bid/ask from last timestep)
    competitorSpread = {
        "bid":env.states[-1]["tightestSpread"]["bid"],
        "ask":env.states[-1]["tightestSpread"]["ask"],
    }
    qbid, qask = Qagent.quote(price,competitorSpread)
    if(Qagent.inventory[-1] + buyOrder <= maxInv):
        bids.append([qbid,numCompetitors])
    if(Qagent.inventory[-1] - sellOrder >= minInv):
        asks.append([qask,numCompetitors])

    #sort bids/asks to be ascending by first col (value)
    bids.sort(key = lambda x: x[0])
    asks.sort(key = lambda x: x[0])

    #pick buy/sell winner
    bestBid, buyWinner = bids[-1]
    bestAsk, sellWinner = asks[0]
    #profit calculations for each agent
    for i in range(numCompetitors):
            agents[i].settle(sellOrder, bestBid, buyWinner, buyOrder, bestAsk, sellWinner)
    Qagent.settle(sellOrder, bestBid, buyWinner, buyOrder, bestAsk, sellWinner, price, lastPrice)

    #prevent QL agent from being a part of tightest spread
    env.updateState({
        "tightestSpread": {"bid": bestBid, "ask": bestAsk},
        "refPrice": price
        })

    env.updateCurrentTimeStep()

    #finish once simulation time is reached
    if(currentTimeStep > steps -1):
        done = True
        print("Simulation Complete")
        print("Total Timesteps: " + str(steps))
        print("Execution Time: - %s seconds -" % (TIME.time() - start_time))

#plot results
def plotResults():
    plt.figure(0, figsize=(18, 10))

    #plot ref price over time
    plt.subplot(121)
    plt.plot(env.refPrices)
    plt.grid(True)
    plt.xlabel('Timestep')
    plt.ylabel('Reference Price ($)')
    plt.title('Reference Price')

    #plot competitor agent performance
    plt.subplot(122)
    for agent in agents:
        plt.plot(agent.profit)  #plot the agents
    plt.plot(Qagent.profit)
    plt.legend(["0","1","2","3","4","Q"])
    plt.ylabel('Profit ($)')
    plt.xlabel('Timestep')
    plt.title('Agent Profit over time')
    plt.grid(True)

    #plot agent trades over time
    # for i in range (len(agents)):
    #     plt.figure(i+1)
    #     plt.plot(agents[i].trades)
    #     plt.ylabel('Volume')
    #     plt.xlabel('Timestep')
    #     plt.title('Agent '+ str(agents[i]._id) + ' trade activity')
    #     plt.grid(True)

   
    plt.figure(2)
    plt.hist([i[0] for i in Qagent.spreadRatios], density = True, bins = 30)
    plt.ylabel('Probability')
    plt.xlabel('Bid Epsilon')
    plt.title('QL Bid Epsilon')
    plt.grid(True)
    

    plt.figure(3)
    plt.hist([i[1] for i in Qagent.spreadRatios], density = True, bins = 30)
    plt.ylabel('Probability')
    plt.xlabel('Ask Epsilon')
    plt.title('QL ask Epsilon')
    plt.grid(True)
    
    #plot Qlearner
    plt.figure(numCompetitors+1)
    plt.plot(Qagent.rewards)
    plt.ylabel('Volume')
    plt.xlabel('Timestep')
    plt.title('QL rewards activity')
    plt.grid(True)

    #plot Qlearner learning curve
    # plt.figure(numCompetitors+2)
    # plt.plot(Qagent.learningCurve)
    # plt.ylabel('Learned amount')
    # plt.xlabel('Timestep')
    # plt.title('QL Agent Learning Curve')
    # plt.grid(True)

    #plot agent inventories
    plt.figure(numCompetitors+2)
    plt.plot(Qagent.inventory)
    plt.ylabel('inventory')
    plt.xlabel('Timestep')
    plt.title('Agent inventory')
    plt.grid(True)


    plt.show()



plotResults()
profitTotal = 0
for i in range(0,int(steps)):
    profitTotal = profitTotal + Qagent.profit[i]*qConfig["gamma"]**i
     
print("Total Discounted Profit: ", profitTotal)
