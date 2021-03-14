import numpy as np
import time as TIME

from Environment import Environment
from agent import Agent
from agentQ import AgentQ
import matplotlib.pyplot as plt

#simulation configuration
step = 0.25
time = 1000000
steps = time/step
numCompetitors = 5
emax = 0.05
refPriceConfig = {
    "step": step,
    "time": time,
    "drift": 0.02,
    "volatility": 0.2,
    "initValue": 20
}

#Qlearner configuration
qConfig = {
    "mu": 0.5, #exploration coefficient (%80 of time it is greedy) *change this
    "gamma": 0.4, #discount factor
    "alpha": 0.2, #learning rate
    "nudge": 0.002, # nudge constant for epsilon_bid and epsilon_ask
    "init_epsilon_bid": 0.1,
    "init_epsilon_ask": 0.1,
    "max_inventory": 1000,
    "min_inventory": 1000,
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

    #spreads for competitors and Qlearner
    bid = np.zeros(numCompetitors+1)
    ask = np.zeros(numCompetitors+1)


    #get bids/asks from market maker competitors
    for i in range(numCompetitors):
        bid[i], ask[i] = agents[i].quote(price, buyOrder, sellOrder)

    #get bid/ask from qlearner (with bid/ask from last timestep)

    competitorSpread = {
        "bid":env.states[-1]["tightestSpread"]["bid"],
        "ask":env.states[-1]["tightestSpread"]["ask"],
    }
    bid[-1], ask[-1] = Qagent.quote(price,competitorSpread)# add qagent bid and ask to end of bid and ask arrays
    #profit calc for learner
    sellWinner = ask.argmin()
    buyWinner = bid.argmax()
    #cap inventory at max, agent cannot buy if it will max out
    if(Qagent.inventory[-1] + buyOrder > qConfig["max_inventory"]):
        buyWinner = bid[:-1].argmax()
    #cap inventory at min, agent cannot buy if it will drop below
    if(Qagent.inventory[-1] - sellOrder < -qConfig["min_inventory"]):
        sellWinner = ask[:-1].argmin()

    #profit calculations for each agent
    for i in range(numCompetitors):
            agents[i].settle(sellOrder, bid[buyWinner], buyWinner, buyOrder, ask[sellWinner], sellWinner)
    Qagent.settle(sellOrder, bid[buyWinner], buyWinner, buyOrder, ask[sellWinner], sellWinner, price, lastPrice)

    #prevent QL agent from being a part of tightest spread
    env.updateState({
        "tightestSpread": {"bid": max(bid[:-1]), "ask": min(ask[:-1])},
        "refPrice": price
        })

    env.updateCurrentTimeStep()

    #finish once simulation time is reached
    if(currentTimeStep > steps -1):
        done = True
        print("Simulation Complete")
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
    plt.plot([i[0] for i in Qagent.spreadRatios])
    plt.ylabel('epsilon')
    plt.xlabel('Timestep')
    plt.title('QL Bid Epsilon')
    plt.grid(True)
    plt.figure(3)
    plt.plot([i[1] for i in Qagent.spreadRatios])
    plt.ylabel('epsilon')
    plt.xlabel('Timestep')
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

    #plot Qlearner inventory
    plt.figure(numCompetitors+3)
    plt.plot(Qagent.inventory)
    plt.ylabel('inventory')
    plt.xlabel('Timestep')
    plt.title('QL Agent inventory')
    plt.grid(True)


    plt.show()



plotResults()
