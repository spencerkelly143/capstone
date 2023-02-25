# Capstone
In this project, we designed an agent to learn to maximize profits in a financial markets by "market-making", that is posting simulatenous bid and offer prices. We designed the problem as a Markov Decision Process and used Q-Learning to teach our agent an optimal policy to maximize it's profits.

## Running simulation
Ensure that you have git and Python3 installed. Clone the git repo, enter the directory.
    
    git clone https://github.com/spencerkelly143/capstone.git
    cd captone 
    git pull

To run the simulation, run the simulation.py file.

    python simulation.py

If executed correctly, the program should create the simulation environment and produce graphs with the ref price and agent profits/inventory over time. The simulation can be configured in the simulation.py file.

## demand.py
Generates the market buy and sell orders. These are randomly generated and are normalized to be around
100. Returns an array of size *Size*.

## agent.py
This is the file that defines the competitor 1. We initialize the competitor with emax. That's the
max spread value percentage. We call *quote()* when we want to get bi/ask prices from the agent.
Currently I am using five agents.

## agent2.py
This is the file that defines the competitor 2. It has all the same properties and functions of type 1 competitors, but it's bid and ask epsilons are constant.


## brownian.py
This file is deprecated, use randWalk.py instead.

This file returns a list of reference prices. Considers the initial value, volatility,
and drift. You can also specify the time and step values. The total number of iterations
is calculated time divided by step. For example, if you wanted prices every hour for 5 days,
you could do *time = 5 x 24* and *step = 1*.

## environment.py
Currently set up as main market environment for the simulation. Acts as the environment object for keeping track of states and ref price.


## simulation.py
This is the new main file where the simulation is executed. Running this file executes initializes the environment.py, generates random walk ref prices based on confguration, creates agents, and runs the market simulation. 

## agentQ.py
This is the main Q Learning agent in this simulation. it has the same quote() and settle() functions as the competitor agents, but its quoting behavior is determined by it's Q matrix. It will pick bid and ask epsilons based on it's current state and its Q-matrix. This Q matrix is updated each timestep based on the reward in the last timestep in the settle() function. 

## Q-Matrix format
The q matrix is shaped as a 4 dimentional tensor (8,10,10,9). The first 3 indices are the state of the QLearner/market. They are binned based on the ranges shown below to increase computational efficiency. The 4th index represents the action space. 

    inventory(8): {<-900, -900 - -100, -100 - -50, -50 - 0, 0 - 50, 50 - 100, 100 - 900, >900}
    bid ratio(10): {<-0.2, 
                -0.2:-0.15, 
                -0.15:-0.1,
                -0.1:-0.05,
                -0.05:0,
                0:0.05,
                0.05:0.1,
                0.1:0.15,
                0.15:0.2,
                >0.2,
                }
    ask ratio(10): {<-0.2, 
                -0.2:-0.15, 
                -0.15:-0.1,
                -0.1:-0.05,
                -0.05:0,
                0:0.05,
                0.05:0.1,
                0.1:0.15,
                0.15:0.2,
                >0.2,
                }
    actions(9):{
        "increaseBid",
        "decreaseBid", 
        "increaseAsk", 
        "decreaseAsk",
        "increaseBid & increaseAsk"
        "increaseBid & decreaseAsk"
        "decreaseBid & increaseAsk"
        "decreaseBid & decreaseAsk"
        "do nothing"
        }

## Issues
* A mix of python dicts and small arrays are used throughout the project to pass multiple values, this should be standardized to just one or the other


## Tasks (not started)
* create GUI
* Generalize our agent so that it can use any RL algorithm (Q-learning, etc.)
* make trader demand function more realistic 

## Tasks (started)
* double check inventory min/max logic for all agents

## tasks (done)
* plot ref price
* track trades
* Visualize trades over time, plot 
* Create more detailed simulation diagram
* update Q table with bellman
* Fix bug with q learner not being able to buy and sell at once
* get Q-Learning agent working in simple terms
* get Qlearning agent working more consistently
    * setup simulation to have many training episodes
    * setup testing for all methods
    * formalize all math and make sure its good
* simulation assumes all market makers can buy/sell as much as the market demands
* might need to do more actions at once
* add functionality for negative inventory
* add functionality for more than one margin increase/decrease at once ()
* check bid/ask ratio calculations and make sure it's good. (I think it's ignoring if we have the best price)
* might not need negative bid/ask ratios since we can just pick our own values as the tightest (but maybe we do since we want to have a wider margin to make more profit)
* agents might just be selling as much as possible without having inventory 
* seems like ref price can go below zero

