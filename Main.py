#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:40:44 2022

@author: charlieroadhouse
"""

#Enviroment is imported from the seoul gym
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Thoughts fo the coursework 
#Will have to create agent in a seperate class - with methods operations
#I assume you train the agent against itself ? - **Ask Ryan** 


import seoulai_gym as gym
import time
from seoulai_gym.envs.checkers.agents import RandomAgentLight 
import MyRandomAgentLight as testingAgent
from seoulai_gym.envs.checkers.base import Constants
from seoulai_gym.envs.checkers.agents import RandomAgentDark

env = gym.make("Checkers")


agent_one = RandomAgentDark()
agent_two = testingAgent()

observation = env.reset()


current_agent = agent_one
next_agent = agent_two

while True:
    time.sleep(1)
    env.render()
    from_row, from_col, to_row, to_col = current_agent.act(observation)
    observation, reward, done, info = env.step(current_agent, from_row, from_col, to_row, to_col)
    current_agent.consume(observation, reward, done)
    
    if done:
        print(f"Game over! {current_agent} agent wins.")
        observation = env.reset()
        break;
    
    #Change around to whcih one is currently moving
    temp_agent = current_agent
    current_agent = next_agent
    next_agent = temp_agent
    
env.close()
        