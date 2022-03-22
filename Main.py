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
import seoulai_gym as gym
import time
from seoulai_gym.envs.checkers.agents import RandomAgentLight 
from MyAgentFile import *
from seoulai_gym.envs.checkers.base import Constants
from seoulai_gym.envs.checkers.agents import RandomAgentDark
from seoulai_gym.envs.checkers.utils import board_list2numpy

env = gym.make("Checkers")
#Lets you decided what you want to do. 
gameChoice =0
while True:
    gameChoice = input("1 to play: 2 to watch: ")
    if(int(gameChoice)==1 or int(gameChoice)==2):
        break
    
if(int(gameChoice)==1):
    agent_two = MyKeyboardAgentLight()
elif(int(gameChoice)==2):
    agent_two = MyRandomAgentLight()


agent_one = MyRandomAgentDark()
observation = env.reset()

current_agent = agent_two
next_agent = agent_one


while True:
    time.sleep(1)
    env.render()
    #Agent gets the action from the current enviroment
    from_row, from_col, to_row, to_col = current_agent.act(observation)
    #
    observation, reward, done, info = env.step(current_agent, from_row, from_col, to_row, to_col)
    print(f"Current agent is - {current_agent}: Info is {info}")
    print(f"Reward for this move is {reward}")
    current_agent.consume(observation, reward, done)
    
    print(board_list2numpy(observation))
    
    if done:
        print(f"Game over! {current_agent} agent wins.")
        observation = env.reset()
        #Can track the number of games here
        break;
    
    #Change around to whcih one is currently moving
    temp_agent = current_agent
    current_agent = next_agent
    next_agent = temp_agent
    
env.close()


#Try using a keyboard agent with a random agent 
