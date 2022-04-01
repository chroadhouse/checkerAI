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
reward_map= {
    "default" : 0.0,
    "invalid_move": 0.0,
    "move_opponent_piece": 0.0,
    "remove_opponent_piece": 0.0,
    "become_king": 0.0,
    "opponent_no_pieces":1.0,
    "opponent_no_valid_move":1.0
}
env.update_rewards(reward_map)
#Note that I don't need to remap the rewards here , i can do it in the other section
#Don't have to do this here i don't think but I will anyway 
#Lets you decided what you want to do. 
gameChoice =0
while True:
    gameChoice = input("1 to play: 2 to watch Random 3 to to watch MST ")
    if(int(gameChoice)==1 or int(gameChoice)==2 or int(gameChoice)==3):
        break
    
if(int(gameChoice)==1):
    agent_two = MyKeyboardAgentLight()
elif(int(gameChoice)==2):
    agent_two = MyRandomAgentLight()
elif(int(gameChoice)==3):
    agent_two = MCTSAgentLight()
    

agent_one = MCTSAgentDark()
#agent_one = MyRandomAgentDark()
observation = env.reset()
current_agent = agent_two
next_agent = agent_one
episode_count = 0

while True:
    if episode_count == 5:
        break
    env.render()
    #Agent gets the action from the current enviroment
    from_row, from_col, to_row, to_col = current_agent.act(observation,env.board)
    #
    observation, reward, done, info = env.step(current_agent, from_row, from_col, to_row, to_col)
    print(f"Current agent is - {current_agent}: Info is {info}")
    #print(f"Reward for this move is {reward}")
    #current_agent.consume(observation, reward, done)
    
    #print(board_list2numpy(observation))
    
    if done:
        print(f"Game over! {current_agent} agent wins.")
        observation = env.reset()
        #Can track the number of games here
        episode_count += 1
        print(f"Episode - {episode_count}")
        
    
    #Change around to whcih one is currently moving
    temp_agent = current_agent
    current_agent = next_agent
    next_agent = temp_agent
    
env.close()


#Try using a keyboard agent with a random agent 
