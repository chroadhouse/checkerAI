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
import datetime
from seoulai_gym.envs.checkers.agents import RandomAgentLight 
from MyAgentFile import *
from seoulai_gym.envs.checkers.base import Constants
from seoulai_gym.envs.checkers.agents import RandomAgentDark
from seoulai_gym.envs.checkers.utils import board_list2numpy
import DataLogger


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
    gameChoice = input("1 to play: \n2 to watch Random: \n3 to to watch 2 MST: ")
    if(int(gameChoice)==1 or int(gameChoice)==2 or int(gameChoice)==3): # or int(gameChoice)==4):
        break
    
if(int(gameChoice)==1):
    agent_two = MyKeyboardAgentLight()
elif(int(gameChoice)==2):
    agent_two = MyRandomAgentLight()
elif(int(gameChoice)==3):
    agent_two = MCTSAgentLight()

agent_one = MCTSAgentDark() 
#Going to make this all run and look ab

#agent_one = MyRandomAgentDark()
env.reset()
current_agent = agent_two
next_agent = agent_one
#get rid of the random option as it is not needed
endCount =0
n =0
increment =0

while True:
    #Number of games 
    #Number of cycles for the agent 
    #How much you want to increment the simulationm number
    tempEndCount = input("Please input a of games you want it to play")
    
    tempN = input('Please input a number of cycles you want ')
    tempIncrement = input("Please input a number you want to increment the cycles by")
    if(tempN.strip().isdigit() and tempIncrement.strip().isdigit() and tempEndCount.strip().isdigit()):
        if(int(tempN) > 0 and int(tempIncrement) >= 0 and int(tempEndCount) >0):
            endCount = int(tempEndCount)
            n = int(tempN)
            increment = int(tempIncrement)
            break
        else:
            print("The digit must be greater than 0")
    else:
        print("Please input a digit")
        
playFromRandom =0
while True:
    fromRandomString = input('Would you like to have the agent play random to begin: y/n')
    if fromRandomString == 'y':
        fromRandomString = input('What number of moves would you like random between 0 and 170: ')
        if(fromRandomString.strip().isdigit()):
            if(int(fromRandomString) >= 0 and int(fromRandomString) <= 170):
                playFromRandom = int(fromRandomString)
                break
    elif fromRandomString == 'n':
        playFromRandom = 0
        break
    else:
        print('Please input y/n only')
# =============================================================================
#     else:
#         if(tempEndCount.strip().isdigit()):
#             if(int(tempEndCount) > 0):
#                 endCount = int(tempEndCount)
#                 break
#             else:
#                 print('Number must be greater than 0')
#         else:
#             print('Number must be a digit')
# =============================================================================
        

    
#Parameters for running 
episode_count = 0
tally = 0
winCount = []
episodeInfo = []
tallyList = []
simNumberList =[]


#Test the functionality of the 
startTime = datetime.datetime.now()
while True:
    if episode_count == endCount:# or tally >250:
        break
    #env.render()
    print(f'Currently in game {episode_count+1} on move {tally}')
    #Agent gets the action from the current enviroment
    from_row, from_col, to_row, to_col = current_agent.act(env.board,n,tally,playFromRandom)
    #
    _, _, done, info = env.step(current_agent, from_row, from_col, to_row, to_col)
    print(f"Current agent - {current_agent}: {info}")
    #print(f"Reward for this move is {reward}")
    #current_agent.consume(observat6,ion, reward, done)
    episodeInfo.append(f" {current_agent}: {info}")
    #print(board_list2numpy(observation))
    
    if done:
        print(f"Game over! {current_agent} agent wins.")
        winCount.append(f"{current_agent} agent wins")
        env.reset()
        tallyList.append(tally)
        tally = 0
        simNumberList.append(n)
        n = n + increment
        print(f'N is noww {n}')
        #Can track the number of games here
        episode_count += 1
        #print(f"Episode - {episode_count}")
        
    # Some thoughts on what is going to be need to bsdhgh\
    tally += 1
    #Change around to whcih one is currently moving
    temp_agent = current_agent
    current_agent = next_agent
    next_agent = temp_agent
    
print('With consequence - Rollout stop 50')
for i in range(0,len(winCount)):
    print(f"Sim No.{simNumberList[i]} {winCount[i]} with N steps {tallyList[i]}")


endTime = datetime.datetime.now()
time_delta = (endTime - startTime)
total_seconds = time_delta.total_seconds()
minutes = total_seconds/60
print(f"This took {minutes} minutes to complete the game")
#Try using a keyboard agent with a random agent 


# =============================================================================
# Performance improved - now to work on selecting better choice
# =============================================================================
