#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:40:44 2022

@author: charlieroadhouse
"""

#Enviroment is imported from the seoul gym
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import seoulai_gym as gym
import datetime
from seoulai_gym.envs.checkers.agents import RandomAgentLight 
from MyAgentFile import *
from seoulai_gym.envs.checkers.base import Constants
from seoulai_gym.envs.checkers.agents import RandomAgentDark
from seoulai_gym.envs.checkers.utils import board_list2numpy
from seoulai_gym.envs.checkers.rules import Rules


env = gym.make("Checkers")
env.reset()

#Choses what the agent will play 
gameChoice =0
while True:
    gameChoice = input("1 to play: \n2 to watch Random: \n3 to to watch 2 MST: ")
    if(int(gameChoice)==1 or int(gameChoice)==2 or int(gameChoice)==3): 
        break
    
if(int(gameChoice)==1):
    agent_two = MyKeyboardAgentLight()
elif(int(gameChoice)==2):
    agent_two = MyRandomAgentLight()
elif(int(gameChoice)==3):
    agent_two = MCTSAgentLight()
#Agents are set here 
agent_one = MCTSAgentDark() 
current_agent = agent_two
next_agent = agent_one

endCount =0
n =0
increment =0

#Choses parameters for running the game
while True:
    
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


#Game data is recorded
episode_count = 0
tally = 0
winCount = []
episodeInfo = []
tallyList = []
simNumberList =[]

currentPicesList = [[0 for x in range(2)] for y in range(endCount)]
startTime = datetime.datetime.now()
while True:
    if episode_count == endCount:
        break
    env.render()
    
    if playFromRandom != 0:
        if playFromRandom == tally:
            print()
            print('Random Play now over')
            print(f'{current_agent} has {len(Rules.get_positions(env.board.board_list, current_agent.ptype, 8))} pieces on the board')
            print(f"{next_agent} has {len(Rules.get_positions(env.board.board_list, next_agent.ptype, 8))} pieces on the board")
            print()
  
    print(f'Currently in game {episode_count+1} on move {tally}')
    from_row, from_col, to_row, to_col = current_agent.act(env.board,n,tally,playFromRandom)
    _, _, done, info = env.step(current_agent, from_row, from_col, to_row, to_col)
    print(f"Current agent - {current_agent}: {info}")
    episodeInfo.append(f" {current_agent}: {info}")
  
    currentPicesList[episode_count][0] = len(env.board.get_positions(env.board.board_list, Constants.DARK, 8))
    currentPicesList[episode_count][1] = len(env.board.get_positions(env.board.board_list, Constants.LIGHT, 8))
    
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

    tally += 1
    # 0 for Dark 
    # 1 for light
    #Change around to whcih one is currently moving
    temp_agent = current_agent
    current_agent = next_agent
    next_agent = temp_agent
    
for i in range(0,len(winCount)):
    print(f"Sim No.{simNumberList[i]} {winCount[i]} with N steps {tallyList[i]} Dark Piece {currentPicesList[i][0]}  Light Pieces {currentPicesList[i][1]} ")
   

endTime = datetime.datetime.now()
minutes = (endTime - startTime).total_seconds()/60
print(f"This took {minutes} minutes to complete the game")


env.close()
