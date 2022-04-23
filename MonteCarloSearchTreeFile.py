#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:13:15 2022

@author: charlieroadhouse
"""
from math import sqrt, log, e
import numpy as np
from collections import defaultdict
import copy
from seoulai_gym.envs.checkers.rules import Rules
from seoulai_gym.envs.checkers.board import Board
from seoulai_gym.envs.checkers.utils import board_list2numpy
import time
"""
    Notes:
    Upper Confidence Bound (UCB)
"""
#Remap the rewars of the game board - that can be done 
#High level thoughts - when it comes to selecting the move i want the one that comes back as a dictionary 
#Need to remap the rewards that are shown

#Need to adjust the player types for -1 and 1 
class MCTS:
    def __init__(self, state, playerType):
        self.root = Node(state, playerType) 
        
    def bestAction(self, simNumber, consequence, rolloutStop):
        #print(f'The input player type - {self.root.ptype}')
        #print(f'The opposite player type - {Rules.get_opponent_type(self.root.ptype)}')
        for i in range(0,simNumber):
            #print(i)
            node = self.treePolicy()
            #print(f'Current node type is - {node.ptype}')
            reward = node.rollout(node.state, consequence, rolloutStop)
            node.backpropagate(reward)
        if consequence:
            print(f'With consequence: {rolloutStop}')
        else:
            print(f'without consequence: {rolloutStop}')
        return self.root.bestChildFinal()
        #return self.root.bestChild(c_param=0.)
        #Look for the number of sim numbers
            #get the node from tree polocy 
            #reward is set from the rollout
            #Then back propagate 
        #Then will have the best action
        
    
    def treePolicy(self):
        #current node is set to root
        #while the current node is not terminal 
            #if the current node isn't expanded
                #Expand the current node
            #else
                #Current node is set to the best child 
        #Retyurn the current node
        count =0
        current_node = self.root
        while not current_node.terminalNode():
            count +=1
            #print("Current nodes state")
            #print(f"{board_list2numpy(current_node.state.board_list)}")
            if not current_node.isFullyExpanded():
                return current_node.expand()
            else:
                current_node = current_node.bestChild()
        return current_node
        

class Node():
    """
        Node stores specific information
        state of the board
        any children 
        any parent nodes
        -Number of times the parent node has been visited 
        -Number of times the child nonde has been visited
        -the average reward/value of all the nodes beneath this nodre
    """
    def __init__(self, state, playerType, depth=0,  actionPlayed = None,parent=None):
        self.state = state
        self.depth = depth
        self.parent = parent
        self.children = []
        self.numberVisits = 0
        self.results = defaultdict(int)
        self.ptype = playerType
        self.actionPlayed = actionPlayed
        
    @property
    def untriedActions(self):
        """
            output:
                A list of actions that the node could take
        """
        if not hasattr(self, '_untriedActions'):
            #Get a list of all the actions not done by this node
            actions = Rules.generate_valid_moves(self.state.board_list, self.ptype, 8)
            #List of pairs is needed
            self._untriedActions = [[key,value] for key in actions for value in actions[key]]
            #for key in actions:
            #    for value in actions[key]:
            #        self._untriedActions.append([key, value])
            
        return self._untriedActions
      
    @property
    def q(self):
        #Original ones
        loses = self.results[Rules.get_opponent_type(self.parent.ptype)]
        wins = self.results[self.parent.ptype]
        
    
        #wins = self.results[Rules.get_opponent_type(self.parent.ptype)]
        #loses = self.results[self.parent.ptype]
        #print(f"Loses normal - {loses} type - {Rules.get_opponent_type(self.parent.ptype)}")
        #print(f"Wins normal- {wins} type - {Rules.get_opponent_type(Rules.get_opponent_type(self.parent.ptype))}")
        return wins - loses
        
    
    @property
    def n(self):
        #Get the number of visits
        return self.numberVisits
    
        
    def expand(self):
        """
            output: 
                new child node from the expanded action
            action is popped from stack and a new state is created
        """
        newState = copy.deepcopy(self.state)
        action = self.untriedActions.pop()
        test, reward, done, info = newState.move(self.ptype,action[0][0],action[0][1],action[1][0],action[1][1])
        childPlayerType = Rules.get_opponent_type(self.ptype)
        #print(f'child type - {childPlayerType}')
        #print(f'child of child type - {Rules.get_opponent_type(childPlayerType)}')
        child_node = Node(newState, childPlayerType, depth=self.depth+1, actionPlayed=action,parent=self)
        #print(f'child node type = {child_node.ptype}')
        self.children.append(child_node)
        return child_node
        #Set the action to the first action not done
        #set the next state to the board when that action has been taken
        #create a child node passing that state
        #append the child node to the list
        #return the child node

    def terminalNode(self):
        #print("Terminal node call ")
        #Check if the game is over
        #Need a way to check if the game is over
       # print(f"Number of my pieces - {len(self.gameBoard.get_positions(self.gameBoard.board_list, self.ptype, 8)) }")
        #print(f"Number of opponent pieces - {len(self.gameBoard.get_positions(self.gameBoard.board_list, self.gameBoard.get_opponent_type(self.ptype), 8))}")
        if len(self.state.get_positions(self.state.board_list, self.ptype, 8)) == 0 or len(self.state.get_positions(self.state.board_list, Rules.get_opponent_type(self.ptype), 8)) == 0 or len(Rules.generate_valid_moves(self.state.board_list, self.ptype, 8)) ==0 or len(Rules.generate_valid_moves(self.state.board_list,  Rules.get_opponent_type(self.ptype), 8)) ==0:
            #print('Is a terminal node')
            #print(f"{board_list2numpy(self.state.board_list)}")
            return True
        else:
            #print("Not Terminal")
            return False
        #Working on making it all happen 
        
# =============================================================================
#     def oldgetMoves(self, state, playerType):
#         actions = Rules.generate_valid_moves(state, playerType, 8)
#         moveList = []
#         #print(f"actions are {actions}")
#         for key in actions:
#             for value in actions[key]:
#                 moveList.append([key,value])
#         return moveList
# =============================================================================
    
    def getMovesNow(self, state, playerType):
        actions = Rules.generate_valid_moves(state, playerType, 8)
        moveList = [[key,value] for key in actions for value in actions[key]]
        return moveList
    
    #Incentivise winning more
    def rollout(self, tempState, consequence, rolloutStop):
        """
            Simulates the game and returns the player time that won from the simulation
        """
        rolloutState = copy.deepcopy(self.state)
        #rolloutState = tempState
        rewardWeight = 0
        done = False
        count =0
        #print(self.depth)
        #At the moment only my pieces are moving
        #Top level thoughs is it uses the ptype and  retruns the ptype that won
        #PLayer type is working so the problem is not there
        currentPlayerType = self.ptype
        #print(f'Rollout Starting with - {currentPlayerType}')
        while True : #or count < 50:
            #print(f"Roll: {count}")
            #print(f'current player type - {currentPlayerType}')
            #print("Beginning of rollout State ")
            #print(f"{board_list2numpy(rolloutState.board_list)}")
            potencialMoves = self.getMovesNow(rolloutState.board_list, currentPlayerType)
            #Might not need this methods after all
            
            if len(potencialMoves)==0 or len(rolloutState.get_positions(rolloutState.board_list, currentPlayerType, 8))==0:
                #Currentplayer has lost then swap the player
                currentPlayerType = Rules.get_opponent_type(currentPlayerType)
                #print(f"{currentPlayerType} - wins")
                break
            if len(Rules.generate_valid_moves(rolloutState.board_list, Rules.get_opponent_type(currentPlayerType) ,8))==0 or len(rolloutState.get_positions(rolloutState.board_list, Rules.get_opponent_type(currentPlayerType), 8)) ==0:
                #Current player wins
                #print(f"{currentPlayerType} - wins")
                break
            #print("Before Action")
            #print(f"{board_list2numpy(rolloutState.board_list)}")
            #action = self.rolloutPolicy(potencialMoves,rolloutState,currentPlayerType)
            action = self.oldRolloutPolicy(potencialMoves)
            #testingAction = self.newRolloutPolicy(potencialMoves, board_list2numpy(rolloutState.board_list), currentPlayerType)
            #break
            test, reward, done, info = rolloutState.move(currentPlayerType, action[0][0],action[0][1],action[1][0],action[1][1])
            
            #print(f'Player - {currentPlayerType}: {info}')
            if done:
                #print('Rollout ending in normal way')
                break
            #print(f"after {board_list2numpy(rolloutState.board_list)}")
            
            currentPlayerType = Rules.get_opponent_type(currentPlayerType)
            count += 1
            if count > rolloutStop:
                #print('Count is over 300')
                #if(count > 600):
                break
                
        #Reward needs to be changed to only look at player one 
        #maybe the current player type should be the one that looks to see if they have won and just return that
        #If the current player won get the inverse of the reward
        
# =============================================================================
#         if count < 50:
#             #print('Under 50')
#             rewardWeight = 0.5
#             if count < 30:
#                 rewardWeight = 0.7
#                 if count < 10:
#                     rewardWeight = 0.9
#                     if count < 3:
#                         rewardWeight = 1.1(c.q / (c.n)) + c_param * np.sqrt((2 * np.log(self.n) / (c.n)))
#         
# =============================================================================
# =============================================================================
#         if currentPlayerType != self.ptype:
#             reward = -1
#         else:
#             reward = 1
#     
#     
#     
#         if count <=50:
#             print('under 50')
#             rewardWeight = 0.5
#             if count <=5:
#                 print('Under 5')
#                 rewardWeight = 0.7
#         
# =============================================================================
        # 1 is dark - 2 is light ?
# =============================================================================
#         print(currentPlayerType)        
#         if currentPlayerType == 2:
#             return -1, rewardWeight
#         else:
#             return 1, rewardWeight
# =============================================================================
        #print(f"The end of the rollout state{board_list2numpy(rolloutState.board_list)}")
        if consequence:
            if count > 50:
                currentPlayerType = Rules.get_opponent_type(self.ptype)
        return currentPlayerType
        #return reward, 0#rewardWeight
        #create a current rollout state
        #While this state is not over
        # get all possible moves from the state
        #Get an action from the rollout polocy 
        #set the current rollout state to the next state after the action
        #After the while return the game result - reward
    
        
    
    def backpropagate(self, result):
        #increment the number of vsisits 
        #print(f' result is - {result}')
        #incrememnt the result 
        #if the node has a parent - backpropagate to that
        self.numberVisits += 1
        self.results[result] +=1
        if self.parent:
            self.parent.backpropagate(result)

    
    def isFullyExpanded(self):
        #print(f'Expanding is - {len(self.untriedActions)==0}') 
        #print(f'Untried actions = {self.untriedActions}')
        return len(self.untriedActions)==0
        
    
    def bestChild(self, c_param=1.4):
        #Work out the UCT and return the child with the - Choose weights ?
        choice_weight = [
            (c.q / (c.n)) + c_param * np.sqrt((2 * np.log(self.n) / (c.n)))
            for c in self.children    
        ]
        try:
            temp = np.argmax(choice_weight)
            return self.children[temp]
        except Exception:
            return None
    
    def bestChildFinal(self, c_param=1.4):
        #Work out the UCT and return the child with the - Choose weights ?
        choice_weight = [
            (c.q / (c.n)) + c_param * np.sqrt((2 * np.log(self.n) / (c.n)))
            for c in self.children    
        ]
        q_list = [c.q for c in self.children]
        n_list = [c.n for c in self.children]
        temp = [c.actionPlayed for c in self.children]
        print(choice_weight)
        print('*******************************************************************')
        for i in range(0,len(choice_weight)):
            print(f'Action - {temp[i]} -- Weight: {choice_weight[i]} -- Q Score: {q_list[i]} -- No. Visited: {n_list[i]}')
        print(f'Total visits - {self.n}')
        try:
            temp = np.argmax(choice_weight)
            print(temp)
            return self.children[temp]
        except Exception:
            return None
    
    #White box testing where you look at the code as it runs thorught a game to see the expeceted algorithm 
    def oldRolloutPolicy(self, potencialMoves):
        return potencialMoves[np.random.randint(len(potencialMoves))]
    
# =============================================================================
#     def rolloutPolicy(self, potencialMoves, rolloutState, currentPlayerType):
#         #print(len(potencialMoves))
#         rewardList = []
#         for move in potencialMoves:
#             tempState = copy.deepcopy(rolloutState)
#             action = move
#             test, reward, done, info = tempState.move(currentPlayerType, action[0][0],action[0][1],action[1][0],action[1][1])
#             rewardList.append(reward)
#         index = np.argmax(rewardList)
#         #print(f"index is  {index}")
#         #print(f" Number of potencial moves{len(potencialMoves)}")
#         #return potencialMoves[np.random.randint(len(potencialMoves))]
#         return potencialMoves[index]
#     
# 
#     def newRolloutPolicy(self, potencialMoves, rolloutState, currentPlayerType):
#         if currentPlayerType == 2: #Current player is dark 
#             #Row needs to be the row that will be there at the end 
#             endRow = 7 
#             endColList = [0,2,4,6]
#             #[0][0] From row 
#             #[0][1] From col 
#             #[1][0] To Row
#             #[1][1] To col 
#             for move in potencialMoves:
#                 reward = self.rewardFromMove(move, endRow, endColList, rolloutState, True)
#         else:
#             endRow = 0
#             endColList = [1,3,5,7]
#             for move in potencialMoves:
#                 startPoint = move[0]
#                 endPoint = move[1]
#                 print(f"Start point is {startPoint}")
#                 print(f"End Point is {endPoint}")
#                 
#         
# #Thought process of why this isn't currently working:
#     #So the child needs to be changed so that both states move? 
#     #
#     
#     def rewardFromMove(move, endRow, endColList, rolloutState, darkPiece):
#         darkPiece = 10
#         darkKing = 11
#         lightPiece = 20 
#         lightKing = 21
#         
#         startPoint = move[0]
#         endPoint = move[1]
#         #Look like i need to look at the idea of why the Q value is not doing what it should before i move onto to look at other things 
#         #Doe this i should probably think about what can be done 
#         #Check if it's a king
#         
#         return 0
#         
# =============================================================================
        
        