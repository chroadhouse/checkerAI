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
class MCTS:
    def __init__(self, state, playerType, gameBoard):
        self.root = Node(state, playerType, gameBoard) 
        
    def bestAction(self, simNumber):
        for i in range(0,simNumber):
            print(f"Sim number - {i}")
            node = self.treePolicy()
            reward = node.rollout()
            node.backpropagate(reward)
        #For now - a try and catch would work here
        
        return self.root.bestChild(c_param=0.)
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
    def __init__(self, state, playerType, gameBoard, actionPlayed = None,parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.numberVisits = 0
        self.results = defaultdict(int)
        self.ptype = playerType
        self.gameBoard = gameBoard
        self.actionPlayed = actionPlayed
        
    @property
    def untriedActions(self):
        """
            output:
                A list of actions that the node could take
        """
        if not hasattr(self, '_untriedActions'):
            #Get a list of all the actions not done by this node
            actions = Rules.generate_valid_moves(self.state, self.ptype, 8)
            #List of pairs is needed
            self._untriedActions = []
            for key in actions:
                for value in actions[key]:
                    self._untriedActions.append([key, value])
       
        
        return self._untriedActions
      
    @property
    def q(self):
        #Get the wins - the losses from the parent node
        wins = self.results[Rules.get_opponent_type(self.parent.ptype)]
        loses = self.results[-1 * Rules.get_opponent_type(self.parent.ptype)]
        #wins = self.results[self.ptype]
        #loses = self.results[-1 * self.ptype]
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
        newState = copy.deepcopy(self.gameBoard)
        #print(f"length of the action list before pop - {len(self.untriedActions)}")
        action = self.untriedActions.pop()
        #print(f"length of the action list after pop - {len(self.untriedActions)}")
        
        test, reward, done, info = newState.move(self.ptype,action[0][0],action[0][1],action[1][0],action[1][1])
        childPlayerType = Rules.get_opponent_type(self.ptype)
        opponentAction = self.getMoves(newState.board_list, Rules.get_opponent_type(self.ptype))
        #opponentAction = opponentAction.pop()
        #test, reward, done, info = newState.move(self.ptype, opponentAction[0][0], opponentAction[0][1], opponentAction[1][0], opponentAction[1][1])
        #childPlayerType = self.ptype
        child_node = Node(test, childPlayerType, newState, actionPlayed=action,parent=self)
        self.children.append(child_node)
        return child_node
        #Set the action to the first action not done
        #set the next state to the board when that action has been taken
        #create a child node passing that state
        #append the child node to the list
        #return the child node

    def terminalNode(self):
        #Check if the game is over
        #Need a way to check if the game is over
       # print(f"Number of my pieces - {len(self.gameBoard.get_positions(self.gameBoard.board_list, self.ptype, 8)) }")
        #print(f"Number of opponent pieces - {len(self.gameBoard.get_positions(self.gameBoard.board_list, self.gameBoard.get_opponent_type(self.ptype), 8))}")
        if len(self.gameBoard.get_positions(self.gameBoard.board_list, self.ptype, 8)) == 0 or len(self.gameBoard.get_positions(self.gameBoard.board_list, self.gameBoard.get_opponent_type(self.ptype), 8)) == 0 or len(Rules.generate_valid_moves(self.gameBoard.board_list, self.ptype, 8)) ==0 or len(Rules.generate_valid_moves(self.gameBoard.board_list,  Rules.get_opponent_type(self.ptype), 8)) ==0:
            print("terminal node")
            print(f"{board_list2numpy(self.gameBoard.board_list)}")
            return True
        else:
           # print("Not Terminal")
            return False
        
    
    def getMoves(self, state, playerType):
        actions = Rules.generate_valid_moves(state, playerType, 8)
        moveList = []
        #print(f"actions are {actions}")
        for key in actions:
            for value in actions[key]:
                moveList.append([key,value])
        return moveList
    
    
    def rollout(self):
        """
            Simulates the game and returns the player time that won from the simulation
        """
        rolloutState = copy.deepcopy(self.gameBoard)
        reward = 0
        done = False
        
        #At the moment only my pieces are moving
        #Top level thoughs is it uses the ptype and  retruns the ptype that won
        #PLayer type is working so the problem is not there
        currentPlayerType = self.ptype
        while not done:
            #print(f"{board_list2numpy(rolloutState.board_list)}")
            potencialMoves = self.getMoves(rolloutState.board_list, currentPlayerType)
            if len(potencialMoves)==0:
                done = True
                #inverseres the 
                currentPlayerType = Rules.get_opponent_type(currentPlayerType)
                break
            action = self.rolloutPolicy(potencialMoves)
            
            #print(f"actions number - {len(action)}")
           # print(f"before {board_list2numpy(rolloutState.board_list)}")
            test, reward, done, info = rolloutState.move(currentPlayerType, action[0][0],action[0][1],action[1][0],action[1][1])
            #print(f"after {board_list2numpy(rolloutState.board_list)}")
            if currentPlayerType == self.ptype:
                currentPlayerType = Rules.get_opponent_type(self.ptype)
            else:
                currentPlayerType = self.ptype
        #Reward needs to be changed to only look at player one 
        if currentPlayerType != self.ptype:
            #If the current player won get the inverse of the reward
            reward *= -1
        return reward
        #create a current rollout state
        #While this state is not over
        # get all possible moves from the state
        #Get an action from the rollout polocy 
        #set the current rollout state to the next state after the action
        #After the while return the game result - reward
    
        
    
    def backpropagate(self, result):
        #increment the number of vsisits 
        #incrememnt the result 
        #if the node has a parent - backpropagate to that
        self.numberVisits += 1
        self.results[result] +=1
        if self.parent:
            self.parent.backpropagate(result)

    
    def isFullyExpanded(self):
        return len(self.untriedActions)==0
        
        
    def bestChild(self, c_param=1.4):
        #Work out the UCT and return the child with the - Choose weights ? 
        choice_weight = [
            (c.q / (c.n)) + c_param * np.sqrt((2 * np.log(self.n) / (c.n)))
            for c in self.children    
        ]
        #print(f"{len(self.children)}")
        try:
            return self.children[np.argmax(choice_weight)]
        except Exception:
            return None
        
    def rolloutPolicy(self, potencialMoves):
        #print(len(potencialMoves))
        #Return a random move
        #print(f" Number of potencial moves{len(potencialMoves)}")
        return potencialMoves[np.random.randint(len(potencialMoves))]
        
#Thought process of why this isn't currently working:
    #So the child needs to be changed so that both states move? 
    #
    
        
        