#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:13:15 2022

@author: charlieroadhouse
"""
#from math import sqrt, log
import numpy as np
from collections import defaultdict
import copy
from seoulai_gym.envs.checkers.rules import Rules

class MCTS:
    def __init__(self, state, playerType):
        self.root = Node(state, playerType) 
        
    def search(self, n):
        """Runs the 4 phases of the MCTS algorithm and at the end it picks the best action
        
        Args:
            n: The number of iterations - which is equivelent to the search space
        """
        for i in range(0,n):
            node = self.treePolicy()
            reward = node.simulation()
            node.backpropagate(reward)
        return self.root.UCTShow()
        
    
    def treePolicy(self):
        current_node = self.root
        while not current_node.terminalNode():
            if not current_node.isFullyExpanded():
                return current_node.expand()
            else:
                current_node = current_node.UCT()
        return current_node
        

class Node():
    def __init__(self, state, playerType, actionPlayed = None,parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.numberVisits = 0
        self.results = defaultdict(int)
        self.ptype = playerType
        self.actionPlayed = actionPlayed
        
    @property
    def untriedActions(self):
        if not hasattr(self, '_untriedActions'):
            
            #actions = Rules.generate_valid_moves(self.state.board_list, self.ptype, 8)
            #self._untriedActions = [[key,value] for key in actions for value in actions[key]]
            self._untriedActions = self.getMoves(self.state.board_list, self.ptype)
        return self._untriedActions
      
    def q(self):
        return self.results[self.parent.ptype] - self.results[self.ptype]
        
    def n(self):
        return self.numberVisits
    
        
    def expand(self):
        newState = copy.deepcopy(self.state)
        action = self.untriedActions.pop()
        newState.move(self.ptype,action[0][0],action[0][1],action[1][0],action[1][1])
        childPlayerType = Rules.get_opponent_type(self.ptype)
        child_node = Node(newState, childPlayerType, actionPlayed=action,parent=self)
        self.children.append(child_node)
        return child_node


    def terminalNode(self):
        if len(self.state.get_positions(self.state.board_list, self.ptype, 8)) == 0 or len(self.state.get_positions(self.state.board_list, Rules.get_opponent_type(self.ptype), 8)) == 0 or len(Rules.generate_valid_moves(self.state.board_list, self.ptype, 8)) ==0 or len(Rules.generate_valid_moves(self.state.board_list,  Rules.get_opponent_type(self.ptype), 8)) ==0:
            return True
        else:
            return False
        
    def getMoves(self, state, playerType):
        actions = Rules.generate_valid_moves(state, playerType, 8)
        moveList = [[key,value] for key in actions for value in actions[key]]
        return moveList

    def simulation(self):
        rolloutState = copy.deepcopy(self.state)
        count =0
        currentPlayerType = self.ptype

        while True:
            potencialMoves = self.getMoves(rolloutState.board_list, currentPlayerType)
            
            if len(potencialMoves)==0 or len(rolloutState.get_positions(rolloutState.board_list, currentPlayerType, 8))==0:
                currentPlayerType = Rules.get_opponent_type(currentPlayerType)
                break
            
            if len(Rules.generate_valid_moves(rolloutState.board_list, Rules.get_opponent_type(currentPlayerType) ,8))==0 or len(rolloutState.get_positions(rolloutState.board_list, Rules.get_opponent_type(currentPlayerType), 8)) ==0:
                break

            action = self.rolloutPolicy(potencialMoves)
            _, _, done, info = rolloutState.move(currentPlayerType, action[0][0],action[0][1],action[1][0],action[1][1])
            if done:
                break

            currentPlayerType = Rules.get_opponent_type(currentPlayerType)
            count += 1
            if count > 50:
                break
            
        if count > 50:
            currentPlayerType = Rules.get_opponent_type(self.ptype)
        return currentPlayerType

    
        
    
    def backpropagate(self, simulationResult):
        self.numberVisits += 1
        self.results[simulationResult] +=1
        if self.parent:
            self.parent.backpropagate(simulationResult)

    
    def isFullyExpanded(self):
        return len(self.untriedActions)==0
        
    
    def UCT(self, c_param=1.6):
        uctScore = [ (c.q() / (c.n())) + c_param * np.sqrt((2 * np.log(self.n()) / (c.n()))) for c in self.children]
        try:
            index = np.argmax(uctScore)
            return self.children[index]
        except Exception:
            return None
    
    #This method is here for testing purposes only - Shows the scores as well as the estimated wins and number of visits
    def UCTShow(self, c_param=1.6):
        uctScore = [ (c.q() / (c.n())) + c_param * np.sqrt((2 * np.log(self.n()) / (c.n()))) for c in self.children]
        q_list = [c.q() for c in self.children]
        n_list = [c.n() for c in self.children]
        temp = [c.actionPlayed for c in self.children]
        print()
        for i in range(0,len(uctScore)):
            print(f'Action - {temp[i]} -- Weight: {uctScore[i]} -- Q Score: {q_list[i]} -- No. Visited: {n_list[i]}')
        
        try:
            index = np.argmax(uctScore)
            print(f'Action picked was - {self.children[index].actionPlayed}')
            return self.children[index]
        except Exception:
            return None
    
    def rolloutPolicy(self, potencialMoves):
        n_moves = len(potencialMoves)
        return potencialMoves[np.random.randint(n_moves)]
    

        