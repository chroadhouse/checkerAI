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
        """State is copied and action picked from list, this is then perfomed and the new state is 
        created, this is then added to the child list and returned 
        """
        newState = copy.deepcopy(self.state)
        action = self.untriedActions.pop()
        newState.move(self.ptype,action[0][0],action[0][1],action[1][0],action[1][1])
        childPlayerType = Rules.get_opponent_type(self.ptype)
        child_node = Node(newState, childPlayerType, actionPlayed=action,parent=self)
        self.children.append(child_node)
        return child_node


    def terminalNode(self):
        """Checks if the node is termnial by looking to see if there are any pieces on the board
        and whether to see if the pieces on the board can move
        """
        if len(self.state.get_positions(self.state.board_list, self.ptype, 8)) == 0 or len(self.state.get_positions(self.state.board_list, Rules.get_opponent_type(self.ptype), 8)) == 0 or len(Rules.generate_valid_moves(self.state.board_list, self.ptype, 8)) ==0 or len(Rules.generate_valid_moves(self.state.board_list,  Rules.get_opponent_type(self.ptype), 8)) ==0:
            return True
        else:
            return False
        
    def getMoves(self, state, playerType):
        """Moves are extracted from the 3rd party libraray and uses list comprehension to 
        extract the data and returns this in a lists
        
        Args:
            state: Board class that represents the game board state
            playerType: an integer value that represents the playertype 
        """
        actions = Rules.generate_valid_moves(state, playerType, 8)
        moveList = [[key,value] for key in actions for value in actions[key]]
        return moveList

    def simulation(self):
        """Rollout state is used to play a game which will run for 50 moves or will stop before that 
        if someone wins the game. The output is the player type that wins
        """
        rolloutState = copy.deepcopy(self.state)
        count =0
        currentPlayerType = self.ptype

        while True:
            potencialMoves = self.getMoves(rolloutState.board_list, currentPlayerType)
            
            #
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

    def rolloutPolicy(self, potencialMoves):
        """Randomly chooses a move out of the options
        
        Args:
            potencialMoves: a list of moves that have being generated
        """
        n_moves = len(potencialMoves)
        return potencialMoves[np.random.randint(n_moves)]
        
        
    
    def backpropagate(self, simulationResult):
        """Backpropagates up the tree incrementing the number of visits and 
        the wins for that player
        
        Args:
            simulationResult: the player type that won the simulation
        """
        self.numberVisits += 1
        self.results[simulationResult] +=1
        if self.parent:
            self.parent.backpropagate(simulationResult)

    
    def isFullyExpanded(self):
        """Retuns a boolean value dependant on whether the node has more
        actions that can be created into nodes
        """
        return len(self.untriedActions)==0
        
    
    def UCT(self, c=1.6):
        """Weights for the different children of the node are calcualted using 
        the UCT algorithm. The max of this is then used to return the child with the best score
        
        Args:
            c: constant value that can be changed but has a default of 1.6
        """
        uctScore = [ (child.q() / (child.n())) + c * np.sqrt((2 * np.log(self.n()) / (child.n()))) for child in self.children]
        try:
            index = np.argmax(uctScore)
            return self.children[index]
        except Exception:
            return None
    
    #This method is here for testing purposes only - Shows the scores as well as the estimated wins and number of visits
    def UCTShow(self, c=1.6):
        """Weights for the different children of the node are calcualted using 
        the UCT algorithm. The values are printed for testing
        Args:
            c: constant value that can be changed but has a default of 1.6
        """
        uctScore = [ (child.q() / (child.n())) + c * np.sqrt((2 * np.log(self.n()) / (child.n()))) for child in self.children]
        q_list = [child.q() for child in self.children]
        n_list = [child.n() for child in self.children]
        temp = [child.actionPlayed for child in self.children]
        print()
        for i in range(0,len(uctScore)):
            print(f'Action - {temp[i]} -- Weight: {uctScore[i]} -- Q Score: {q_list[i]} -- No. Visited: {n_list[i]}')
        
        try:
            index = np.argmax(uctScore)
            print(f'Action picked was - {self.children[index].actionPlayed}')
            return self.children[index]
        except Exception:
            return None
    


        