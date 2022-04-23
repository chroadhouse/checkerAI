#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:06:26 2022

@author: charlieroadhouse
"""

#Agent is inheriting from the data - this is the for the random agent

from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Tuple

from seoulai_gym.envs.checkers.base import Constants
from seoulai_gym.envs.checkers.rules import Rules
from seoulai_gym.envs.checkers.utils import generate_random_move
from seoulai_gym.envs.checkers.utils import board_list2numpy
from seoulai_gym.envs.checkers.agents import Agent
from MonteCarloSearchTreeFile import MCTS
from seoulai_gym.envs.checkers.board import Board


class MyRandomAgent(Agent):
    def __init__(
            self,
            ptype: int,
    ):
        
        if ptype == Constants().DARK:
            name = "MyRandomAgentDark"
        elif ptype == Constants().LIGHT:
            name = "MyRandomAgentLight"
        else:
            raise ValueError
    
        super().__init__(name, ptype)
    
    #The act algorithm 
    #Can probably get rid of the board variable and just have the board class being passed
    def act(self,gameBoard: Board,  n,move,playfromRandom) -> Tuple[int, int, int, int]:
        """
            input: state of the board
        """
        from_row, from_col, to_row, to_col = generate_random_move(
            gameBoard.board_list,
            self.ptype,
            len(gameBoard.board_list),
        )
        #test.bestAction(10)
        #print("Type is {0}".format(self.ptype))
        return from_row, from_col, to_row, to_col

    def consume(
        self,
        obs: List[List],
        reward: float,
        done: bool,
    ) -> None:
        """Agent processes information returned by environment based on agent's latest action.
        Random agent does not need `reward` or `done` variables, but this method is called anyway
        when used with other agents.

        Args:
            board: information about positions of pieces.
            reward: reward for perfomed step.
            done: information about end of game.
        """
        pass

class MCTSAgent(Agent):
    def __init__(
            self,
            ptype: int,
    ):
        
        if ptype == Constants().DARK:
            name = "MCTSAgentDark"
        elif ptype == Constants().LIGHT:
            name = "MCTSAgentLight"
        else:
            raise ValueError
    
        super().__init__(name, ptype)
    
    #The act algorithm 
    #Can probably get rid of the board variable and just have the board class being passed
    def act(self,gameBoard: Board,  n,move,playfromRandom) -> Tuple[int, int, int, int]:
        """
            input: state of the board
        """
        from_row, from_col, to_row, to_col = generate_random_move(
            gameBoard.board_list,
            self.ptype,
            len(gameBoard.board_list),
        )
        if(move >playfromRandom):
            testing50 = MCTS(gameBoard, self.ptype)
            testingConsequence50 = MCTS(gameBoard, self.ptype)
            testing300 = MCTS(gameBoard, self.ptype)
            testingConsequence300 = MCTS(gameBoard, self.ptype)
            #Pass in number of rollout stop number 0
            #No conseqeunce 50
            node = testing50.bestAction(n,False,50)
           # print(f'Action Chosen: {nodeTemp.actionPlayed}')
            #node = testingConsequence50.bestAction(n,True,50)
            #print(f"Action Chosen: {node.actionPlayed}")
           # node = testing300.bestAction(n, False, 300)
           # print(f"Action Chosen: {node300.actionPlayed}")
            #consequence 500
            #nodeConsequence300 = testingConsequence300.bestAction(n,True,300)
            #print(f'Action Chosen: {nodeConsequence300.actionPlayed}')
            
            if node != None:
                print(node.actionPlayed)
                return node.actionPlayed[0][0], node.actionPlayed[0][1], node.actionPlayed[1][0], node.actionPlayed[1][1]
        return from_row, from_col, to_row, to_col

    def consume(
        self,
        obs: List[List],
        reward: float,
        done: bool,
    ) -> None:
        """Agent processes information returned by environment based on agent's latest action.
        Random agent does not need `reward` or `done` variables, but this method is called anyway
        when used with other agents.
        
        Have a cancel on the after so many steps to stop it from runnnig forever 

        Args:
            board: information about positions of pieces.
            reward: reward for perfomed step.
            done: information about end of game.
        """
        pass

class MCTSAgentDark(MCTSAgent):
    def __init__(
            self,
    ):
        super().__init__(Constants().DARK)


class MCTSAgentLight(MCTSAgent):
    def __init__(
            self,
    ):
        super().__init__(Constants().LIGHT)
                    
    
    
class KeyboardAgent(Agent):
    def __init__(
        self,
        ptype: int,        
    ):
        if ptype == Constants().DARK:
            name = "MyAgentDark"
        elif ptype == Constants().LIGHT:
            name = "MyAgentLight"
        else:
            raise ValueError
         
        super().__init__(name, ptype)
    
    def act(self,board: List[List],n,move,playFromRandom, gameBoard: Board) -> Tuple[int, int, int, int]:
        """
        Keyboard input - Take the input X and Y 
        
        Output the possible move
        
        """
        from_row, from_col, to_row, to_col = generate_random_move(
            gameBoard.board_list,
            self.ptype,
            len(gameBoard.board_list),
        )
        if(move > playFromRandom):
            print(board_list2numpy(gameBoard.board_list))
            start = []
            end = []
            while True:
                #Makes sure the move you enter is a valid move 
                start = [int(pos) for pos in input("Enter start posistion (e.g x,y): ").split(",")]
                end = [int(pos) for pos in input("Enter end posistion (e.g x,y): ").split(",")]
                if(Rules.validate_move(board, start[0], start[1], end[0], end[1])):
                   break;
            print("Enter a valid move")
            from_row = start[0]
            from_col = start[1]
            to_row = end[0]
            to_col = end[1]
        return from_row, from_col, to_row, to_col
    
    def consume(
        self,
        obs: List[List],
        reward: float,
        done: bool,
    ) -> None:
        pass

class MyKeyboardAgentLight(KeyboardAgent):
    def __init__(
        self,        
    ):
        super().__init__(Constants().LIGHT)

class MyKeyboardAgentDark(KeyboardAgent):
    def __init__(
        self,
    ):
        super().__init__(Constants().DARK)


class MyRandomAgentLight(MyRandomAgent):
    def __init__(
        self,
    ):
        super().__init__(Constants().LIGHT)
        

class MyRandomAgentDark(MyRandomAgent):
    def __init__(
        self,
    ):
        super().__init__(Constants().DARK)
