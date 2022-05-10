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
    
    def act(self,gameBoard: Board,  n,move,playfromRandom) -> Tuple[int, int, int, int]:
        """Agent takes board and decides what move should be played on the board, this move is chosen randomly
        Args:
            gameBoard: Board class - represents the state of the of the enviroment
            n: number of search space 
            move: what move the game is on
            playfromRandom: the value that the agent plays randomly
        """
        from_row, from_col, to_row, to_col = generate_random_move(
            gameBoard.board_list,
            self.ptype,
            len(gameBoard.board_list),
        )
        return from_row, from_col, to_row, to_col
    
    #This method is not used but is needed for the agent to work
    def consume(
        self,
        obs: List[List],
        reward: float,
        done: bool,
    ) -> None:

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
    
    def act(self,gameBoard,  n,move,playfromRandom) -> Tuple[int, int, int, int]:
        """Agent takes board and and uses the MCTS algorithm to pick an action that is returned and 
        played on the board
        
        Args:
            gameBoard: Board class - represents the state of the of the enviroment
            n: number of search space 
            move: what move the game is on
            playfromRandom: the value that the agent plays randomly
        """
        
        from_row, from_col, to_row, to_col = generate_random_move(
            gameBoard.board_list,
            self.ptype,
            len(gameBoard.board_list),
        )
        
        if(move >playfromRandom):
            mcts = MCTS(gameBoard, self.ptype)        
            node = mcts.search(n)
            if node != None:
                return node.actionPlayed[0][0], node.actionPlayed[0][1], node.actionPlayed[1][0], node.actionPlayed[1][1]
        return from_row, from_col, to_row, to_col
    #This method is not used but is needed for the agent to work
    def consume(
        self,
        obs: List[List],
        reward: float,
        done: bool,
    ) -> None:
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
    
    def act(self,gameBoard,n,move,playFromRandom) -> Tuple[int, int, int, int]:
        """Agent takes board and and the player makes the move 
        Args:
            gameBoard: Board class - represents the state of the of the enviroment
            n: number of search space 
            move: what move the game is on
            playfromRandom: the value that the agent plays randomly
        """
        from_row, from_col, to_row, to_col = generate_random_move(
            gameBoard.board_list,
            self.ptype,
            len(gameBoard.board_list),
        )
        
        #Board is printed to show the user
        if(move >= playFromRandom):
            print(board_list2numpy(gameBoard.board_list))
            start = []
            end = []
            while True:
                try:
                    start = [int(pos) for pos in input("Enter start posistion (e.g x,y): ").split(",")]
                    end = [int(pos) for pos in input("Enter end posistion (e.g x,y): ").split(",")]
                    if(Rules.validate_move(gameBoard.board_list, start[0], start[1], end[0], end[1])):
                        break
                    else:
                        print('Enter a valid move')
                    
                except Exception:
                    print('Must be X,Y')
               
            from_row = start[0]
            from_col = start[1]
            to_row = end[0]
            to_col = end[1]
        return from_row, from_col, to_row, to_col
    
    #Method is not used
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
