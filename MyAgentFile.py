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
from seoulai_gym.envs.checkers.agents import Agent

class MyAgent(Agent):
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
    
    #The act algorithm 
    def act(self,board: List[List],) -> Tuple[int, int, int, int]:
        """
        MonteCarlo search method can be called in here 
        
        best action method
        back progagation
        best child
        simulate
        
        """
        rand_from_row, rand_from_col, rand_to_row, rand_to_col = generate_random_move(
            board,
            self.ptype,
            len(board),
        )
        
        print("Start Row:{0} Start Col:{1}".format(rand_from_row, rand_from_col))
        print("End Row:{0} End Col:{1}".format(rand_to_row, rand_to_col))
        print("----------------------")
        return rand_from_row, rand_from_col, rand_to_row, rand_to_col

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
    
    def act(self,board: List[List],) -> Tuple[int, int, int, int]:
        """
        Keyboard input - Take the input X and Y 
        
        Output the possible move
        
        """
        start = []
        end = []
        while True:
            
        #Check whether you have entered two numbers
            start = [int(pos) for pos in input("Enter start posistion (e.g x,y): ").split(",")]
            end = [int(pos) for pos in input("Enter end posistion (e.g x,y): ").split(",")]
            if(Rules.validate_move(board, start[0], start[1], end[0], end[1])):
               break;
                         
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


class MyRandomAgentLight(MyAgent):
    def __init__(
        self,
    ):
        super().__init__(Constants().LIGHT)
        

class MyRandomAgentDark(MyAgent):
    def __init__(
        self,
    ):
        super().__init__(Constants().DARK)
#Pseudo code for the MCST:
    #4 Steps: Select, Expand, Simulate and backup
    
    
    
    
    
    
    