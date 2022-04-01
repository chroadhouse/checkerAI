#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 08:58:34 2022

@author: charlieroadhouse
"""

#Data logger needs to store the moves of the last game with the state
#Just needs 



class DataLogger:
    
    def logMatch(self, matchData):
        w = open('matchData.text','w')
        
        for i in range(0,len(matchData),3):
            #for every 3 we should have the state of the board
            #The positiion of the one of the agents
            #And the position of the tohter agent
            pass
        #I want to log the state of the board, and i want to log the move that was made 
        #Number of pieces on the board ? 
        #Called at the end of the game
        
        