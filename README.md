# checkerAI
This is the code for the MCTS learning agent written in python using Seoul AI gym

You cann find a copy of the code at: https://github.com/chroadhouse/checkerAIDiss/edit/main/README.md

To do this you must install the seoul ai gym libarary from github: https://github.com/seoulai/gym

Or install via pip
```
pip3 install seoulai-gym
```

Once install simply 

```
cd checkerAI

python3 Main.py
```

To play the game you have 3 options 
- 1: To play against the agent yourself 
- 2: watch the agent play against a random agent 
- 3: Watch the agent play against a version of itself. 


Once in you will be asked:
- How many games you want to play 
- How big the search space of the agent will be 
- Whether you want to increase the search space in the next game 
- Whether you want to play some of the game at random (up to 170 moves)


When playing the game against the opponent use the numbered board (board map.png file)below: 

![use the board map file](https://github.com/chroadhouse/checkerAIDiss/blob/main/board%20map.png)

which is also included in the in the file to make a move.

The input must be in the format of 
X,Y

for both the start and end position of the piece. If you do not enter a valid 
move then you will be asked to move again. If you try and move an opponent piece, one of your 
pieces will be moved randomly 
