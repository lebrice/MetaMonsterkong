__author__ = 'Batchu Vishal'
import pygame
import math
import sys
import os
import numpy as np
from .person import Person
from .onBoard import OnBoard
from .coin import Coin
from .player import Player
import numpy as np
import random

class Board(object):
    '''
    This class defines our gameboard.
    A gameboard contains everthing related to our game on it like our characters, walls, ladders, coins etc
    The generation of the level also happens in this class.
    '''

    def __init__(self, width, height, rewards, rng, _dir, mk_config, level):
        self.mk_config = mk_config
        self.__width = width
        self.__actHeight = height
        self.__height = self.__actHeight + int(self.mk_config['MapWidthInTiles'])        
        self.score = 0
        self.rng = rng
        self.rewards = rewards
        self.cycles = 0  # For the characters animation
        self.direction = 0
        self._dir = _dir
        self.level = level
        # self.playerPosition = (120, 180) #change here depending on game width and height. for map_short it was (120,150), for short2 it was (120,230), for 3 (50,100)
        # self.playerPosition = (120, 210)
        myoptions = [(18,10),(15,6),(15,13),(12,3),(12,16),(8,13),(6,3),(6,16),(3,7),(3,12)]
        myindex = random.randint(0,9)
        posy,posx = myoptions[myindex]
        #print("position=",posx,posy)
        #posx = random.randint(9,11) -- 6 start positions
        #posy = random.randint(17,18) -- 6 start positions
        self.playerStartPosition = ((int(posx)//2) * 15 + 15 / 2, (int(posy) - 2) * 15 + 15 / 2)
        # old init: self.playerStartPosition = ((int(self.mk_config['MapWidthInTiles'])//2) * 15 + 15 / 2, (int(self.mk_config['MapHeightInTiles']) - 2) * 15 + 15 / 2)
        # old start position is fixed at grid cell 18,10 (center of bottom row in grid)
        
        # map2: (200, 90), 
        # map3: (200, 90), 
        # map4: (200, 90) \
        # map5: (200, 120)
        # 6: (200, 120)
        # 7: (200, 120)
        # 8: (200, 180)
        # 9: (180, 60)
        # 10: (200, 110)
        #enemies are numbered from 11-15	
        self.IMAGES = {
            "still": pygame.image.load(os.path.join(_dir, 'assets/still.png')).convert_alpha(),
            "monster0": pygame.image.load(os.path.join(_dir, 'assets/monster0.png')).convert_alpha(),
            "princess": pygame.image.load(os.path.join(_dir, 'assets/princess.png')).convert_alpha(),
            "coin1": pygame.image.load(os.path.join(_dir, 'assets/coin1.png')).convert_alpha(),
            "coin2": pygame.image.load(os.path.join(_dir, 'assets/fire.png')).convert_alpha(),
            "wood_block": pygame.image.load(os.path.join(_dir, 'assets/wood_block.png')).convert_alpha(),
            "wood_block2": pygame.image.load(os.path.join(_dir, 'assets/wood_block2.png')).convert_alpha(),
            "wood_block3": pygame.image.load(os.path.join(_dir, 'assets/wood_block3.png')).convert_alpha(),
            "wood_block4": pygame.image.load(os.path.join(_dir, 'assets/wood_block4.png')).convert_alpha(),
            "wood_block5": pygame.image.load(os.path.join(_dir, 'assets/wood_block5.png')).convert_alpha(),
            "boundary": pygame.image.load(os.path.join(_dir, 'assets/boundary.png')).convert_alpha(),            
            "ladder": pygame.image.load(os.path.join(_dir, 'assets/ladder.png')).convert_alpha()
        }

        self.white = (255, 255, 255)

        '''
        The map is essentially an array of 30x80 in which we store what each block on our map is.
        1 represents a wall, 2 for a ladder and 3 for a coin.
        '''
        self.map = []
        # These are the arrays in which we store our instances of different
        # classes
        self.Players = []
        self.Allies = []
        self.Coins = [] #enemy #1 is destroyable
        self.Coins2 = [] #enemy #2 is not destroyable
        self.Walls = []
        self.Ladders = []
        self.Boards = []

        # Resets the above groups and initializes the game for us
        self.resetGroups()

        # Initialize the instance groups which we use to display our instances
        # on the screen
        self.playerGroup = pygame.sprite.RenderPlain(self.Players)
        self.wallGroup = pygame.sprite.RenderPlain(self.Walls)
        self.ladderGroup = pygame.sprite.RenderPlain(self.Ladders)
        self.coinGroup = pygame.sprite.RenderPlain(self.Coins)
        self.coinGroup2 = pygame.sprite.RenderPlain(self.Coins2)
        self.allyGroup = pygame.sprite.RenderPlain(self.Allies)

    def resetGroups(self):
        self.score = 0
        self.lives = 1
        self.map = []  # We will create the map again when we reset the game
        self.Players = [Player(self.IMAGES["still"], self.playerStartPosition, 15, 15)] #initial position of the player
        mapi = self.level
        # self.Allies = [Person(self.IMAGES["princess"], self.princessPositions[mapi-1], 18, 25)] #initial position of the goal i.e. princess
        princessPath = self.mk_config['MapsDir'] + '/princess' + str(mapi) + '.txt'
        pp = np.loadtxt(princessPath, dtype='int16')
        # self.Allies[0].updateWH(self.Allies[0].image, "H", 0, 25, 25)
        self.Allies = [Person(self.IMAGES["princess"], (pp[1], pp[0]), 15, 15)] #initial position of the goal i.e. princess
        # self.Allies[0].updateWH(self.Allies[0].image, "H", 0, 14, 14)
        self.Coins = []
        self.Coins2 = []
        self.Walls = []
        self.Ladders = []
        self.initializeGame(mapi)  # This initializes the game and generates our map
        self.createGroups()  # This creates the instance groups

    # Given a position and checkNo ( 1 for wall, 2 for ladder, 3 for coin) the
    # function tells us if its a valid position to place or not
    def checkMapForMatch(self, placePosition, floor, checkNo, offset):
        if floor < 1:
            return 0
        for i in range(
                0, 5):  # We will get things placed atleast 5-1 blocks away from each other
            if self.map[floor * 5 - offset][placePosition + i] == checkNo:
                return 1
            if self.map[floor * 5 - offset][placePosition - i] == checkNo:
                return 1
        return 0

    def populateMapFromFile(self, mapi):
        path = self.mk_config['MapsDir'] + '/map' + str(mapi) + '.txt'
        f = open (path, 'r')
        self.map = [list(map(int,line.split(','))) for line in f if line.strip() != "" ] #load your own custom map here
        for x in range(len(self.map)):
            for y in range(len(self.map[x])):
                if self.map[x][y] == 1:
                    # Add a wall at that position
                    self.Walls.append(
                        OnBoard(
                            self.IMAGES["wood_block"],
                            (y * 15 + 15 / 2,
                             x * 15 + 15 / 2)))
                elif self.map[x][y] == 2:
                    # Add a ladder at that position
                    self.Ladders.append(
                        OnBoard(
                            self.IMAGES["ladder"],
                            (y * 15 + 15 / 2,
                             x * 15 + 15 / 2)))
                elif self.map[x][y] == 3: 
                    # Add the enemy to our enemy list
                    self.Coins.append(
                        Coin(
                            self.IMAGES["coin1"],
                            (y * 15 + 15 / 2,
                             x * 15 + 15 / 2),
                             self._dir))
                elif self.map[x][y] == 12: 
                    # Add the enemy to our enemy list
                    self.Coins2.append(
                        Coin(
                            self.IMAGES["coin2"],
                            (y * 15 + 15 / 2,
                             x * 15 + 15 / 2),
                             self._dir)) 
                # Add a wall at that position
                elif self.map[x][y] == 4:
                    self.Walls.append(
                        OnBoard(
                            self.IMAGES["wood_block2"],
                            (y * 15 + 15 / 2,
                             x * 15 + 15 / 2)))                 
                elif self.map[x][y] == 5:
                    self.Walls.append(
                        OnBoard(
                            self.IMAGES["wood_block3"],
                            (y * 15 + 15 / 2,
                             x * 15 + 15 / 2)))
                elif self.map[x][y] == 6:
                    self.Walls.append(
                        OnBoard(
                            self.IMAGES["wood_block4"],
                            (y * 15 + 15 / 2,
                             x * 15 + 15 / 2))) 
                elif self.map[x][y] == 7:
                    self.Walls.append(
                        OnBoard(
                            self.IMAGES["wood_block5"],
                            (y * 15 + 15 / 2,
                             x * 15 + 15 / 2))) 
                elif self.map[x][y] == 9:
                    self.Walls.append(
                        OnBoard(
                            self.IMAGES["boundary"], #9 is to create boundary walls so that player doesn't cross over the game
                            (y * 15 + 15 / 2,
                             x * 15 + 15 / 2)))

    # Check if the player is on a ladder or not
    def ladderCheck(self, laddersCollidedBelow,
                    wallsCollidedBelow, wallsCollidedAbove, player):
        if laddersCollidedBelow:
            for ladder in laddersCollidedBelow:
                if ladder.getPosition()[1] >= self.Players[0].getPosition()[1] and not player.isJumping:
                    self.Players[0].onLadder = 1
                    self.Players[0].isJumping = 0
                    # Move the player down if he collides a wall above
                    # if wallsCollidedAbove:
                    #     self.Players[0].updateY(3)
        else:
            self.Players[0].onLadder = 0

    # Check for coins collided and add the appropriate score
    def coinCheck(self, coinsCollected):
        for coin in coinsCollected:
            self.score += self.rewards["positive"]
            # We also remove the coin entry from our map
            self.map[int((coin.getPosition()[1] - 15 / 2) /
                     15)][int((coin.getPosition()[0] - 15 / 2) / 15)] = 0
            # Remove the coin entry from our list
            self.Coins.remove(coin)
            # Update the coin group since we modified the coin list
            self.createGroups()

        # Check for coins collided and add the appropriate score
    def coinCheck2(self, coinsCollected2):
        for coin2 in coinsCollected2:
            #self.score += self.rewards["negative"]
            self.Players[0].setPosition(self.playerStartPosition) #player dies when reaches coin (coin is basically the enemy)
            self.lives = 0
            # Update the coin group since we modified the coin list
            self.createGroups()

    # Check if the player wins
    def checkVictory(self):
        # If you touch the princess or reach the floor with the princess you
        # win!
        
        # if self.Players[0].checkCollision(self.allyGroup): # if one cell away from princess
        if np.linalg.norm(np.asarray(self.Players[0].getPosition()) - np.asarray(self.Allies[0].getPosition())) <= 2:
            self.score += self.rewards["win"]
            # This is just the next level so we only clear the fireballs and
            # regenerate the coins
            self.lives = 0 #set lives to be zero when you win to restart the game
            #self.populateMap()
            self.createGroups()
            # print("WON")
            # self.resetGroups()

    # Redraws the entire game screen for us
    def redrawScreen(self, screen, width, height):
        screen.fill((40, 20, 0))  # Fill it with black
        # Draw all our groups on the background
        self.ladderGroup.draw(screen)
        self.playerGroup.draw(screen)
        self.coinGroup2.draw(screen)
        self.coinGroup.draw(screen)
        self.wallGroup.draw(screen)
        self.allyGroup.draw(screen)

    # Update all the groups from their corresponding lists
    def createGroups(self):
        self.playerGroup = pygame.sprite.RenderPlain(self.Players)
        self.wallGroup = pygame.sprite.RenderPlain(self.Walls)
        self.ladderGroup = pygame.sprite.RenderPlain(self.Ladders)
        self.coinGroup = pygame.sprite.RenderPlain(self.Coins)
        self.coinGroup2 = pygame.sprite.RenderPlain(self.Coins2)
        self.allyGroup = pygame.sprite.RenderPlain(self.Allies)

    '''
    Initialize the game by making the map, generating walls, generating princess chamber, generating ladders randomly,
    generating broken ladders randomly, generating holes, generating coins randomly, adding the ladders and walls to our lists
    and finally updating the groups.
    '''

    def initializeGame(self, mapi):
        self.populateMapFromFile(mapi)
        self.createGroups()
