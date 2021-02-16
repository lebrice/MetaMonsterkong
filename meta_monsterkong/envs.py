__author__ = 'Richard'
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import PIL
import pygame
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
from ple.games.base.pygamewrapper import PyGameWrapper
from pygame.constants import K_SPACE, KEYDOWN, QUIT, K_a, K_d, K_s, K_w

from .board import Board

source_dir = Path(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class MkConfig:
    MapsDir : str = f"{source_dir}/maps"
    MapHeightInTiles: int = 20
    MapWidthInTiles: int = 20
    IsRender: bool = True
    SingleID: bool = None
    DenseRewardCoeff: float = 0.0
    RewardsWin: float = 50.0
    StartLevel: int = 0
    NumLevels: int = 1
    TextureFixed: bool = True  # False
    Mode: str = "Test"
    
    def __getitem__(self, key: str):
        return getattr(self, key)



class MonsterKong_Base(PyGameWrapper):

    def __init__(self, mk_config: Union[MkConfig, Dict]):
        if isinstance(mk_config, dict):
            mk_config = MkConfig(**mk_config)
        self.mk_config: MkConfig = mk_config
        self.reward_type = None
        self.height = self.mk_config.MapHeightInTiles * 15 #modify height accordingly based on how long the game level is , for map_short it was 210*210, for short2 it was 270*270, for short3 it is 170*170
        self.width = self.mk_config.MapWidthInTiles * 15

        actions = {
            "left": K_a,
            "right": K_d,
            "jump": K_SPACE,
            "up": K_w,
            "down": K_s
        }

        PyGameWrapper.__init__(
            self, self.width, self.height, actions=actions)

        self.rewards = {
            "positive": 5,  # original was 5
            "win": self.mk_config.RewardsWin,
            "negative": 0,  # original was -25
            "tick": 0
        }
        self.allowed_fps = 30
        self._dir = os.path.dirname(os.path.abspath(__file__))

        self.IMAGES = {
            "right": pygame.image.load(os.path.join(self._dir, 'assets/right.png')),
            "right2": pygame.image.load(os.path.join(self._dir, 'assets/right2.png')),
            "left": pygame.image.load(os.path.join(self._dir, 'assets/left.png')),
            "left2": pygame.image.load(os.path.join(self._dir, 'assets/left2.png')),
            "still": pygame.image.load(os.path.join(self._dir, 'assets/still.png'))
        }
        if not self.mk_config.IsRender:
            os.environ["SDL_VIDEODRIVER"] = "dummy"


    def init(self):
        # Create a new instance of the Board class
        self.currentBoard = Board(
            self.width,
            self.height,
            self.rewards,
            self.rng,
            self._dir,
            self.mk_config)

        # Assign groups from the Board instance that was created
        self.playerGroup = self.currentBoard.playerGroup
        self.wallGroup = self.currentBoard.wallGroup
        self.ladderGroup = self.currentBoard.ladderGroup
        self.currentBoard.redrawScreen(self.screen, self.width, self.height)


    def getScore(self):
        return self.currentBoard.score

    def game_over(self):
        if (self.currentBoard.lives <= 0):
            return 1
        else:
            return 0

    def step(self, dt):
        # Check if you have reached the princess
        self.currentBoard.checkVictory()

        keyPress = None
        def dist2princess(board):
            princessPos= np.asarray(board.Allies[0].getPosition())
            playerPos = np.asarray(board.Players[0].getPosition())
            return np.linalg.norm(princessPos-playerPos)

        self.currentBoard.score += self.rewards["tick"]
        assert self.reward_type is not None
        if self.reward_type == 'dense':
            self.currentBoard.score += self.mk_config.DenseRewardCoeff * dist2princess(self.currentBoard) # Dense reward, euclidean distance to princess
            assert self.mk_config.DenseRewardCoeff is not None
        # This is where the actual game is run
        # Get the appropriate groups
        self.coinGroup = self.currentBoard.coinGroup
        self.coinGroup2 = self.currentBoard.coinGroup2

        before = str(self.currentBoard.Players[0].getPosition())
        # To check collisions below, we move the player downwards then check
        # and move him back to his original location
        self.currentBoard.Players[0].updateY(2)
        self.laddersCollidedBelow = self.currentBoard.Players[
            0].checkCollision(self.ladderGroup)
        self.laddersCollidedAbove = self.currentBoard.Players[0].checkTopOnlyCollision(self.ladderGroup)
        # self.laddersCollidedBelow = []
        self.wallsCollidedBelow = self.currentBoard.Players[
            0].checkCollision(self.wallGroup)
        self.currentBoard.Players[0].updateY(-2)
        after = str(self.currentBoard.Players[0].getPosition())
        # if after != before:
        #     print("init1 before " + str(before) + " after " + str(after))

        before = str(self.currentBoard.Players[0].getPosition())
        # To check for collisions above, we move the player up then check and
        # then move him back down
        self.currentBoard.Players[0].updateY(-2)
        self.wallsCollidedAbove = self.currentBoard.Players[
            0].checkCollision(self.wallGroup)
        self.currentBoard.Players[0].updateY(2)
        after = str(self.currentBoard.Players[0].getPosition())

        before = str(self.currentBoard.Players[0].getPosition())
        # Sets the onLadder state of the player
        self.currentBoard.ladderCheck(
            self.laddersCollidedAbove,
            self.wallsCollidedBelow,
            self.wallsCollidedAbove, self.currentBoard.Players[0])
        after = str(self.currentBoard.Players[0].getPosition())

        for event in pygame.event.get():
            # Exit to desktop
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            if event.type == KEYDOWN:
                # print("isJumping" + str(self.currentBoard.Players[0].isJumping))
                # print("Going")
                # Get the ladders collided with the player
                # print("Event key" + str(event.key))
                self.laddersCollidedExact = self.currentBoard.Players[
                    0].checkCollision(self.ladderGroup)
                if (event.key == self.actions["jump"] and self.currentBoard.Players[0].onLadder == 0) \
                        or (event.key == self.actions["jump"] and \
                            self.currentBoard.Players[0].checkBottomOnlyCollision(self.ladderGroup) \
                            and not self.currentBoard.Players[0].checkTopOnlyCollision(self.ladderGroup)):
                    # Set the player to move up
                    self.direction = 2
                    if (self.currentBoard.Players[
                            0].isJumping == 0 and self.wallsCollidedBelow) or \
                            (self.currentBoard.Players[0].isJumping == 0 and \
                             self.currentBoard.Players[0].checkBottomOnlyCollision(self.ladderGroup) \
                             and not self.currentBoard.Players[0].checkTopOnlyCollision(self.ladderGroup)):
                        # We can make the player jump and set his
                        # currentJumpSpeed
                        # print("is Jumping")
                        self.currentBoard.Players[0].isJumping = 1
                        self.currentBoard.Players[0].currentJumpSpeed = 15
                # if (event.key == self.actions["up"] and self.laddersCollidedExact):
                #     self.currentBoard.Players[0].updateWH(self.IMAGES["left"], "V",
                #                                      self.currentBoard.Players[0].getSpeed(), 15, 15)
                keyState = pygame.key.get_pressed()
                before = str(self.currentBoard.Players[0].getPosition())

                if keyState[pygame.K_d] or event.key == pygame.K_d:
                    if self.currentBoard.direction != 4:
                        self.currentBoard.direction = 4
                        self.currentBoard.cycles = -1  # Reset cycles
                    self.currentBoard.cycles = (self.currentBoard.cycles + 1) % 4
                    if self.currentBoard.cycles < 2:
                        # Display the first image for half the cycles
                        self.currentBoard.Players[0].updateWH(self.IMAGES["right"], "H",
                                                         self.currentBoard.Players[0].getSpeed(), 15, 15)
                    else:
                        # Display the second image for half the cycles
                        self.currentBoard.Players[0].updateWH(self.IMAGES["right2"], "H",
                                                         self.currentBoard.Players[0].getSpeed(), 15, 15)
                    wallsCollidedExact = self.currentBoard.Players[
                        0].checkCollision(self.wallGroup)
                    # wallsCollidedExact = []
                    if wallsCollidedExact:
                        # print("Right collide")
                        # print(wallsCollidedExact[0].getPosition())
                        # If we have collided a wall, move the player back to
                        # where he was in the last state
                        self.currentBoard.Players[0].updateWH(self.IMAGES["right"], "H",
                                                         -self.currentBoard.Players[0].getSpeed(), 15, 15)
                    # if self.currentBoard.Players[0].checkBottomOnlyCollision(self.ladderGroup):
                    #     self.currentBoard.Players[0].onLadder = 1
                    # self.currentBoard.Players[0].isJumping = 0
                after = str(self.currentBoard.Players[0].getPosition())
                # if after != before:
                # print("KD before " + str(before) + " after " + str(after))
                # print(self.currentBoard.Players[0].isJumping)

                if keyState[pygame.K_a] or event.key == pygame.K_a:
                    if self.currentBoard.direction != 3:
                        self.currentBoard.direction = 3
                        self.currentBoard.cycles = -1  # Reset cycles
                    self.currentBoard.cycles = (self.currentBoard.cycles + 1) % 4
                    if self.currentBoard.cycles < 2:
                        # Display the first image for half the cycles
                        self.currentBoard.Players[0].updateWH(self.IMAGES["left"], "H",
                                                         -self.currentBoard.Players[0].getSpeed(), 15, 15)
                    else:
                        # Display the second image for half the cycles
                        self.currentBoard.Players[0].updateWH(self.IMAGES["left2"], "H",
                                                         -self.currentBoard.Players[0].getSpeed(), 15, 15)
                    # wallsCollidedExact = []
                    wallsCollidedExact = self.currentBoard.Players[
                        0].checkCollision(self.wallGroup)
                    if wallsCollidedExact:
                        # If we have collided a wall, move the player back to
                        # where he was in the last state
                        # print("Wall collided")
                        self.currentBoard.Players[0].updateWH(self.IMAGES["left"], "H",
                                                         self.currentBoard.Players[0].getSpeed(), 15, 15)
                    # if self.currentBoard.Players[0].checkBottomOnlyCollision(self.ladderGroup):
                    #     self.currentBoard.Players[0].onLadder = 1
                    # self.currentBoard.Players[0].isJumping = 0

                # If we are on a ladder, then we can move up
                if (keyState[pygame.K_w] or event.key == pygame.K_w) and self.currentBoard.Players[0].onLadder and \
                        self.currentBoard.Players[0].isJumping == 0:
                    for i in range(1):
                        self.currentBoard.Players[0].updateWH(self.IMAGES["still"], "V",
                                                         -self.currentBoard.Players[0].getSpeed(), 15, 15)
                        if len(self.currentBoard.Players[0].checkCollision(self.wallGroup)) != 0:
                            self.currentBoard.Players[0].updateWH(self.IMAGES["still"], "V",
                                                             self.currentBoard.Players[0].getSpeed(), 15, 15)

                # If we are on a ladder, then we can move down
                if (keyState[pygame.K_s] or event.key == pygame.K_s) and self.currentBoard.Players[
                    0].isJumping == 0 and not self.wallsCollidedBelow:  # and (self.currentBoard.Players[0].onLadder or self.currentBoard.Players[0].checkBottomOnlyCollision(self.ladderGroup))
                    for i in range(1):
                        self.currentBoard.Players[0].updateWH(self.IMAGES["still"], "V",
                                                         self.currentBoard.Players[0].getSpeed(), 15, 15)
                        if len(self.currentBoard.Players[0].checkCollision(self.wallGroup)) != 0:
                            self.currentBoard.Players[0].updateWH(self.IMAGES["still"], "V",
                                                             -self.currentBoard.Players[0].getSpeed(), 15, 15)
                keyPress = event.key

        # Update the player's position and process his jump if he is jumping
        before = str(self.currentBoard.Players[0].getPosition())
        self.currentBoard.Players[0].continuousUpdate(
            self.wallGroup, self.ladderGroup)
        after = str(self.currentBoard.Players[0].getPosition())
        # if after != before:
        #     print("CU before " + str(before) + " after " + str(after))

        '''
        We use cycles to animate the character, when we change direction we also reset the cycles
        We also change the direction according to the key pressed
        '''

        # Collect a coin
        coinsCollected = pygame.sprite.spritecollide(
            self.currentBoard.Players[0], self.coinGroup, True)
        coinsCollected2 = pygame.sprite.spritecollide(
            self.currentBoard.Players[0], self.coinGroup2, True)
        # print(self.currentBoard.Players[0])
        self.currentBoard.coinCheck(coinsCollected)
        self.currentBoard.coinCheck2(coinsCollected2)

        # Redraws all our instances onto the screen
        self.currentBoard.redrawScreen(self.screen, self.width, self.height)
        if keyPress:
            return keyPress
        else:
            return None

    def get_state(self):
        board_state = np.ndarray.flatten(np.asarray(self.currentBoard.map)).astype(int)
        agent_state = np.asarray(self.currentBoard.Players[0].getPosition()).astype(int)
        return np.concatenate((board_state, agent_state))

class meta_monsterkong(MonsterKong_Base):
    def __init__(self, mk_config):
        super().__init__(mk_config=mk_config)
        self._level: Optional[int] = None
        end_level = self.mk_config.StartLevel + self.mk_config.NumLevels
        # print(f"Creating a new meta_monsterkong wrapper for levels: "
        #       f"[{self.mk_config.StartLevel}:{end_level})")

    def init(self):
        # Create a new instance of the Board class
        # self.playerPosition = (120, 180) #change here depending on game width and height. for map_short it was (120,150), for short2 it was (120,230), for 3 (50,100)
        # self.playerPosition = (120, 210)
        self.texture_fixed = self.mk_config.TextureFixed
        if self._level is not None:
            # If a level was manually fixed using the `level` property
            level = self._level
        else:
            level = np.random.randint(low=self.mk_config.StartLevel,
                                      high=self.mk_config.StartLevel+self.mk_config.NumLevels)
        #print("level=",level)
        self.currentBoard = Board(
            self.width,
            self.height,
            self.rewards,
            self.rng,
            self._dir,
            self.mk_config,
            level,
        )

        # Assign groups from the Board instance that was created
        self.playerGroup = self.currentBoard.playerGroup
        self.wallGroup = self.currentBoard.wallGroup
        self.ladderGroup = self.currentBoard.ladderGroup

        if self.mk_config.Mode == 'Test':
            level += 10000

        if not self.texture_fixed:
            rng = np.random.default_rng(seed=level)
            self.size = (3,3)
            self.kernel = [rng.random(self.size).flatten() for _ in range(3)]
            self.offset = [128*rng.random() for _ in range(3)]

    def getScreenRGB(self):
        img = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8)        
        if not self.texture_fixed:
            red, green, blue = (Image.fromarray(img)).split()
            n_c_channels = []
            for (i, c_channel) in enumerate([red, green, blue]):                    
                n_c_channels.append(np.asarray((c_channel).filter(PIL.ImageFilter.Kernel(size=self.size, kernel=self.kernel[i], offset=self.offset[i])))[:,:,None])
            return np.concatenate(n_c_channels, axis=2)
        return img
 
