__author__ = 'Batchu Vishal'
from .person import Person

'''
This class defines our player.
It inherits from the Person class since a Player is also a person.
We specialize the person by adding capabilities such as jump etc..
'''


class Player(Person):

    def __init__(self, raw_image, position, width, height):
        super(Player, self).__init__(raw_image, position, width, height)
        self.isJumping = 0
        self.onLadder = 0
        self.currentJumpSpeed = 0
        self.__gravity = 2 # Gravity affecting the jump velocity of the player
        self.__speed = 15  # Movement speed of the player
        self.isFalling = 0
        # self.jumpLagCount = 0
        # self.stayStill = 0

    # Getters and Setters
    def getSpeed(self):
        return self.__speed

    def setSpeed(self):
        return self.__speed

    # This manages the players jump
    # Only the player can jump (For the player's jump)
    def continuousUpdate(self, wallGroupList, ladderGroupList):
        # Only gets run when the player is not on the ladder
        if (self.onLadder == 0) or (self.checkBottomOnlyCollision(ladderGroupList) and not self.checkTopOnlyCollision(ladderGroupList)):
        # if (self.onLadder == 0):
            wallsCollided = self.checkCollision(wallGroupList)
            # print("bot2 collision")
            # print(self.checkBotCollision(ladderGroupList))
            # print("top2 collision")
            # print(self.checkTopCollision(ladderGroupList))
            # If the player is not jumping
            if self.isJumping == 0:
                # We move down a little and check if we collide with anything
                self.updateY(2)
                laddersCollided = self.checkCollision(ladderGroupList)
                wallsCollided = self.checkCollision(wallGroupList)
                self.updateY(-2)
                # If we are not colliding with anything below, then we start a
                # jump with 0 speed so that we just fall down
                # if len(wallsCollided) == 0 and len(laddersCollided) == 0:
                if len(wallsCollided) == 0 and len(laddersCollided) == 0:
                    # print("IN")
                    # print(len(wallsCollided))
                    # print(len(laddersCollided))
                    self.isJumping = 1
                    self.currentJumpSpeed = 0

            # If the player is jumping
            if self.isJumping:
                # print("JLC " + str(self.jumpLagCount))
                self.updateY(-2)
                if self.checkCollision(wallGroupList):
                    self.currentJumpSpeed = -1
                self.updateY(2)
                if wallsCollided:
                    # If you collide a wall while jumping  and its below you,
                    # then you stop the jump
                    # print(wallsCollided[0].getPosition())
                    if wallsCollided[0].getPosition()[1] > self.getPosition()[
                            1]:  # wallsize/2 and charsize/2 and +1
                        self.isJumping = 0
                        self.setPosition(((self.getPosition()[0], wallsCollided[0].getPosition()[
                            1] - (self.height))))  # Wall size/2 and charactersize/2 and +1
                        # print "HIT FLOOR"
                        # self.jumpLagCount = 0
                    # If you collide a wall while jumping and its above you,
                    # then you hit the ceiling so you make jump speed 0 so he
                    # falls down
                    elif wallsCollided[0].getPosition()[1] < self.getPosition()[1]:
                        # print("Hit ceiling")
                        self.currentJumpSpeed = 0
                        # self.setPosition((self.getPosition()[0], wallsCollided[
                        #                  0].getPosition()[1] + (self.height + 1)))
                        self.setPosition((self.getPosition()[0], wallsCollided[
                                         0].getPosition()[1] + self.height))
                        # if self.jumpLagCount <1:
                        #     self.stayStill = 1
                        #     self.jumpLagCount +=1
                        # else:
                        #     self.stayStill = 0
                        # print "HIT TOP"
                self.setCenter(self.getPosition())
                # If he is still jumping (ie. hasnt touched the floor yet)
                if self.isJumping:
                    # We move him down by the currentJumpSpeed
                    self.updateY(-self.currentJumpSpeed)
                    self.setCenter(self.getPosition())
                    self.currentJumpSpeed -= self.__gravity  # Affect the jump speed with gravity
                    if self.currentJumpSpeed < -8:
                        self.currentJumpSpeed = -8
