#   # # # #        _____ _     __        ______      _____      _                 
#   # # # #? ?    / ___/(_)___/ /__     / ____/___  / / (_)____(_)___  ____  _____
#   # # # #? ?    \__ \/ / __  / _ \   / /   / __ \/ / / / ___/ / __ \/ __ \/ ___/
#   # # # #? ?   ___/ / / /_/ /  __/  / /___/ /_/ / / / (__  ) / /_/ / / / (__  ) 
#      ? ? ? ?  /____/_/\__,_/\___/   \____/\____/_/_/_/____/_/\____/_/ /_/____/  
#
# Checks collisions on wich side of sprite
# v0.0.1
# By Death_Miner
# MIT License

import pygame

#
# Sets velocity properties for a sprite
# 
# @class Velocity
class Velocity():

	#
	# Instantiates the class
	# 
	# @param  object self     The class itself
	# @param  list   velocity The velocity to apply, default is 0 on both axis
	# @return void
	def __init__(self, velocity=[0,0]):

		#
		# Velocity in x axis
		# 
		# @var int
		self.x = velocity[0]
	
		#
		# Velocity in x axis
		# 
		# @var int
		self.y = velocity[1]

#
# Checks if sprites has velocity, if not add it
# 
# @param  object A The first sprite
# @param  object B The second sprite
# @return void
def check_velocity(A, B):
	if hasattr(A, "velocity") == False:
		A.velocity = Velocity()

	if hasattr(B, "velocity") == False:
		B.velocity = Velocity()

#
# Check if B collide on left of A
# 
# @param  object A The First sprite to check collision to
# @param  object B The second sprite wich would collide A on left
# @return bool     The result of the test
def left(A, B):

	# Set velocity to both sprites if there's no
	check_velocity(A, B)

	# First check if A & B collide themselves
	if pygame.sprite.collide_rect(A, B) == True:
		
		# Check if right points of B are in A but not left points of B
		if A.rect.collidepoint(B.rect.midright) == True and (\
		   A.rect.collidepoint(B.rect.topright) == True or \
		   A.rect.collidepoint(B.rect.bottomright) == True) and \
		   A.rect.collidepoint(B.rect.midleft) == False and (\
		   A.rect.collidepoint(B.rect.topleft) == False or \
		   A.rect.collidepoint(B.rect.bottomleft) == False):

		    # Check if B velocity moves to the left
			if B.velocity.x > 0:
				return True

	# Instead return False        
	return False

#
# Check if B collide on right of A
# 
# @param  object A The First sprite to check collision to
# @param  object B The second sprite wich would collide A on right
# @return bool     The result of the test
def right(A, B):

	# Set velocity to both sprites if there's no
	check_velocity(A, B)

	# First check if A & B collide themselves
	if pygame.sprite.collide_rect(A, B) == True:
		
		# Check if left points of B are in A but not right points of B
		if A.rect.collidepoint(B.rect.midleft) == True and (\
		   A.rect.collidepoint(B.rect.topleft) == True or \
		   A.rect.collidepoint(B.rect.bottomleft) == True) and \
		   A.rect.collidepoint(B.rect.midright) == False and (\
		   A.rect.collidepoint(B.rect.topright) == False or \
		   A.rect.collidepoint(B.rect.bottomright) == False):

		    # Check if B velocity moves to the right
			if B.velocity.x < 0:
				return True

	# Instead return False        
	return False

#
# Check if B collide on top of A
# 
# @param  object A The First sprite to check collision to
# @param  object B The second sprite wich would collide A on top
# @return bool     The result of the test
def top(A, B):

	# Set velocity to both sprites if there's no
	check_velocity(A, B)

	# First check if A & B collide themselves
	if pygame.sprite.collide_rect(A, B) == True:
		# print("124")
		# Check if bottom points of B are in A but not top points of B
		if A.rect.collidepoint(B.rect.midbottom) == True or (\
		   A.rect.collidepoint(B.rect.bottomleft) == True or \
		   A.rect.collidepoint(B.rect.bottomright) == True) and \
		   A.rect.collidepoint(B.rect.midtop) == False and (\
		   A.rect.collidepoint(B.rect.topleft) == False or \
		   A.rect.collidepoint(B.rect.topright) == False):
			# print("132")
		    # Check if B velocity moves to the top
			return True
			if B.velocity.y < 0:
				print("Velocity check true")
				return True

	# Instead return False        
	return False

#
# Check if B collide on bottom of A
# 
# @param  object A The First sprite to check collision to
# @param  object B The second sprite wich would collide A on bottom
# @return bool     The result of the test
def bottom(A, B):

	# Set velocity to both sprites if there's no
	check_velocity(A, B)

	# First check if A & B collide themselves
	if pygame.sprite.collide_rect(A, B) == True:
		
		# Check if top points of B are in A but not bottom points of B #and or and and or
		if A.rect.collidepoint(B.rect.midtop) == True or (\
		   A.rect.collidepoint(B.rect.topleft) == True or \
		   A.rect.collidepoint(B.rect.topright) == True) and \
		   A.rect.collidepoint(B.rect.midbottom) == False and (\
		   A.rect.collidepoint(B.rect.bottomleft) == False or \
		   A.rect.collidepoint(B.rect.bottomright) == False):
			return True
		    # Check if B velocity moves to the bottom
			if B.velocity.y > 0:
				return True

	# Instead return False        
	return False

#
# Sets multiple side detection
#
class _Multiple():
	
	#
	# Inits __Multiple class
	#
	# @param  object self  The class itself
	# @param  list   sides The sides wich has to be detected
	# @return void
	def __init__(self, sides):
		self.sides = sides
	
	#
	# Checks detection on multiple sides
	#
	# @param  object self The class itself
	# @param  object A    The first sprite
	# @param  object B    The second sprite
	# @return bool        The result of the tests
	def check_sides(self, A, B):

		# Array keeping already done sides test for not repeating tasks
		done_sides = []
		
		# Navigate trough list
		for side in self.sides:

			# Check if side test was done before
			if side not in done_sides:

				# Insert that element
				done_sides.append(side)

				# Check for left if selected
				if side == "left":
					if left(A, B) == True:
						return True
						
				# Check for right if selected
				elif side == "right":
					if right(A, B) == True:
						return True
						
				# Check for top if selected
				elif side == "top":
					if top(A, B) == True:
						return True
						
				# Check for bottom if selected
				elif side == "bottom":
					if bottom(A, B) == True:
						return True
				
				# Error if invalid side name
				else:
					print("Unknown side name passed in multiple side detections: \""+str(side)+"\"")
				
		# Return False if no collision detected
		return False

	#
	# Checks detection on multiple sides (with group support)
	#
	# @param  object self The class itself
	# @param  object A    The first sprite
	# @param  object B    The second sprite
	# @return bool        The result of the tests 
	def check_sides_group(self, A, B):
		return self.check_sides(B, A)

#
# Check for collisions on multiple sides
#
# @param  list     sides A list of collisions: left, right, top, bottom available
# @return function       The function passed as collided callback
def multiple(sides):
	multiple_class = _Multiple(sides)
	return multiple_class.check_sides

#
# Detects collisions in inversed mode for support groupcollide()
#
class Group():

	#
	# Check if B collide on left of A
	# 
	# @param  object A The First sprite to check collision to
	# @param  object B The second sprite wich would collide A on left
	# @return bool     The result of the test
	def left(A, B):
		return left(B, A)

	#
	# Check if B collide on right of A
	# 
	# @param  object A The First sprite to check collision to
	# @param  object B The second sprite wich would collide A on right
	# @return bool     The result of the test
	def right(A, B):
		return right(B, A)

	#
	# Check if B collide on top of A
	# 
	# @param  object A The First sprite to check collision to
	# @param  object B The second sprite wich would collide A on top
	# @return bool     The result of the test
	def top(A, B):
		return top(B, A)

	#
	# Check if B collide on bottom of A
	# 
	# @param  object A The First sprite to check collision to
	# @param  object B The second sprite wich would collide A on bottom
	# @return bool     The result of the test
	def bottom(A, B):
		return bottom(B, A)

	#
	# Check for collisions on multiple sides
	# 
	# @param  list     sides A list of collisions: left, right, top, bottom available
	# @return function       The function passed as collided callback
	def multiple(sides):
		multiple_class = _Multiple(sides)
		return multiple_class.check_sides_group
