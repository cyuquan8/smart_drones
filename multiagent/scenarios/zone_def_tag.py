# ==========================================================================================================================================================
# scenario class 
# purpose: scenario for zone tag environment
# ==========================================================================================================================================================

import time
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from utils.utils import random_cartesian_from_polar, within_drone_radius

# wrapper class around agent class for additional attributes of drones
class DroneAgent(Agent):

	def __init__(self, drone_radius):


		""" function to generate the drone agent along with its attributes """

		# class constructor from Agent to inherit its attributes
		super().__init__()

		# drone radius that determines observation fidelity
		self.drone_radius = drone_radius

# scenario class for zone defense
class Scenario(BaseScenario):

	def make_world(self, dim_c, num_good_agents, num_adversaries, num_landmarks, r_rad, i_rad, r_noise_pos, r_noise_vel, big_rew_cnst, rew_multiplier_cnst, ep_time_step_limit, drone_radius, 
				   agent_size, agent_density, agent_initial_mass, agent_accel, agent_max_speed, agent_collide, agent_silent, agent_u_noise, agent_c_noise, agent_u_range, landmark_size):

		""" function to generate the world along with attributes of the world """

		# create world instance
		world = World()

		# set any world properties first

		# dimension of communication action space
		world.dim_c = dim_c

		# create episode start time step attribute to determine episode length
		world.ep_time_step = 0

		# create attributes for the number of adversaries, good agents
		world.num_adversaries = num_adversaries
		world.num_good_agents = num_good_agents

		# number of agents
		num_agents = num_adversaries + num_good_agents

		# restricted (r) and intercept(i) radius from location of base (player start location)
		# i_rad > r_rad and beyond i_rad is the monitoring zone
		self.r_rad = r_rad
		self.i_rad = i_rad
		
		# radar noise for position and velocity of agents
		self.r_noise_pos = r_noise_pos
		self.r_noise_vel = r_noise_vel

		# constants for reward functions
		self.big_rew_cnst = big_rew_cnst
		self.rew_multiplier_cnst = rew_multiplier_cnst

		# time limit for episode
		self.ep_time_step_limit = ep_time_step_limit

		# add agents

		# generate agents instances in list
		world.agents = [DroneAgent(drone_radius[1] if i < num_adversaries else drone_radius[0]) for i in range(num_agents)]

		# iterate over agent instances to amend agent attributes
		for i, agent in enumerate(world.agents):
			
			# update relevant agent attributes for good and adversarial agents
			agent.adversary = True if i < num_adversaries else False
			agent.name = 'red_agent_%d' % i if agent.adversary else 'blue_agent_%d' % (i - num_adversaries)
			agent.size = agent_size[1] if agent.adversary else agent_size[0]
			agent.density = agent_density[1] if agent.adversary else agent_density[0]
			agent.initial_mass = agent_initial_mass[1] if agent.adversary else agent_initial_mass[0]
			agent.accel = agent_accel[1] if agent.adversary else agent_accel[0]
			agent.max_speed = agent_max_speed[1] if agent.adversary else agent_max_speed[0]
			agent.collide = agent_collide[1] if agent.adversary else agent_collide[0]
			agent.silent = agent_silent[1] if agent.adversary else agent_silent[0]
			agent.u_noise = agent_u_noise[1] if agent.adversary else agent_u_noise[0]
			agent.c_noise = agent_c_noise[1] if agent.adversary else agent_c_noise[0]
			agent.u_range = agent_u_range[1] if agent.adversary else agent_u_range[0]
			
		# add landmarks

		# generate landmarks instances in list
		world.landmarks = [Landmark() for i in range(num_landmarks)]

		# iterate over landmark instances to amend landmark attributes
		for i, landmark in enumerate(world.landmarks):

			# update relevant landmark attributes
			landmark.name = 'landmark_%d' % i
			landmark.collide = True
			landmark.movable = False
			landmark.size = landmark_size

		# store silent the case of zone_def_tag
		self.agent_silent = agent_silent

		# make initial conditions
		self.reset_world(world)
		
		return world

	def reset_world(self, world):

		""" function to reset world when episode terminates """

		# random properties for agents
		for i, agent in enumerate(world.agents):

			# good = blue, adversary = red
			agent.color = np.array([0.35, 0.35, 0.85]) if not agent.adversary else np.array([0.85, 0.35, 0.35])

			# make sure all agents are movable with correct silent
			agent.movable = True
			agent.silent = self.agent_silent[1] if agent.adversary else self.agent_silent[0]

		# random properties for landmarks
		for i, landmark in enumerate(world.landmarks):

			# black-grey color for landmark
			landmark.color = np.array([0.25, 0.25, 0.25])

		# set random initial states
		for agent in world.agents:

			# spawn good agents within r_rad
			if not agent.adversary:

				# generate random cartesian position for good agents
				agent.state.p_pos = random_cartesian_from_polar(r_low = 0, r_high = self.r_rad)

			# spawn adversarial agents beyond i_rad
			elif agent.adversary:

				# generate random cartesian position for adversarial agents
				agent.state.p_pos = random_cartesian_from_polar(r_low = self.i_rad, r_high = 1)

			# reset velocity and comms actions to zero
			agent.state.p_vel = np.zeros(world.dim_p)
			agent.state.c = np.zeros(world.dim_c)

		# set random positions for landmarks within r_rad and i_rad
		for i, landmark in enumerate(world.landmarks):

			# generate random cartesian position for landmarks 
			landmark.state.p_pos = random_cartesian_from_polar(r_low = self.r_rad, r_high = self.i_rad)

			# reset velocity to zero
			landmark.state.p_vel = np.zeros(world.dim_p)

		# reset ep_time_step to zero
		world.ep_time_step = 0

	def benchmark_data(self, agent, world):

		""" function that returns relevant metrics for benchmarking purposes """

		# track number of collisions between team and opponent drones
		number_of_team_collisions = 0
		number_of_oppo_collisions = 0
	
		# for agent drones 
		if not agent.adversary:

			# iterate over all drones
			for agt in world.agents:

				# check if agt in current iteration is agent
				if agt is agent:

					# ignore agt if agt is agent
					continue 

				# check for collisions
				if self.is_collision(agent, agt):

					# check if collision against agent drone
					if not agt.adversary:

						# add to team collision counter
						number_of_team_collisions += 1

					# check if collision against adversarial drone
					elif agt.adversary:

						 # add to oppo collision counter
						number_of_oppo_collisions += 1

		# for adversarial drones
		elif agent.adversary:

			# iterate over all drones
			for agt in world.agents:

				# check if agt in current iteration is agent
				if agt is agent:

					# ignore agt if agt is agent
					continue 

				# check for collisions
				if self.is_collision(agent, agt):

					# check if collision against agent drone
					if not agt.adversary:

						# add to oppo collision counter
						number_of_oppo_collisions += 1

					# check if collision against adversarial drone
					elif agt.adversary:

						 # add to team collision counter
						number_of_team_collisions += 1

		return [number_of_team_collisions, number_of_oppo_collisions]

	def is_collision(self, agent1, agent2):

		""" function to check if there are collisions between two agents """

		# obtain difference in position in terms of vector
		delta_pos = agent1.state.p_pos - agent2.state.p_pos

		# obtain difference in position in terms of magnitude
		dist = np.sqrt(np.sum(np.square(delta_pos)))

		# obtain minimal distance between the two agents
		dist_min = agent1.size + agent2.size

		# collision if difference in position is smaller than minimal distance
		return True if dist < dist_min else False

	def good_agents(self, world):

		""" function that returns all agents that are not adversaries """

		return [agent for agent in world.agents if not agent.adversary]

	def adversaries(self, world):

		""" fucntion that returns all agents that are adversarial """

		return [agent for agent in world.agents if agent.adversary]

	def reward(self, agent, world):

		""" function that returns reward for any agent """

		# different reward for good and adverserial agents
		main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

		return main_reward

	def agent_reward(self, agent, world):

		""" function that returns reward for any good agent """

		# initialise reward to zero
		rew = 0

		# obtain list of adversaries
		adver_list = self.adversaries(world)

		# iterate over list of adversaries
		for adver in adver_list:

			# check if there are collisions between agent and advers
			if self.is_collision(adver, agent):
				
				# reward to good agent if it collide and disabled adversarial drone
				rew += self.big_rew_cnst / 10

				# disable adversarial drone
				adver.movable = False
				adver.silent = True

		# iterate over list of adversaries
		for adver in adver_list:

			# obtain radius from origin of adversarial drone 
			radius = np.sqrt(np.sum(np.square(adver.state.p_pos)))

			# check if adversarial drone entered the restricted zone
			if radius <= self.r_rad:

				# big penalty to good agent if adversarial drone entered the restricted zone 
				rew -= self.big_rew_cnst 

				# # reward for keeping away adv drone from r_rad
				# rew -= self.rew_multiplier_cnst * (1.1 - radius)

				return rew

		# reward good agent for acheiving objective
		if world.ep_time_step >= self.ep_time_step_limit:

			rew += self.big_rew_cnst

			return rew

		# reward for exiting screen. Remember to uncomment if EXIT_SCREEN_TERMINATE == True 

		# # check if agent exceeds screen boundary
		if np.sqrt(np.sum(np.square(agent.state.p_pos))) > 1:

			# big penalty to agent if it exits screen boundary
			rew -= self.big_rew_cnst

			return rew

		# # reward for exiting screen. Remember to uncomment if EXIT_SCREEN_TERMINATE == False

		# def bound(x):

		# 	""" function to penalise agents for exiting the screen"""

		# 	# case 1: x < 0.9
		# 	if x < 0.9:

		# 		return 0

		# 	# case 2 x < 1.0
		# 	if x < 1.0:

		# 		return (x - 0.9) * 10

		# 	# case 3: x >= 1
		# 	return min(np.exp(2 * x - 2), 10)

		# # iterate over agent position dimension
		# for p in range(world.dim_p):

		# 	# obtain absolute value for position for that dimension
		# 	x = abs(agent.state.p_pos[p])

		# 	# penalty for leaivng screen
		# 	rew -= bound(x)

		return rew

	def adversary_reward(self, agent, world):

		""" function that returns reward for any adversarial agent """

		# initialise reward to zero
		rew = 0

		# obtain list of agent and adversarial drones
		adver_list = self.adversaries(world)
		agent_list = sefl.good_agents(world)

		# iterate over list of agent
		for agt in agent_list:

			# check if there are collisions between agent and advers
			if self.is_collision(agt, agent):
				
				# reward to good agent if it collide and disabled adversarial drone
				rew -= self.big_rew_cnst / 10

				# disable adversarial drone
				agent.movable = False
				agent.silent = True

		# iterate over list of adversaries
		for adver in adver_list:

			# obtain radius from origin of adversarial drone 
			radius = np.sqrt(np.sum(np.square(adver.state.p_pos)))

			# check if adversarial drone entered the restricted zone
			if radius <= self.r_rad:

				# big reward to adversarial agent if it entered the restricted zone weighted based on time of occurence
				rew += self.big_rew_cnst

				# # reward coming close to r_rad
				# rew += self.rew_multiplier_cnst * (1.1 - radius)

				return rew

		# reward good agent for acheiving objective
		if world.ep_time_step >= self.ep_time_step_limit:

			rew -= self.big_rew_cnst

			return rew

		# reward for exiting screen. Remember to uncomment if EXIT_SCREEN_TERMINATE == True 

		# # check if agent exceeds screen boundary
		if np.sqrt(np.sum(np.square(agent.state.p_pos))) > 1:

			# big penalty to agent if it exits screen boundary
			rew -= self.big_rew_cnst

			return rew

		# # reward for exiting screen. Remember to uncomment if EXIT_SCREEN_TERMINATE == False

		# def bound(x):

		# 	""" function to penalise agents for exiting the screen"""

		# 	# case 1: x < 0.9
		# 	if x < 0.9:

		# 		return 0

		# 	# case 2 x < 1.0
		# 	if x < 1.0:

		# 		return (x - 0.9) * 10

		# 	# case 3: x >= 1
		# 	return min(np.exp(2 * x - 2), 10)

		# # iterate over agent position dimension
		# for p in range(world.dim_p):

		# 	# obtain absolute value for position for that dimension
		# 	x = abs(agent.state.p_pos[p])

		# 	# penalty for leaivng screen
		# 	rew -= bound(x)

		return rew

	def reward_goal(self, agent, world, goal):

		""" function that returns reward for any agent for a given goal"""

		# different reward for good and adverserial agents
		main_reward = self.adversary_goal_reward(agent, world, goal) if agent.adversary else self.agent_goal_reward(agent, world, goal)

		return main_reward

	def agent_goal_reward(self, agent, world, goal):

		""" function that returns reward for any good agent for a given goal """

		# initialise reward to zero
		rew = 0

		# obtain list of adversaries
		adver_list = self.adversaries(world)

		# iterate over list of adversaries
		for adver in adver_list:

			# check if there are collisions between agent and advers
			if self.is_collision(adver, agent):
				
				# reward to good agent if it collide and disabled adversarial drone
				rew += self.big_rew_cnst / 10

				# disable adversarial drone
				adver.movable = False
				adver.silent = True

		# iterate over list of adversaries
		for adver in adver_list:

			# obtain radius from origin of adversarial drone 
			radius = np.sqrt(np.sum(np.square(adver.state.p_pos)))

			# check if adversarial drone entered the restricted zone
			if radius <= self.r_rad:

				# big penalty to good agent if adversarial drone entered the restricted zone 
				rew -= self.big_rew_cnst 

				# # reward for keeping away adv drone from r_rad
				# rew -= self.rew_multiplier_cnst * (1.1 - radius)

				return rew

		# reward good agent for acheiving goal
		if world.ep_time_step >= goal:

			# check if goal is original goal
			if goal >= self.ep_time_step_limit:

				rew += self.big_rew_cnst

			# half of big_rew_cnst if goal is not original goal
			else: 

				rew += self.big_rew_cnst / 10

			return rew

		# reward for exiting screen. Remember to uncomment if EXIT_SCREEN_TERMINATE == True 

		# check if agent exceeds screen boundary
		if np.sqrt(np.sum(np.square(agent.state.p_pos))) > 1:

			# big penalty to agent if it exits screen boundary
			rew -= self.big_rew_cnst

			return rew

		# reward for exiting screen. Remember to uncomment if EXIT_SCREEN_TERMINATE == False

		# def bound(x):

		# 	""" function to penalise agents for exiting the screen"""

		# 	# case 1: x < 0.9
		# 	if x < 0.9:

		# 		return 0

		# 	# case 2 x < 1.0
		# 	if x < 1.0:

		# 		return (x - 0.9) * 10

		# 	# case 3: x >= 1
		# 	return min(np.exp(2 * x - 2), 10)

		# # iterate over agent position dimension
		# for p in range(world.dim_p):

		# 	# obtain absolute value for position for that dimension
		# 	x = abs(agent.state.p_pos[p])

		# 	# penalty for leaivng screen
		# 	rew -= bound(x)

		return rew

	def adversary_goal_reward(self, agent, world, goal):

		""" function that returns reward for any adversarial agent """

		# initialise reward to zero
		rew = 0

		# obtain list of agent and adversarial drones
		adver_list = self.adversaries(world)
		agent_list = self.good_agents(world)

		# iterate over list of adversaries
		for agt in agent_list:

			# check if there are collisions between agent and advers
			if self.is_collision(agt, agent):
				
				# reward to good agent if it collide and disabled adversarial drone
				rew += self.big_rew_cnst / 10

				# disable adversarial drone
				agent.movable = False
				agent.silent = True

		# iterate over list of adversaries
		for adver in adver_list:

			# obtain radius from origin of adversarial drone 
			radius = np.sqrt(np.sum(np.square(adver.state.p_pos)))

			# check if adversarial drone entered the restricted zone
			if radius <= goal:

				# check if goal is original goal
				if goal <= self.r_rad:

					# big reward to adversarial agent if it entered the restricted zone weighted based on time of occurence
					rew += self.big_rew_cnst

				# half of original reward if goal is not original goal
				else: 

					rew += self.big_rew_cnst / 10

				# # reward coming close to r_rad
				# rew += self.rew_multiplier_cnst * (1.1 - radius)

				return rew

		# reward good agent for acheiving objective
		if world.ep_time_step >= self.ep_time_step_limit:

			rew -= self.big_rew_cnst

			return rew

		# reward for exiting screen. Remember to uncomment if EXIT_SCREEN_TERMINATE == True 

		# check if agent exceeds screen boundary
		if np.sqrt(np.sum(np.square(agent.state.p_pos))) > 1:

			# big penalty to agent if it exits screen boundary
			rew -= self.big_rew_cnst

			return rew

		# reward for exiting screen. Remember to uncomment if EXIT_SCREEN_TERMINATE == False

		# def bound(x):

		# 	""" function to penalise agents for exiting the screen"""

		# 	# case 1: x < 0.9
		# 	if x < 0.9:

		# 		return 0

		# 	# case 2 x < 1.0
		# 	if x < 1.0:

		# 		return (x - 0.9) * 10

		# 	# case 3: x >= 1
		# 	return min(np.exp(2 * x - 2), 10)

		# # iterate over agent position dimension
		# for p in range(world.dim_p):

		# 	# obtain absolute value for position for that dimension
		# 	x = abs(agent.state.p_pos[p])

		# 	# penalty for leaivng screen
		# 	rew -= bound(x)

		return rew

	def is_terminal(self, agent, world):

		""" function to return to check if episode has terminated """
		""" terminal condition 1: adversarial drone enters restricted drone (adversarial drones win) """
		""" terminal condition 2: exceed time limit (agent drones win) """
		""" terminal condition 3: agent drone exceeds screen boundary (draw) """
		""" terminal condition 4: adversarial drone exceeds screen boundary (draw) """

		# check if adversarial drone has inflitrated restricted zone
		if np.sqrt(np.sum(np.square(agent.state.p_pos))) < self.r_rad and agent.adversary == True:

			return [True, 1]

		# if episode time exceeds episode time limit, episode terminates with agent drones succeeding in defending the restricted zone
		if world.ep_time_step >= self.ep_time_step_limit:

			return [True, 2] 

		# terminal condition for exiting screen. Remember to uncomment if EXIT_SCREEN_TERMINATE == True and vice-versa

		# check if agent drone have exceeded screen boundary 
		if np.sqrt(np.sum(np.square(agent.state.p_pos))) > 1 and agent.adversary == False:

			return [True, 3]

		# check if adversarial drone have exceeded screen boundary 
		if np.sqrt(np.sum(np.square(agent.state.p_pos))) > 1 and agent.adversary == True:

			return [True, 4]

		return [False, 0]

	def is_terminal_goal(self, agent, world, agent_goal, adver_goal):

		""" function to return to check if episode has terminated """
		""" terminal condition 1: adversarial drone enters restricted drone (adversarial drones win) """
		""" terminal condition 2: exceed time limit (agent drones win) """
		""" terminal condition 3: agent drone exceeds screen boundary (draw) """
		""" terminal condition 4: adversarial drone exceeds screen boundary (draw) """

		# check if adversarial drone has inflitrated restricted zone
		if np.sqrt(np.sum(np.square(agent.state.p_pos))) < adver_goal and agent.adversary == True:

			return [True, 1]

		# if episode time exceeds episode time limit, episode terminates with agent drones succeeding in defending the restricted zone
		if world.ep_time_step >= agent_goal:

			return [True, 2] 

		# terminal condition for exiting screen. Remember to uncomment if EXIT_SCREEN_TERMINATE == True and vice-versa

		# check if agent drone have exceeded screen boundary 
		if np.sqrt(np.sum(np.square(agent.state.p_pos))) > 1 and agent.adversary == False:

			return [True, 3]

		# check if adversarial drone have exceeded screen boundary 
		if np.sqrt(np.sum(np.square(agent.state.p_pos))) > 1 and agent.adversary == True:

			return [True, 4]

		return [False, 0]
					
	def observation(self, agent, world):

		""" function that returns observation for any agent """

		# empty list to store landmark positions
		landmark_pos = []

		# iterate over landmarks entities
		for landmark in world.landmarks:

			# check if landmark is within agent's drone radius
			if within_drone_radius(agent, landmark):

				# obtain position of landmark relative to agent's reference frame without noise if within agent's drone radius
				landmark_pos.append(landmark.state.p_pos - agent.state.p_pos)

			# if outside agent's drone radius
			else:

				# landmark position is unknown and filled with zeros if not within agent's drone radius
				landmark_pos.append(np.array([0, 0]))

		# empty list for comms, position and veloctiy of good and adverserial agents other than the agent itself
		good_comms = []
		good_pos = []
		good_vel = []
		adv_comms = []
		adv_pos = []
		adv_vel = []

		# iterate over all agents
		for other in world.agents:

			# check if agent in current iteration is the current agent
			if other is agent:  

				# ignore current agent for observation
				continue

			# check if agent is good agent 
			if not agent.adversary:

				# check if agent in current iteration is good agent
				if not other.adversary:

					# append communication of agent in current iteration
					good_comms.append(other.state.c)

					# check if agent in current iteration is within agent's drone radius
					if within_drone_radius(agent, other):

						# append position of agent in current iteration relative to agent's reference frame without radar noise if within agent's drone radius
						good_pos.append(other.state.p_pos - agent.state.p_pos)

						# append velocity of agent in current iteration without radar noise if within agent's drone radius 
						good_vel.append(other.state.p_vel)

					# if outside agent's drone radius, radar gives info
					else:

						# obtain noise for position and velocity from normal distribution
						pos_noise = np.random.normal(loc = 0.0, scale = self.r_noise_pos, size = None)
						vel_noise = np.random.normal(loc = 0.0, scale = self.r_noise_vel, size = None)

						# append position of agent in current iteration relative to agent's reference frame with noise if within agent's drone radius
						good_pos.append(other.state.p_pos - agent.state.p_pos + pos_noise) 

						# append velocity of agent in current iteration with radar noise if within agent's drone radius 
						good_vel.append(other.state.p_vel + vel_noise)

				# check if agent in current iteration is adversarial agent
				elif other.adversary:

					# append communication of agent in current iteration with zeros given that agent is good
					adv_comms.append(np.zeros(world.dim_c))

					# check if agent in current iteration is within agent's drone radius
					if within_drone_radius(agent, other):

						# append position of agent in current iteration relative to agent's reference frame without radar noise if within agent's drone radius
						adv_pos.append(other.state.p_pos - agent.state.p_pos)

						# append velocity of agent in current iteration without radar noise if within agent's drone radius 
						adv_vel.append(other.state.p_vel)

					# if outside agent's drone radius
					else:

						# obtain noise for position and velocity from normal distribution
						pos_noise = np.random.normal(loc = 0.0, scale = self.r_noise_pos, size = None)
						vel_noise = np.random.normal(loc = 0.0, scale = self.r_noise_vel, size = None)

						# append position of agent in current iteration relative to agent's reference frame with noise if within agent's drone radius
						adv_pos.append(other.state.p_pos - agent.state.p_pos + pos_noise) 

						# append velocity of agent in current iteration with radar noise if within agent's drone radius 
						adv_vel.append(other.state.p_vel + vel_noise)

			# check if agent is adversarial agent 
			elif agent.adversary:

				# check if agent in current iteration is good agent
				if not other.adversary:

					# append communication of agent in current iteration with zeros given that agent is adversarial
					good_comms.append(np.zeros(world.dim_c))

					# check if agent in current iteration is within agent's drone radius
					if within_drone_radius(agent, other):

						# append position of agent in current iteration relative to agent's reference frame without radar noise if within agent's drone radius
						good_pos.append(other.state.p_pos - agent.state.p_pos)

						# append velocity of agent in current iteration without radar noise if within agent's drone radius 
						good_vel.append(other.state.p_vel)

					# if outside agent's drone radius, no info
					else:

						# append position of agent in current iteration with zeros given that agent is adversarial
						good_pos.append(np.zeros(world.dim_p))

						# append velocity of agent in current iteration with zeros given that agent is adversarial
						good_vel.append(np.zeros(world.dim_p))

				# check if agent in current iteration is adversarial agent
				elif other.adversary:

					# append communication of agent in current iteration
					adv_comms.append(other.state.c)

					# check if agent in current iteration is within agent's drone radius
					if within_drone_radius(agent, other):

						# append position of agent in current iteration relative to agent's reference frame without radar noise if within agent's drone radius
						adv_pos.append(other.state.p_pos - agent.state.p_pos)

						# append velocity of agent in current iteration without radar noise if within agent's drone radius 
						adv_vel.append(other.state.p_vel)

					# if outside agent's drone radius
					else:

						# obtain noise for position and velocity from normal distribution
						pos_noise = np.random.normal(loc = 0.0, scale = self.r_noise_pos, size = None)
						vel_noise = np.random.normal(loc = 0.0, scale = self.r_noise_vel, size = None)

						# append position of agent in current iteration relative to agent's reference frame with noise if within agent's drone radius
						adv_pos.append(other.state.p_pos - agent.state.p_pos + pos_noise) 

						# append velocity of agent in current iteration with radar noise if within agent's drone radius 
						adv_vel.append(other.state.p_vel + vel_noise)
		
		# return concenated observation
		return np.concatenate([[world.ep_time_step]] + [agent.state.c] + [agent.state.p_pos] + [agent.state.p_vel] + landmark_pos + good_comms + adv_comms + good_pos + adv_pos + good_vel + adv_vel + \
							  [[1 if adver.movable == True else 0 for adver in self.adversaries(world)]])