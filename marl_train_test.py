# ==========================================================================================================================================================
# multi agent reinforcement learning (marl) train / test function
# purpose: train/test models for agent and opponent drones using marl algorithms
# algorithms: maddpg 
# ==========================================================================================================================================================

import os
import shutil
import time
import numpy as np
import torch as T
import pandas as pd
from maddpg import maddpg
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from utils import make_env, complete_graph_edge_index, update_noise_exponential_decay
																			
# general options
AGENT_MODEL 												= "maddpg" 
ADVER_MODEL 												= "maddpg"
AGENT_MODE 													= "train"
ADVER_MODE 													= "train"
TRAINING_NAME 												= "agent_" + AGENT_MODEL + "_" + AGENT_MODE + "_vs_opp_"  + ADVER_MODEL + "_" + ADVER_MODE + "_1"
TENSORBOARD_LOG_DIRECTORY 									= "tensorboard_log" + '/' + TRAINING_NAME
CSV_LOG_DIRECTORY											= "csv_log" 
NUMBER_OF_EPISODES 											= 50000
EPISODE_TIME_LIMIT											= 60
RENDER_ENV													= True
SAVE_MODEL_RATE 											= 10

# env and drone options
POSITION_DIMENSIONS  										= 2
COMMUNICATION_DIMENSIONS  									= 1
NUMBER_OF_AGENT_DRONES 										= 1
NUMBER_OF_ADVER_DRONES	 									= 1
NUMBER_OF_LANDMARKS											= 4
RESTRICTED_RADIUS 											= 0.2
INTERCEPT_RADIUS 											= 0.7
RADAR_NOISE_POSITION										= 0.1
RADAR_NOISE_VELOCITY										= 0.25
BIG_REWARD_CONSTANT											= 10.0
REWARD_MULTIPLIER_CONSTANT									= 2.0
LANDMARK_SIZE												= 0.05
EXPONENTIAL_NOISE_DECAY										= True
EXPONENTIAL_NOISE_DECAY_CONSTANT							= 0.0002

AGENT_DRONE_RADIUS											= 0.2
AGENT_DRONE_SIZE      										= 0.025
AGENT_DRONE_DENSITY											= 25.0
AGENT_DRONE_INITIAL_MASS									= 1.0
AGENT_DRONE_ACCEL											= 4
AGENT_DRONE_MAX_SPEED										= 1.0
AGENT_DRONE_COLLIDE											= True
AGENT_DRONE_SILENT											= False												
AGENT_DRONE_U_NOISE											= 5.0
AGENT_DRONE_C_NOISE											= 3.0
AGENT_DRONE_U_RANGE											= 1.0

ADVER_DRONE_RADIUS											= 0.2
ADVER_DRONE_SIZE      										= 0.025
ADVER_DRONE_DENSITY											= 25.0
ADVER_DRONE_INITIAL_MASS									= 1.0
ADVER_DRONE_ACCEL											= 4
ADVER_DRONE_MAX_SPEED										= 1.0
ADVER_DRONE_COLLIDE											= True
ADVER_DRONE_SILENT											= False												
ADVER_DRONE_U_NOISE											= 5.0
ADVER_DRONE_C_NOISE											= 3.0
ADVER_DRONE_U_RANGE											= 1.0

# maddpg options for agent
AGENT_MADDPG_DISCOUNT_RATE 									= 0.99
AGENT_MADDPG_LEARNING_RATE_ACTOR 							= 0.001
AGENT_MADDPG_LEARNING_RATE_CRITIC 							= 0.001
AGENT_MADDPG_ACTOR_DROPOUT									= 0
AGENT_MADDPG_CRITIC_DROPOUT									= 0
AGENT_MADDPG_TAU 											= 0.01	  
AGENT_MADDPG_MEMORY_SIZE 									= 100000
AGENT_MADDPG_BATCH_SIZE 									= 64
AGENT_MADDPG_UPDATE_TARGET 									= None
AGENT_GRADIENT_CLIPPING										= True
AGENT_GRADIENT_NORM_CLIP									= 2

AGENT_MADDPG_ACTOR_INPUT_DIMENSIONS 						= [(NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2 
															   + COMMUNICATION_DIMENSIONS)) for i in range(NUMBER_OF_AGENT_DRONES)]
AGENT_MADDPG_ACTOR_OUTPUT_DIMENSIONS						= [128, 128, 128]
AGENT_MADDPG_U_ACTIONS_DIMENSIONS							= POSITION_DIMENSIONS
AGENT_MADDPG_C_ACTIONS_DIMENSIONS							= COMMUNICATION_DIMENSIONS
AGENT_MADDPG_ACTIONS_DIMENSIONS 							= AGENT_MADDPG_U_ACTIONS_DIMENSIONS + AGENT_MADDPG_C_ACTIONS_DIMENSIONS 

AGENT_MADDPG_CRITIC_GNN_INPUT_DIMS							= [(NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2 
															   + COMMUNICATION_DIMENSIONS)) for i in range(NUMBER_OF_AGENT_DRONES)]
AGENT_MADDPG_CRITIC_GNN_NUM_HEADS							= 4
AGENT_MADDPG_CRITIC_BOOL_CONCAT								= True
AGENT_MADDPG_CRITIC_GNN_OUTPUT_DIMS							= [128, 128, 128]

AGENT_MADDPG_CRITIC_GMT_HIDDEN_DIMS							= 128		
AGENT_MADDPG_CRITIC_GMT_OUTPUT_DIMS							= 128			

AGENT_MADDPG_CRITIC_U_ACTIONS_FC_INPUT_DIMS					= POSITION_DIMENSIONS * NUMBER_OF_AGENT_DRONES
AGENT_MADDPG_CRITIC_C_ACTIONS_FC_INPUT_DIMS					= COMMUNICATION_DIMENSIONS * NUMBER_OF_AGENT_DRONES
AGENT_MADDPG_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS				= [64, 64]
AGENT_MADDPG_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS				= [64, 64]
AGENT_MADDPG_CRITIC_CONCAT_FC_OUTPUT_DIMS 					= [128, 128]

# maddpg options for adversary
ADVER_MADDPG_DISCOUNT_RATE 									= 0.99
ADVER_MADDPG_LEARNING_RATE_ACTOR 							= 0.001
ADVER_MADDPG_LEARNING_RATE_CRITIC 							= 0.001
ADVER_MADDPG_ACTOR_DROPOUT									= 0
ADVER_MADDPG_CRITIC_DROPOUT									= 0
ADVER_MADDPG_TAU 											= 0.01	  
ADVER_MADDPG_MEMORY_SIZE 									= 100000
ADVER_MADDPG_BATCH_SIZE 									= 64
ADVER_MADDPG_UPDATE_TARGET 									= None
ADVER_GRADIENT_CLIPPING										= True
ADVER_GRADIENT_NORM_CLIP									= 2

ADVER_MADDPG_ACTOR_INPUT_DIMENSIONS 						= [(NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2 
															   + COMMUNICATION_DIMENSIONS)) for i in range(NUMBER_OF_ADVER_DRONES)]
ADVER_MADDPG_ACTOR_OUTPUT_DIMENSIONS						= [128, 128, 128]
ADVER_MADDPG_U_ACTIONS_DIMENSIONS							= POSITION_DIMENSIONS
ADVER_MADDPG_C_ACTIONS_DIMENSIONS							= COMMUNICATION_DIMENSIONS
ADVER_MADDPG_ACTIONS_DIMENSIONS 							= ADVER_MADDPG_U_ACTIONS_DIMENSIONS + ADVER_MADDPG_C_ACTIONS_DIMENSIONS 

ADVER_MADDPG_CRITIC_GNN_INPUT_DIMS							= [(NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2 
															   + COMMUNICATION_DIMENSIONS)) for i in range(NUMBER_OF_ADVER_DRONES)]
ADVER_MADDPG_CRITIC_GNN_NUM_HEADS							= 4
ADVER_MADDPG_CRITIC_BOOL_CONCAT								= True
ADVER_MADDPG_CRITIC_GNN_OUTPUT_DIMS							= [128, 128, 128]

ADVER_MADDPG_CRITIC_GMT_HIDDEN_DIMS							= 128		
ADVER_MADDPG_CRITIC_GMT_OUTPUT_DIMS							= 128			

ADVER_MADDPG_CRITIC_U_ACTIONS_FC_INPUT_DIMS					= POSITION_DIMENSIONS * NUMBER_OF_ADVER_DRONES 
ADVER_MADDPG_CRITIC_C_ACTIONS_FC_INPUT_DIMS					= COMMUNICATION_DIMENSIONS * NUMBER_OF_ADVER_DRONES 
ADVER_MADDPG_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS				= [64, 64]
ADVER_MADDPG_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS				= [64, 64]
ADVER_MADDPG_CRITIC_CONCAT_FC_OUTPUT_DIMS 					= [128, 128]

def train_test():

	""" function to execute experiments to train or test models based on different algorithms """

	# check agent model
	if AGENT_MODEL == "maddpg":

		# generate maddpg agents for agent drones
		agent_maddpg_agents = maddpg(mode = AGENT_MODE, training_name = TRAINING_NAME, discount_rate = AGENT_MADDPG_DISCOUNT_RATE, lr_actor = AGENT_MADDPG_LEARNING_RATE_ACTOR, 
									 lr_critic = AGENT_MADDPG_LEARNING_RATE_CRITIC, num_agents = NUMBER_OF_AGENT_DRONES, num_opp = NUMBER_OF_ADVER_DRONES, 
									 actor_dropout_p = AGENT_MADDPG_ACTOR_DROPOUT, critic_dropout_p = AGENT_MADDPG_CRITIC_DROPOUT, state_fc_input_dims = AGENT_MADDPG_ACTOR_INPUT_DIMENSIONS, 
									 state_fc_output_dims = AGENT_MADDPG_ACTOR_OUTPUT_DIMENSIONS, u_action_dims = AGENT_MADDPG_U_ACTIONS_DIMENSIONS, 
									 c_action_dims = AGENT_MADDPG_C_ACTIONS_DIMENSIONS, num_heads = AGENT_MADDPG_CRITIC_GNN_NUM_HEADS, bool_concat = AGENT_MADDPG_CRITIC_BOOL_CONCAT, 
									 gnn_input_dims = AGENT_MADDPG_CRITIC_GNN_INPUT_DIMS, gnn_output_dims = AGENT_MADDPG_CRITIC_GNN_INPUT_DIMS, gmt_hidden_dims = AGENT_MADDPG_CRITIC_GMT_HIDDEN_DIMS, 
									 gmt_output_dims = AGENT_MADDPG_CRITIC_GMT_OUTPUT_DIMS, u_actions_fc_input_dims = AGENT_MADDPG_CRITIC_U_ACTIONS_FC_INPUT_DIMS, 
									 u_actions_fc_output_dims = AGENT_MADDPG_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS, c_actions_fc_input_dims = AGENT_MADDPG_CRITIC_C_ACTIONS_FC_INPUT_DIMS, 
									 c_actions_fc_output_dims = AGENT_MADDPG_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS, concat_fc_output_dims = AGENT_MADDPG_CRITIC_CONCAT_FC_OUTPUT_DIMS, 
									 tau = AGENT_MADDPG_TAU, mem_size = AGENT_MADDPG_MEMORY_SIZE, batch_size = AGENT_MADDPG_BATCH_SIZE, update_target = AGENT_MADDPG_UPDATE_TARGET, 
									 grad_clipping = AGENT_GRADIENT_CLIPPING, grad_norm_clip = AGENT_GRADIENT_NORM_CLIP)
	
	# check adversarial model
	if ADVER_MODEL == "maddpg":

		# generate maddpg agents for adversarial drones 
		adver_maddpg_agents = maddpg(mode = ADVER_MODE, training_name = TRAINING_NAME, discount_rate = ADVER_MADDPG_DISCOUNT_RATE, lr_actor = ADVER_MADDPG_LEARNING_RATE_ACTOR, 
								   	 lr_critic = ADVER_MADDPG_LEARNING_RATE_CRITIC, num_agents = NUMBER_OF_AGENT_DRONES, num_opp = NUMBER_OF_ADVER_DRONES, 
								   	 actor_dropout_p = ADVER_MADDPG_ACTOR_DROPOUT, critic_dropout_p = ADVER_MADDPG_CRITIC_DROPOUT, state_fc_input_dims = ADVER_MADDPG_ACTOR_INPUT_DIMENSIONS, 
								   	 state_fc_output_dims = ADVER_MADDPG_ACTOR_OUTPUT_DIMENSIONS, u_action_dims = ADVER_MADDPG_U_ACTIONS_DIMENSIONS, 
								   	 c_action_dims = ADVER_MADDPG_C_ACTIONS_DIMENSIONS, num_heads = ADVER_MADDPG_CRITIC_GNN_NUM_HEADS, bool_concat = ADVER_MADDPG_CRITIC_BOOL_CONCAT, 
								   	 gnn_input_dims = ADVER_MADDPG_CRITIC_GNN_INPUT_DIMS, gnn_output_dims = ADVER_MADDPG_CRITIC_GNN_INPUT_DIMS, gmt_hidden_dims = ADVER_MADDPG_CRITIC_GMT_HIDDEN_DIMS, 
								   	 gmt_output_dims = ADVER_MADDPG_CRITIC_GMT_OUTPUT_DIMS, u_actions_fc_input_dims = ADVER_MADDPG_CRITIC_U_ACTIONS_FC_INPUT_DIMS, 
								   	 u_actions_fc_output_dims = ADVER_MADDPG_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS, c_actions_fc_input_dims = ADVER_MADDPG_CRITIC_C_ACTIONS_FC_INPUT_DIMS, 
								   	 c_actions_fc_output_dims = ADVER_MADDPG_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS, concat_fc_output_dims = ADVER_MADDPG_CRITIC_CONCAT_FC_OUTPUT_DIMS, 
								   	 tau = ADVER_MADDPG_TAU, mem_size = ADVER_MADDPG_MEMORY_SIZE, batch_size = ADVER_MADDPG_BATCH_SIZE, update_target = ADVER_MADDPG_UPDATE_TARGET, 
								   	 grad_clipping = ADVER_GRADIENT_CLIPPING, grad_norm_clip = ADVER_GRADIENT_NORM_CLIP)

	# generate environment during evaluation
	env = make_env(scenario_name = "zone_def", dim_c = COMMUNICATION_DIMENSIONS, num_good_agents = NUMBER_OF_AGENT_DRONES, num_adversaries = NUMBER_OF_ADVER_DRONES, 
				   num_landmarks = NUMBER_OF_LANDMARKS, r_rad = RESTRICTED_RADIUS, i_rad = INTERCEPT_RADIUS, r_noise_pos = RADAR_NOISE_POSITION, r_noise_vel = RADAR_NOISE_VELOCITY, 
				   big_rew_cnst = BIG_REWARD_CONSTANT, rew_multiplier_cnst = REWARD_MULTIPLIER_CONSTANT, ep_time_limit = EPISODE_TIME_LIMIT, 
				   drone_radius = [AGENT_DRONE_RADIUS, ADVER_DRONE_RADIUS], agent_size = [AGENT_DRONE_SIZE, ADVER_DRONE_SIZE], agent_density = [AGENT_DRONE_DENSITY, ADVER_DRONE_DENSITY], 
				   agent_initial_mass = [AGENT_DRONE_INITIAL_MASS, ADVER_DRONE_INITIAL_MASS], agent_accel = [AGENT_DRONE_ACCEL, ADVER_DRONE_ACCEL], 
				   agent_max_speed = [AGENT_DRONE_MAX_SPEED, ADVER_DRONE_MAX_SPEED], agent_collide = [AGENT_DRONE_COLLIDE, ADVER_DRONE_COLLIDE], 
				   agent_silent = [AGENT_DRONE_SILENT, ADVER_DRONE_SILENT], agent_u_noise = [AGENT_DRONE_U_NOISE, ADVER_DRONE_U_NOISE], 
				   agent_c_noise = [AGENT_DRONE_C_NOISE, ADVER_DRONE_C_NOISE], agent_u_range = [AGENT_DRONE_U_RANGE, ADVER_DRONE_U_RANGE], landmark_size = LANDMARK_SIZE, benchmark = True)

	# if log directory for tensorboard exist
	if os.path.exists(TENSORBOARD_LOG_DIRECTORY):
		
		# remove entire directory
		shutil.rmtree(TENSORBOARD_LOG_DIRECTORY)

	# generate writer for tensorboard logging
	writer = SummaryWriter(log_dir = TENSORBOARD_LOG_DIRECTORY)

	# variables to track the sum of agent and opp wins
	sum_agent_wins = 0
	sum_adver_wins = 0
	sum_agent_exceed_screen = 0 
	sum_adver_exceed_screen = 0 

	# list to store metrics to be converted to csv for postprocessing
	sum_agent_wins_list = []
	sum_adver_wins_list = []
	sum_agent_exceed_screen_list = [] 
	sum_adver_exceed_screen_list = []
	sum_agent_number_of_team_collisions_list = []
	sum_agent_number_of_oppo_collisions_list = []
	sum_adver_number_of_team_collisions_list = []
	sum_adver_number_of_oppo_collisions_list = [] 
	avg_agent_actor_loss_list = []
	avg_agent_critic_loss_list = []
	avg_adver_actor_loss_list = []
	avg_adver_critic_loss_list = []
	avg_agent_number_of_team_collisions_list = []
	avg_agent_number_of_oppo_collisions_list = []
	avg_adver_number_of_team_collisions_list = []
	avg_adver_number_of_oppo_collisions_list = []

	# generate edge_index for complete graph for gnn for critic models
	agent_edge_index = complete_graph_edge_index(num_nodes = NUMBER_OF_AGENT_DRONES)
	adver_edge_index = complete_graph_edge_index(num_nodes = NUMBER_OF_ADVER_DRONES)

	# iterate over number of episodes
	for eps in range(1, NUMBER_OF_EPISODES): 

		# boolean to check if episode is terminal
		is_terminal = 0

		# variable to track terminal condition
		terminal_condition = 0

		# print episode number 
		print("episode " + str(eps) + ":") 

		# obtain states of agent and adverserial agents
		actor_states = env.reset()

		# set episode start time
		env.world.ep_start_time = time.time()

		# check if exponential decay for noise is desired
		if EXPONENTIAL_NOISE_DECAY == True:

			update_noise_exponential_decay(env = env, expo_decay_cnst = EXPONENTIAL_NOISE_DECAY_CONSTANT, num_adver = NUMBER_OF_ADVER_DRONES, eps_timestep = eps, 
										   agent_u_noise_cnst = AGENT_DRONE_U_NOISE, agent_c_noise_cnst = AGENT_DRONE_C_NOISE, adver_u_noise_cnst = ADVER_DRONE_U_NOISE, 
										   adver_c_noise_cnst = ADVER_DRONE_C_NOISE)

		# obtain numpy array of actor_states, adver_actor_states, agent_actor_states
		actor_states = np.array(actor_states)
		adver_actor_states = np.array(actor_states[:NUMBER_OF_ADVER_DRONES])
		agent_actor_states = np.array(actor_states[NUMBER_OF_ADVER_DRONES:])

		# iterate till episode terminates
		while is_terminal == 0:

			# check if environment is required to be rendered
			if RENDER_ENV == True:

				# render env
				env.render()

			# obtain actions for agent_maddpg_agents  
			if AGENT_MODEL == "maddpg":

				# obtain motor and communication actions for agent drones
				# mode is always 'test' as the environment handles the addition of noise to the actions
				agent_u_actions, agent_c_actions, agent_actions_list  = agent_maddpg_agents.select_actions(mode = "test", env_agents = env.agents[NUMBER_OF_ADVER_DRONES:], 
																										   actor_state_list = agent_actor_states)

			# obtain actions for agent_maddpg_agents 
			if ADVER_MODEL == "maddpg":

				# obtain actions from fc_state and cam_state for all opp drones
				# mode is always 'test' as the environment handles the addition of noise to the actions
				adver_u_actions, adver_c_actions, adver_actions_list = adver_maddpg_agents.select_actions(mode = "test", env_agents = env.agents[:NUMBER_OF_ADVER_DRONES], 
																										  actor_state_list = adver_actor_states)

			# iterate over agent_maddpg_agents
			for i in range(NUMBER_OF_AGENT_DRONES):

				# append agent drones actions to adversarial drones actions
				adver_actions_list.append(agent_actions_list[i])

			# update state of the world and obtain information of the updated state
			actor_states_prime, rewards, terminates_p_terminal_con, benchmark_data = env.step(adver_actions_list)
			
			# obtain numpy array of actor_states_prime, adver_actor_states_prime, agent_actor_states_prime, adver_rewards, agent_rewards
			actor_states_prime = np.array(actor_states_prime)
			adver_actor_states_prime = np.array(actor_states_prime[:NUMBER_OF_ADVER_DRONES])
			agent_actor_states_prime = np.array(actor_states_prime[NUMBER_OF_ADVER_DRONES:])
			adver_rewards = np.array(rewards[:NUMBER_OF_ADVER_DRONES])
			agent_rewards = np.array(rewards[NUMBER_OF_ADVER_DRONES:])
			adver_benchmark_data = np.array(benchmark_data['n'][:NUMBER_OF_ADVER_DRONES])
			agent_benchmark_data = np.array(benchmark_data['n'][NUMBER_OF_ADVER_DRONES:])

			# empty list for adver_terminates, agent_terminates, terminal_con
			adver_terminates = []
			agent_terminates = []
			terminal_con = []

			# iterate over all drones
			for i in range(NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES):

				# check for adversarial drones
				if i < NUMBER_OF_ADVER_DRONES:

					# append terminates and terminal_con
					adver_terminates.append(terminates_p_terminal_con[i][0])
					terminal_con.append(terminates_p_terminal_con[i][1])

					# check if episode has terminated and is_terminal is false
					if adver_terminates[i] == True and is_terminal == False:

						# update is_terminal
						is_terminal = True

						# obtain corresponding terminal condition
						terminal_condition = terminal_con[i]

				# check for agent drones
				if i >= NUMBER_OF_ADVER_DRONES:

					# append terminates and terminal_con
					agent_terminates.append(terminates_p_terminal_con[i - NUMBER_OF_ADVER_DRONES][0])
					terminal_con.append(terminates_p_terminal_con[i][1])

					# check if episode has terminated and is_terminal is false
					if agent_terminates[i - NUMBER_OF_ADVER_DRONES] == True and is_terminal == False:

						# update is_terminal
						is_terminal = True

						# obtain corresponding terminal condition
						terminal_condition = terminal_con[i]

			# obtain numpy array of adver_terminates, agent_terminates, terminal_con
			adver_terminates = np.array(adver_terminates, dtype = bool)
			agent_terminates = np.array(agent_terminates, dtype = bool)
			terminal_con = np.array(terminal_con)

			# for maddpg agent drones to store memory in replay buffer and train model and check if training agent model
			if AGENT_MODEL == "maddpg" and AGENT_MODE != "test":

				# obtain agent_critic_states and agent_critic_states_prime in gnn data format
				agent_critic_states = Data(x = T.tensor(agent_actor_states, dtype = T.float), edge_index = T.tensor(agent_edge_index, dtype = T.long).t().contiguous())
				agent_critic_states_prime = Data(x = T.tensor(agent_actor_states_prime, dtype = T.float), edge_index = T.tensor(agent_edge_index, dtype = T.long).t().contiguous())

				# set num_nodes for agent_critic_states, agent_critic_states_prime
				agent_critic_states.num_nodes = NUMBER_OF_AGENT_DRONES 
				agent_critic_states_prime.num_nodes = NUMBER_OF_AGENT_DRONES

				# store states and actions in replay buffer
				agent_maddpg_agents.replay_buffer.log(actor_state = agent_actor_states, actor_state_prime = agent_actor_states_prime, critic_state = agent_critic_states, 
													  critic_state_prime = agent_critic_states_prime, u_action = agent_u_actions, c_action = agent_c_actions, rewards = agent_rewards, 
													  is_done = agent_terminates)

				# train agent models and obtain actor and critic loss for each agent drone for logging
				agent_actor_loss_list, agent_critic_loss_list = agent_maddpg_agents.apply_gradients_maddpg()

			else:

				agent_actor_loss_list, agent_critic_loss_list = np.nan, np.nan

			# for maddpg adversarial drones to store memory in replay buffer and check if training adversarial model
			if ADVER_MODEL == "maddpg" and ADVER_MODE != "test":

				# obtain adver_critic_states and adver_critic_states_prime in gnn data format
				adver_critic_states = Data(x = T.tensor(adver_actor_states, dtype = T.float), edge_index = T.tensor(adver_edge_index, dtype = T.long).t().contiguous())
				adver_critic_states_prime = Data(x = T.tensor(adver_actor_states_prime, dtype = T.float), edge_index = T.tensor(adver_edge_index, dtype = T.long).t().contiguous())

				# set num_nodes for adver_critic_states, adver_critic_states_prime
				adver_critic_states.num_nodes = NUMBER_OF_ADVER_DRONES
				adver_critic_states_prime.num_nodes = NUMBER_OF_ADVER_DRONES
				
				# store states and actions in replay buffer
				adver_maddpg_agents.replay_buffer.log(actor_state = adver_actor_states, actor_state_prime = adver_actor_states_prime, critic_state = adver_critic_states, 
													  critic_state_prime = adver_critic_states_prime, u_action = adver_u_actions, c_action = adver_c_actions, rewards = adver_rewards, 
													  is_done = adver_terminates)

				# train adversarial models and obtain actor and critic loss for each adversarial drone for logging
				adver_actor_loss_list, adver_critic_loss_list = adver_maddpg_agents.apply_gradients_maddpg()

			else:

				adver_actor_loss_list, adver_critic_loss_list = np.nan, np.nan

			# log metrics for agent_model
			if AGENT_MODEL == "maddpg":

				# variables to track sum of actor and critic loss
				sum_agent_actor_loss = 0
				sum_agent_critic_loss = 0
				sum_agent_number_of_team_collisions = 0
				sum_agent_number_of_oppo_collisions = 0

				# iterate over num of agent drones
				for i in range(NUMBER_OF_AGENT_DRONES): 

					# check if list is not nan and agent model is training
					if np.isnan(agent_actor_loss_list) == False and np.isnan(agent_critic_loss_list) == False and AGENT_MODE != "test":

						# update actor and critic loss sums
						sum_agent_actor_loss += agent_actor_loss_list[i]
						sum_agent_critic_loss += agent_critic_loss_list[i]

						# add actor and critic loss for agent drone
						writer.add_scalar(tag = "agent_drone_" + str(i + 1) + "/actor_loss", scalar_value = agent_actor_loss_list[i], global_step = eps)
						writer.add_scalar(tag = "agent_drone_" + str(i + 1) + "/critic_loss", scalar_value = agent_critic_loss_list[i], global_step = eps)

					# update sum of team and opponent collisions
					sum_agent_number_of_team_collisions += agent_benchmark_data[i, 0]
					sum_agent_number_of_oppo_collisions += agent_benchmark_data[i, 1]

					# add actor and critic loss for agent drone
					writer.add_scalar(tag = "agent_drone_" + str(i + 1) + "/team_collisions", scalar_value = agent_benchmark_data[i, 0], global_step = eps)
					writer.add_scalar(tag = "agent_drone_" + str(i + 1) + "/opponent_collisions", scalar_value = agent_benchmark_data[i, 1], global_step = eps)

				# check if agent model is training
				if AGENT_MODE != "test":

					# obtain avg actor and critic loss
					avg_agent_actor_loss = sum_agent_actor_loss / float(NUMBER_OF_AGENT_DRONES)
					avg_agent_critic_loss = sum_agent_critic_loss / float(NUMBER_OF_AGENT_DRONES)

					# add avg actor and critic loss for agent drones
					writer.add_scalar(tag = "avg_agent_actor_loss", scalar_value = avg_agent_actor_loss, global_step = eps)
					writer.add_scalar(tag = "avg_agent_critic_loss", scalar_value = avg_agent_critic_loss, global_step = eps)

					# append avg_agent_actor_loss and avg_agent_critic_loss to their respective list
					avg_agent_actor_loss_list.append(avg_agent_actor_loss)
					avg_agent_critic_loss_list.append(avg_agent_critic_loss)

				# add sum team and oppo collisions for agent drones
				writer.add_scalar(tag = "sum_agent_number_of_team_collisions", scalar_value = sum_agent_number_of_team_collisions, global_step = eps)
				writer.add_scalar(tag = "sum_agent_number_of_oppo_collisions", scalar_value = sum_agent_number_of_oppo_collisions, global_step = eps)

				# append sum_agent_number_of_team_collisisum and sum_agent_number_of_oppo_collisions to their respective list
				sum_agent_number_of_team_collisions_list.append(sum_agent_number_of_team_collisions)
				sum_agent_number_of_oppo_collisions_list.append(sum_agent_number_of_oppo_collisions)

				# obtain avg team and oppo collisions 
				avg_agent_number_of_team_collisions = sum_agent_number_of_team_collisions / float(NUMBER_OF_AGENT_DRONES)
				avg_agent_number_of_oppo_collisions = sum_agent_number_of_oppo_collisions / float(NUMBER_OF_AGENT_DRONES)

				# add avg team and oppo collisions for agent drones
				writer.add_scalar(tag = "avg_agent_number_of_team_collisions", scalar_value = avg_agent_number_of_team_collisions, global_step = eps)
				writer.add_scalar(tag = "avg_agent_number_of_oppo_collisions", scalar_value = avg_agent_number_of_oppo_collisions, global_step = eps)

				# append avg_agent_number_of_team_collisions and avg_agent_number_of_oppo_collisions to their respective list
				avg_agent_number_of_team_collisions_list.append(avg_agent_number_of_team_collisions)
				avg_agent_number_of_oppo_collisions_list.append(avg_agent_number_of_oppo_collisions)

			# log metrics for adver_model
			if ADVER_MODEL == "maddpg":

				# variables to track sum of actor and critic loss
				sum_adver_actor_loss = 0
				sum_adver_critic_loss = 0
				sum_adver_number_of_team_collisions = 0
				sum_adver_number_of_oppo_collisions = 0

				# iterate over num of adver drones
				for i in range(NUMBER_OF_ADVER_DRONES): 

					# check if list is not nan and adversarial model is training
					if np.isnan(adver_actor_loss_list) == False and np.isnan(adver_critic_loss_list) == False and ADVER_MODE != "test":

						# update actor and critic loss sums
						sum_adver_actor_loss += adver_actor_loss_list[i]
						sum_adver_critic_loss += adver_critic_loss_list[i]

						# add actor and critic loss for adver drone
						writer.add_scalar(tag = "adver_drone_" + str(i + 1) + "/actor_loss", scalar_value = adver_actor_loss_list[i], global_step = eps)
						writer.add_scalar(tag = "adver_drone_" + str(i + 1) + "/critic_loss", scalar_value = adver_critic_loss_list[i], global_step = eps)

					# update sum of team and opponent collisions
					sum_adver_number_of_team_collisions += adver_benchmark_data[i, 0]
					sum_adver_number_of_oppo_collisions += adver_benchmark_data[i, 1]

					# add actor and critic loss for adver drone
					writer.add_scalar(tag = "adver_drone_" + str(i + 1) + "/team_collisions", scalar_value = adver_benchmark_data[i, 0], global_step = eps)
					writer.add_scalar(tag = "adver_drone_" + str(i + 1) + "/opponent_collisions", scalar_value = adver_benchmark_data[i, 1], global_step = eps)

				# check if adver model is training
				if AGENT_MODE != "test":

					# obtain avg actor and critic loss
					avg_adver_actor_loss = sum_adver_actor_loss / float(NUMBER_OF_ADVER_DRONES)
					avg_adver_critic_loss = sum_adver_critic_loss / float(NUMBER_OF_ADVER_DRONES)

					# add avg actor and critic loss for adver drones
					writer.add_scalar(tag = "avg_adver_actor_loss", scalar_value = avg_adver_actor_loss, global_step = eps)
					writer.add_scalar(tag = "avg_adver_critic_loss", scalar_value = avg_adver_critic_loss, global_step = eps)

					# append avg_adver_actor_loss and avg_adver_critic_loss to their respective list
					avg_adver_actor_loss_list.append(avg_adver_actor_loss)
					avg_adver_critic_loss_list.append(avg_adver_critic_loss)
				
				# add sum team and oppo collisions for adver drones
				writer.add_scalar(tag = "sum_adver_number_of_team_collisions", scalar_value = sum_adver_number_of_team_collisions, global_step = eps)
				writer.add_scalar(tag = "sum_adver_number_of_oppo_collisions", scalar_value = sum_adver_number_of_oppo_collisions, global_step = eps)

				# append sum_adver_number_of_team_collisisum and sum_adver_number_of_oppo_collisions to their respective list
				sum_adver_number_of_team_collisions_list.append(sum_adver_number_of_team_collisions)
				sum_adver_number_of_oppo_collisions_list.append(sum_adver_number_of_oppo_collisions)

				# obtain avg team and oppo collisions 
				avg_adver_number_of_team_collisions = sum_adver_number_of_team_collisions / float(NUMBER_OF_ADVER_DRONES)
				avg_adver_number_of_oppo_collisions = sum_adver_number_of_oppo_collisions / float(NUMBER_OF_ADVER_DRONES)

				# add avg team and oppo collisions for adver drones
				writer.add_scalar(tag = "avg_adver_number_of_team_collisions", scalar_value = avg_adver_number_of_team_collisions, global_step = eps)
				writer.add_scalar(tag = "avg_adver_number_of_oppo_collisions", scalar_value = avg_adver_number_of_oppo_collisions, global_step = eps)

				# append avg_adver_number_of_team_collisions and avg_adver_number_of_oppo_collisions to their respective list
				avg_adver_number_of_team_collisions_list.append(avg_adver_number_of_team_collisions)
				avg_adver_number_of_oppo_collisions_list.append(avg_adver_number_of_oppo_collisions)

			# log wins for agent drones
			if terminal_condition == 1:

				# add sum of wins for agent drones
				sum_agent_wins += 1

				# append sums to all lists
				sum_agent_wins_list.append(sum_agent_wins)
				sum_adver_wins_list.append(sum_adver_wins)
				sum_agent_exceed_screen_list.append(sum_agent_exceed_screen)
				sum_adver_exceed_screen_list.append(sum_adver_exceed_screen)

				# add opp win and sum of agent win
				writer.add_scalar(tag = "terminate_info/agent_drone_win", scalar_value = 1, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_agent_drone_wins", scalar_value = sum_agent_wins, global_step = eps)

				# add opp win and sum of opp win
				writer.add_scalar(tag = "terminate_info/adver_drone_win", scalar_value = 0, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_adver_drone_wins", scalar_value = sum_adver_wins, global_step = eps)

				# add agent_exceed_screen and sum_agent_exceed_screen
				writer.add_scalar(tag = "terminate_info/agent_exceed_screen", scalar_value = 0, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_agent_exceed_screen", scalar_value = sum_agent_exceed_screen, global_step = eps)

				# add agent_exceed_screen and sum_agent_exceed_screen
				writer.add_scalar(tag = "terminate_info/adver_exceed_screen", scalar_value = 0, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_adver_exceed_screen", scalar_value = sum_adver_exceed_screen, global_step = eps)

				# print log
				print(f'terminal_condition {terminal_condition}: agent drones win')

			# log wins for adversarial drones
			elif terminal_condition == 2:

				# add sum of wins for adversarial drones
				sum_adver_wins += 1

				# append sums to all lists
				sum_agent_wins_list.append(sum_agent_wins)
				sum_adver_wins_list.append(sum_adver_wins)
				sum_agent_exceed_screen_list.append(sum_agent_exceed_screen)
				sum_adver_exceed_screen_list.append(sum_adver_exceed_screen)

				# add opp win and sum of agent win
				writer.add_scalar(tag = "terminate_info/agent_drone_win", scalar_value = 0, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_agent_drone_wins", scalar_value = sum_agent_wins, global_step = eps)

				# add opp win and sum of opp win
				writer.add_scalar(tag = "terminate_info/adver_drone_win", scalar_value = 1, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_adver_drone_wins", scalar_value = sum_adver_wins, global_step = eps)

				# add agent_exceed_screen and sum_agent_exceed_screen
				writer.add_scalar(tag = "terminate_info/agent_exceed_screen", scalar_value = 0, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_agent_exceed_screen", scalar_value = sum_agent_exceed_screen, global_step = eps)

				# add agent_exceed_screen and sum_agent_exceed_screen
				writer.add_scalar(tag = "terminate_info/adver_exceed_screen", scalar_value = 0, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_adver_exceed_screen", scalar_value = sum_adver_exceed_screen, global_step = eps)

				# print log
				print(f'terminal_condition {terminal_condition}: adversarial drones win')

			# log agent drones exceeding screen boundary
			elif terminal_condition == 3:

				# add sum_agent_exceed_screen
				sum_agent_exceed_screen += 1

				# append sums to all lists
				sum_agent_wins_list.append(sum_agent_wins)
				sum_adver_wins_list.append(sum_adver_wins)
				sum_agent_exceed_screen_list.append(sum_agent_exceed_screen)
				sum_adver_exceed_screen_list.append(sum_adver_exceed_screen)

				# add opp win and sum of agent win
				writer.add_scalar(tag = "terminate_info/agent_drone_win", scalar_value = 0, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_agent_drone_wins", scalar_value = sum_agent_wins, global_step = eps)

				# add opp win and sum of opp win
				writer.add_scalar(tag = "terminate_info/adver_drone_win", scalar_value = 0, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_adver_drone_wins", scalar_value = sum_adver_wins, global_step = eps)

				# add agent_exceed_screen and sum_agent_exceed_screen
				writer.add_scalar(tag = "terminate_info/agent_exceed_screen", scalar_value = 1, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_agent_exceed_screen", scalar_value = sum_agent_exceed_screen, global_step = eps)

				# add agent_exceed_screen and sum_agent_exceed_screen
				writer.add_scalar(tag = "terminate_info/adver_exceed_screen", scalar_value = 0, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_adver_exceed_screen", scalar_value = sum_adver_exceed_screen, global_step = eps)
				
				# print log
				print(f'terminal_condition {terminal_condition}: agent drones exceeded screen boundary')

			# log adversarial drones exceeding screen boundary
			elif terminal_condition == 4:

				# add sum_adver_exceed_screen
				sum_adver_exceed_screen += 1

				# append sums to all lists
				sum_agent_wins_list.append(sum_agent_wins)
				sum_adver_wins_list.append(sum_adver_wins)
				sum_agent_exceed_screen_list.append(sum_agent_exceed_screen)
				sum_adver_exceed_screen_list.append(sum_adver_exceed_screen)

				# add opp win and sum of agent win
				writer.add_scalar(tag = "terminate_info/agent_drone_win", scalar_value = 0, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_agent_drone_wins", scalar_value = sum_agent_wins, global_step = eps)

				# add opp win and sum of opp win
				writer.add_scalar(tag = "terminate_info/adver_drone_win", scalar_value = 0, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_adver_drone_wins", scalar_value = sum_adver_wins, global_step = eps)

				# add agent_exceed_screen and sum_agent_exceed_screen
				writer.add_scalar(tag = "terminate_info/agent_exceed_screen", scalar_value = 0, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_agent_exceed_screen", scalar_value = sum_agent_exceed_screen, global_step = eps)

				# add agent_exceed_screen and sum_agent_exceed_screen
				writer.add_scalar(tag = "terminate_info/adver_exceed_screen", scalar_value = 1, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_adver_exceed_screen", scalar_value = sum_adver_exceed_screen, global_step = eps)
				
				# print log
				print(f'terminal_condition {terminal_condition}: adversarial drones exceeded screen boundary')

			# update actor_states, adver_actor_states, agent_actor_states
			actor_states = actor_states_prime
			adver_actor_states = adver_actor_states_prime
			agent_actor_states = agent_actor_states_prime

			# check if agent and adversarial model are both training
			if AGENT_MODE != "test" and ADVER_MODE != "test":

				# generate pandas dataframe to store logs
				df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
										   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
										   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
										   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
										   avg_adver_actor_loss_list, avg_adver_critic_loss_list)), 
								  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 'sum_agent_number_of_team_collisions', 
								  			 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 
								  			 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 
								  			 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_adver_actor_loss', 'avg_adver_critic_loss'])

			# check if agent model is testing and adversarial model is training
			elif AGENT_MODE == "test" and ADVER_MODE != "test":

				# generate pandas dataframe to store logs
				df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list,
										   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
										   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
										   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list)), 
								  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 'sum_agent_number_of_team_collisions', 
								  			 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 
								  			 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 
								  			 'avg_adver_number_of_oppo_collisions', 'avg_adver_actor_loss', 'avg_adver_critic_loss'])

			# check if agent model is training and adversarial model is testing
			elif AGENT_MODE !=  "test" and ADVER_MODE == "test":

				# generate pandas dataframe to store logs
				df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
										   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
										   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
										   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list)), 
								  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 'sum_agent_number_of_team_collisions', 
								  			 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 
								  			 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 
								  			 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss'])

			# check if agent and adversarial model are both testing
			if AGENT_MODE ==  "test" and ADVER_MODE == "test":

				# generate pandas dataframe to store logs
				df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
										   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
										   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
										   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list)), 
								  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 'sum_agent_number_of_team_collisions', 
								  			 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 
								  			 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 
								  			 'avg_adver_number_of_oppo_collisions'])

			# store training logs
			df.to_csv(CSV_LOG_DIRECTORY + '/' + TRAINING_NAME + "_logs.csv", index = False)

		# check if agent is training and at correct episode to save
		if AGENT_MODE != "test" and eps % SAVE_MODEL_RATE == 0:

			# save all models
			agent_maddpg_agents.save_all_models()

		# check if adver is training and at correct episode to save
		if ADVER_MODE != "test" and eps % SAVE_MODEL_RATE == 0:

			# save all models
			adver_maddpg_agents.save_all_models()

if __name__ == "__main__":

	train_test()