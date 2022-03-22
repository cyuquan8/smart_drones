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
from maddpg.maddpg import maddpg
from mappo.mappo import mappo
from maddpgv2.maddpgv2 import maddpgv2
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from utils.utils import make_env, complete_graph_edge_index, update_noise_exponential_decay, calculate_elo_rating, update_agent_goals_softmax_weights, update_adver_goals_softmax_weights
																			
# general options
AGENT_MODEL 												= "maddpgv2" 
ADVER_MODEL 												= "maddpgv2"
AGENT_MODE 													= "train"
ADVER_MODE 													= "train"
GENERAL_TRAINING_NAME 										= "agent_" + AGENT_MODEL + "_vs_opp_"  + ADVER_MODEL + "_2_vs_1_1_big"
AGENT_TRAINING_NAME											= GENERAL_TRAINING_NAME + "_agent"
ADVER_TRAINING_NAME											= GENERAL_TRAINING_NAME + "_adver"
TENSORBOARD_LOG_DIRECTORY 									= "tensorboard_log" + '/' + GENERAL_TRAINING_NAME
CSV_LOG_DIRECTORY											= "csv_log" 
NUMBER_OF_EPISODES 											= 500000
EPISODE_TIME_LIMIT											= 10
RENDER_ENV													= True
SAVE_MODEL_RATE 											= 100
SAVE_CSV_LOG												= True

# elo options
INITIAL_ELO													= 1000.0
ELO_DIFFRENCE												= 1000.0
ELO_D    													= 400
RESULTS_REWARD_DICT											= {'1': [0, 1], '2': [1, 0], '3': [0.5, 0.5], '4': [0.5, 0.5]}

AGENT_TASK_DIFFICULTY_COEFFICIENT							= 1
ADVER_TASK_DIFFICULTY_COEFFICIENT							= 1
AGENT_ELO_K 												= 32 * AGENT_TASK_DIFFICULTY_COEFFICIENT
ADVER_ELO_K  												= 32 * ADVER_TASK_DIFFICULTY_COEFFICIENT

# env and drone options
POSITION_DIMENSIONS  										= 2
COMMUNICATION_DIMENSIONS  									= 1
NUMBER_OF_AGENT_DRONES 										= 2
NUMBER_OF_ADVER_DRONES	 									= 1
NUMBER_OF_LANDMARKS											= 0
RESTRICTED_RADIUS 											= 0.2
INTERCEPT_RADIUS 											= 0.7
RADAR_NOISE_POSITION										= 0.1
RADAR_NOISE_VELOCITY										= 0.5
BIG_REWARD_CONSTANT											= 10.0
REWARD_MULTIPLIER_CONSTANT									= 2.0
LANDMARK_SIZE												= 0.05
EXPONENTIAL_NOISE_DECAY										= True
EXPONENTIAL_NOISE_DECAY_CONSTANT							= 0.0002
EXIT_SCREEN_TERMINATE										= True

AGENT_DRONE_RADIUS											= 0.25
AGENT_DRONE_SIZE      										= 0.075
AGENT_DRONE_DENSITY											= 25.0
AGENT_DRONE_INITIAL_MASS									= 1.0
AGENT_DRONE_ACCEL											= 4.0
AGENT_DRONE_MAX_SPEED										= 1.0
AGENT_DRONE_COLLIDE											= True
AGENT_DRONE_SILENT											= False												
AGENT_DRONE_U_NOISE											= 0.5 
AGENT_DRONE_C_NOISE											= 0.5  
AGENT_DRONE_U_RANGE											= 1.0

ADVER_DRONE_RADIUS											= 0.25
ADVER_DRONE_SIZE      										= 0.075
ADVER_DRONE_DENSITY											= 25.0
ADVER_DRONE_INITIAL_MASS									= 1.0
ADVER_DRONE_ACCEL											= 4.0
ADVER_DRONE_MAX_SPEED										= 1.0
ADVER_DRONE_COLLIDE											= True
ADVER_DRONE_SILENT											= False												
ADVER_DRONE_U_NOISE											= 0.5   
ADVER_DRONE_C_NOISE											= 0.5  
ADVER_DRONE_U_RANGE											= 1.0

# maddpg options for agent
AGENT_MADDPG_DISCOUNT_RATE 									= 0.99
AGENT_MADDPG_LEARNING_RATE_ACTOR 							= 0.0005
AGENT_MADDPG_LEARNING_RATE_CRITIC 							= 0.0005
AGENT_MADDPG_ACTOR_DROPOUT									= 0
AGENT_MADDPG_CRITIC_DROPOUT									= 0
AGENT_MADDPG_TAU 											= 0.01	  
AGENT_MADDPG_MEMORY_SIZE 									= 10000
AGENT_MADDPG_BATCH_SIZE 									= 128
AGENT_MADDPG_UPDATE_TARGET 									= None
AGENT_MADDPG_GRADIENT_CLIPPING								= True
AGENT_MADDPG_GRADIENT_NORM_CLIP								= 1

AGENT_MADDPG_ACTOR_INPUT_DIMENSIONS 						= [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2 
															   + COMMUNICATION_DIMENSIONS)) for i in range(NUMBER_OF_AGENT_DRONES)]
AGENT_MADDPG_ACTOR_OUTPUT_DIMENSIONS						= [128, 128, 128]
AGENT_MADDPG_U_ACTIONS_DIMENSIONS							= POSITION_DIMENSIONS
AGENT_MADDPG_C_ACTIONS_DIMENSIONS							= COMMUNICATION_DIMENSIONS
AGENT_MADDPG_ACTIONS_DIMENSIONS 							= AGENT_MADDPG_U_ACTIONS_DIMENSIONS + AGENT_MADDPG_C_ACTIONS_DIMENSIONS 

AGENT_MADDPG_CRITIC_GNN_INPUT_DIMS							= [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2 
															   + COMMUNICATION_DIMENSIONS)) for i in range(NUMBER_OF_AGENT_DRONES)]
AGENT_MADDPG_CRITIC_GNN_NUM_HEADS							= 1
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
ADVER_MADDPG_LEARNING_RATE_ACTOR 							= 0.0005
ADVER_MADDPG_LEARNING_RATE_CRITIC 							= 0.0005
ADVER_MADDPG_ACTOR_DROPOUT									= 0
ADVER_MADDPG_CRITIC_DROPOUT									= 0
ADVER_MADDPG_TAU 											= 0.01	  
ADVER_MADDPG_MEMORY_SIZE 									= 10000
ADVER_MADDPG_BATCH_SIZE 									= 128
ADVER_MADDPG_UPDATE_TARGET 									= None
ADVER_MADDPG_GRADIENT_CLIPPING								= True
ADVER_MADDPG_GRADIENT_NORM_CLIP								= 1

ADVER_MADDPG_ACTOR_INPUT_DIMENSIONS 						= [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2 
															   + COMMUNICATION_DIMENSIONS)) for i in range(NUMBER_OF_ADVER_DRONES)]
ADVER_MADDPG_ACTOR_OUTPUT_DIMENSIONS						= [128, 128, 128]
ADVER_MADDPG_U_ACTIONS_DIMENSIONS							= POSITION_DIMENSIONS
ADVER_MADDPG_C_ACTIONS_DIMENSIONS							= COMMUNICATION_DIMENSIONS
ADVER_MADDPG_ACTIONS_DIMENSIONS 							= ADVER_MADDPG_U_ACTIONS_DIMENSIONS + ADVER_MADDPG_C_ACTIONS_DIMENSIONS 

ADVER_MADDPG_CRITIC_GNN_INPUT_DIMS							= [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2 
															   + COMMUNICATION_DIMENSIONS)) for i in range(NUMBER_OF_ADVER_DRONES)]
ADVER_MADDPG_CRITIC_GNN_NUM_HEADS							= 1
ADVER_MADDPG_CRITIC_BOOL_CONCAT								= True
ADVER_MADDPG_CRITIC_GNN_OUTPUT_DIMS							= [128, 128, 128]

ADVER_MADDPG_CRITIC_GMT_HIDDEN_DIMS							= 128		
ADVER_MADDPG_CRITIC_GMT_OUTPUT_DIMS							= 128			

ADVER_MADDPG_CRITIC_U_ACTIONS_FC_INPUT_DIMS					= POSITION_DIMENSIONS * NUMBER_OF_ADVER_DRONES 
ADVER_MADDPG_CRITIC_C_ACTIONS_FC_INPUT_DIMS					= COMMUNICATION_DIMENSIONS * NUMBER_OF_ADVER_DRONES 
ADVER_MADDPG_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS				= [64, 64]
ADVER_MADDPG_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS				= [64, 64]
ADVER_MADDPG_CRITIC_CONCAT_FC_OUTPUT_DIMS 					= [128, 128]

# mappo options for agent
AGENT_MAPPO_LEARNING_RATE_ACTOR 							= 0.0005
AGENT_MAPPO_LEARNING_RATE_CRITIC 							= 0.0005
AGENT_MAPPO_ACTOR_DROPOUT									= 0
AGENT_MAPPO_CRITIC_DROPOUT									= 0
AGENT_MAPPO_BATCH_SIZE 										= 20
AGENT_MAPPO_GAMMA 											= 0.99
AGENT_MAPPO_CLIP_COEFFICIENT								= 0.2
AGENT_MAPPO_NUMBER_OF_EPOCHS								= 5
AGENT_MAPPO_GAE_LAMBDA										= 0.95
AGENT_MAPPO_ENTROPY_COEFFICIENT								= 0.01
AGENT_MAPPO_USE_HUBER_LOSS									= True
AGENT_MAPPO_HUBER_DELTA										= 10.0
AGENT_MAPPO_USE_CLIPPED_VALUE_LOSS							= True
AGENT_MAPPO_CRITIC_LOSS_COEFFICIENT							= 0.5
AGENT_MAPPO_GRADIENT_CLIPPING								= True
AGENT_MAPPO_GRADIENT_NORM_CLIP								= 1
AGENT_MAPPO_EPISODE_LENGTH									= AGENT_MAPPO_BATCH_SIZE * AGENT_MAPPO_NUMBER_OF_EPOCHS

AGENT_MAPPO_ACTOR_INPUT_DIMENSIONS 							= [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2 
															   + COMMUNICATION_DIMENSIONS)) for i in range(NUMBER_OF_AGENT_DRONES)]
AGENT_MAPPO_ACTOR_OUTPUT_DIMENSIONS							= [128, 128, 128]
AGENT_MAPPO_U_ACTIONS_DIMENSIONS							= POSITION_DIMENSIONS
AGENT_MAPPO_C_ACTIONS_DIMENSIONS							= COMMUNICATION_DIMENSIONS
AGENT_MAPPO_ACTIONS_DIMENSIONS 								= AGENT_MAPPO_U_ACTIONS_DIMENSIONS + AGENT_MAPPO_C_ACTIONS_DIMENSIONS 

AGENT_MAPPO_CRITIC_GNN_INPUT_DIMS							= [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2 
															   + COMMUNICATION_DIMENSIONS)) for i in range(NUMBER_OF_AGENT_DRONES)]
AGENT_MAPPO_CRITIC_GNN_NUM_HEADS							= 1
AGENT_MAPPO_CRITIC_BOOL_CONCAT								= True
AGENT_MAPPO_CRITIC_GNN_OUTPUT_DIMS							= [128, 128, 128]

AGENT_MAPPO_CRITIC_GMT_HIDDEN_DIMS							= 128		
AGENT_MAPPO_CRITIC_GMT_OUTPUT_DIMS							= 128			
AGENT_MAPPO_CRITIC_FC_OUTPUT_DIMS 							= [128, 128]

# mappo options for adver
ADVER_MAPPO_LEARNING_RATE_ACTOR 							= 0.0005
ADVER_MAPPO_LEARNING_RATE_CRITIC 							= 0.0005
ADVER_MAPPO_ACTOR_DROPOUT									= 0
ADVER_MAPPO_CRITIC_DROPOUT									= 0
ADVER_MAPPO_BATCH_SIZE 										= 20
ADVER_MAPPO_GAMMA 											= 0.99
ADVER_MAPPO_CLIP_COEFFICIENT								= 0.2
ADVER_MAPPO_NUMBER_OF_EPOCHS								= 5
ADVER_MAPPO_GAE_LAMBDA										= 0.95
ADVER_MAPPO_ENTROPY_COEFFICIENT								= 0.01
ADVER_MAPPO_USE_HUBER_LOSS									= True
ADVER_MAPPO_HUBER_DELTA										= 10.0
ADVER_MAPPO_USE_CLIPPED_VALUE_LOSS							= True
ADVER_MAPPO_CRITIC_LOSS_COEFFICIENT							= 0.5
ADVER_MAPPO_GRADIENT_CLIPPING								= True
ADVER_MAPPO_GRADIENT_NORM_CLIP								= 1
ADVER_MAPPO_EPISODE_LENGTH									= ADVER_MAPPO_BATCH_SIZE * ADVER_MAPPO_NUMBER_OF_EPOCHS

ADVER_MAPPO_ACTOR_INPUT_DIMENSIONS 							= [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2 
															   + COMMUNICATION_DIMENSIONS)) for i in range(NUMBER_OF_ADVER_DRONES)]
ADVER_MAPPO_ACTOR_OUTPUT_DIMENSIONS							= [128, 128, 128]
ADVER_MAPPO_U_ACTIONS_DIMENSIONS							= POSITION_DIMENSIONS
ADVER_MAPPO_C_ACTIONS_DIMENSIONS							= COMMUNICATION_DIMENSIONS
ADVER_MAPPO_ACTIONS_DIMENSIONS 								= AGENT_MAPPO_U_ACTIONS_DIMENSIONS + AGENT_MAPPO_C_ACTIONS_DIMENSIONS 

ADVER_MAPPO_CRITIC_GNN_INPUT_DIMS							= [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2 
															   + COMMUNICATION_DIMENSIONS)) for i in range(NUMBER_OF_ADVER_DRONES)]
ADVER_MAPPO_CRITIC_GNN_NUM_HEADS							= 1
ADVER_MAPPO_CRITIC_BOOL_CONCAT								= True
ADVER_MAPPO_CRITIC_GNN_OUTPUT_DIMS							= [128, 128, 128]

ADVER_MAPPO_CRITIC_GMT_HIDDEN_DIMS							= 128		
ADVER_MAPPO_CRITIC_GMT_OUTPUT_DIMS							= 128			
ADVER_MAPPO_CRITIC_FC_OUTPUT_DIMS 							= [128, 128]

# maddpgv2 options for agent
AGENT_MADDPGV2_DISCOUNT_RATE 								= 0.99
AGENT_MADDPGV2_LEARNING_RATE_ACTOR 							= 0.0005
AGENT_MADDPGV2_LEARNING_RATE_CRITIC 						= 0.0005
AGENT_MADDPGV2_ACTOR_DROPOUT								= 0
AGENT_MADDPGV2_CRITIC_DROPOUT								= 0
AGENT_MADDPGV2_TAU 											= 0.01	  
AGENT_MADDPGV2_MEMORY_SIZE 									= 100000
AGENT_MADDPGV2_BATCH_SIZE 									= 128
AGENT_MADDPGV2_UPDATE_TARGET 								= None
AGENT_MADDPGV2_GRADIENT_CLIPPING							= True
AGENT_MADDPGV2_GRADIENT_NORM_CLIP							= 1
AGENT_MADDPGV2_GOAL 										= EPISODE_TIME_LIMIT
AGENT_MADDPGV2_NUMBER_OF_GOALS								= 8
AGENT_MADDPGV2_GOAL_DIFFERENCE								= 1
AGENT_MADDPGV2_GOAL_DISTRIBUTION 							= [AGENT_MADDPGV2_GOAL + i * AGENT_MADDPGV2_GOAL_DIFFERENCE for i in range(- AGENT_MADDPGV2_NUMBER_OF_GOALS + 1, 1)]
AGENT_MADDPGV2_ADDITIONAL_GOALS								= 4
AGENT_MADDPGV2_GOAL_STRATEGY								= "goal_distribution"

AGENT_MADDPGV2_ACTOR_INPUT_DIMENSIONS 						= [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2 
															   + COMMUNICATION_DIMENSIONS)) for i in range(NUMBER_OF_AGENT_DRONES)]
AGENT_MADDPGV2_ACTOR_OUTPUT_DIMENSIONS						= [128, 128, 128]
AGENT_MADDPGV2_U_ACTIONS_DIMENSIONS							= POSITION_DIMENSIONS
AGENT_MADDPGV2_C_ACTIONS_DIMENSIONS							= COMMUNICATION_DIMENSIONS
AGENT_MADDPGV2_ACTIONS_DIMENSIONS 							= AGENT_MADDPG_U_ACTIONS_DIMENSIONS + AGENT_MADDPG_C_ACTIONS_DIMENSIONS 

AGENT_MADDPGV2_CRITIC_GNN_INPUT_DIMS						= [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2 
															   + COMMUNICATION_DIMENSIONS)) for i in range(NUMBER_OF_AGENT_DRONES)]
AGENT_MADDPGV2_CRITIC_GNN_NUM_HEADS							= 1
AGENT_MADDPGV2_CRITIC_BOOL_CONCAT							= True
AGENT_MADDPGV2_CRITIC_GNN_OUTPUT_DIMS						= [128, 128, 128]

AGENT_MADDPGV2_CRITIC_GMT_HIDDEN_DIMS						= 128		
AGENT_MADDPGV2_CRITIC_GMT_OUTPUT_DIMS						= 128			

AGENT_MADDPGV2_CRITIC_U_ACTIONS_FC_INPUT_DIMS				= POSITION_DIMENSIONS * NUMBER_OF_AGENT_DRONES
AGENT_MADDPGV2_CRITIC_C_ACTIONS_FC_INPUT_DIMS				= COMMUNICATION_DIMENSIONS * NUMBER_OF_AGENT_DRONES
AGENT_MADDPGV2_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS				= [64, 64]
AGENT_MADDPGV2_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS				= [64, 64]

AGENT_MADDPGV2_CRITIC_GOAL_FC_INPUT_DIMS 					= len([AGENT_MADDPGV2_GOAL])
AGENT_MADDPGV2_CRITIC_GOAL_FC_OUTPUT_DIMS 					= [8, 8]

AGENT_MADDPGV2_CRITIC_CONCAT_FC_OUTPUT_DIMS 				= [128, 128]

# maddpgv2 options for adversary
ADVER_MADDPGV2_DISCOUNT_RATE 								= 0.99
ADVER_MADDPGV2_LEARNING_RATE_ACTOR 							= 0.0005
ADVER_MADDPGV2_LEARNING_RATE_CRITIC 						= 0.0005
ADVER_MADDPGV2_ACTOR_DROPOUT								= 0
ADVER_MADDPGV2_CRITIC_DROPOUT								= 0
ADVER_MADDPGV2_TAU 											= 0.01	  
ADVER_MADDPGV2_MEMORY_SIZE 									= 100000
ADVER_MADDPGV2_BATCH_SIZE 									= 128
ADVER_MADDPGV2_UPDATE_TARGET 								= None
ADVER_MADDPGV2_GRADIENT_CLIPPING							= True
ADVER_MADDPGV2_GRADIENT_NORM_CLIP							= 1
ADVER_MADDPGV2_GOAL 										= RESTRICTED_RADIUS
ADVER_MADDPGV2_NUMBER_OF_GOALS								= 5
ADVER_MADDPGV2_GOAL_DIFFERENCE								= 0.025
ADVER_MADDPGV2_GOAL_DISTRIBUTION 							= [ADVER_MADDPGV2_GOAL + i * ADVER_MADDPGV2_GOAL_DIFFERENCE for i in range(ADVER_MADDPGV2_NUMBER_OF_GOALS)]																																	  
ADVER_MADDPGV2_ADDITIONAL_GOALS								= 0
ADVER_MADDPGV2_GOAL_STRATEGY								= "goal_distribution"

ADVER_MADDPGV2_ACTOR_INPUT_DIMENSIONS 						= [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2 
															   + COMMUNICATION_DIMENSIONS)) for i in range(NUMBER_OF_ADVER_DRONES)]
ADVER_MADDPGV2_ACTOR_OUTPUT_DIMENSIONS						= [128, 128, 128]
ADVER_MADDPGV2_U_ACTIONS_DIMENSIONS							= POSITION_DIMENSIONS
ADVER_MADDPGV2_C_ACTIONS_DIMENSIONS							= COMMUNICATION_DIMENSIONS
ADVER_MADDPGV2_ACTIONS_DIMENSIONS 							= ADVER_MADDPG_U_ACTIONS_DIMENSIONS + ADVER_MADDPG_C_ACTIONS_DIMENSIONS 

ADVER_MADDPGV2_CRITIC_GNN_INPUT_DIMS						= [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2 
															   + COMMUNICATION_DIMENSIONS)) for i in range(NUMBER_OF_ADVER_DRONES)]
ADVER_MADDPGV2_CRITIC_GNN_NUM_HEADS							= 1
ADVER_MADDPGV2_CRITIC_BOOL_CONCAT							= True
ADVER_MADDPGV2_CRITIC_GNN_OUTPUT_DIMS						= [128, 128, 128]

ADVER_MADDPGV2_CRITIC_GMT_HIDDEN_DIMS						= 128		
ADVER_MADDPGV2_CRITIC_GMT_OUTPUT_DIMS						= 128			

ADVER_MADDPGV2_CRITIC_U_ACTIONS_FC_INPUT_DIMS				= POSITION_DIMENSIONS * NUMBER_OF_ADVER_DRONES 
ADVER_MADDPGV2_CRITIC_C_ACTIONS_FC_INPUT_DIMS				= COMMUNICATION_DIMENSIONS * NUMBER_OF_ADVER_DRONES 
ADVER_MADDPGV2_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS				= [64, 64]
ADVER_MADDPGV2_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS				= [64, 64]

ADVER_MADDPGV2_CRITIC_GOAL_FC_INPUT_DIMS 					= len([ADVER_MADDPGV2_GOAL])
ADVER_MADDPGV2_CRITIC_GOAL_FC_OUTPUT_DIMS 					= [8, 8]

ADVER_MADDPGV2_CRITIC_CONCAT_FC_OUTPUT_DIMS 				= [128, 128]

def train_test():

	""" function to execute experiments to train or test models based on different algorithms """

	# check agent model
	if AGENT_MODEL == "maddpg":

		# generate maddpg agents for agent drones
		agent_maddpg_agents = maddpg(mode = AGENT_MODE, training_name = AGENT_TRAINING_NAME, discount_rate = AGENT_MADDPG_DISCOUNT_RATE, lr_actor = AGENT_MADDPG_LEARNING_RATE_ACTOR, 
									 lr_critic = AGENT_MADDPG_LEARNING_RATE_CRITIC, num_agents = NUMBER_OF_AGENT_DRONES, num_opp = NUMBER_OF_ADVER_DRONES, 
									 actor_dropout_p = AGENT_MADDPG_ACTOR_DROPOUT, critic_dropout_p = AGENT_MADDPG_CRITIC_DROPOUT, state_fc_input_dims = AGENT_MADDPG_ACTOR_INPUT_DIMENSIONS, 
									 state_fc_output_dims = AGENT_MADDPG_ACTOR_OUTPUT_DIMENSIONS, u_action_dims = AGENT_MADDPG_U_ACTIONS_DIMENSIONS, 
									 c_action_dims = AGENT_MADDPG_C_ACTIONS_DIMENSIONS, num_heads = AGENT_MADDPG_CRITIC_GNN_NUM_HEADS, bool_concat = AGENT_MADDPG_CRITIC_BOOL_CONCAT, 
									 gnn_input_dims = AGENT_MADDPG_CRITIC_GNN_INPUT_DIMS, gnn_output_dims = AGENT_MADDPG_CRITIC_GNN_INPUT_DIMS, gmt_hidden_dims = AGENT_MADDPG_CRITIC_GMT_HIDDEN_DIMS, 
									 gmt_output_dims = AGENT_MADDPG_CRITIC_GMT_OUTPUT_DIMS, u_actions_fc_input_dims = AGENT_MADDPG_CRITIC_U_ACTIONS_FC_INPUT_DIMS, 
									 u_actions_fc_output_dims = AGENT_MADDPG_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS, c_actions_fc_input_dims = AGENT_MADDPG_CRITIC_C_ACTIONS_FC_INPUT_DIMS, 
									 c_actions_fc_output_dims = AGENT_MADDPG_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS, concat_fc_output_dims = AGENT_MADDPG_CRITIC_CONCAT_FC_OUTPUT_DIMS, 
									 tau = AGENT_MADDPG_TAU, mem_size = AGENT_MADDPG_MEMORY_SIZE, batch_size = AGENT_MADDPG_BATCH_SIZE, update_target = AGENT_MADDPG_UPDATE_TARGET, 
									 grad_clipping = AGENT_MADDPG_GRADIENT_CLIPPING, grad_norm_clip = AGENT_MADDPG_GRADIENT_NORM_CLIP, is_adversary = False)
	
	elif AGENT_MODEL == "mappo":

		# generate mappo agents for agent drones
		agent_mappo_agents = mappo(mode = AGENT_MODE, training_name = AGENT_TRAINING_NAME, lr_actor = AGENT_MAPPO_LEARNING_RATE_ACTOR, lr_critic = AGENT_MAPPO_LEARNING_RATE_CRITIC, 
								   num_agents = NUMBER_OF_AGENT_DRONES, num_opp = NUMBER_OF_ADVER_DRONES, u_range = AGENT_DRONE_U_RANGE, u_noise = AGENT_DRONE_U_NOISE, c_noise = AGENT_DRONE_C_NOISE,
								   is_adversary = False, actor_dropout_p = AGENT_MAPPO_ACTOR_DROPOUT, critic_dropout_p = AGENT_MAPPO_CRITIC_DROPOUT, 
								   state_fc_input_dims = AGENT_MAPPO_ACTOR_INPUT_DIMENSIONS, state_fc_output_dims = AGENT_MAPPO_ACTOR_OUTPUT_DIMENSIONS, 
								   u_action_dims = AGENT_MAPPO_U_ACTIONS_DIMENSIONS, c_action_dims = AGENT_MAPPO_C_ACTIONS_DIMENSIONS, num_heads = AGENT_MAPPO_CRITIC_GNN_NUM_HEADS, 
								   bool_concat = AGENT_MAPPO_CRITIC_BOOL_CONCAT, gnn_input_dims = AGENT_MAPPO_CRITIC_GNN_INPUT_DIMS, gnn_output_dims = AGENT_MAPPO_CRITIC_GNN_OUTPUT_DIMS, 
								   gmt_hidden_dims = AGENT_MAPPO_CRITIC_GMT_HIDDEN_DIMS, gmt_output_dims = AGENT_MAPPO_CRITIC_GMT_OUTPUT_DIMS, fc_output_dims = AGENT_MAPPO_CRITIC_FC_OUTPUT_DIMS, 
								   batch_size = AGENT_MAPPO_BATCH_SIZE, gamma = AGENT_MAPPO_GAMMA, clip_coeff = AGENT_MAPPO_CLIP_COEFFICIENT, num_epochs = AGENT_MAPPO_NUMBER_OF_EPOCHS, 
								   gae_lambda = AGENT_MAPPO_GAE_LAMBDA, entropy_coeff = AGENT_MAPPO_ENTROPY_COEFFICIENT, use_huber_loss = AGENT_MAPPO_USE_HUBER_LOSS, 
								   huber_delta = AGENT_MAPPO_HUBER_DELTA, use_clipped_value_loss = AGENT_MAPPO_USE_CLIPPED_VALUE_LOSS, critic_loss_coeff = AGENT_MAPPO_CRITIC_LOSS_COEFFICIENT, 
								   grad_clipping = AGENT_MAPPO_GRADIENT_CLIPPING, grad_norm_clip = AGENT_MAPPO_GRADIENT_NORM_CLIP)

	elif AGENT_MODEL == "maddpgv2":

		# generate maddpgv2 agents for agent drones
		agent_maddpgv2_agents = maddpgv2(mode = AGENT_MODE, training_name = AGENT_TRAINING_NAME, discount_rate = AGENT_MADDPGV2_DISCOUNT_RATE, lr_actor = AGENT_MADDPGV2_LEARNING_RATE_ACTOR, 
										 lr_critic = AGENT_MADDPGV2_LEARNING_RATE_CRITIC, num_agents = NUMBER_OF_AGENT_DRONES, num_opp = NUMBER_OF_ADVER_DRONES, 
										 actor_dropout_p = AGENT_MADDPGV2_ACTOR_DROPOUT, critic_dropout_p = AGENT_MADDPGV2_CRITIC_DROPOUT, state_fc_input_dims = AGENT_MADDPGV2_ACTOR_INPUT_DIMENSIONS, 
										 state_fc_output_dims = AGENT_MADDPGV2_ACTOR_OUTPUT_DIMENSIONS, u_action_dims = AGENT_MADDPGV2_U_ACTIONS_DIMENSIONS, 
										 c_action_dims = AGENT_MADDPGV2_C_ACTIONS_DIMENSIONS, num_heads = AGENT_MADDPGV2_CRITIC_GNN_NUM_HEADS, bool_concat = AGENT_MADDPGV2_CRITIC_BOOL_CONCAT, 
										 gnn_input_dims = AGENT_MADDPGV2_CRITIC_GNN_INPUT_DIMS, gnn_output_dims = AGENT_MADDPGV2_CRITIC_GNN_INPUT_DIMS, 
										 gmt_hidden_dims = AGENT_MADDPGV2_CRITIC_GMT_HIDDEN_DIMS, gmt_output_dims = AGENT_MADDPGV2_CRITIC_GMT_OUTPUT_DIMS, 
										 u_actions_fc_input_dims = AGENT_MADDPGV2_CRITIC_U_ACTIONS_FC_INPUT_DIMS, u_actions_fc_output_dims = AGENT_MADDPGV2_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS, 
										 c_actions_fc_input_dims = AGENT_MADDPGV2_CRITIC_C_ACTIONS_FC_INPUT_DIMS, c_actions_fc_output_dims = AGENT_MADDPGV2_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS, 
										 goal_fc_input_dims = AGENT_MADDPGV2_CRITIC_GOAL_FC_INPUT_DIMS, goal_fc_output_dims = AGENT_MADDPGV2_CRITIC_GOAL_FC_OUTPUT_DIMS, 
										 concat_fc_output_dims = AGENT_MADDPGV2_CRITIC_CONCAT_FC_OUTPUT_DIMS, tau = AGENT_MADDPGV2_TAU, mem_size = AGENT_MADDPGV2_MEMORY_SIZE, 
										 batch_size = AGENT_MADDPGV2_BATCH_SIZE, update_target = AGENT_MADDPGV2_UPDATE_TARGET, grad_clipping = AGENT_MADDPGV2_GRADIENT_CLIPPING, 
										 grad_norm_clip = AGENT_MADDPGV2_GRADIENT_NORM_CLIP, num_of_add_goals = AGENT_MADDPGV2_ADDITIONAL_GOALS, goal_strategy = AGENT_MADDPGV2_GOAL_STRATEGY, 
										 is_adversary = False, ep_time_limit = EPISODE_TIME_LIMIT, r_rad = RESTRICTED_RADIUS, big_rew_cnst = BIG_REWARD_CONSTANT, 
										 rew_multiplier_cnst = REWARD_MULTIPLIER_CONSTANT, pos_dims = POSITION_DIMENSIONS, exit_screen_terminate = EXIT_SCREEN_TERMINATE)

	# check adversarial model
	if ADVER_MODEL == "maddpg":

		# generate maddpg agents for adversarial drones 
		adver_maddpg_agents = maddpg(mode = ADVER_MODE, training_name = ADVER_TRAINING_NAME, discount_rate = ADVER_MADDPG_DISCOUNT_RATE, lr_actor = ADVER_MADDPG_LEARNING_RATE_ACTOR, 
									 lr_critic = ADVER_MADDPG_LEARNING_RATE_CRITIC, num_agents = NUMBER_OF_AGENT_DRONES, num_opp = NUMBER_OF_ADVER_DRONES, 
									 actor_dropout_p = ADVER_MADDPG_ACTOR_DROPOUT, critic_dropout_p = ADVER_MADDPG_CRITIC_DROPOUT, state_fc_input_dims = ADVER_MADDPG_ACTOR_INPUT_DIMENSIONS, 
									 state_fc_output_dims = ADVER_MADDPG_ACTOR_OUTPUT_DIMENSIONS, u_action_dims = ADVER_MADDPG_U_ACTIONS_DIMENSIONS, 
									 c_action_dims = ADVER_MADDPG_C_ACTIONS_DIMENSIONS, num_heads = ADVER_MADDPG_CRITIC_GNN_NUM_HEADS, bool_concat = ADVER_MADDPG_CRITIC_BOOL_CONCAT, 
									 gnn_input_dims = ADVER_MADDPG_CRITIC_GNN_INPUT_DIMS, gnn_output_dims = ADVER_MADDPG_CRITIC_GNN_INPUT_DIMS, gmt_hidden_dims = ADVER_MADDPG_CRITIC_GMT_HIDDEN_DIMS, 
									 gmt_output_dims = ADVER_MADDPG_CRITIC_GMT_OUTPUT_DIMS, u_actions_fc_input_dims = ADVER_MADDPG_CRITIC_U_ACTIONS_FC_INPUT_DIMS, 
									 u_actions_fc_output_dims = ADVER_MADDPG_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS, c_actions_fc_input_dims = ADVER_MADDPG_CRITIC_C_ACTIONS_FC_INPUT_DIMS, 
									 c_actions_fc_output_dims = ADVER_MADDPG_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS, concat_fc_output_dims = ADVER_MADDPG_CRITIC_CONCAT_FC_OUTPUT_DIMS, 
									 tau = ADVER_MADDPG_TAU, mem_size = ADVER_MADDPG_MEMORY_SIZE, batch_size = ADVER_MADDPG_BATCH_SIZE, update_target = ADVER_MADDPG_UPDATE_TARGET, 
									 grad_clipping = ADVER_MADDPG_GRADIENT_CLIPPING, grad_norm_clip = ADVER_MADDPG_GRADIENT_NORM_CLIP, is_adversary = True)

	elif ADVER_MODEL == "mappo":

		# generate mappo agents for adver drones
		adver_mappo_agents = mappo(mode = ADVER_MODE, training_name = ADVER_TRAINING_NAME, lr_actor = ADVER_MAPPO_LEARNING_RATE_ACTOR, lr_critic = ADVER_MAPPO_LEARNING_RATE_CRITIC, 
								   num_agents = NUMBER_OF_AGENT_DRONES, num_opp = NUMBER_OF_ADVER_DRONES, u_range = ADVER_DRONE_U_RANGE, u_noise = ADVER_DRONE_U_NOISE, c_noise = ADVER_DRONE_C_NOISE, 
								   is_adversary = True, actor_dropout_p = ADVER_MAPPO_ACTOR_DROPOUT, critic_dropout_p = ADVER_MAPPO_CRITIC_DROPOUT, 
								   state_fc_input_dims = ADVER_MAPPO_ACTOR_INPUT_DIMENSIONS, state_fc_output_dims = ADVER_MAPPO_ACTOR_OUTPUT_DIMENSIONS, 
								   u_action_dims = ADVER_MAPPO_U_ACTIONS_DIMENSIONS, c_action_dims = ADVER_MAPPO_C_ACTIONS_DIMENSIONS, num_heads = ADVER_MAPPO_CRITIC_GNN_NUM_HEADS, 
								   bool_concat = ADVER_MAPPO_CRITIC_BOOL_CONCAT, gnn_input_dims = ADVER_MAPPO_CRITIC_GNN_INPUT_DIMS, gnn_output_dims = ADVER_MAPPO_CRITIC_GNN_OUTPUT_DIMS, 
								   gmt_hidden_dims = ADVER_MAPPO_CRITIC_GMT_HIDDEN_DIMS, gmt_output_dims = ADVER_MAPPO_CRITIC_GMT_OUTPUT_DIMS, fc_output_dims = ADVER_MAPPO_CRITIC_FC_OUTPUT_DIMS, 
								   batch_size = ADVER_MAPPO_BATCH_SIZE, gamma = ADVER_MAPPO_GAMMA, clip_coeff = ADVER_MAPPO_CLIP_COEFFICIENT, num_epochs = ADVER_MAPPO_NUMBER_OF_EPOCHS, 
								   gae_lambda = ADVER_MAPPO_GAE_LAMBDA, entropy_coeff = ADVER_MAPPO_ENTROPY_COEFFICIENT, use_huber_loss = ADVER_MAPPO_USE_HUBER_LOSS, 
								   huber_delta = ADVER_MAPPO_HUBER_DELTA, use_clipped_value_loss = ADVER_MAPPO_USE_CLIPPED_VALUE_LOSS, critic_loss_coeff = ADVER_MAPPO_CRITIC_LOSS_COEFFICIENT, 
								   grad_clipping = ADVER_MAPPO_GRADIENT_CLIPPING, grad_norm_clip = ADVER_MAPPO_GRADIENT_NORM_CLIP) 

	elif ADVER_MODEL == "maddpgv2":

		# generate maddpgv2 agents for adver drones
		adver_maddpgv2_agents = maddpgv2(mode = ADVER_MODE, training_name = ADVER_TRAINING_NAME, discount_rate = ADVER_MADDPGV2_DISCOUNT_RATE, lr_actor = ADVER_MADDPGV2_LEARNING_RATE_ACTOR, 
										 lr_critic = ADVER_MADDPGV2_LEARNING_RATE_CRITIC, num_agents = NUMBER_OF_AGENT_DRONES, num_opp = NUMBER_OF_ADVER_DRONES, 
										 actor_dropout_p = ADVER_MADDPGV2_ACTOR_DROPOUT, critic_dropout_p = ADVER_MADDPGV2_CRITIC_DROPOUT, state_fc_input_dims = ADVER_MADDPGV2_ACTOR_INPUT_DIMENSIONS, 
										 state_fc_output_dims = ADVER_MADDPGV2_ACTOR_OUTPUT_DIMENSIONS, u_action_dims = ADVER_MADDPGV2_U_ACTIONS_DIMENSIONS, 
										 c_action_dims = ADVER_MADDPGV2_C_ACTIONS_DIMENSIONS, num_heads = ADVER_MADDPGV2_CRITIC_GNN_NUM_HEADS, bool_concat = ADVER_MADDPGV2_CRITIC_BOOL_CONCAT, 
										 gnn_input_dims = ADVER_MADDPGV2_CRITIC_GNN_INPUT_DIMS, gnn_output_dims = ADVER_MADDPGV2_CRITIC_GNN_INPUT_DIMS, 
										 gmt_hidden_dims = ADVER_MADDPGV2_CRITIC_GMT_HIDDEN_DIMS, gmt_output_dims = ADVER_MADDPGV2_CRITIC_GMT_OUTPUT_DIMS, 
										 u_actions_fc_input_dims = ADVER_MADDPGV2_CRITIC_U_ACTIONS_FC_INPUT_DIMS, u_actions_fc_output_dims = ADVER_MADDPGV2_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS, 
										 c_actions_fc_input_dims = ADVER_MADDPGV2_CRITIC_C_ACTIONS_FC_INPUT_DIMS, c_actions_fc_output_dims = ADVER_MADDPGV2_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS, 
										 goal_fc_input_dims = ADVER_MADDPGV2_CRITIC_GOAL_FC_INPUT_DIMS, goal_fc_output_dims = ADVER_MADDPGV2_CRITIC_GOAL_FC_OUTPUT_DIMS,
										 concat_fc_output_dims = ADVER_MADDPGV2_CRITIC_CONCAT_FC_OUTPUT_DIMS, tau = ADVER_MADDPGV2_TAU, mem_size = ADVER_MADDPGV2_MEMORY_SIZE, 
										 batch_size = ADVER_MADDPGV2_BATCH_SIZE, update_target = ADVER_MADDPGV2_UPDATE_TARGET, grad_clipping = ADVER_MADDPGV2_GRADIENT_CLIPPING, 
										 grad_norm_clip = ADVER_MADDPGV2_GRADIENT_NORM_CLIP, num_of_add_goals = ADVER_MADDPGV2_ADDITIONAL_GOALS, goal_strategy = ADVER_MADDPGV2_GOAL_STRATEGY, 
										 is_adversary = True, ep_time_limit = EPISODE_TIME_LIMIT, r_rad = RESTRICTED_RADIUS, big_rew_cnst = BIG_REWARD_CONSTANT, 
										 rew_multiplier_cnst = REWARD_MULTIPLIER_CONSTANT, pos_dims = POSITION_DIMENSIONS, exit_screen_terminate = EXIT_SCREEN_TERMINATE)

	if AGENT_MODE != "test" and ADVER_MODE != "test":

		# generate environment during evaluation
		env = make_env(scenario_name = "zone_def", dim_c = COMMUNICATION_DIMENSIONS, num_good_agents = NUMBER_OF_AGENT_DRONES, num_adversaries = NUMBER_OF_ADVER_DRONES, 
					   num_landmarks = NUMBER_OF_LANDMARKS, r_rad = RESTRICTED_RADIUS, i_rad = INTERCEPT_RADIUS, r_noise_pos = RADAR_NOISE_POSITION, r_noise_vel = RADAR_NOISE_VELOCITY, 
					   big_rew_cnst = BIG_REWARD_CONSTANT, rew_multiplier_cnst = REWARD_MULTIPLIER_CONSTANT, ep_time_limit = EPISODE_TIME_LIMIT, 
					   drone_radius = [AGENT_DRONE_RADIUS, ADVER_DRONE_RADIUS], agent_size = [AGENT_DRONE_SIZE, ADVER_DRONE_SIZE], agent_density = [AGENT_DRONE_DENSITY, ADVER_DRONE_DENSITY], 
					   agent_initial_mass = [AGENT_DRONE_INITIAL_MASS, ADVER_DRONE_INITIAL_MASS], agent_accel = [AGENT_DRONE_ACCEL, ADVER_DRONE_ACCEL], 
					   agent_max_speed = [AGENT_DRONE_MAX_SPEED, ADVER_DRONE_MAX_SPEED], agent_collide = [AGENT_DRONE_COLLIDE, ADVER_DRONE_COLLIDE], 
					   agent_silent = [AGENT_DRONE_SILENT, ADVER_DRONE_SILENT], agent_u_noise = [AGENT_DRONE_U_NOISE, ADVER_DRONE_U_NOISE], 
					   agent_c_noise = [AGENT_DRONE_C_NOISE, ADVER_DRONE_C_NOISE], agent_u_range = [AGENT_DRONE_U_RANGE, ADVER_DRONE_U_RANGE], landmark_size = LANDMARK_SIZE, benchmark = True)

	elif AGENT_MODE == "test" and ADVER_MODE != "test":

		# generate environment during evaluation
		env = make_env(scenario_name = "zone_def", dim_c = COMMUNICATION_DIMENSIONS, num_good_agents = NUMBER_OF_AGENT_DRONES, num_adversaries = NUMBER_OF_ADVER_DRONES, 
					   num_landmarks = NUMBER_OF_LANDMARKS, r_rad = RESTRICTED_RADIUS, i_rad = INTERCEPT_RADIUS, r_noise_pos = RADAR_NOISE_POSITION, r_noise_vel = RADAR_NOISE_VELOCITY, 
					   big_rew_cnst = BIG_REWARD_CONSTANT, rew_multiplier_cnst = REWARD_MULTIPLIER_CONSTANT, ep_time_limit = EPISODE_TIME_LIMIT, 
					   drone_radius = [AGENT_DRONE_RADIUS, ADVER_DRONE_RADIUS], agent_size = [AGENT_DRONE_SIZE, ADVER_DRONE_SIZE], agent_density = [AGENT_DRONE_DENSITY, ADVER_DRONE_DENSITY], 
					   agent_initial_mass = [AGENT_DRONE_INITIAL_MASS, ADVER_DRONE_INITIAL_MASS], agent_accel = [AGENT_DRONE_ACCEL, ADVER_DRONE_ACCEL], 
					   agent_max_speed = [AGENT_DRONE_MAX_SPEED, ADVER_DRONE_MAX_SPEED], agent_collide = [AGENT_DRONE_COLLIDE, ADVER_DRONE_COLLIDE], 
					   agent_silent = [AGENT_DRONE_SILENT, ADVER_DRONE_SILENT], agent_u_noise = [0.0, ADVER_DRONE_U_NOISE], agent_c_noise = [0.0, ADVER_DRONE_C_NOISE], 
					   agent_u_range = [AGENT_DRONE_U_RANGE, ADVER_DRONE_U_RANGE], landmark_size = LANDMARK_SIZE, benchmark = True)

	elif AGENT_MODE != "test" and ADVER_MODE == "test":

		# generate environment during evaluation
		env = make_env(scenario_name = "zone_def", dim_c = COMMUNICATION_DIMENSIONS, num_good_agents = NUMBER_OF_AGENT_DRONES, num_adversaries = NUMBER_OF_ADVER_DRONES, 
					   num_landmarks = NUMBER_OF_LANDMARKS, r_rad = RESTRICTED_RADIUS, i_rad = INTERCEPT_RADIUS, r_noise_pos = RADAR_NOISE_POSITION, r_noise_vel = RADAR_NOISE_VELOCITY, 
					   big_rew_cnst = BIG_REWARD_CONSTANT, rew_multiplier_cnst = REWARD_MULTIPLIER_CONSTANT, ep_time_limit = EPISODE_TIME_LIMIT, 
					   drone_radius = [AGENT_DRONE_RADIUS, ADVER_DRONE_RADIUS], agent_size = [AGENT_DRONE_SIZE, ADVER_DRONE_SIZE], agent_density = [AGENT_DRONE_DENSITY, ADVER_DRONE_DENSITY], 
					   agent_initial_mass = [AGENT_DRONE_INITIAL_MASS, ADVER_DRONE_INITIAL_MASS], agent_accel = [AGENT_DRONE_ACCEL, ADVER_DRONE_ACCEL], 
					   agent_max_speed = [AGENT_DRONE_MAX_SPEED, ADVER_DRONE_MAX_SPEED], agent_collide = [AGENT_DRONE_COLLIDE, ADVER_DRONE_COLLIDE], 
					   agent_silent = [AGENT_DRONE_SILENT, ADVER_DRONE_SILENT], agent_u_noise = [AGENT_DRONE_U_NOISE, 0.0], agent_c_noise = [AGENT_DRONE_C_NOISE, 0.0], 
					   agent_u_range = [AGENT_DRONE_U_RANGE, ADVER_DRONE_U_RANGE], landmark_size = LANDMARK_SIZE, benchmark = True)

	elif AGENT_MODE == "test" and ADVER_MODE == "test":

		# generate environment during evaluation
		env = make_env(scenario_name = "zone_def", dim_c = COMMUNICATION_DIMENSIONS, num_good_agents = NUMBER_OF_AGENT_DRONES, num_adversaries = NUMBER_OF_ADVER_DRONES, 
					   num_landmarks = NUMBER_OF_LANDMARKS, r_rad = RESTRICTED_RADIUS, i_rad = INTERCEPT_RADIUS, r_noise_pos = RADAR_NOISE_POSITION, r_noise_vel = RADAR_NOISE_VELOCITY, 
					   big_rew_cnst = BIG_REWARD_CONSTANT, rew_multiplier_cnst = REWARD_MULTIPLIER_CONSTANT, ep_time_limit = EPISODE_TIME_LIMIT, 
					   drone_radius = [AGENT_DRONE_RADIUS, ADVER_DRONE_RADIUS], agent_size = [AGENT_DRONE_SIZE, ADVER_DRONE_SIZE], agent_density = [AGENT_DRONE_DENSITY, ADVER_DRONE_DENSITY], 
					   agent_initial_mass = [AGENT_DRONE_INITIAL_MASS, ADVER_DRONE_INITIAL_MASS], agent_accel = [AGENT_DRONE_ACCEL, ADVER_DRONE_ACCEL], 
					   agent_max_speed = [AGENT_DRONE_MAX_SPEED, ADVER_DRONE_MAX_SPEED], agent_collide = [AGENT_DRONE_COLLIDE, ADVER_DRONE_COLLIDE], 
					   agent_silent = [AGENT_DRONE_SILENT, ADVER_DRONE_SILENT], agent_u_noise = [0.0, 0.0], agent_c_noise = [0.0, 0.0], agent_u_range = [AGENT_DRONE_U_RANGE, ADVER_DRONE_U_RANGE], 
					   landmark_size = LANDMARK_SIZE, benchmark = True)

	# if log directory for tensorboard exist
	if os.path.exists(TENSORBOARD_LOG_DIRECTORY):
		
		# remove entire directory
		shutil.rmtree(TENSORBOARD_LOG_DIRECTORY)

	# generate writer for tensorboard logging
	writer = SummaryWriter(log_dir = TENSORBOARD_LOG_DIRECTORY)

	# variables to track elo of agent and adver models
	agent_elo = INITIAL_ELO
	adver_elo = INITIAL_ELO

	# variables to track the sum of agent and opp wins
	sum_agent_wins = 0
	sum_adver_wins = 0

	# list to store metrics to be converted to csv for postprocessing
	sum_agent_wins_list = []
	sum_adver_wins_list = []
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
	avg_agent_actor_grad_norm_list = []
	avg_agent_critic_grad_norm_list = []
	avg_adver_actor_grad_norm_list = []
	avg_adver_critic_grad_norm_list = []

	# list to store agent and adver elo
	agent_elo_list = []
	adver_elo_list = []

	# check if exit screen is terminate
	if EXIT_SCREEN_TERMINATE == True: 

		# variables to track for exiting screen
		sum_agent_exceed_screen = 0 
		sum_adver_exceed_screen = 0 

		# list for exiting screen
		sum_agent_exceed_screen_list = [] 
		sum_adver_exceed_screen_list = []

	# check if agent model is mappo
	if AGENT_MODEL == "mappo":

		# list for policy ratio
		avg_agent_policy_ratio_list = []

		# generate batch tensor for graph multiset transformer in critic model for agent for mappo
		agent_critic_batch = T.tensor([i for i in range(1) for j in range(NUMBER_OF_AGENT_DRONES)], dtype = T.long).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu'))

	# check if adver model is mappo
	if ADVER_MODEL == "mappo":

		# list for policy ratio
		avg_adver_policy_ratio_list = []

		# generate batch tensor for graph multiset transformer in critic model for adver for mappo
		adver_critic_batch = T.tensor([i for i in range(1) for j in range(NUMBER_OF_ADVER_DRONES)], dtype = T.long).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu'))

	# generate edge_index for complete graph for gnn for critic models
	agent_edge_index = complete_graph_edge_index(num_nodes = NUMBER_OF_AGENT_DRONES)
	adver_edge_index = complete_graph_edge_index(num_nodes = NUMBER_OF_ADVER_DRONES)

	# track steps of episodes for agent model using mappo
	if AGENT_MODEL == "mappo":

		# variable to track number of steps in episode
		agent_eps_steps = 0

	# track steps of episodes for adver model using mappo
	if ADVER_MODEL == "mappo":

		# variable to track number of steps in episode
		adver_eps_steps = 0

	# goals based variables for agent maddppv2 
	if AGENT_MODEL == "maddpgv2":

		# list to store agent goals
		agent_goals_list = []

		# initialise softmax weights for agent goal distribution
		agent_goals_softmax_weights = np.zeros(len(AGENT_MADDPGV2_GOAL_DISTRIBUTION))

	# goals based variables for agent maddppv2 
	if ADVER_MODEL == "maddpgv2":

		# list to store agent goals
		adver_goals_list = []

		# initialise softmax weights for adversarial goal distribution
		adver_goals_softmax_weights = np.zeros(len(ADVER_MADDPGV2_GOAL_DISTRIBUTION))

	# iterate over number of episodes
	for eps in range(1, NUMBER_OF_EPISODES + 1): 

		# boolean to check if episode is terminal
		is_terminal = 0

		# variable to track terminal condition
		terminal_condition = 0

		# print episode number 
		print("episode " + str(eps) + ":") 

		# obtain states of agent and adverserial agents
		actor_states = env.reset()

		# check if exponential decay for noise is desired
		if EXPONENTIAL_NOISE_DECAY == True:

			update_noise_exponential_decay(env = env, expo_decay_cnst = EXPONENTIAL_NOISE_DECAY_CONSTANT, num_adver = NUMBER_OF_ADVER_DRONES, eps_timestep = eps, 
										   agent_u_noise_cnst = AGENT_DRONE_U_NOISE, agent_c_noise_cnst = AGENT_DRONE_C_NOISE, adver_u_noise_cnst = ADVER_DRONE_U_NOISE, 
										   adver_c_noise_cnst = ADVER_DRONE_C_NOISE)

		# set up agent actor goals for maddpg and mappo
		if AGENT_MODEL == "maddpg" or AGENT_MODEL == "mappo":
			
			agent_goal = AGENT_MADDPGV2_GOAL
			agent_actor_goals = np.array([[AGENT_MADDPGV2_GOAL] for i in range(NUMBER_OF_AGENT_DRONES)])

		# set up agent actor and critic goals for maddpgv2
		elif AGENT_MODEL == "maddpgv2":

			# check if training 
			if AGENT_MODE != "test":

				# obtain probability distribution from agent_goals_softmax_weights
				prob_dist = np.nan_to_num(np.exp(agent_goals_softmax_weights), nan = 10**-5)/sum(np.nan_to_num(np.exp(agent_goals_softmax_weights), nan = 10**-5))

				# sample goal from probability distribution
				agent_goal = np.random.choice(a = AGENT_MADDPGV2_GOAL_DISTRIBUTION, p = prob_dist)

				# obtain agent_actor_goals and agent_critic_goals
				agent_actor_goals = np.array([[agent_goal] for i in range(NUMBER_OF_AGENT_DRONES)])
				agent_critic_goals = np.array([[agent_goal] for i in range(NUMBER_OF_AGENT_DRONES)]).reshape(1, -1)

			else: 

				# set for original goals
				agent_goal = AGENT_MADDPGV2_GOAL
				agent_actor_goals = np.array([[agent_goal] for i in range(NUMBER_OF_AGENT_DRONES)])
				agent_critic_goals = np.array([[agent_goal] for i in range(NUMBER_OF_AGENT_DRONES)]).reshape(1, -1)

		# set up adver actor goals for maddpg and mappo
		if ADVER_MODEL == "maddpg" or ADVER_MODEL == "mappo":
			
			adver_goal = ADVER_MADDPGV2_GOAL
			adver_actor_goals = np.array([[ADVER_MADDPGV2_GOAL] for i in range(NUMBER_OF_ADVER_DRONES)])

		# set up adver actor and critic goals for maddpgv2
		if ADVER_MODEL == "maddpgv2":

			# check if training 
			if ADVER_MODE != "test":

				# obtain probability distribution from adver_goals_softmax_weights
				prob_dist = np.nan_to_num(np.exp(adver_goals_softmax_weights), nan = 10**-5)/sum(np.nan_to_num(np.exp(adver_goals_softmax_weights), nan = 10**-5))

				# sample goal from probability distribution
				adver_goal = np.random.choice(a = ADVER_MADDPGV2_GOAL_DISTRIBUTION, p = prob_dist)

				# obtain adver_actor_goals and adver_critic_goals
				adver_actor_goals = np.array([[adver_goal] for i in range(NUMBER_OF_ADVER_DRONES)])
				adver_critic_goals = np.array([[adver_goal] for i in range(NUMBER_OF_ADVER_DRONES)]).reshape(1, -1)

			else: 

				# set for original goals 
				adver_goal = ADVER_MADDPGV2_GOAL
				adver_actor_goals = np.array([[adver_goal] for i in range(NUMBER_OF_ADVER_DRONES)])
				adver_critic_goals = np.array([[adver_goal] for i in range(NUMBER_OF_ADVER_DRONES)]).reshape(1, -1)

		# obtain numpy array of actor_states, adver_actor_states, agent_actor_states
		actor_states = np.array(actor_states)
		adver_actor_states = np.array(actor_states[:NUMBER_OF_ADVER_DRONES])
		agent_actor_states = np.array(actor_states[NUMBER_OF_ADVER_DRONES:])

		# set episode start time
		env.world.ep_start_time = time.time()

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
				agent_u_actions, agent_c_actions, agent_actions_list = agent_maddpg_agents.select_actions(mode = "test", env_agents = env.agents[NUMBER_OF_ADVER_DRONES:], 
																										  actor_state_list = agent_actor_states)
			
			# obtain actions for agent_mappo_agents
			elif AGENT_MODEL == "mappo":

				# obtain motor and communication actions for agent drones
				# mode is always 'test' as the environment handles the addition of noise to the actions
				agent_u_actions, agent_c_actions, agent_u_actions_log_probs, agent_c_actions_log_probs, agent_actions_list = \
				agent_mappo_agents.select_actions(mode = "test", env_agents = env.agents[NUMBER_OF_ADVER_DRONES:], actor_state_list = agent_actor_states)

			# obtain actions for agent_maddpgv2_agents  
			elif AGENT_MODEL == "maddpgv2":
				
				# agent_actor_states concatenated with agent goals
				agent_actor_states_p_goal = np.concatenate((agent_actor_states, agent_actor_goals), axis = -1)

				# obtain motor and communication actions for agent drones
				# mode is always 'test' as the environment handles the addition of noise to the actions
				agent_u_actions, agent_c_actions, agent_actions_list = agent_maddpgv2_agents.select_actions(mode = "test", env_agents = env.agents[NUMBER_OF_ADVER_DRONES:], 
																											actor_state_list = agent_actor_states_p_goal)

			# obtain actions for adver_maddpg_agents 
			if ADVER_MODEL == "maddpg":

				# obtain actions from fc_state and cam_state for all opp drones
				# mode is always 'test' as the environment handles the addition of noise to the actions
				adver_u_actions, adver_c_actions, adver_actions_list = adver_maddpg_agents.select_actions(mode = "test", env_agents = env.agents[:NUMBER_OF_ADVER_DRONES], 
																										  actor_state_list = adver_actor_states)

			# obtain actions for adver_mappo_agents
			elif AGENT_MODEL == "mappo":

				# obtain motor and communication actions for adver drones
				# mode is always 'test' as the environment handles the addition of noise to the actions
				adver_u_actions, adver_c_actions, adver_u_actions_log_probs, adver_c_actions_log_probs, adver_actions_list = \
				adver_mappo_agents.select_actions(mode = "test", env_agents = env.agents[:NUMBER_OF_ADVER_DRONES], actor_state_list = adver_actor_states)

			# obtain actions for adver_maddpgv2_agents 
			elif ADVER_MODEL == "maddpgv2":

				# agent_actor_states concatenated with agent goals
				adver_actor_states_p_goal = np.concatenate((adver_actor_states, adver_actor_goals), axis = -1)

				# obtain actions from fc_state and cam_state for all opp drones
				# mode is always 'test' as the environment handles the addition of noise to the actions
				adver_u_actions, adver_c_actions, adver_actions_list = adver_maddpgv2_agents.select_actions(mode = "test", env_agents = env.agents[:NUMBER_OF_ADVER_DRONES], 
																											actor_state_list = adver_actor_states_p_goal)

			# iterate over agent_maddpg_agents
			for i in range(NUMBER_OF_AGENT_DRONES):

				# append agent drones actions to adversarial drones actions
				adver_actions_list.append(agent_actions_list[i])

			# update state of the world and obtain information of the updated state
			actor_states_prime, rewards, terminates_p_terminal_con, benchmark_data = env.step(action_n = adver_actions_list, agent_goal = AGENT_MADDPGV2_GOAL, adver_goal = ADVER_MADDPGV2_GOAL)
			
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
					agent_terminates.append(terminates_p_terminal_con[i][0])
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

			# for maddpg agent drones to store memory in replay buffer 
			if AGENT_MODEL == "maddpg":

				# check if agent is training
				if AGENT_MODE != "test": 

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

					# train model
					if not(abs(agent_elo - adver_elo) > ELO_DIFFRENCE and agent_elo > adver_elo):

						# train agent models and obtain metrics for each agent drone for logging
						agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list = \
						agent_maddpg_agents.apply_gradients_maddpg(num_of_agents = NUMBER_OF_AGENT_DRONES)

					else:

						agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list = np.nan, np.nan, np.nan, np.nan							

				else: 

					agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list = np.nan, np.nan, np.nan, np.nan					

			# for mappo agent to store memory in replay buffer and train model 
			elif AGENT_MODEL == "mappo":

				# train model
				if AGENT_MODE != "test" and not(abs(agent_elo - adver_elo) > ELO_DIFFRENCE and agent_elo > adver_elo):

					# obtain agent_critic_states in gnn data format
					agent_critic_states = Data(x = T.tensor(agent_actor_states, dtype = T.float), edge_index = \
					T.tensor(agent_edge_index, dtype = T.long).t().contiguous()).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu'))

					# set num_nodes for agent_critic_states
					agent_critic_states.num_nodes = NUMBER_OF_AGENT_DRONES 

					# list to store critic state values
					agent_critic_state_value = []

					# iterate over agent critic models to obtain agent_critic_state_value:
					for agent_index, agent in enumerate(agent_mappo_agents.mappo_agents_list):

						# turn critic to eval mode
						agent.mappo_critic.eval()

						# append critic value to list
						agent_critic_state_value.append(agent.mappo_critic.forward(agent_critic_states, agent_critic_batch).item())

						# turn critic to train mode
						agent.mappo_critic.train()

					# obtain numpy array of agent_critic_state_value
					agent_critic_state_value = np.array(agent_critic_state_value)
					
					# obtain cpu copy of critic states
					agent_critic_states = agent_critic_states.cpu()

					# store states and actions in replay buffer
					agent_mappo_agents.replay_buffer.log(actor_state = agent_actor_states, critic_state = agent_critic_states, critic_state_value = agent_critic_state_value, 
														 u_action = agent_u_actions, c_action = agent_c_actions, u_action_log_probs = agent_u_actions_log_probs, 
														 c_action_log_probs = agent_c_actions_log_probs, rewards = agent_rewards, is_done = agent_terminates)

					# update agent_eps_steps
					agent_eps_steps += 1

					# train model
					if agent_eps_steps % AGENT_MAPPO_EPISODE_LENGTH == 0:

						# train agent models and obtain metrics for each agent drone for logging
						agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list, agent_policy_ratio_list = \
						agent_mappo_agents.apply_gradients_mappo(num_of_agents = NUMBER_OF_AGENT_DRONES)

					else:

						agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list, agent_policy_ratio_list = np.nan, np.nan, np.nan, np.nan, np.nan

				else:

					agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list, agent_policy_ratio_list = np.nan, np.nan, np.nan, np.nan, np.nan

			# for maddpgv2 agent drones to store memory in replay buffer 
			elif AGENT_MODEL == "maddpgv2" :

				# check if agent is training
				if AGENT_MODE != "test":

					# obtain agent_critic_states and agent_critic_states_prime in gnn data format
					agent_critic_states = Data(x = T.tensor(agent_actor_states, dtype = T.float), edge_index = T.tensor(agent_edge_index, dtype = T.long).t().contiguous())
					agent_critic_states_prime = Data(x = T.tensor(agent_actor_states_prime, dtype = T.float), edge_index = T.tensor(agent_edge_index, dtype = T.long).t().contiguous())

					# set num_nodes for agent_critic_states, agent_critic_states_prime
					agent_critic_states.num_nodes = NUMBER_OF_AGENT_DRONES 
					agent_critic_states_prime.num_nodes = NUMBER_OF_AGENT_DRONES

					# store states and actions in replay buffer
					agent_maddpgv2_agents.replay_buffer.log(actor_state = agent_actor_states, actor_state_prime = agent_actor_states_prime, org_actor_goals = agent_actor_goals, 
															critic_state = agent_critic_states, critic_state_prime = agent_critic_states_prime, org_critic_goals = agent_critic_goals, 
															u_action = agent_u_actions, c_action = agent_c_actions, org_rewards = agent_rewards, is_done = agent_terminates)

					# train model
					if not(abs(agent_elo - adver_elo) > ELO_DIFFRENCE and agent_elo > adver_elo):

						# train agent models and obtain metrics for each agent drone for logging
						agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list = \
						agent_maddpgv2_agents.apply_gradients_maddpgv2(num_of_agents = NUMBER_OF_AGENT_DRONES)

					else:

						agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list = np.nan, np.nan, np.nan, np.nan	

				else: 

					agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list = np.nan, np.nan, np.nan, np.nan	

			# for maddpg adversarial drones to store memory in replay buffer
			if ADVER_MODEL == "maddpg" :

				# check if adversarial is training
				if ADVER_MODE != "test":

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

					# train model
					if not(abs(adver_elo - agent_elo) > ELO_DIFFRENCE and adver_elo > agent_elo):

						# train adversarial models and obtain metrics for each adversarial drone for logging
						adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list = \
						adver_maddpg_agents.apply_gradients_maddpg(num_of_agents = NUMBER_OF_ADVER_DRONES)

					else: 

						adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list = np.nan, np.nan, np.nan, np.nan						

				else:

					adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list = np.nan, np.nan, np.nan, np.nan

			# for mappo agent to store memory in replay buffer and train model 
			elif ADVER_MODEL == "mappo":

				# train model
				if ADVER_MODE != "test" and not(abs(adver_elo - agent_elo) > ELO_DIFFRENCE and adver_elo > agent_elo):

					# obtain adver_critic_states in gnn data format
					adver_critic_states = Data(x = T.tensor(adver_actor_states, dtype = T.float), edge_index = T.tensor(adver_edge_index, dtype = \
					T.long).t().contiguous()).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu'))

					# set num_nodes for adver_critic_states
					adver_critic_states.num_nodes = NUMBER_OF_ADVER_DRONES 

					# list to store critic state values
					adver_critic_state_value = []

					# iterate over agent critic models to obtain agent_critic_state_value:
					for agent_index, agent in enumerate(adver_mappo_agents.mappo_agents_list):

						# turn critic to eval mode
						agent.mappo_critic.eval()

						# append critic value to list
						adver_critic_state_value.append(agent.mappo_critic.forward(adver_critic_states, adver_critic_batch).item())

						# turn critic to train mode
						agent.mappo_critic.train()

					# obtain numpy array of adver_critic_state_value
					adver_critic_state_value = np.array(adver_critic_state_value)

					# obtain cpu copy of critic states
					adver_critic_states = adver_critic_states.cpu()

					# store states and actions in replay buffer
					adver_mappo_agents.replay_buffer.log(actor_state = adver_actor_states, critic_state = adver_critic_states, critic_state_value = adver_critic_state_value, 
														 u_action = adver_u_actions, c_action = adver_c_actions, u_action_log_probs = adver_u_actions_log_probs, 
														 c_action_log_probs = adver_c_actions_log_probs, rewards = adver_rewards, is_done = adver_terminates)

					# update adver_eps_steps
					adver_eps_steps += 1

					# train model
					if adver_eps_steps % ADVER_MAPPO_EPISODE_LENGTH == 0:

						# train agent models and obtain metrics for each adver drone for logging
						adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list, adver_policy_ratio_list = \
						adver_mappo_agents.apply_gradients_mappo(num_of_agents = NUMBER_OF_ADVER_DRONES)

					else:

						adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list, adver_policy_ratio_list = np.nan, np.nan, np.nan, np.nan, np.nan

				else:

					adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list, adver_policy_ratio_list = np.nan, np.nan, np.nan, np.nan, np.nan

			# for maddpgv2 adversarial drones to store memory in replay buffer
			elif ADVER_MODEL == "maddpgv2":

				# check if adversarial is training
				if ADVER_MODE != "test":

					# obtain adver_critic_states and adver_critic_states_prime in gnn data format
					adver_critic_states = Data(x = T.tensor(adver_actor_states, dtype = T.float), edge_index = T.tensor(adver_edge_index, dtype = T.long).t().contiguous())
					adver_critic_states_prime = Data(x = T.tensor(adver_actor_states_prime, dtype = T.float), edge_index = T.tensor(adver_edge_index, dtype = T.long).t().contiguous())

					# set num_nodes for adver_critic_states, adver_critic_states_prime
					adver_critic_states.num_nodes = NUMBER_OF_ADVER_DRONES
					adver_critic_states_prime.num_nodes = NUMBER_OF_ADVER_DRONES

					# store states and actions in replay buffer
					adver_maddpgv2_agents.replay_buffer.log(actor_state = adver_actor_states, actor_state_prime = adver_actor_states_prime, org_actor_goals = adver_actor_goals, 
															critic_state = adver_critic_states, critic_state_prime = adver_critic_states_prime, org_critic_goals = adver_critic_goals, 
															u_action = adver_u_actions, c_action = adver_c_actions, org_rewards = adver_rewards, is_done = adver_terminates)

					# train model
					if not(abs(adver_elo - agent_elo) > ELO_DIFFRENCE and adver_elo > agent_elo):

						# train adversarial models and obtain metrics for each adversarial drone for logging
						adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list = \
						adver_maddpgv2_agents.apply_gradients_maddpgv2(num_of_agents = NUMBER_OF_ADVER_DRONES)

					else:

						adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list = np.nan, np.nan, np.nan, np.nan

				else:

					adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list = np.nan, np.nan, np.nan, np.nan

			# check if agent is training
			if AGENT_MODE != "test":

				# populate addtional replay buffer for agent maddpgv2:
				if AGENT_MODEL == "maddpgv2" and agent_maddpgv2_agents.replay_buffer.org_replay_buffer.is_ep_terminal == True and AGENT_MADDPGV2_ADDITIONAL_GOALS != 0:

					# populate her replay buffer
					agent_maddpgv2_agents.replay_buffer.generate_her_replay_buffer(opp_org_replay_buffer = adver_maddpgv2_agents.replay_buffer.org_replay_buffer, agent_goal = agent_goal, 
																				   adver_goal = adver_goal, agent_goal_dist = AGENT_MADDPGV2_GOAL_DISTRIBUTION, 
																				   adver_goal_dist = ADVER_MADDPGV2_GOAL_DISTRIBUTION)

			# check if adver is training
			if ADVER_MODE != "test":

				# populate addtional replay buffer for adver maddpgv2:
				if ADVER_MODEL == "maddpgv2" and adver_maddpgv2_agents.replay_buffer.org_replay_buffer.is_ep_terminal == True and ADVER_MADDPGV2_ADDITIONAL_GOALS != 0:

					# populate her replay buffer
					adver_maddpgv2_agents.replay_buffer.generate_her_replay_buffer(opp_org_replay_buffer = agent_maddpgv2_agents.replay_buffer.org_replay_buffer, agent_goal = agent_goal, 
																				   adver_goal = adver_goal, agent_goal_dist = AGENT_MADDPGV2_GOAL_DISTRIBUTION, 
																				   adver_goal_dist = ADVER_MADDPGV2_GOAL_DISTRIBUTION)

			# log metrics for agent_model for maddpg or maddpgv2
			if AGENT_MODEL == "maddpg" or AGENT_MODEL == "maddpgv2":

				# variables to track metrics
				sum_agent_actor_loss = 0
				sum_agent_critic_loss = 0
				sum_agent_number_of_team_collisions = 0
				sum_agent_number_of_oppo_collisions = 0
				sum_agent_actor_grad_norm = 0
				sum_agent_critic_grad_norm = 0

				# iterate over num of agent drones
				for i in range(NUMBER_OF_AGENT_DRONES): 

					# check if list is not nan and agent model is training
					if np.any(np.isnan(agent_actor_loss_list)) == False and AGENT_MODE != "test":

						# update sums
						sum_agent_actor_loss += agent_actor_loss_list[i]

						# add actor loss for agent drone
						writer.add_scalar(tag = "agent_drone_" + str(i + 1) + "/actor_loss", scalar_value = agent_actor_loss_list[i], global_step = eps)

					# check if list is not nan and agent model is training
					if np.any(np.isnan(agent_critic_loss_list)) == False and AGENT_MODE != "test":

						# update sums
						sum_agent_critic_loss += agent_critic_loss_list[i]

						# add critic loss for agent drone
						writer.add_scalar(tag = "agent_drone_" + str(i + 1) + "/critic_loss", scalar_value = agent_critic_loss_list[i], global_step = eps)

					# check if list is not nan and agent model is training
					if np.any(np.isnan(agent_actor_grad_norm_list)) == False and AGENT_MODE != "test":

						# update sums
						sum_agent_actor_grad_norm += agent_actor_grad_norm_list[i]

						# add actor grad norms for agent drone
						writer.add_scalar(tag = "agent_drone_" + str(i + 1) + "/actor_grad_norm", scalar_value = agent_actor_grad_norm_list[i], global_step = eps)

					# check if list is not nan and agent model is training
					if np.any(np.isnan(agent_critic_grad_norm_list)) == False and AGENT_MODE != "test":

						# update sums
						sum_agent_critic_grad_norm += agent_critic_grad_norm_list[i]

						# add critic grad norms for agent drone
						writer.add_scalar(tag = "agent_drone_" + str(i + 1) + "/critic_grad_norm", scalar_value = agent_critic_grad_norm_list[i], global_step = eps)

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

					# obtain avg actor and critic grad norms
					avg_agent_actor_grad_norm = sum_agent_actor_grad_norm / float(NUMBER_OF_AGENT_DRONES)
					avg_agent_critic_grad_norm = sum_agent_critic_grad_norm / float(NUMBER_OF_AGENT_DRONES)

					# add avg actor and critic grad norms for agent drones
					writer.add_scalar(tag = "avg_agent_actor_grad_norm", scalar_value = avg_agent_actor_grad_norm, global_step = eps)
					writer.add_scalar(tag = "avg_agent_critic_grad_norm", scalar_value = avg_agent_critic_grad_norm, global_step = eps)

					# append avg actor and critic grad norms to their respective list
					avg_agent_actor_grad_norm_list.append(avg_agent_actor_grad_norm)
					avg_agent_critic_grad_norm_list.append(avg_agent_critic_grad_norm)

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

				# add agent goal
				writer.add_scalar(tag = "goal/agent_goal", scalar_value = agent_goal, global_step = eps)

				# check if maddpgv2
				if AGENT_MODEL == "maddpgv2":

					# append agent goal to agent_goals_list
					agent_goals_list.append(agent_goal)

			# log metrics for agent_model for mappo
			elif AGENT_MODEL == "mappo":

				# variables to track metrics
				sum_agent_actor_loss = 0
				sum_agent_critic_loss = 0
				sum_agent_number_of_team_collisions = 0
				sum_agent_number_of_oppo_collisions = 0
				sum_agent_actor_grad_norm = 0
				sum_agent_critic_grad_norm = 0
				sum_agent_policy_ratio = 0

				# iterate over num of agent drones
				for i in range(NUMBER_OF_AGENT_DRONES): 

					# check if list is not nan and agent model is training
					if np.any(np.isnan(agent_actor_loss_list)) == False and AGENT_MODE != "test":

						# update sums
						sum_agent_actor_loss += agent_actor_loss_list[i]

						# add actor loss for agent drone
						writer.add_scalar(tag = "agent_drone_" + str(i + 1) + "/actor_loss", scalar_value = agent_actor_loss_list[i], global_step = eps)

					# check if list is not nan and agent model is training
					if np.any(np.isnan(agent_critic_loss_list)) == False and AGENT_MODE != "test":

						# update sums
						sum_agent_critic_loss += agent_critic_loss_list[i]

						# add critic loss for agent drone
						writer.add_scalar(tag = "agent_drone_" + str(i + 1) + "/critic_loss", scalar_value = agent_critic_loss_list[i], global_step = eps)

					# check if list is not nan and agent model is training
					if np.any(np.isnan(agent_actor_grad_norm_list)) == False and AGENT_MODE != "test":

						# update sums
						sum_agent_actor_grad_norm += agent_actor_grad_norm_list[i]

						# add actor grad norms for agent drone
						writer.add_scalar(tag = "agent_drone_" + str(i + 1) + "/actor_grad_norm", scalar_value = agent_actor_grad_norm_list[i], global_step = eps)

					# check if list is not nan and agent model is training
					if np.any(np.isnan(agent_critic_grad_norm_list)) == False and AGENT_MODE != "test":

						# update sums
						sum_agent_critic_grad_norm += agent_critic_grad_norm_list[i]

						# add critic grad norms for agent drone
						writer.add_scalar(tag = "agent_drone_" + str(i + 1) + "/critic_grad_norm", scalar_value = agent_critic_grad_norm_list[i], global_step = eps)

					# check if list is not nan and agent model is training
					if np.any(np.isnan(agent_policy_ratio_list)) == False and AGENT_MODE != "test":

						# update sums
						sum_agent_policy_ratio += agent_policy_ratio_list[i]

						# add critic grad norms for agent drone
						writer.add_scalar(tag = "agent_drone_" + str(i + 1) + "/policy_ratio", scalar_value = agent_policy_ratio_list[i], global_step = eps)

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

					# obtain avg actor and critic grad norms
					avg_agent_actor_grad_norm = sum_agent_actor_grad_norm / float(NUMBER_OF_AGENT_DRONES)
					avg_agent_critic_grad_norm = sum_agent_critic_grad_norm / float(NUMBER_OF_AGENT_DRONES)

					# add avg actor and critic grad norms for agent drones
					writer.add_scalar(tag = "avg_agent_actor_grad_norm", scalar_value = avg_agent_actor_grad_norm, global_step = eps)
					writer.add_scalar(tag = "avg_agent_critic_grad_norm", scalar_value = avg_agent_critic_grad_norm, global_step = eps)

					# append avg actor and critic grad norms to their respective list
					avg_agent_actor_grad_norm_list.append(avg_agent_actor_grad_norm)
					avg_agent_critic_grad_norm_list.append(avg_agent_critic_grad_norm)

					# obtain avg avg_agent_policy_ratio
					avg_agent_policy_ratio = sum_agent_policy_ratio / float(NUMBER_OF_AGENT_DRONES)

					# add avg_agent_policy_ratio for agent drones
					writer.add_scalar(tag = "avg_agent_policy_ratio", scalar_value = avg_agent_policy_ratio, global_step = eps)

					# append avg_agent_policy_ratio to list
					avg_agent_policy_ratio_list.append(avg_agent_policy_ratio)  

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

			# log metrics for adver_model for maddpg or maddpgv2
			if ADVER_MODEL == "maddpg" or ADVER_MODEL == "maddpgv2":

				# variables to track sum of actor and critic loss
				sum_adver_actor_loss = 0
				sum_adver_critic_loss = 0
				sum_adver_number_of_team_collisions = 0
				sum_adver_number_of_oppo_collisions = 0
				sum_adver_actor_grad_norm = 0
				sum_adver_critic_grad_norm = 0

				# iterate over num of adver drones
				for i in range(NUMBER_OF_ADVER_DRONES): 

					# check if list is not nan and adversarial model is training
					if np.any(np.isnan(adver_actor_loss_list)) == False and ADVER_MODE != "test":

						# update sums
						sum_adver_actor_loss += adver_actor_loss_list[i]

						# add actor loss for adver drone
						writer.add_scalar(tag = "adver_drone_" + str(i + 1) + "/actor_loss", scalar_value = adver_actor_loss_list[i], global_step = eps)

					# check if list is not nan and adversarial model is training
					if np.any(np.isnan(adver_critic_loss_list)) == False and ADVER_MODE != "test":

						# update sums
						sum_adver_critic_loss += adver_critic_loss_list[i]

						# add critic loss for adver drone
						writer.add_scalar(tag = "adver_drone_" + str(i + 1) + "/critic_loss", scalar_value = adver_critic_loss_list[i], global_step = eps)

					# check if list is not nan and adversarial model is training
					if np.any(np.isnan(adver_actor_grad_norm_list)) == False and ADVER_MODE != "test":

						# update sums
						sum_adver_actor_grad_norm += adver_actor_grad_norm_list[i]

						# add actor and critic grad norms for adver drone
						writer.add_scalar(tag = "adver_drone_" + str(i + 1) + "/actor_grad_norm", scalar_value = adver_actor_grad_norm_list[i], global_step = eps)

					# check if list is not nan and adversarial model is training
					if np.any(np.isnan(adver_critic_grad_norm_list)) == False and ADVER_MODE != "test":

						# update sums
						sum_adver_critic_grad_norm += adver_critic_grad_norm_list[i]

						# add actor and critic grad norms for adver drone
						writer.add_scalar(tag = "adver_drone_" + str(i + 1) + "/critic_grad_norm", scalar_value = adver_critic_grad_norm_list[i], global_step = eps)

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

					# obtain avg actor and critic grad norms
					avg_adver_actor_grad_norm = sum_adver_actor_grad_norm / float(NUMBER_OF_ADVER_DRONES)
					avg_adver_critic_grad_norm = sum_adver_critic_grad_norm / float(NUMBER_OF_ADVER_DRONES)

					# add avg actor and critic grad norms for adver drones
					writer.add_scalar(tag = "avg_adver_actor_grad_norm", scalar_value = avg_adver_actor_grad_norm, global_step = eps)
					writer.add_scalar(tag = "avg_adver_critic_grad_norm", scalar_value = avg_adver_critic_grad_norm, global_step = eps)

					# append avg actor and critic grad norms to their respective list
					avg_adver_actor_grad_norm_list.append(avg_adver_actor_grad_norm)
					avg_adver_critic_grad_norm_list.append(avg_adver_critic_grad_norm)
				
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

				# add adver goal
				writer.add_scalar(tag = "goal/adver_goal", scalar_value = adver_goal, global_step = eps)

				# check if maddpgv2
				if ADVER_MODEL == "maddpgv2":

					# append adver goal to adver_goals_list
					adver_goals_list.append(adver_goal)

			# log metrics for adver_model for mappo
			elif ADVER_MODEL == "mappo":

				# variables to track sum of actor and critic loss
				sum_adver_actor_loss = 0
				sum_adver_critic_loss = 0
				sum_adver_number_of_team_collisions = 0
				sum_adver_number_of_oppo_collisions = 0
				sum_adver_actor_grad_norm = 0
				sum_adver_critic_grad_norm = 0
				sum_adver_policy_ratio = 0

				# iterate over num of adver drones
				for i in range(NUMBER_OF_ADVER_DRONES): 

					# check if list is not nan and adversarial model is training
					if np.any(np.isnan(adver_actor_loss_list)) == False and ADVER_MODE != "test":

						# update sums
						sum_adver_actor_loss += adver_actor_loss_list[i]

						# add actor loss for adver drone
						writer.add_scalar(tag = "adver_drone_" + str(i + 1) + "/actor_loss", scalar_value = adver_actor_loss_list[i], global_step = eps)

					# check if list is not nan and adversarial model is training
					if np.any(np.isnan(adver_critic_loss_list)) == False and ADVER_MODE != "test":

						# update sums
						sum_adver_critic_loss += adver_critic_loss_list[i]

						# add critic loss for adver drone
						writer.add_scalar(tag = "adver_drone_" + str(i + 1) + "/critic_loss", scalar_value = adver_critic_loss_list[i], global_step = eps)

					# check if list is not nan and adversarial model is training
					if np.any(np.isnan(adver_actor_grad_norm_list)) == False and ADVER_MODE != "test":

						# update sums
						sum_adver_actor_grad_norm += adver_actor_grad_norm_list[i]

						# add actor and critic grad norms for adver drone
						writer.add_scalar(tag = "adver_drone_" + str(i + 1) + "/actor_grad_norm", scalar_value = adver_actor_grad_norm_list[i], global_step = eps)

					# check if list is not nan and adversarial model is training
					if np.any(np.isnan(adver_critic_grad_norm_list)) == False and ADVER_MODE != "test":

						# update sums
						sum_adver_critic_grad_norm += adver_critic_grad_norm_list[i]

						# add actor and critic grad norms for adver drone
						writer.add_scalar(tag = "adver_drone_" + str(i + 1) + "/critic_grad_norm", scalar_value = adver_critic_grad_norm_list[i], global_step = eps)

					# check if list is not nan and adversarial model is training
					if np.any(np.isnan(adver_policy_ratio_list)) == False and ADVER_MODE != "test":

						# update sums
						sum_adver_policy_ratio += adver_policy_ratio_list[i]

						# add critic grad norms for adver drone
						writer.add_scalar(tag = "adver_drone_" + str(i + 1) + "/policy_ratio", scalar_value = adver_policy_ratio_list[i], global_step = eps)

					# update sum of team and opponent collisions
					sum_adver_number_of_team_collisions += adver_benchmark_data[i, 0]
					sum_adver_number_of_oppo_collisions += adver_benchmark_data[i, 1]

					# add actor and critic loss for adver drone
					writer.add_scalar(tag = "adver_drone_" + str(i + 1) + "/team_collisions", scalar_value = adver_benchmark_data[i, 0], global_step = eps)
					writer.add_scalar(tag = "adver_drone_" + str(i + 1) + "/opponent_collisions", scalar_value = adver_benchmark_data[i, 1], global_step = eps)

				# check if adver model is training
				if ADVER_MODE != "test":

					# obtain avg actor and critic loss
					avg_adver_actor_loss = sum_adver_actor_loss / float(NUMBER_OF_ADVER_DRONES)
					avg_adver_critic_loss = sum_adver_critic_loss / float(NUMBER_OF_ADVER_DRONES)

					# add avg actor and critic loss for adver drones
					writer.add_scalar(tag = "avg_adver_actor_loss", scalar_value = avg_adver_actor_loss, global_step = eps)
					writer.add_scalar(tag = "avg_adver_critic_loss", scalar_value = avg_adver_critic_loss, global_step = eps)

					# append avg_adver_actor_loss and avg_adver_critic_loss to their respective list
					avg_adver_actor_loss_list.append(avg_adver_actor_loss)
					avg_adver_critic_loss_list.append(avg_adver_critic_loss)

					# obtain avg actor and critic grad norms
					avg_adver_actor_grad_norm = sum_adver_actor_grad_norm / float(NUMBER_OF_ADVER_DRONES)
					avg_adver_critic_grad_norm = sum_adver_critic_grad_norm / float(NUMBER_OF_ADVER_DRONES)

					# add avg actor and critic grad norms for adver drones
					writer.add_scalar(tag = "avg_adver_actor_grad_norm", scalar_value = avg_adver_actor_grad_norm, global_step = eps)
					writer.add_scalar(tag = "avg_adver_critic_grad_norm", scalar_value = avg_adver_critic_grad_norm, global_step = eps)

					# append avg actor and critic grad norms to their respective list
					avg_adver_actor_grad_norm_list.append(avg_adver_actor_grad_norm)
					avg_adver_critic_grad_norm_list.append(avg_adver_critic_grad_norm)

					# obtain avg avg_adver_policy_ratio
					avg_adver_policy_ratio = sum_adver_policy_ratio / float(NUMBER_OF_ADVER_DRONES)

					# add avg_adver_policy_ratio for agent drones
					writer.add_scalar(tag = "avg_adver_policy_ratio", scalar_value = avg_adver_policy_ratio, global_step = eps)

					# append avg_adver_policy_ratio to list
					avg_adver_policy_ratio_list.append(avg_adver_policy_ratio)  
				
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

			# log wins for adversarial drones
			if terminal_condition == 1:

				# add sum of wins for adversarial drones
				sum_adver_wins += 1

				# append sums to all lists
				sum_agent_wins_list.append(sum_agent_wins)
				sum_adver_wins_list.append(sum_adver_wins)

				# add opp win and sum of agent win
				writer.add_scalar(tag = "terminate_info/agent_drone_win", scalar_value = 0, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_agent_drone_wins", scalar_value = sum_agent_wins, global_step = eps)

				# add opp win and sum of opp win
				writer.add_scalar(tag = "terminate_info/adver_drone_win", scalar_value = 1, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_adver_drone_wins", scalar_value = sum_adver_wins, global_step = eps)

				# check if experiment terminates if screen exits
				if EXIT_SCREEN_TERMINATE == True:

					# append sums to exceed screen lists
					sum_agent_exceed_screen_list.append(sum_agent_exceed_screen)
					sum_adver_exceed_screen_list.append(sum_adver_exceed_screen)

					# add agent_exceed_screen and sum_agent_exceed_screen
					writer.add_scalar(tag = "terminate_info/agent_exceed_screen", scalar_value = 0, global_step = eps)
					writer.add_scalar(tag = "terminate_info/sum_agent_exceed_screen", scalar_value = sum_agent_exceed_screen, global_step = eps)

					# add agent_exceed_screen and sum_agent_exceed_screen
					writer.add_scalar(tag = "terminate_info/adver_exceed_screen", scalar_value = 0, global_step = eps)
					writer.add_scalar(tag = "terminate_info/sum_adver_exceed_screen", scalar_value = sum_adver_exceed_screen, global_step = eps)

				# update agent_goals_softmax_weights for maddpgv2
				if AGENT_MODEL == "maddpgv2":

					update_agent_goals_softmax_weights(agent_goals_softmax_weights = agent_goals_softmax_weights, agent_goal_distribution = AGENT_MADDPGV2_GOAL_DISTRIBUTION, agent_elo = agent_elo, 
													   adver_elo = adver_elo, agent_goal = agent_goal, terminal_condition = terminal_condition)

				# update adver_goals_softmax_weights for maddpgv2
				if ADVER_MODEL == "maddpgv2":

					update_adver_goals_softmax_weights(adver_goals_softmax_weights = adver_goals_softmax_weights, adver_goal_distribution = ADVER_MADDPGV2_GOAL_DISTRIBUTION, agent_elo = agent_elo, 
													   adver_elo = adver_elo, adver_goal = adver_goal, terminal_condition = terminal_condition)

				# calculate elo for agent and adver
				agent_elo, adver_elo = calculate_elo_rating(agent_curr_elo = agent_elo, adver_curr_elo = adver_elo, k_agent = AGENT_ELO_K, k_adver = ADVER_ELO_K, d = ELO_D, 
															results_list = [terminal_condition], results_reward_dict = RESULTS_REWARD_DICT)

				# add agent and adver elo
				writer.add_scalar(tag = "elo/agent_elo", scalar_value = agent_elo, global_step = eps)
				writer.add_scalar(tag = "elo/adver_elo", scalar_value = adver_elo, global_step = eps)

				# append elo to list
				agent_elo_list.append(agent_elo)
				adver_elo_list.append(adver_elo)

				# print log
				print(f'terminal_condition {terminal_condition}: adversarial drones win')

				# additional log message of goals for maddpgv2 for agent
				if AGENT_MODEL == "maddpgv2":

					# print agent_goal
					print(f'agent_goal (episode time limit): {agent_goal}')

				# additional log message of goals for maddpgv2 for adversarial
				if ADVER_MODEL == "maddpgv2":

					# print adver_goal
					print(f'adver_goal (restricted radius): {adver_goal}')

			# log wins for agent drones
			elif terminal_condition == 2:

				# add sum of wins for agent drones
				sum_agent_wins += 1

				# append sums to all lists
				sum_agent_wins_list.append(sum_agent_wins)
				sum_adver_wins_list.append(sum_adver_wins)

				# add opp win and sum of agent win
				writer.add_scalar(tag = "terminate_info/agent_drone_win", scalar_value = 1, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_agent_drone_wins", scalar_value = sum_agent_wins, global_step = eps)

				# add opp win and sum of opp win
				writer.add_scalar(tag = "terminate_info/adver_drone_win", scalar_value = 0, global_step = eps)
				writer.add_scalar(tag = "terminate_info/sum_adver_drone_wins", scalar_value = sum_adver_wins, global_step = eps)

				# check if experiment terminates if screen exits
				if EXIT_SCREEN_TERMINATE == True:

					# append sums to exceed screen lists
					sum_agent_exceed_screen_list.append(sum_agent_exceed_screen)
					sum_adver_exceed_screen_list.append(sum_adver_exceed_screen)

					# add agent_exceed_screen and sum_agent_exceed_screen
					writer.add_scalar(tag = "terminate_info/agent_exceed_screen", scalar_value = 0, global_step = eps)
					writer.add_scalar(tag = "terminate_info/sum_agent_exceed_screen", scalar_value = sum_agent_exceed_screen, global_step = eps)

					# add agent_exceed_screen and sum_agent_exceed_screen
					writer.add_scalar(tag = "terminate_info/adver_exceed_screen", scalar_value = 0, global_step = eps)
					writer.add_scalar(tag = "terminate_info/sum_adver_exceed_screen", scalar_value = sum_adver_exceed_screen, global_step = eps)

				# update agent_goals_softmax_weights for maddpgv2
				if AGENT_MODEL == "maddpgv2":

					update_agent_goals_softmax_weights(agent_goals_softmax_weights = agent_goals_softmax_weights, agent_goal_distribution = AGENT_MADDPGV2_GOAL_DISTRIBUTION, agent_elo = agent_elo, 
													   adver_elo = adver_elo, agent_goal = agent_goal, terminal_condition = terminal_condition)

				# update adver_goals_softmax_weights for maddpgv2
				if ADVER_MODEL == "maddpgv2":

					update_adver_goals_softmax_weights(adver_goals_softmax_weights = adver_goals_softmax_weights, adver_goal_distribution = ADVER_MADDPGV2_GOAL_DISTRIBUTION, agent_elo = agent_elo, 
													   adver_elo = adver_elo, adver_goal = adver_goal, terminal_condition = terminal_condition)

				# calculate elo for agent and adver
				agent_elo, adver_elo = calculate_elo_rating(agent_curr_elo = agent_elo, adver_curr_elo = adver_elo, k_agent = AGENT_ELO_K, k_adver = ADVER_ELO_K, d = ELO_D, 
															results_list = [terminal_condition], results_reward_dict = RESULTS_REWARD_DICT)

				# add agent and adver elo
				writer.add_scalar(tag = "elo/agent_elo", scalar_value = agent_elo, global_step = eps)
				writer.add_scalar(tag = "elo/adver_elo", scalar_value = adver_elo, global_step = eps)

				# append elo to list
				agent_elo_list.append(agent_elo)
				adver_elo_list.append(adver_elo)

				# print log
				print(f'terminal_condition {terminal_condition}: agent drones win')

				# additional log message of goals for maddpgv2 for agent
				if AGENT_MODEL == "maddpgv2":

					# print agent_goal
					print(f'agent_goal (episode time limit): {agent_goal}')

				# additional log message of goals for maddpgv2 for adversarial
				if ADVER_MODEL == "maddpgv2":

					# print adver_goal
					print(f'adver_goal (restricted radius): {adver_goal}')

			# check if experiment terminates if screen exits
			if EXIT_SCREEN_TERMINATE == True:

				# log agent drones exceeding screen boundary
				if terminal_condition == 3:

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

					# update agent_goals_softmax_weights for maddpgv2
					if AGENT_MODEL == "maddpgv2":

						update_agent_goals_softmax_weights(agent_goals_softmax_weights = agent_goals_softmax_weights, agent_goal_distribution = AGENT_MADDPGV2_GOAL_DISTRIBUTION, 
														   agent_elo = agent_elo, adver_elo = adver_elo, agent_goal = agent_goal, terminal_condition = terminal_condition)

					# update adver_goals_softmax_weights for maddpgv2
					if ADVER_MODEL == "maddpgv2":

						update_adver_goals_softmax_weights(adver_goals_softmax_weights = adver_goals_softmax_weights, adver_goal_distribution = ADVER_MADDPGV2_GOAL_DISTRIBUTION, 
														   agent_elo = agent_elo, adver_elo = adver_elo, adver_goal = adver_goal, terminal_condition = terminal_condition)

					# calculate elo for agent and adver
					agent_elo, adver_elo = calculate_elo_rating(agent_curr_elo = agent_elo, adver_curr_elo = adver_elo, k_agent = AGENT_ELO_K, k_adver = ADVER_ELO_K, d = ELO_D, 
																results_list = [terminal_condition], results_reward_dict = RESULTS_REWARD_DICT)

					# add agent and adver elo
					writer.add_scalar(tag = "elo/agent_elo", scalar_value = agent_elo, global_step = eps)
					writer.add_scalar(tag = "elo/adver_elo", scalar_value = adver_elo, global_step = eps)

					# append elo to list
					agent_elo_list.append(agent_elo)
					adver_elo_list.append(adver_elo)

					# print log
					print(f'terminal_condition {terminal_condition}: agent drones exceeded screen boundary')

					# additional log message of goals for maddpgv2 for agent
					if AGENT_MODEL == "maddpgv2":

						# print agent_goal
						print(f'agent_goal (episode time limit): {agent_goal}')

					# additional log message of goals for maddpgv2 for adversarial
					if ADVER_MODEL == "maddpgv2":

						# print adver_goal
						print(f'adver_goal (restricted radius): {adver_goal}')

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

					# update agent_goals_softmax_weights for maddpgv2
					if AGENT_MODEL == "maddpgv2":

						update_agent_goals_softmax_weights(agent_goals_softmax_weights = agent_goals_softmax_weights, agent_goal_distribution = AGENT_MADDPGV2_GOAL_DISTRIBUTION, 
														   agent_elo = agent_elo, adver_elo = adver_elo, agent_goal = agent_goal, terminal_condition = terminal_condition)

					# update adver_goals_softmax_weights for maddpgv2
					if ADVER_MODEL == "maddpgv2":

						update_adver_goals_softmax_weights(adver_goals_softmax_weights = adver_goals_softmax_weights, adver_goal_distribution = ADVER_MADDPGV2_GOAL_DISTRIBUTION, 
														   agent_elo = agent_elo, adver_elo = adver_elo, adver_goal = adver_goal, terminal_condition = terminal_condition)

					# calculate elo for agent and adver
					agent_elo, adver_elo = calculate_elo_rating(agent_curr_elo = agent_elo, adver_curr_elo = adver_elo, k_agent = AGENT_ELO_K, k_adver = ADVER_ELO_K, d = ELO_D, 
																results_list = [terminal_condition], results_reward_dict = RESULTS_REWARD_DICT)

					# add agent and adver elo
					writer.add_scalar(tag = "elo/agent_elo", scalar_value = agent_elo, global_step = eps)
					writer.add_scalar(tag = "elo/adver_elo", scalar_value = adver_elo, global_step = eps)

					# append elo to list
					agent_elo_list.append(agent_elo)
					adver_elo_list.append(adver_elo)

					# print log
					print(f'terminal_condition {terminal_condition}: adversarial drones exceeded screen boundary')

					# additional log message of goals for maddpgv2 for agent
					if AGENT_MODEL == "maddpgv2":

						# print agent_goal
						print(f'agent_goal (episode time limit): {agent_goal}')

					# additional log message of goals for maddpgv2 for adversarial
					if ADVER_MODEL == "maddpgv2":

						# print adver_goal
						print(f'adver_goal (restricted radius): {adver_goal}')

			# update actor_states, adver_actor_states, agent_actor_states
			actor_states = actor_states_prime
			adver_actor_states = adver_actor_states_prime
			agent_actor_states = agent_actor_states_prime

			# check if metrics is to be saved in csv log
			if SAVE_CSV_LOG == True:

				# check if exit screen is terminate
				if EXIT_SCREEN_TERMINATE == True:  

					# for agent and adver using maddpg
					if AGENT_MODEL == "maddpg" and ADVER_MODEL == "maddpg":

						# check if agent and adversarial model are both training
						if AGENT_MODE != "test" and ADVER_MODE != "test":

							# both gradient clipping
							if AGENT_MADDPG_GRADIENT_CLIPPING == True and ADVER_MADDPG_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'avg_adver_actor_grad_norm', 
															 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# adver gradient clipping only
							elif AGENT_MADDPG_GRADIENT_CLIPPING == False and ADVER_MADDPG_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# agent gradient clipping only
							elif AGENT_MADDPG_GRADIENT_CLIPPING == True and ADVER_MADDPG_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list,
														   avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# no gradient clipping
							elif AGENT_MADDPG_GRADIENT_CLIPPING == False and ADVER_MADDPG_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'agent_elo', 'adver_elo'])

						# check if agent model is testing and adversarial model is training
						elif AGENT_MODE == "test" and ADVER_MODE != "test":

							# both gradient clipping
							if AGENT_MADDPG_GRADIENT_CLIPPING == True and ADVER_MADDPG_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list,
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 
															 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# adver gradient clipping only
							elif AGENT_MADDPG_GRADIENT_CLIPPING == False and ADVER_MADDPG_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list,
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 
															 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# agent gradient clipping only
							elif AGENT_MADDPG_GRADIENT_CLIPPING == True and ADVER_MADDPG_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list,
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 
															 'agent_elo', 'adver_elo'])

							# no gradient clipping
							elif AGENT_MADDPG_GRADIENT_CLIPPING == False and ADVER_MADDPG_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list,
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 
															 'agent_elo', 'adver_elo'])

						# check if agent model is training and adversarial model is testing
						elif AGENT_MODE !=  "test" and ADVER_MODE == "test":

							# both gradient clipping
							if AGENT_MADDPG_GRADIENT_CLIPPING == True and ADVER_MADDPG_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# adver gradient clipping only
							elif AGENT_MADDPG_GRADIENT_CLIPPING == False and ADVER_MADDPG_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'agent_elo', 'adver_elo'])
							# agent gradient clipping only
							elif AGENT_MADDPG_GRADIENT_CLIPPING == True and ADVER_MADDPG_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# no gradient clipping
							elif AGENT_MADDPG_GRADIENT_CLIPPING == False and ADVER_MADDPG_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'agent_elo', 'adver_elo'])

						# check if agent and adversarial model are both testing
						if AGENT_MODE ==  "test" and ADVER_MODE == "test":

							# generate pandas dataframe to store logs
							df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
													   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
													   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
													   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, agent_elo_list, adver_elo_list)), 
											  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 'sum_agent_number_of_team_collisions', 
														 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 
														 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 
														 'avg_adver_number_of_oppo_collisions', 'agent_elo', 'adver_elo'])

					# for agent and adver using mappo 
					elif AGENT_MODEL == "mappo" and ADVER_MODEL == "mappo":

						# check if agent and adversarial model are both training
						if AGENT_MODE != "test" and ADVER_MODE != "test":

							# both gradient clipping
							if AGENT_MAPPO_GRADIENT_CLIPPING == True and ADVER_MAPPO_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, avg_agent_policy_ratio_list, avg_adver_policy_ratio_list, 
														   agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'avg_adver_actor_grad_norm', 
															 'avg_adver_critic_grad_norm', 'avg_agent_policy_ratio_list', 'avg_adver_policy_ratio_list', 'agent_elo', 'adver_elo'])

							# adver gradient clipping only
							elif AGENT_MAPPO_GRADIENT_CLIPPING == False and ADVER_MAPPO_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, avg_agent_policy_ratio_list, avg_adver_policy_ratio_list, 
														   agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 
															 'avg_agent_policy_ratio_list', 'avg_adver_policy_ratio_list', 'agent_elo', 'adver_elo'])

							# agent gradient clipping only
							elif AGENT_MAPPO_GRADIENT_CLIPPING == True and ADVER_MAPPO_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list,
														   avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, avg_agent_policy_ratio_list, avg_adver_policy_ratio_list, 
														   agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 
															 'avg_agent_policy_ratio_list', 'avg_adver_policy_ratio_list', 'agent_elo', 'adver_elo'])

							# no gradient clipping
							elif AGENT_MAPPO_GRADIENT_CLIPPING == False and ADVER_MAPPO_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, avg_agent_policy_ratio_list, avg_adver_policy_ratio_list, 
														   agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_agent_policy_ratio_list', 'avg_adver_policy_ratio_list', 
															 'agent_elo', 'adver_elo'])

						# check if agent model is testing and adversarial model is training
						elif AGENT_MODE == "test" and ADVER_MODE != "test":

							# both gradient clipping
							if AGENT_MAPPO_GRADIENT_CLIPPING == True and ADVER_MAPPO_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list,
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, avg_adver_policy_ratio_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 
															 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'avg_adver_policy_ratio_list', 'agent_elo', 'adver_elo'])

							# adver gradient clipping only
							elif AGENT_MAPPO_GRADIENT_CLIPPING == False and ADVER_MAPPO_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list,
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, avg_adver_policy_ratio_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 
															 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'avg_adver_policy_ratio_list', 'agent_elo', 'adver_elo'])

							# agent gradient clipping only
							elif AGENT_MAPPO_GRADIENT_CLIPPING == True and ADVER_MAPPO_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list,
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, avg_adver_policy_ratio_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 
															 'avg_adver_policy_ratio_list', 'agent_elo', 'adver_elo'])

							# no gradient clipping
							elif AGENT_MAPPO_GRADIENT_CLIPPING == False and ADVER_MAPPO_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list,
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, avg_adver_policy_ratio_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 
															 'avg_adver_policy_ratio_list', 'agent_elo', 'adver_elo'])

						# check if agent model is training and adversarial model is testing
						elif AGENT_MODE !=  "test" and ADVER_MODE == "test":

							# both gradient clipping
							if AGENT_MAPPO_GRADIENT_CLIPPING == True and ADVER_MAPPO_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, avg_agent_policy_ratio_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'avg_agent_policy_ratio_list', 'agent_elo', 'adver_elo'])

							# adver gradient clipping only
							elif AGENT_MAPPO_GRADIENT_CLIPPING == False and ADVER_MAPPO_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, avg_agent_policy_ratio_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_agent_policy_ratio_list', 'agent_elo', 'adver_elo'])
							# agent gradient clipping only
							elif AGENT_MAPPO_GRADIENT_CLIPPING == True and ADVER_MAPPO_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, avg_agent_policy_ratio_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'avg_agent_policy_ratio_list', 'agent_elo', 'adver_elo'])

							# no gradient clipping
							elif AGENT_MAPPO_GRADIENT_CLIPPING == False and ADVER_MAPPO_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, avg_agent_policy_ratio_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_agent_policy_ratio_list', 'agent_elo', 'adver_elo'])

						# check if agent and adversarial model are both testing
						if AGENT_MODE ==  "test" and ADVER_MODE == "test":

							# generate pandas dataframe to store logs
							df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
													   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
													   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
													   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, agent_elo_list, adver_elo_list)), 
											  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 'sum_agent_number_of_team_collisions', 
														 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 
														 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 
														 'avg_adver_number_of_oppo_collisions', 'agent_elo', 'adver_elo'])

					# for agent and adver using maddpgv2
					elif AGENT_MODEL == "maddpgv2" and ADVER_MODEL == "maddpgv2":

						# check if agent and adversarial model are both training
						if AGENT_MODE != "test" and ADVER_MODE != "test":

							# both gradient clipping
							if AGENT_MADDPGV2_GRADIENT_CLIPPING == True and ADVER_MADDPGV2_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'avg_adver_actor_grad_norm', 
															 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])

							# adver gradient clipping only
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == False and ADVER_MADDPGV2_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo', 
															 'agent_goal', 'adver_goal'])

							# agent gradient clipping only
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == True and ADVER_MADDPGV2_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list,
														   avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'agent_elo', 'adver_elo', 
															 'agent_goal', 'adver_goal'])

							# no gradient clipping
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == False and ADVER_MADDPGV2_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])

						# check if agent model is testing and adversarial model is training
						elif AGENT_MODE == "test" and ADVER_MODE != "test":

							# both gradient clipping
							if AGENT_MADDPGV2_GRADIENT_CLIPPING == True and ADVER_MADDPGV2_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list,
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 
															 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])

							# adver gradient clipping only
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == False and ADVER_MADDPGV2_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list,
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 
															 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])

							# agent gradient clipping only
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == True and ADVER_MADDPGV2_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list,
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 
															 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])

							# no gradient clipping
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == False and ADVER_MADDPGV2_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list,
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, 
														   avg_adver_actor_grad_norm_list, avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 
															 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])

						# check if agent model is training and adversarial model is testing
						elif AGENT_MODE !=  "test" and ADVER_MODE == "test":

							# both gradient clipping
							if AGENT_MADDPGV2_GRADIENT_CLIPPING == True and ADVER_MADDPGV2_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])

							# adver gradient clipping only
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == False and ADVER_MADDPGV2_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])
							# agent gradient clipping only
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == True and ADVER_MADDPGV2_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])

							# no gradient clipping
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == False and ADVER_MADDPGV2_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
														   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
														   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
														   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, 
														   avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 
															 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 
															 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 
															 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 'avg_agent_actor_loss', 'avg_agent_critic_loss', 
															 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])

						# check if agent and adversarial model are both testing
						if AGENT_MODE ==  "test" and ADVER_MODE == "test":

							# generate pandas dataframe to store logs
							df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list, sum_adver_exceed_screen_list, 
													   sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, 
													   sum_adver_number_of_oppo_collisions_list, avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, 
													   avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list, agent_elo_list, adver_elo_list, agent_goals_list, 
													   adver_goals_list)), 
											  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen', 'sum_adver_exceed_screen', 'sum_agent_number_of_team_collisions', 
														 'sum_agent_number_of_oppo_collisions', 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 
														 'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 
														 'avg_adver_number_of_oppo_collisions', 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])

				# for not terminating on exit screen
				elif EXIT_SCREEN_TERMINATE == False:

					# for agent and adver using maddpg
					if AGENT_MODEL == "maddpg" and ADVER_MODEL == "maddpg":

						# check if agent and adversarial model are both training
						if AGENT_MODE != "test" and ADVER_MODE != "test":

							# both gradient clipping
							if AGENT_MADDPG_GRADIENT_CLIPPING == True and ADVER_MADDPG_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_adver_actor_loss_list, 
														   avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_agent_actor_grad_norm', 
															 'avg_agent_critic_grad_norm', 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# adver gradient clipping only
							elif AGENT_MADDPG_GRADIENT_CLIPPING == False and ADVER_MADDPG_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_adver_actor_loss_list, 
														   avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_adver_actor_grad_norm', 
															 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# agent gradient clipping only
							elif AGENT_MADDPG_GRADIENT_CLIPPING == True and ADVER_MADDPG_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_adver_actor_loss_list, 
														   avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_agent_actor_grad_norm', 
															 'avg_agent_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# no gradient clipping
							elif AGENT_MADDPG_GRADIENT_CLIPPING == False and ADVER_MADDPG_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_adver_actor_loss_list, 
														   avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'agent_elo', 'adver_elo'])

						# check if agent model is testing and adversarial model is training
						elif AGENT_MODE == "test" and ADVER_MODE != "test":

							# both gradient clipping
							if AGENT_MADDPG_GRADIENT_CLIPPING == True and ADVER_MADDPG_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# adver gradient clipping only
							elif AGENT_MADDPG_GRADIENT_CLIPPING == False and ADVER_MADDPG_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'agent_elo', 'adver_elo'])

							# agent gradient clipping only
							elif AGENT_MADDPG_GRADIENT_CLIPPING == True and ADVER_MADDPG_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# no gradient clipping
							elif AGENT_MADDPG_GRADIENT_CLIPPING == False and ADVER_MADDPG_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'agent_elo', 'adver_elo'])


						# check if agent model is training and adversarial model is testing
						elif AGENT_MODE !=  "test" and ADVER_MODE == "test":

							# both gradient clipping
							if AGENT_MADDPG_GRADIENT_CLIPPING == True and ADVER_MADDPG_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_agent_actor_grad_norm_list, 
														   avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# adver gradient clipping only
							elif AGENT_MADDPG_GRADIENT_CLIPPING == False and ADVER_MADDPG_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_agent_actor_grad_norm_list, 
														   avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'agent_elo', 'adver_elo'])

							# agent gradient clipping only
							elif AGENT_MADDPG_GRADIENT_CLIPPING == True and ADVER_MADDPG_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_agent_actor_grad_norm_list, 
														   avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# no gradient clipping
							elif AGENT_MADDPG_GRADIENT_CLIPPING == False and ADVER_MADDPG_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_agent_actor_grad_norm_list, 
														   avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'agent_elo', 'adver_elo'])

						# check if agent and adversarial model are both testing
						if AGENT_MODE ==  "test" and ADVER_MODE == "test":

							# generate pandas dataframe to store logs
							df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
													   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
													   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
													   avg_adver_number_of_oppo_collisions_list, agent_elo_list, adver_elo_list)), 
											  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
														 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
														 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
														 'agent_elo', 'adver_elo'])

					# for agent and adver using mappo
					elif AGENT_MODEL == "mappo" and ADVER_MODEL == "mappo":

						# check if agent and adversarial model are both training
						if AGENT_MODE != "test" and ADVER_MODE != "test":

							# both gradient clipping
							if AGENT_MAPPO_GRADIENT_CLIPPING == True and ADVER_MAPPO_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_adver_actor_loss_list, 
														   avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_agent_actor_grad_norm', 
															 'avg_agent_critic_grad_norm', 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# adver gradient clipping only
							elif AGENT_MAPPO_GRADIENT_CLIPPING == False and ADVER_MAPPO_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_adver_actor_loss_list, 
														   avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_adver_actor_grad_norm', 
															 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# agent gradient clipping only
							elif AGENT_MAPPO_GRADIENT_CLIPPING == True and ADVER_MAPPO_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_adver_actor_loss_list, 
														   avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_agent_actor_grad_norm', 
															 'avg_agent_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# no gradient clipping
							elif AGENT_MAPPO_GRADIENT_CLIPPING == False and ADVER_MAPPO_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_adver_actor_loss_list, 
														   avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'agent_elo', 'adver_elo'])

						# check if agent model is testing and adversarial model is training
						elif AGENT_MODE == "test" and ADVER_MODE != "test":

							# both gradient clipping
							if AGENT_MAPPO_GRADIENT_CLIPPING == True and ADVER_MAPPO_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# adver gradient clipping only
							elif AGENT_MAPPO_GRADIENT_CLIPPING == False and ADVER_MAPPO_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'agent_elo', 'adver_elo'])

							# agent gradient clipping only
							elif AGENT_MAPPO_GRADIENT_CLIPPING == True and ADVER_MAPPO_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# no gradient clipping
							elif AGENT_MAPPO_GRADIENT_CLIPPING == False and ADVER_MAPPO_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'agent_elo', 'adver_elo'])


						# check if agent model is training and adversarial model is testing
						elif AGENT_MODE !=  "test" and ADVER_MODE == "test":

							# both gradient clipping
							if AGENT_MAPPO_GRADIENT_CLIPPING == True and ADVER_MAPPO_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_agent_actor_grad_norm_list, 
														   avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# adver gradient clipping only
							elif AGENT_MAPPO_GRADIENT_CLIPPING == False and ADVER_MAPPO_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_agent_actor_grad_norm_list, 
														   avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'agent_elo', 'adver_elo'])

							# agent gradient clipping only
							elif AGENT_MAPPO_GRADIENT_CLIPPING == True and ADVER_MAPPO_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_agent_actor_grad_norm_list, 
														   avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'agent_elo', 'adver_elo'])

							# no gradient clipping
							elif AGENT_MAPPO_GRADIENT_CLIPPING == False and ADVER_MAPPO_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_agent_actor_grad_norm_list, 
														   avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'agent_elo', 'adver_elo'])

						# check if agent and adversarial model are both testing
						if AGENT_MODE ==  "test" and ADVER_MODE == "test":

							# generate pandas dataframe to store logs
							df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
													   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
													   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
													   avg_adver_number_of_oppo_collisions_list, agent_elo_list, adver_elo_list)), 
											  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
														 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
														 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
														 'agent_elo', 'adver_elo'])

					# for agent and adver using maddpgv2
					if AGENT_MODEL == "maddpgv2" and ADVER_MODEL == "maddpgv2":

						# check if agent and adversarial model are both training
						if AGENT_MODE != "test" and ADVER_MODE != "test":

							# both gradient clipping
							if AGENT_MADDPGV2_GRADIENT_CLIPPING == True and ADVER_MADDPGV2_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_adver_actor_loss_list, 
														   avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_agent_actor_grad_norm', 
															 'avg_agent_critic_grad_norm', 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo', 'agent_goal', 
															 'adver_goal'])

							# adver gradient clipping only
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == False and ADVER_MADDPGV2_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_adver_actor_loss_list, 
														   avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_adver_actor_grad_norm', 
															 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])

							# agent gradient clipping only
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == True and ADVER_MADDPGV2_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_adver_actor_loss_list, 
														   avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_agent_actor_grad_norm', 
															 'avg_agent_critic_grad_norm', 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])

							# no gradient clipping
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == False and ADVER_MADDPGV2_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_adver_actor_loss_list, 
														   avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list, avg_agent_critic_grad_norm_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'agent_elo', 'adver_elo', 'agent_goal', 
															 'adver_goal'])

						# check if agent model is testing and adversarial model is training
						elif AGENT_MODE == "test" and ADVER_MODE != "test":

							# both gradient clipping
							if AGENT_MADDPGV2_GRADIENT_CLIPPING == True and ADVER_MADDPGV2_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo', 
															 'agent_goal', 'adver_goal'])

							# adver gradient clipping only
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == False and ADVER_MADDPGV2_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])

							# agent gradient clipping only
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == True and ADVER_MADDPGV2_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_adver_actor_grad_norm', 'avg_adver_critic_grad_norm', 'agent_elo', 'adver_elo', 
															 'agent_goal', 'adver_goal'])

							# no gradient clipping
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == False and ADVER_MADDPGV2_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_adver_actor_grad_norm_list, 
														   avg_adver_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_adver_actor_loss', 'avg_adver_critic_loss', 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])


						# check if agent model is training and adversarial model is testing
						elif AGENT_MODE !=  "test" and ADVER_MODE == "test":

							# both gradient clipping
							if AGENT_MADDPGV2_GRADIENT_CLIPPING == True and ADVER_MADDPGV2_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_agent_actor_grad_norm_list, 
														   avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'agent_elo', 'adver_elo', 
															 'agent_goal', 'adver_goal'])

							# adver gradient clipping only
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == False and ADVER_MADDPGV2_GRADIENT_CLIPPING == True:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_agent_actor_grad_norm_list, 
														   avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])

							# agent gradient clipping only
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == True and ADVER_MADDPGV2_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_agent_actor_grad_norm_list, 
														   avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'avg_agent_actor_grad_norm', 'avg_agent_critic_grad_norm', 'agent_elo', 'adver_elo', 
															 'agent_goal', 'adver_goal'])

							# no gradient clipping
							elif AGENT_MADDPGV2_GRADIENT_CLIPPING == False and ADVER_MADDPGV2_GRADIENT_CLIPPING == False:

								# generate pandas dataframe to store logs
								df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
														   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
														   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
														   avg_adver_number_of_oppo_collisions_list, avg_agent_actor_loss_list, avg_agent_critic_loss_list, avg_agent_actor_grad_norm_list, 
														   avg_agent_critic_grad_norm_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
												  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
															 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
															 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
															 'avg_agent_actor_loss', 'avg_agent_critic_loss', 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])

						# check if agent and adversarial model are both testing
						if AGENT_MODE ==  "test" and ADVER_MODE == "test":

							# generate pandas dataframe to store logs
							df = pd.DataFrame(list(zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_number_of_team_collisions_list, 
													   sum_agent_number_of_oppo_collisions_list, sum_adver_number_of_team_collisions_list, sum_adver_number_of_oppo_collisions_list, 
													   avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list, avg_adver_number_of_team_collisions_list, 
													   avg_adver_number_of_oppo_collisions_list, agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)), 
											  columns = ['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions', 
														 'sum_adver_number_of_team_collisions', 'sum_adver_number_of_oppo_collisions', 'avg_agent_number_of_team_collisions', 
														 'avg_agent_number_of_oppo_collisions', 'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions', 
														 'agent_elo', 'adver_elo', 'agent_goal', 'adver_goal'])	

				# store training logs
				df.to_csv(CSV_LOG_DIRECTORY + '/' + GENERAL_TRAINING_NAME + "_" + AGENT_MODE + "_" + ADVER_MODE + "_logs.csv", index = False)

			# reset agent_eps_steps
			if AGENT_MODEL == "mappo" and agent_eps_steps % AGENT_MAPPO_EPISODE_LENGTH == 0:

				agent_eps_steps = 0

			# reset adver_eps_steps
			if ADVER_MODEL == "mappo" and adver_eps_steps % ADVER_MAPPO_EPISODE_LENGTH == 0:

				adver_eps_steps = 0

		# check if agent is training and at correct episode to save
		if AGENT_MODE != "test" and eps % SAVE_MODEL_RATE == 0:

			# check if agent model is maddpg
			if AGENT_MODEL == "maddpg":

				# save all models
				agent_maddpg_agents.save_all_models()

			# check if agent model is mappo
			elif AGENT_MODEL == "mappo":

				# save all models
				agent_mappo_agents.save_all_models()

			# check if agent model is maddpgv2
			elif AGENT_MODEL == "maddpgv2":

				# save all models
				agent_maddpgv2_agents.save_all_models()

		# check if adver is training and at correct episode to save
		if ADVER_MODE != "test" and eps % SAVE_MODEL_RATE == 0:

			# check if adver model is maddpg
			if ADVER_MODEL == "maddpg":

				# save all models
				adver_maddpg_agents.save_all_models()

			# check if adver model is mappo
			elif ADVER_MODEL == "mappo":

				# save all models
				adver_mappo_agents.save_all_models()

			# check if adver model is maddpgv2
			elif ADVER_MODEL == "maddpgv2":

				# save all models
				adver_maddpgv2_agents.save_all_models()

if __name__ == "__main__":

	train_test()