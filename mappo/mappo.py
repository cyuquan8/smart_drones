# ==========================================================================================================================================================
# mappo class
# purpose: class to train multiple agents
# ==========================================================================================================================================================

import os
import numpy as np
import torch as T
import torch.nn.functional as F
from mappo.mappo_agent import mappo_agent
from mappo.mappo_replay_buffer import mappo_replay_buffer
from torch_geometric.data import Batch

class mappo:
    
    def __init__(self, mode, training_name, lr_actor, lr_critic, num_agents, num_opp, u_range, u_noise, c_noise, is_adversary, actor_dropout_p, critic_dropout_p, state_fc_input_dims, 
                 state_fc_output_dims, u_action_dims, c_action_dims, num_heads, bool_concat, gnn_input_dims, gnn_output_dims, gmt_hidden_dims, gmt_output_dims, fc_output_dims, batch_size, gamma, 
                 clip_coeff, num_epochs, gae_lambda, entropy_coeff, use_huber_loss, huber_delta, use_clipped_value_loss, critic_loss_coeff, grad_clipping, grad_norm_clip):
            
        """ class constructor for attributes of the mappo class (for multiple agents) """
        
        # list to store mappo agents
        self.mappo_agents_list = []
        
        # number of agent and adversarial drones
        self.num_agents = num_agents
        self.num_opp = num_opp

        # dimensions of action for motor and communications
        self.u_action_dims = u_action_dims
        self.c_action_dims = c_action_dims

        # dimensions of action space
        self.actions_dims = self.u_action_dims + self.c_action_dims
        
        # batch of memory to sample
        self.batch_size = batch_size

        # factor in discount rate for general advantage estimation
        self.gamma = gamma

        # variable for clip
        self.clip_coeff = clip_coeff

        # number of epochs
        self.num_epochs = num_epochs

        # factor in discount rate for general advantage estimation
        self.gae_lambda = gae_lambda

        # constant to scale entropy
        self.entropy_coeff = entropy_coeff

        # boolean to determine to use huber loss for value loss
        self.use_huber_loss = use_huber_loss

        # huber loss variable
        self.huber_delta = huber_delta

        # boolean to choose to use clipped or original value loss
        self.use_clipped_value_loss = use_clipped_value_loss

        # constant to scale critic_loss
        self.critic_loss_coeff = critic_loss_coeff

        # gradient clipping
        self.grad_clipping = grad_clipping
        self.grad_norm_clip = grad_norm_clip

        # if mode is not test
        if mode == 'train':
            
            # create replay buffer
            self.replay_buffer = mappo_replay_buffer(num_agents = num_agents, batch_size = batch_size)

        # if test mode
        elif mode == 'test':
            
            # load all models
            self.load_all_models()

        elif mode == "load_and_train":

            # create replay buffer
            self.replay_buffer = mappo_replay_buffer(num_agents = num_agents, batch_size = batch_size)

            # load all models
            self.load_all_models()

        # iterate over num_agents
        for i in range(num_agents):
            
            # append mappo agent to list
            self.mappo_agents_list.append(mappo_agent(mode = mode, training_name = training_name, lr_actor = lr_actor, lr_critic = lr_critic, num_agents = num_agents, 
                                                      num_opp = num_opp, u_range = u_range, u_noise = u_noise, c_noise = c_noise, is_adversary = is_adversary, actor_dropout_p = actor_dropout_p, 
                                                      critic_dropout_p = critic_dropout_p, state_fc_input_dims = state_fc_input_dims[i], state_fc_output_dims = state_fc_output_dims, 
                                                      u_action_dims = u_action_dims, c_action_dims = c_action_dims, num_heads = num_heads, bool_concat = bool_concat, 
                                                      gnn_input_dims = gnn_input_dims[i], gnn_output_dims = gnn_output_dims, gmt_hidden_dims = gmt_hidden_dims, gmt_output_dims = gmt_output_dims, 
                                                      fc_output_dims = fc_output_dims))
            
            # update actor model_names attributes for checkpoints
            self.mappo_agents_list[i].mappo_actor.model_name = "mappo_actor"
    
            # update actor checkpoints_path attributes
            self.mappo_agents_list[i].mappo_actor.checkpoint_path = os.path.join(self.mappo_agents_list[i].mappo_actor.checkpoint_dir, 
                                                                                 self.mappo_agents_list[i].mappo_actor.model_name + "_" + str(i) + ".pt")

            # update critic model_names attributes for checkpoints
            self.mappo_agents_list[i].mappo_critic.model_name = "mappo_critic"
    
            # update critic checkpoints_path attributes
            self.mappo_agents_list[i].mappo_critic.checkpoint_path = os.path.join(self.mappo_agents_list[i].mappo_critic.checkpoint_dir, 
                                                                                  self.mappo_agents_list[i].mappo_critic.model_name + "_" + str(i) + ".pt")

    def select_actions(self, mode, env_agents, actor_state_list):
       
        """ function to select actions for the all agents given state observed by respective agent """
         
        # initialise empty list to store motor, communication actions and their respective log probabiities and all actions from all agents
        u_actions_list = []
        c_actions_list = []
        u_actions_log_probs_list = []
        c_actions_log_probs_list = []
        actions_list = []

        # iterate over num_agents
        for agent_index, agent in enumerate(self.mappo_agents_list):
            
            # select action for respective agent from corresponding list of states observed by agent
            u_action, c_action, u_action_log_probs, c_action_log_probs = agent.select_action(mode = mode, agent = env_agents[agent_index], state = actor_state_list[agent_index])
            
            # append actions to respective lists
            u_actions_list.append(u_action)
            c_actions_list.append(c_action)
            u_actions_log_probs_list.append(u_action_log_probs)
            c_actions_log_probs_list.append(c_action_log_probs)
            actions_list.append([np.array(u_action, dtype = np.float32), np.array(c_action, dtype = np.float32)])

        return np.array(u_actions_list), np.array(c_actions_list), np.array(u_actions_log_probs_list), np.array(c_actions_log_probs_list), actions_list
    
    def apply_gradients_mappo(self, num_of_agents):
        
        """ function to apply gradients for mappo to learn from replay buffer """
        
        # obtain device (should be same for all models)
        device = self.mappo_agents_list[0].mappo_actor.device

        # generate batch tensor for graph multiset transformer in critic model
        critic_batch = T.tensor([i for i in range(self.batch_size) for j in range(num_of_agents)], dtype = T.long).to(device)

        # list to store metrics for logging
        actor_loss_list = []
        critic_loss_list = []
        actor_grad_norm_list = []
        critic_grad_norm_list = []
        policy_ratio_list = []

        # enumerate over agents
        for agent_index, agent in enumerate(self.mappo_agents_list): 

            # obtain value_normaliser
            value_normaliser = agent.mappo_critic.popart

            # variables to store metric
            avg_actor_loss_value = 0.0
            avg_critic_loss_value = 0.0
            avg_actor_grad_norm_value = 0.0
            avg_critic_grad_norm_value = 0.0
            avg_policy_ratio_value = 0.0

            # iterate over number of epochs
            for _ in range(self.num_epochs):

                # sample replay buffer
                actor_state_list, actor_u_action_list, actor_c_action_list, actor_u_action_log_probs_list, actor_c_action_log_probs_list, critic_state_list, critic_state_value_list, rewards, \
                terminal, batches = self.replay_buffer.sample_log(self.batch_size)
                
                # # tensor for advatange and q_values
                # advantages = T.zeros(len(rewards), dtype = T.float, device = device)
                # q_values = T.zeros(len(rewards), dtype = T.float, device = device)

                # convert rewards and critic state value list
                critic_state_value_list_t = T.tensor(critic_state_value_list, dtype = T.float).to(device)

                # numpy arrays for advantage and q_values
                advantages = np.zeros(len(rewards), dtype = np.float32)
                q_values = np.zeros(len(rewards), dtype = np.float32)

                # variable to track gae
                gae = 0

                # iterate over timesteps 
                for step in reversed(range(len(rewards) - 1)):
                    
                    # obtain td_delta error
                    td_delta = rewards[step] + self.gamma * value_normaliser.denormalize(critic_state_value_list_t[agent_index][step + 1]) * (1 - terminal[step + 1]) - \
                               value_normaliser.denormalize(critic_state_value_list_t[agent_index][step])
                    
                    # obtain gae
                    gae = td_delta + self.gamma * self.gae_lambda * gae
                    
                    # obtain advantage and q_values
                    advantages[step] = gae
                    q_values[step] = gae + value_normaliser.denormalize(critic_state_value_list_t[agent_index][step])

                # obtain normalised advantages
                advantages_copy = advantages.copy()
                mean_advantages = np.nanmean(advantages_copy)
                std_advantages = np.nanstd(advantages_copy)
                advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

                # tensor for advatange and q_values
                advantages = T.tensor(advantages, dtype = T.float).to(device)
                q_values = T.tensor(q_values, dtype = T.float).to(device)

                # iterate over batches
                for batch in batches:
                    
                    # covert features to tensors
                    actor_state = T.tensor(actor_state_list[agent_index][batch], dtype = T.float).to(device)
                    critic_state = Batch().from_data_list([critic_state_list[i] for i in batch]).to(device)
                    critic_state_value = T.tensor(critic_state_value_list[agent_index][batch], dtype = T.float).to(device)
                    actor_u_action_log_probs = T.tensor(actor_u_action_log_probs_list[agent_index][batch], dtype = T.float).to(device)
                    actor_c_action_log_probs = T.tensor(actor_c_action_log_probs_list[agent_index][batch], dtype = T.float).to(device)
                    actor_u_action = T.tensor(actor_u_action_list[agent_index][batch], dtype = T.float).to(device)
                    actor_c_action = T.tensor(actor_c_action_list[agent_index][batch], dtype = T.float).to(device)

                    # obtain state value based on current critic
                    critic_state_value_prime = agent.mappo_critic.forward(critic_state, critic_batch)

                    # obtain actions prime distributions
                    actor_u_action_prime_norm_dist, actor_c_action_prime_norm_dist = agent.mappo_actor.forward(actor_state)

                    # obtain log probs of actions prime
                    actor_u_action_prime_log_probs = actor_u_action_prime_norm_dist.log_prob(actor_u_action)
                    actor_c_action_prime_log_probs = actor_c_action_prime_norm_dist.log_prob(actor_c_action)

                    # obtain entropy from actions
                    actor_u_action_prime_entropy = actor_u_action_prime_norm_dist.entropy().mean()
                    actor_c_action_prime_entropy = actor_c_action_prime_norm_dist.entropy().mean()
                   
                    # obtain policy ratio
                    policy_ratio = T.cat((T.exp(actor_u_action_prime_log_probs), T.exp(actor_c_action_prime_log_probs)), axis = 1) / \
                                   T.cat((T.exp(actor_u_action_log_probs), T.exp(actor_c_action_log_probs)), axis = 1)
                    
                    # obtain weighted policy ratio
                    weighted_policy_ratio = policy_ratio * T.unsqueeze(advantages[batch], dim = 1)
                    
                    # obtain weighted clipped policy ratio
                    weighted_clipped_policy_ratio = T.clamp(policy_ratio, 1 - self.clip_coeff, 1 + self.clip_coeff) * T.unsqueeze(advantages[batch], dim = 1)

                    # obtain actor loss 
                    actor_loss = - T.sum(T.min(weighted_policy_ratio, weighted_clipped_policy_ratio), dim = -1, keepdim = True).mean() - \
                                (actor_u_action_prime_entropy + actor_c_action_prime_entropy) * self.entropy_coeff

                    # reset gradients for actor model to zero
                    agent.mappo_actor.optimizer.zero_grad() 

                    # actor model back propagation
                    actor_loss.backward()

                    # check if gradient clipping is needed
                    if self.grad_clipping == True:
                    
                        # gradient norm clipping for critic model
                        actor_grad_norm = T.nn.utils.clip_grad_norm_(agent.mappo_actor.parameters(), max_norm = self.grad_norm_clip, norm_type = 2, error_if_nonfinite = True)

                    # apply gradients to actor model
                    agent.mappo_actor.optimizer.step()
                    
                    # obtain clipped state value
                    critic_state_value_clipped = T.unsqueeze(critic_state_value, dim = 1) + \
                                                (critic_state_value_prime - T.unsqueeze(critic_state_value, dim = 1)).clamp(- self.clip_coeff, self.clip_coeff)
                    
                    # update value normaliser / popart
                    value_normaliser.update(q_values[batch])

                    # check if to use huber loss
                    if self.use_huber_loss == True:

                        # obtain huber loss for clipped and original
                        critic_loss_clipped = T.nn.functional.huber_loss(input = T.reshape(critic_state_value_clipped, (1, -1)), 
                                              target = value_normaliser.normalize(q_values[batch]), delta = self.huber_delta)
                        critic_loss_original = T.nn.functional.huber_loss(input = T.reshape(critic_state_value_prime, (1, -1)), 
                                               target = value_normaliser.normalize(q_values[batch]), delta = self.huber_delta)

                    # else use mse
                    else:

                        # obtain mse  loss for clipped and original
                        critic_loss_clipped = T.nn.functional.mse_loss(input = T.reshape(critic_state_value_clipped, (1, -1)), target = value_normaliser.normalize(q_values[batch]))
                        critic_loss_original = T.nn.functional.mse_loss(input = T.reshape(critic_state_value_prime, (1, -1)), target = value_normaliser.normalize(q_values[batch]))

                    # check if to use clipped losses 
                    if self.use_clipped_value_loss == True:

                        # obtain max of losses
                        critic_loss = T.max(critic_loss_original, critic_loss_clipped)

                    # else use original
                    else:

                        critic_loss = critic_loss_original

                    # reset gradients for critic model to zero
                    agent.mappo_critic.optimizer.zero_grad()

                    # critic model back propagation
                    (critic_loss * self.critic_loss_coeff).backward()

                    # check if gradient clipping is needed
                    if self.grad_clipping == True:
                    
                        # gradient norm clipping for critic model
                        critic_grad_norm = T.nn.utils.clip_grad_norm_(agent.mappo_critic.parameters(), max_norm = self.grad_norm_clip, norm_type = 2, error_if_nonfinite = True)

                    # apply gradients to critic model
                    agent.mappo_critic.optimizer.step()

                    # update metric variables
                    avg_actor_loss_value += actor_loss.item()
                    avg_critic_loss_value += critic_loss.item()
                    avg_policy_ratio_value += policy_ratio.mean().item()

                    # check if gradient clipping is used
                    if self.grad_clipping == True:
                    
                        avg_actor_grad_norm_value += actor_grad_norm.item()
                        avg_critic_grad_norm_value += critic_grad_norm.item()

            # obtain average of metrics
            avg_actor_loss_value += avg_actor_loss_value / (self.num_epochs * len(batches))
            avg_critic_loss_value += avg_critic_loss_value / (self.num_epochs * len(batches))
            avg_actor_grad_norm_value += avg_actor_grad_norm_value / (self.num_epochs * len(batches))
            avg_critic_grad_norm_value += avg_critic_grad_norm_value / (self.num_epochs * len(batches))
            avg_policy_ratio_value += avg_policy_ratio_value / (self.num_epochs * len(batches))

            # append metrics to list
            actor_loss_list.append(avg_actor_loss_value)
            critic_loss_list.append(avg_critic_loss_value)
            actor_grad_norm_list.append(avg_actor_grad_norm_value)
            critic_grad_norm_list.append(avg_critic_grad_norm_value)
            policy_ratio_list.append(avg_policy_ratio_value)

        # clear replay buffer
        self.replay_buffer.clear_log()

        return actor_loss_list, critic_loss_list, actor_grad_norm_list, critic_grad_norm_list, policy_ratio_list
           
    def save_all_models(self):
        
        """ save weights for all models """
        
        print("saving models!")
        
        # iterate over num_agents
        for agent in self.mappo_agents_list:
            
            # save each model
            agent.save_models()
    
    def load_all_models(self):
      
      """ load weights for all models """
      
      print("loading model!")
      
      # iterate over num_agents
      for agent in self.mappo_agents_list:
        
          # load each model
          agent.load_models()