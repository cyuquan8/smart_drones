# ==========================================================================================================================================================
# maddpgv2 gnn class
# purpose: class to train multiple agents
# ==========================================================================================================================================================

import os
import numpy as np
import torch as T
import torch.nn.functional as F
from maddpgv2_gnn.maddpgv2_gnn_agent import maddpgv2_gnn_agent
from maddpgv2_gnn.maddpgv2_gnn_replay_buffer import maddpgv2_gnn_replay_buffer
from torch_geometric.data import Batch

class maddpgv2_gnn:
    
    def __init__(self, mode, scenario_name, training_name, discount_rate, lr_actor, lr_critic, num_agents, num_opp, actor_dropout_p, critic_dropout_p, state_fc_input_dims, state_fc_output_dims, 
                 u_action_dims, c_action_dims, num_heads, bool_concat, gnn_input_dims, gnn_output_dims, gmt_hidden_dims, gmt_output_dims, u_actions_fc_input_dims, u_actions_fc_output_dims, 
                 c_actions_fc_input_dims, c_actions_fc_output_dims, concat_fc_output_dims, goal_fc_input_dims, goal_fc_output_dims, tau, mem_size, batch_size, update_target, grad_clipping, 
                 grad_norm_clip, num_of_add_goals, goal_strategy, is_adversary, ep_time_limit, r_rad, big_rew_cnst, rew_multiplier_cnst, pos_dims, exit_screen_terminate):
            
        """ class constructor for attributes of the maddpg class (for multiple agents) """
        
        # list to store maddpg agents
        self.maddpgv2_gnn_agents_list = []
        
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
        
        # counter for apply gradients
        self.apply_grad_counter = 0 
        
        # step for apply_grad_counter to hardcopy weights of original to target
        self.update_target = update_target
        
        # gradient clipping
        self.grad_clipping = grad_clipping
        self.grad_norm_clip = grad_norm_clip

        # check if agent
        if is_adversary == False:

            # iterate over num_agents
            for i in range(num_agents):
                
                # append maddpg agent to list
                self.maddpgv2_gnn_agents_list.append(maddpgv2_gnn_agent(mode = mode, scenario_name = scenario_name, training_name = training_name, discount_rate = discount_rate, lr_actor = lr_actor, 
                                                                        lr_critic = lr_critic, num_agents = num_agents, num_opp = num_opp, actor_dropout_p = actor_dropout_p, 
                                                                        critic_dropout_p = critic_dropout_p, state_fc_input_dims = state_fc_input_dims[i], 
                                                                        state_fc_output_dims = state_fc_output_dims, u_action_dims = u_action_dims, c_action_dims = c_action_dims, 
                                                                        num_heads = num_heads, bool_concat = bool_concat, gnn_input_dims = gnn_input_dims[i], gnn_output_dims = gnn_output_dims, 
                                                                        gmt_hidden_dims = gmt_hidden_dims, gmt_output_dims = gmt_output_dims, u_actions_fc_input_dims = u_actions_fc_input_dims, 
                                                                        u_actions_fc_output_dims = u_actions_fc_output_dims, c_actions_fc_input_dims = c_actions_fc_input_dims, 
                                                                        c_actions_fc_output_dims = c_actions_fc_output_dims, goal_fc_input_dims = goal_fc_input_dims, 
                                                                        goal_fc_output_dims = goal_fc_output_dims, concat_fc_output_dims = concat_fc_output_dims, tau = tau, 
                                                                        is_adversary = is_adversary))
                
                # update actor model_names attributes for checkpoints
                self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_actor.model_name = "maddpgv2_gnn_actor"
        
                # update actor checkpoints_path attributes
                self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_actor.checkpoint_path = os.path.join(self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_actor.checkpoint_dir, 
                                                                                                   self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_actor.model_name + "_" + str(i) + ".pt")
                
                # update target actor model_names attributes for checkpoints
                self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_target_actor.model_name = "maddpgv2_gnn_target_actor"
        
                # update target actor checkpoints_path attributes
                self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_target_actor.checkpoint_path = os.path.join(self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_target_actor.checkpoint_dir, 
                                                                                                          self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_target_actor.model_name + "_" + str(i) + \
                                                                                                          ".pt")
                
                # update critic model_names attributes for checkpoints
                self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_critic.model_name = "maddpgv2_gnn_critic"
        
                # update critic checkpoints_path attributes
                self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_critic.checkpoint_path = os.path.join(self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_critic.checkpoint_dir, 
                                                                                                    self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_critic.model_name + "_" + str(i) + ".pt")
                
                # update target critic model_names attributes for checkpoints
                self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_target_critic.model_name = "maddpgv2_gnn_target_critic"
        
                # update target critic checkpoints_path attributes
                self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_target_critic.checkpoint_path = os.path.join(self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_target_critic.checkpoint_dir, 
                                                                                                           self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_target_critic.model_name + "_" + str(i) + \
                                                                                                           ".pt")

            # if mode is not test
            if mode == 'train':
                
                # create replay buffer
                self.replay_buffer = maddpgv2_gnn_replay_buffer(mem_size = mem_size, num_agents = num_agents, u_actions_dims = u_action_dims, c_actions_dims = c_action_dims, 
                                                                actor_input_dims = state_fc_input_dims, goal_dims = goal_fc_input_dims, num_of_add_goals = num_of_add_goals, 
                                                                goal_strategy = goal_strategy, is_adversary = is_adversary, ep_time_limit = ep_time_limit, r_rad = r_rad, 
                                                                big_rew_cnst = big_rew_cnst, rew_multiplier_cnst = rew_multiplier_cnst, pos_dims = pos_dims, 
                                                                exit_screen_terminate = exit_screen_terminate)
        
            # if test mode
            elif mode == 'test':
                
                # load all models
                self.load_all_models()

            elif mode == "load_and_train":

                # create replay buffer
                self.replay_buffer = maddpgv2_gnn_replay_buffer(mem_size = mem_size, num_agents = num_agents, u_actions_dims = u_action_dims, c_actions_dims = c_action_dims, 
                                                                actor_input_dims = state_fc_input_dims, goal_dims = goal_fc_input_dims, num_of_add_goals = num_of_add_goals, 
                                                                goal_strategy = goal_strategy, is_adversary = is_adversary, ep_time_limit = ep_time_limit, r_rad = r_rad, 
                                                                big_rew_cnst = big_rew_cnst, rew_multiplier_cnst = rew_multiplier_cnst, pos_dims = pos_dims, 
                                                                exit_screen_terminate = exit_screen_terminate)
                    
                # load all models
                self.load_all_models()

        # check if agent
        elif is_adversary == True:

            # iterate over num_agents
            for i in range(num_opp):
                
                # append maddpg agent to list
                self.maddpgv2_gnn_agents_list.append(maddpgv2_gnn_agent(mode = mode, scenario_name = scenario_name, training_name = training_name, discount_rate = discount_rate, lr_actor = lr_actor, 
                                                                        lr_critic = lr_critic, num_agents = num_agents, num_opp = num_opp, actor_dropout_p = actor_dropout_p, 
                                                                        critic_dropout_p = critic_dropout_p, state_fc_input_dims = state_fc_input_dims[i], 
                                                                        state_fc_output_dims = state_fc_output_dims, u_action_dims = u_action_dims, c_action_dims = c_action_dims, 
                                                                        num_heads = num_heads, bool_concat = bool_concat, gnn_input_dims = gnn_input_dims[i], gnn_output_dims = gnn_output_dims, 
                                                                        gmt_hidden_dims = gmt_hidden_dims, gmt_output_dims = gmt_output_dims, u_actions_fc_input_dims = u_actions_fc_input_dims, 
                                                                        u_actions_fc_output_dims = u_actions_fc_output_dims, c_actions_fc_input_dims = c_actions_fc_input_dims, 
                                                                        c_actions_fc_output_dims = c_actions_fc_output_dims, goal_fc_input_dims = goal_fc_input_dims, 
                                                                        goal_fc_output_dims = goal_fc_output_dims, concat_fc_output_dims = concat_fc_output_dims, tau = tau, 
                                                                        is_adversary = is_adversary))
                
                # update actor model_names attributes for checkpoints
                self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_actor.model_name = "maddpgv2_gnn_actor"
        
                # update actor checkpoints_path attributes
                self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_actor.checkpoint_path = os.path.join(self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_actor.checkpoint_dir, 
                                                                                                   self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_actor.model_name + "_" + str(i) + ".pt")
                
                # update target actor model_names attributes for checkpoints
                self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_target_actor.model_name = "maddpgv2_gnn_target_actor"
        
                # update target actor checkpoints_path attributes
                self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_target_actor.checkpoint_path = os.path.join(self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_target_actor.checkpoint_dir, 
                                                                                                          self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_target_actor.model_name + "_" + str(i) + \
                                                                                                          ".pt")
                
                # update critic model_names attributes for checkpoints
                self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_critic.model_name = "maddpgv2_gnn_critic"
        
                # update critic checkpoints_path attributes
                self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_critic.checkpoint_path = os.path.join(self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_critic.checkpoint_dir, 
                                                                                                    self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_critic.model_name + "_" + str(i) + ".pt")
                
                # update target critic model_names attributes for checkpoints
                self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_target_critic.model_name = "maddpgv2_gnn_target_critic"
        
                # update target critic checkpoints_path attributes
                self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_target_critic.checkpoint_path = os.path.join(self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_target_critic.checkpoint_dir, 
                                                                                                           self.maddpgv2_gnn_agents_list[i].maddpgv2_gnn_target_critic.model_name + "_" + str(i) + \
                                                                                                           ".pt")

            # if mode is not test
            if mode == 'train':
                
                # create replay buffer
                self.replay_buffer = maddpgv2_gnn_replay_buffer(mem_size = mem_size, num_agents = num_agents, u_actions_dims = u_action_dims, c_actions_dims = c_action_dims, 
                                                                actor_input_dims = state_fc_input_dims, goal_dims = goal_fc_input_dims, num_of_add_goals = num_of_add_goals, 
                                                                goal_strategy = goal_strategy, is_adversary = is_adversary, ep_time_limit = ep_time_limit, r_rad = r_rad, 
                                                                big_rew_cnst = big_rew_cnst, rew_multiplier_cnst = rew_multiplier_cnst, pos_dims = pos_dims, 
                                                                exit_screen_terminate = exit_screen_terminate)
        
            # if test mode
            elif mode == 'test':
                
                # load all models
                self.load_all_models()

            elif mode == "load_and_train":

                # create replay buffer
                self.replay_buffer = maddpgv2_gnn_replay_buffer(mem_size = mem_size, num_agents = num_agents, u_actions_dims = u_action_dims, c_actions_dims = c_action_dims, 
                                                                actor_input_dims = state_fc_input_dims, goal_dims = goal_fc_input_dims, num_of_add_goals = num_of_add_goals, 
                                                                goal_strategy = goal_strategy, is_adversary = is_adversary, ep_time_limit = ep_time_limit, r_rad = r_rad, 
                                                                big_rew_cnst = big_rew_cnst, rew_multiplier_cnst = rew_multiplier_cnst, pos_dims = pos_dims, 
                                                                exit_screen_terminate = exit_screen_terminate)
                    
                # load all models
                self.load_all_models()

    def select_actions(self, mode, env_agents, actor_state_list):
       
        """ function to select actions for the all agents given state observed by respective agent """
         
        # initialise empty list to store motor, communication actions and all actions from all agents
        u_actions_list = []
        c_actions_list = []
        actions_list = []

        # iterate over num_agents
        for agent_index, agent in enumerate(self.maddpgv2_gnn_agents_list):
            
            # select action for respective agent from corresponding list of states observed by agent
            u_action, c_action = agent.select_action(mode = mode, agent = env_agents[agent_index], state = actor_state_list[agent_index])
            
            # append actions to respective lists
            u_actions_list.append(u_action)
            c_actions_list.append(c_action)
            actions_list.append([np.array(u_action, dtype = np.float32), np.array(c_action, dtype = np.float32)])

        return np.array(u_actions_list), np.array(c_actions_list), actions_list
    
    def apply_gradients_maddpgv2_gnn(self, num_of_agents):
        
        """ function to apply gradients for maddpg to learn from replay buffer """

        # doesnt not apply gradients if memory does not have at least batch_size number of logs
        if self.replay_buffer.org_replay_buffer.mem_counter < self.batch_size:
            
            return np.nan, np.nan, np.nan, np.nan
        
        # sample replay buffer
        actor_state_list, actor_state_prime_list, actor_u_action_list, actor_c_action_list, actor_goals_list, critic_state_list, critic_state_prime_list, critic_goals_list, rewards, terminal = \
        self.replay_buffer.sample_log(self.batch_size)

        # obtain device (should be same for all models)
        device = self.maddpgv2_gnn_agents_list[0].maddpgv2_gnn_actor.device
    
        # turn features to tensors for critic in device
        critic_state = Batch().from_data_list(critic_state_list).to(device)
        critic_state_prime = Batch().from_data_list(critic_state_prime_list).to(device)
        critic_goals = T.tensor(critic_goals_list, dtype = T.float).to(device)
        actor_u_action = T.tensor(actor_u_action_list, dtype = T.float).to(device)
        actor_c_action = T.tensor(actor_c_action_list, dtype = T.float).to(device)
        rewards = T.tensor(rewards, dtype = T.float).to(device)
        terminal = T.tensor(terminal, dtype = T.bool).to(device)
        
        # generate batch tensor for graph multiset transformer in critic model
        batch = T.tensor([i for i in range(self.batch_size) for j in range(num_of_agents)], dtype = T.long).to(device)

        # generate list to store actor and target actor actions tensor output
        curr_target_actor_u_actions_prime_list = []
        curr_target_actor_c_actions_prime_list = []
        curr_actor_u_actions_list = []
        curr_actor_c_actions_list = []
        past_actor_u_actions_list = []
        past_actor_c_actions_list = []
        
        # enumerate over agents
        for agent_index, agent in enumerate(self.maddpgv2_gnn_agents_list):
            
            # set all models to eval mode to calculate td_target
            agent.maddpgv2_gnn_actor.eval()
            agent.maddpgv2_gnn_critic.eval()
            agent.maddpgv2_gnn_target_actor.eval()
            agent.maddpgv2_gnn_target_critic.eval()

            # convert actor_state_prime to tensor
            actor_state_prime = T.tensor(np.concatenate((actor_state_prime_list[agent_index], actor_goals_list[agent_index]), axis = -1), dtype = T.float).to(device)
            
            # feed actor_state_prime tensor to target actor to obtain actions
            curr_target_actor_u_actions_prime, curr_target_actor_c_actions_prime = agent.maddpgv2_gnn_target_actor.forward(actor_state_prime)
            
            # append actions to curr_target_actor_u_actions_prime_list and curr_target_actor_c_actions_prime_list
            curr_target_actor_u_actions_prime_list.append(curr_target_actor_u_actions_prime)
            curr_target_actor_c_actions_prime_list.append(curr_target_actor_c_actions_prime)
            
            # convert action_state to tensor 
            actor_state = T.tensor(np.concatenate((actor_state_list[agent_index], actor_goals_list[agent_index]), axis = -1), dtype = T.float).to(device)
            
            # feed actor_state tensor to actor to obtain actions
            curr_actor_u_actions, curr_actor_c_actions = agent.maddpgv2_gnn_actor.forward(actor_state)
            
            # append actions to curr_actor_u_actions_list and curr_actor_c_actions_list
            curr_actor_u_actions_list.append(curr_actor_u_actions)
            curr_actor_c_actions_list.append(curr_actor_c_actions)
            
            # append actions from past actor parameters from replay_buffer to past_actor_u_actions_list and past_actor_c_actions_list
            past_actor_u_actions_list.append(actor_u_action[agent_index])
            past_actor_c_actions_list.append(actor_c_action[agent_index])

        # concat actions in list
        curr_target_actor_u_actions_prime_cat = T.cat([action for action in curr_target_actor_u_actions_prime_list], dim = 1)
        curr_target_actor_c_actions_prime_cat = T.cat([action for action in curr_target_actor_c_actions_prime_list], dim = 1)
        curr_actor_u_actions_cat = T.cat([action for action in curr_actor_u_actions_list], dim = 1)
        curr_actor_c_actions_cat = T.cat([action for action in curr_actor_c_actions_list], dim = 1)
        past_actor_u_actions_cat = T.cat([action for action in past_actor_u_actions_list], dim = 1)
        past_actor_c_actions_cat = T.cat([action for action in past_actor_c_actions_list], dim = 1)
        
        # list to store metrics for logging
        actor_loss_list = []
        critic_loss_list = []
        actor_grad_norm_list = []
        critic_grad_norm_list = []

        # enumerate over agents
        for agent_index, agent in enumerate(self.maddpgv2_gnn_agents_list):
          
            # obtain target q value prime
            target_critic_q_value_prime = agent.maddpgv2_gnn_target_critic.forward(critic_state_prime, batch, curr_target_actor_u_actions_prime_cat, curr_target_actor_c_actions_prime_cat, 
                                                                                   critic_goals).flatten()
            
            # mask terminal target q values with 0
            target_critic_q_value_prime[terminal[:, 0]] = 0.0
            
            # obtain critic q value
            critic_q_value = agent.maddpgv2_gnn_critic.forward(critic_state, batch, past_actor_u_actions_cat, past_actor_c_actions_cat, critic_goals).flatten()
            
            # obtain td_target
            td_target = rewards[:, agent_index] + agent.discount_rate * target_critic_q_value_prime
            
            # critic loss is mean squared error between td_target and critic value 
            critic_loss = F.mse_loss(td_target, critic_q_value)
            
            # set critic model to train mode 
            agent.maddpgv2_gnn_critic.train()
             
            # reset gradients for critic model to zero
            agent.maddpgv2_gnn_critic.optimizer.zero_grad()
            
            # critic model back propagation
            critic_loss.backward(retain_graph = True)
            
            # check if gradient clipping is needed
            if self.grad_clipping == True:
            
                # gradient norm clipping for critic model
                critic_grad_norm = T.nn.utils.clip_grad_norm_(agent.maddpgv2_gnn_critic.parameters(), max_norm = self.grad_norm_clip, norm_type = 2, error_if_nonfinite = True)

            # apply gradients to critic model
            agent.maddpgv2_gnn_critic.optimizer.step()
            
            # set critic to eval mode to calculate actor loss
            agent.maddpgv2_gnn_critic.eval()
            
            # set actor model to train mode 
            agent.maddpgv2_gnn_actor.train()
            
            # gradient ascent using critic value ouput as actor loss
            # loss is coupled with actor model from actor actions based on current policy 
            actor_loss = agent.maddpgv2_gnn_critic.forward(critic_state, batch, curr_actor_u_actions_cat, curr_actor_c_actions_cat, critic_goals).flatten()
           
            # reduce mean across batch_size
            actor_loss = -T.mean(actor_loss)
            
            # reset gradients for actor model to zero
            agent.maddpgv2_gnn_actor.optimizer.zero_grad()

            # actor model back propagation
            actor_loss.backward(retain_graph = True, inputs = list(agent.maddpgv2_gnn_actor.parameters()))
            
            # check if gradient clipping is needed
            if self.grad_clipping == True:
            
                # gradient norm clipping for critic model
                actor_grad_norm = T.nn.utils.clip_grad_norm_(agent.maddpgv2_gnn_actor.parameters(), max_norm = self.grad_norm_clip, norm_type = 2, error_if_nonfinite = True)

            # apply gradients to actor model
            agent.maddpgv2_gnn_actor.optimizer.step()
            
            # increment of apply_grad_counter
            self.apply_grad_counter += 1 
            
            # soft copy option: update target models based on user specified tau
            if self.update_target == None:
                
                # iterate over agents
                for agent in self.maddpgv2_gnn_agents_list:

                    # update target models
                    agent.update_maddpgv2_gnn_target_models()    
    
            # hard copy option every update_target steps
            else:
                
                if self.apply_grad_counter % self.update_target == 0: 
                
                    # iterate over agents
                    for agent in self.maddpgv2_gnn_agents_list:

                        # update target models
                        agent.update_maddpgv2_gnn_target_models()     

            # store actor and critic losses in list
            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())

            # check if there is grad clipping
            if self.grad_clipping == True:

                # append clipped gradients
                actor_grad_norm_list.append(actor_grad_norm.item())
                critic_grad_norm_list.append(critic_grad_norm.item())

            else:

                # append 0.0 if there is no gradient clipping
                actor_grad_norm_list.append(0.0)
                critic_grad_norm_list.append(0.0)

        return actor_loss_list, critic_loss_list, actor_grad_norm_list, critic_grad_norm_list
           
    def save_all_models(self):
        
        """ save weights for all models """
        
        print("saving models!")
        
        # iterate over num_agents
        for agent in self.maddpgv2_gnn_agents_list:
            
            # save each model
            agent.save_models()
    
    def load_all_models(self):
      
        """ load weights for all models """

        print("loading model!")

        # iterate over num_agents
        for agent in self.maddpgv2_gnn_agents_list:

            # load each model
            agent.load_models()