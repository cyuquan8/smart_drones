# ==========================================================================================================================================================
# mappo gnn replay buffer class
# purpose: store memory of state, action, state_prime, reward, terminal flag and function to sample them
# ==========================================================================================================================================================

import numpy as np

class mappo_gnn_replay_buffer:
    
    def __init__(self, num_agents, batch_size):
        
        """ class constructor that initialises memory states attributes """
        
        # number of agents
        self.num_agents = num_agents

        # batch size
        self.batch_size = batch_size
        
        # reward_log is list of reward from num_agents of actors
        # terminal_log indicates if episode is terminated
        self.rewards_log = []
        self.terminal_log = []
        
        # list to store num_agents of each actor log of state, actions, actions log probabilities
        self.actor_state_log_list = []
        self.actor_u_action_log_list = []
        self.actor_c_action_log_list = []
        self.actor_u_action_log_probs_log_list = []
        self.actor_c_action_log_probs_log_list = []

        # list to store graph data representation of critic state
        self.critic_state_log_list = []
        
        # list to store critic state value
        self.critic_state_value_log_list = []

        # iterate over num_agents
        for actor_index in range(self.num_agents):
            
            # append each actor log to list
            self.actor_state_log_list.append([])
            self.actor_u_action_log_list.append([])
            self.actor_c_action_log_list.append([])
            self.actor_u_action_log_probs_log_list.append([])
            self.actor_c_action_log_probs_log_list.append([])
            self.critic_state_value_log_list.append([])
            self.rewards_log.append([])
            self.terminal_log.append([])
    
    def log(self, actor_state, critic_state, critic_state_value, u_action, c_action, u_action_log_probs, c_action_log_probs, rewards, is_done):
        
        """ function to log memory """
        
        # iterate over num_agents
        for actor_index in range(self.num_agents):
            
            # log actor_state, motor and communication action and their log probabiities for each actor, critic_state_value, rewards and terminal 
            self.actor_state_log_list[actor_index].append(actor_state[actor_index])
            self.actor_u_action_log_list[actor_index].append(u_action[actor_index])
            self.actor_c_action_log_list[actor_index].append(c_action[actor_index])
            self.actor_u_action_log_probs_log_list[actor_index].append(u_action_log_probs[actor_index])
            self.actor_c_action_log_probs_log_list[actor_index].append(c_action_log_probs[actor_index])
            self.critic_state_value_log_list[actor_index].append(critic_state_value[actor_index])
            self.rewards_log[actor_index].append(rewards[actor_index])
            self.terminal_log[actor_index].append(is_done[actor_index])

        # log critic_fc_state
        self.critic_state_log_list.append(critic_state)
    
    def clear_log(self):

        """ function to clear memory """

        # reinitialise rewards and terminal log
        self.rewards_log = []
        self.terminal_log = []
        
        # reinitialise list to store num_agents of each actor log of state, actions, actions log probabilities
        self.actor_state_log_list = []
        self.actor_u_action_log_list = []
        self.actor_c_action_log_list = []
        self.actor_u_action_log_probs_log_list = []
        self.actor_c_action_log_probs_log_list = []

        # reintialise list to store graph data representation of critic state
        self.critic_state_log_list = []
        
        # list to store critic state value
        self.critic_state_value_log_list = []

        # iterate over num_agents
        for actor_index in range(self.num_agents):
            
            # reinitialise each actor log and critic state value
            self.actor_state_log_list.append([])
            self.actor_u_action_log_list.append([])
            self.actor_c_action_log_list.append([])
            self.actor_u_action_log_probs_log_list.append([])
            self.actor_c_action_log_probs_log_list.append([])
            self.critic_state_value_log_list.append([])
            self.rewards_log.append([])
            self.terminal_log.append([])

    def sample_log(self, batch_size):
        
        """ function to randomly sample a batch of memory """
        
        # obtain number of experiences stored in memory
        num_exp = len(self.critic_state_log_list)

        # obtain starting indices spaced by batch size
        batch_start = np.arange(0, num_exp, self.batch_size)

        # obtain memory indices
        indices = np.arange(num_exp, dtype = np.int64)

        # shuffle indices for stochasticity
        np.random.shuffle(indices)

        # obtain batches 
        batches = [indices[i: i + self.batch_size] for i in batch_start]

        return np.array(self.actor_state_log_list), np.array(self.actor_u_action_log_list), np.array(self.actor_c_action_log_list), np.array(self.actor_u_action_log_probs_log_list), \
               np.array(self.actor_c_action_log_probs_log_list), self.critic_state_log_list, np.array(self.critic_state_value_log_list), np.array(self.rewards_log), \
               np.array(self.terminal_log), batches

        # return self.actor_state_log_list, self.actor_u_action_log_list, self.actor_c_action_log_list, self.actor_u_action_log_probs_log_list, \
        #        self.actor_c_action_log_probs_log_list, self.critic_state_log_list, self.critic_state_value_log_list, self.rewards_log, \
        #        self.terminal_log, batches