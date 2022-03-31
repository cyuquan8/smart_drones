# ==========================================================================================================================================================
# maddpg gnn replay buffer class
# purpose: store memory of state, action, state_prime, reward, terminal flag and function to sample them
# ==========================================================================================================================================================

import numpy as np

class maddpg_gnn_replay_buffer:
    
    def __init__(self, mem_size, num_agents, u_actions_dims, c_actions_dims, actor_input_dims):
        
        """ class constructor that initialises memory states attributes """
        
        # bound for memory log
        self.mem_size = mem_size
        
        # counter for memory logged
        self.mem_counter = 0 
        
        # number of agents
        self.num_agents = num_agents
        
        # dimensions of action for motor and communications
        self.u_action_dims = u_actions_dims
        self.c_action_dims = c_actions_dims

        # dimensions of action space
        self.actions_dims = self.u_action_dims + self.c_action_dims
        
        # reward_log is list of reward from num_agents of actors
        # terminal_log indicates if episode is terminated
        self.rewards_log = np.zeros((self.mem_size, self.num_agents)) 
        self.terminal_log = np.zeros((self.mem_size, self.num_agents), dtype = bool)
        
        # list to store num_agents of each actor log of state, state_prime and actions
        self.actor_state_log_list = []
        self.actor_state_prime_log_list = []
        self.actor_u_action_log_list = []
        self.actor_c_action_log_list = []

        # list to store graph data representation of critic state and state prime
        self.critic_state_log_list = [0 for i in range(self.mem_size)]
        self.critic_state_prime_log_list = [0 for i in range(self.mem_size)]
        
        # iterate over num_agents
        for actor_index in range(self.num_agents):
            
            # append each actor log to list
            # actor_state and actor_state_prime are local observations of environment by each actor
            self.actor_state_log_list.append(np.zeros((self.mem_size, actor_input_dims[actor_index])))
            self.actor_state_prime_log_list.append(np.zeros((self.mem_size, actor_input_dims[actor_index])))
            self.actor_u_action_log_list.append(np.zeros((self.mem_size, u_actions_dims)))
            self.actor_c_action_log_list.append(np.zeros((self.mem_size, c_actions_dims)))
    
    def log(self, actor_state, actor_state_prime, critic_state, critic_state_prime, u_action, c_action, rewards, is_done):
        
        """ function to log memory """
        
        # index for logging. based on first in first out
        index = self.mem_counter % self.mem_size
        
        # iterate over num_agents
        for actor_index in range(self.num_agents):
            
            # log actor_state, actor_state_prime, motor and communication action for each actor
            self.actor_state_log_list[actor_index][index] = actor_state[actor_index]
            self.actor_state_prime_log_list[actor_index][index] = actor_state_prime[actor_index]
            self.actor_u_action_log_list[actor_index][index] = u_action[actor_index]
            self.actor_c_action_log_list[actor_index][index] = c_action[actor_index]
        
        # log critic_fc_state, critic_fc_state_prime, rewards and terminal flag
        self.critic_state_log_list[index] = critic_state
        self.critic_state_prime_log_list[index] = critic_state_prime
        self.rewards_log[index] = rewards
        self.terminal_log[index] = is_done
        
        # increment counter
        self.mem_counter += 1
    
    def sample_log(self, batch_size):
        
        """ function to randomly sample a batch of memory """
        
        # select amongst memory logs that is filled
        max_mem = min(self.mem_counter, self.mem_size)
        
        # randomly select memory from logs
        batch = np.random.choice(max_mem, batch_size, replace = False)
        
        # initialise list for actor_state, actor_state_prime, actions, critic_state, critic_state_prime
        actor_state_log_list = []
        actor_state_prime_log_list = []
        actor_u_action_log_list = []
        actor_c_action_log_list = []
        critic_state_log_list = []
        critic_state_prime_log_list = []

        # iterate over num_agents
        for actor_index in range(self.num_agents):
            
            # obtain corresponding actor_state, actor_state_prime and actions
            actor_state_log_list.append(self.actor_state_log_list[actor_index][batch])
            actor_state_prime_log_list.append(self.actor_state_prime_log_list[actor_index][batch])
            actor_u_action_log_list.append(self.actor_u_action_log_list[actor_index][batch])
            actor_c_action_log_list.append(self.actor_c_action_log_list[actor_index][batch])
        
        # obtain corresponding rewards, terminal flag
        rewards_log = self.rewards_log[batch]
        terminal_log = self.terminal_log[batch]
        
        # iterate over batch for gnn data in critic state
        for index in batch:

            # append relevant state
            critic_state_log_list.append(self.critic_state_log_list[index]) 
            critic_state_prime_log_list.append(self.critic_state_prime_log_list[index])
        
        return actor_state_log_list, actor_state_prime_log_list, actor_u_action_log_list, actor_c_action_log_list, critic_state_log_list, critic_state_prime_log_list, rewards_log, terminal_log