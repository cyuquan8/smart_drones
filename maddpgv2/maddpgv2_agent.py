# ==========================================================================================================================================================
# maddpgv2 agent class 
# purpose: creates and updates neural network 
# ==========================================================================================================================================================

import torch as T
from nn import maddpgv2_actor_model, maddpgv2_critic_model

class maddpgv2_agent:
    
    def __init__(self, mode, training_name, discount_rate, lr_actor, lr_critic, num_agents, num_opp, actor_dropout_p, critic_dropout_p, state_fc_input_dims, state_fc_output_dims, u_action_dims, 
                 c_action_dims, num_heads, bool_concat, gnn_input_dims, gnn_output_dims, gmt_hidden_dims, gmt_output_dims, u_actions_fc_input_dims, u_actions_fc_output_dims, c_actions_fc_input_dims, 
                 c_actions_fc_output_dims, concat_fc_output_dims, goal_fc_input_dims, goal_fc_output_dims, tau, is_adversary):
        
        """ class constructor for maddpg agent attributes """
          
        # discount rate for critic loss (TD error)
        self.discount_rate = discount_rate
        
        # learning rate for actor model
        self.lr_actor = lr_actor
        
        # learning rate for critic model
        self.lr_critic = lr_critic
        
        # number of agent and adversarial drones
        self.num_agents = num_agents
        self.num_opp = num_opp

        # dimensions of action for motor and communications
        self.u_action_dims = u_action_dims
        self.c_action_dims = c_action_dims

        # dimensions of action space
        self.actions_dims = self.u_action_dims + self.c_action_dims

        # softcopy parameter for target network 
        self.tau = tau
        
        # intialise actor model 
        self.maddpgv2_actor = maddpgv2_actor_model(model = "maddpgv2_actor", model_name = None, mode = mode, training_name = training_name, learning_rate = self.lr_actor, num_agents = num_agents, 
                                                   num_opp = num_opp, dropout_p = actor_dropout_p, fc_input_dims = state_fc_input_dims + goal_fc_input_dims, fc_output_dims = state_fc_output_dims, 
                                                   tanh_actions_dims = u_action_dims, sig_actions_dims = c_action_dims)
                         
        # intialise target actor model
        self.maddpgv2_target_actor = maddpgv2_actor_model(model = "maddpgv2_actor", model_name = None, mode = mode, training_name = training_name, learning_rate = self.lr_actor, 
                                                          num_agents = num_agents, num_opp = num_opp, dropout_p = actor_dropout_p, fc_input_dims = state_fc_input_dims + goal_fc_input_dims, 
                                                          fc_output_dims = state_fc_output_dims, tanh_actions_dims = u_action_dims, sig_actions_dims = c_action_dims)

        # intialise critic model
        self.maddpgv2_critic = maddpgv2_critic_model(model = "maddpgv2_critic", model_name = None, mode = mode, training_name = training_name, learning_rate = self.lr_critic, 
                                                     num_agents = num_agents, num_opp = num_opp, num_heads = num_heads, bool_concat = bool_concat, dropout_p = critic_dropout_p, 
                                                     gnn_input_dims = gnn_input_dims, gnn_output_dims = gnn_output_dims, gmt_hidden_dims = gmt_hidden_dims, gmt_output_dims = gmt_output_dims, 
                                                     u_actions_fc_input_dims = u_actions_fc_input_dims, u_actions_fc_output_dims = u_actions_fc_output_dims, 
                                                     c_actions_fc_input_dims = c_actions_fc_input_dims, c_actions_fc_output_dims = c_actions_fc_output_dims, 
                                                     goal_fc_input_dims = goal_fc_input_dims * num_opp if is_adversary == True else goal_fc_input_dims * num_agents, 
                                                     goal_fc_output_dims = goal_fc_output_dims, concat_fc_output_dims = concat_fc_output_dims)

        # intialise target critic model
        self.maddpgv2_target_critic = maddpgv2_critic_model(model = "maddpgv2_critic", model_name = None, mode = mode, training_name = training_name, learning_rate = self.lr_critic, 
                                                            num_agents = num_agents, num_opp = num_opp, num_heads = num_heads, bool_concat = bool_concat, dropout_p = critic_dropout_p, 
                                                            gnn_input_dims = gnn_input_dims, gnn_output_dims = gnn_output_dims, gmt_hidden_dims = gmt_hidden_dims, 
                                                            gmt_output_dims = gmt_output_dims, u_actions_fc_input_dims = u_actions_fc_input_dims, 
                                                            u_actions_fc_output_dims = u_actions_fc_output_dims, c_actions_fc_input_dims = c_actions_fc_input_dims, 
                                                            c_actions_fc_output_dims = c_actions_fc_output_dims, 
                                                            goal_fc_input_dims = goal_fc_input_dims * num_opp if is_adversary == True else goal_fc_input_dims * num_agents, 
                                                            goal_fc_output_dims = goal_fc_output_dims, concat_fc_output_dims = concat_fc_output_dims)

        # hard update target models' weights to online network to match initialised weights
        self.update_maddpgv2_target_models(tau = 1)
        
    def update_maddpgv2_target_models(self, tau = None): 
        
        """ function to soft update target model weights for maddpg. hard update is possible if tau = 1 """
        
        # use tau attribute if tau not specified 
        if tau is None:
            
            tau = self.tau
        
        # iterate over coupled target actor and actor parameters 
        for target_actor_parameters, actor_parameters in zip(self.maddpgv2_target_actor.parameters(), self.maddpgv2_actor.parameters()):
            
            # apply soft update to target actor
            target_actor_parameters.data.copy_((1 - tau) * target_actor_parameters.data + tau * actor_parameters.data)
        
        # iterate over coupled target critic and critic parameters
        for target_critic_parameters, critic_parameters in zip(self.maddpgv2_target_critic.parameters(), self.maddpgv2_critic.parameters()):

            # apply soft update to target critic
            target_critic_parameters.data.copy_((1 - tau) * target_critic_parameters.data + tau * critic_parameters.data)
    
    def select_action(self, mode, agent, state):
        
        """ function to select action for the agent given state observed by local agent """
        
        # set actor model to evaluation mode (for batch norm and dropout) --> remove instances of batch norm, dropout etc. (things that shd only be around in training)
        self.maddpgv2_actor.eval()
        
        # turn actor local state observations to tensor in actor device
        actor_input = T.tensor(state, dtype = T.float).to(self.maddpgv2_actor.device)
        
        # add batch dimension to inputs
        actor_input = actor_input.unsqueeze(0)
    
        # feed actor_input to obtain motor and communication actions 
        u_action, c_action = self.maddpgv2_actor.forward(actor_input)
        
        # multiple u_action by u_range
        u_action = T.mul(u_action, agent.u_range)

        # check if is agent drone
        if agent.adversary == False:

            # multiply c_action by the number of adversarial drones
            c_action = T.mul(c_action, self.num_opp)

        elif agent.adversary == True:

            # multiply c_action by the number of agent drones
            c_action = T.mul(c_action, self.num_agents)
        
        # add gaussian noise if not test
        if mode != "test":

            # generate gaussian noise for motor and communication actions
            u_action_noise = T.mul(T.normal(mean = 0.0, std = 1, size = u_action.size()), agent.u_noise).to(self.maddpg_actor.device)
            c_action_noise = T.mul(T.normal(mean = 0.0, std = 1, size = c_action.size()), agent.c_noise).to(self.maddpg_actor.device)

            # add to noise to motor and communication actions 
            u_action = T.add(u_action, u_action_noise)
            c_action = T.add(c_action, c_action_noise)

        # set actor model to training mode (for batch norm and dropout)
        self.maddpgv2_actor.train()
         
        return u_action.cpu().detach().numpy()[0], c_action.cpu().detach().numpy()[0]
    
    def save_models(self):
        
        """ save weights """
        
        # save weights for each actor, target_actor, critic, target_critic model
        T.save(self.maddpgv2_actor.state_dict(), self.maddpgv2_actor.checkpoint_path)
        T.save(self.maddpgv2_target_actor.state_dict(), self.maddpgv2_target_actor.checkpoint_path)
        T.save(self.maddpgv2_critic.state_dict(), self.maddpgv2_critic.checkpoint_path)
        T.save(self.maddpgv2_target_critic.state_dict(), self.maddpgv2_target_critic.checkpoint_path)
        
    def load_models(self):
        
        """ load weights """
        
        # load weights for each actor, target_actor, critic, target_critic model
        self.maddpgv2_actor.load_state_dict(T.load(self.maddpgv2_actor.checkpoint_path, map_location = T.device('cuda:0' if T.cuda.is_available() else 'cpu')))
        self.maddpgv2_target_actor.load_state_dict(T.load(self.maddpgv2_target_actor.checkpoint_path, map_location = T.device('cuda:0' if T.cuda.is_available() else 'cpu')))
        self.maddpgv2_critic.load_state_dict(T.load(self.maddpgv2_critic.checkpoint_path, map_location = T.device('cuda:0' if T.cuda.is_available() else 'cpu')))
        self.maddpgv2_target_critic.load_state_dict(T.load(self.maddpgv2_target_critic.checkpoint_path, map_location = T.device('cuda:0' if T.cuda.is_available() else 'cpu')))