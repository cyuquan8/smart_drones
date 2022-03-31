# ==========================================================================================================================================================
# mappo gnn agent class 
# purpose: creates and updates neural network 
# ==========================================================================================================================================================

import torch as T
from nn import mappo_mlp_actor_model, mappo_gnn_critic_model

class mappo_gnn_agent:
    
    def __init__(self, mode, scenario_name, training_name, lr_actor, lr_critic, num_agents, num_opp, u_range, u_noise, c_noise, is_adversary, actor_dropout_p, critic_dropout_p, state_fc_input_dims, 
                 state_fc_output_dims, u_action_dims, c_action_dims, num_heads, bool_concat, gnn_input_dims, gnn_output_dims, gmt_hidden_dims, gmt_output_dims, fc_output_dims):
        
        """ class constructor for mappo agent attributes """
        
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
        
        # intialise actor model 
        self.mappo_gnn_actor = mappo_mlp_actor_model(model = "mappo_gnn_actor", model_name = None, mode = mode, scenario_name = scenario_name, training_name = training_name, 
                                                     learning_rate = self.lr_actor, num_agents = num_agents, num_opp = num_opp, u_range = u_range, u_noise = u_noise, c_noise = c_noise, 
                                                     is_adversary = is_adversary, dropout_p = actor_dropout_p, fc_input_dims = state_fc_input_dims, fc_output_dims = state_fc_output_dims, 
                                                     tanh_actions_dims = u_action_dims, sig_actions_dims = c_action_dims)

        # intialise critic model
        self.mappo_gnn_critic = mappo_gnn_critic_model(model = "mappo_gnn_critic", model_name = None, mode = mode, scenario_name = scenario_name, training_name = training_name, 
                                                       learning_rate = self.lr_critic, num_agents = num_agents, num_opp = num_opp, num_heads = num_heads, bool_concat = bool_concat, 
                                                       dropout_p = critic_dropout_p, gnn_input_dims = gnn_input_dims, gnn_output_dims = gnn_output_dims, gmt_hidden_dims = gmt_hidden_dims, 
                                                       gmt_output_dims = gmt_output_dims, fc_output_dims = fc_output_dims)
    
    def select_action(self, mode, agent, state):
        
        """ function to select action for the agent given state observed by local agent """
        
        # set actor model to evaluation mode (for batch norm and dropout) --> remove instances of batch norm, dropout etc. (things that shd only be around in training)
        self.mappo_gnn_actor.eval()
        
        # turn actor local state observations to tensor in actor device
        actor_input = T.tensor(state, dtype = T.float).to(self.mappo_gnn_actor.device)
        
        # add batch dimension to inputs
        actor_input = actor_input.unsqueeze(0)
    
        # feed actor_input to obtain motor and communication actions 
        u_action_norm_dist, c_action_norm_dist = self.mappo_gnn_actor.forward(actor_input)

        # set actor model to training mode (for batch norm and dropout)
        self.mappo_gnn_actor.train()

        # sample from distribution if not test
        if mode != "test":

            # obtain samples
            u_action_sample = u_action_norm_dist.sample()
            c_action_sample = c_action_norm_dist.sample()

            # obtain log probs of actions
            u_action_log_prob = u_action_norm_dist.log_prob(u_action_sample)
            c_action_log_prob = c_action_norm_dist.log_prob(c_action_sample)

            return u_action_sample.cpu().detach().numpy()[0], c_action_sample.cpu().detach().numpy()[0], u_action_log_prob.cpu().detach().numpy()[0], c_action_log_prob.cpu().detach().numpy()[0]

        # take mean during evaluation
        else:
        
            # obtain mean
            u_action_mean = u_action_norm_dist.mean
            c_action_mean = c_action_norm_dist.mean

            # obtain log probs of actions
            u_action_log_prob = u_action_norm_dist.log_prob(u_action_mean)
            c_action_log_prob = c_action_norm_dist.log_prob(c_action_mean)

            return u_action_mean.cpu().detach().numpy()[0], c_action_mean.cpu().detach().numpy()[0], u_action_log_prob.cpu().detach().numpy()[0], c_action_log_prob.cpu().detach().numpy()[0]
    
    def save_models(self):
        
        """ save weights """
        
        # save weights for each actor, target_actor, critic, target_critic model
        T.save(self.mappo_gnn_actor.state_dict(), self.mappo_gnn_actor.checkpoint_path)
        T.save(self.mappo_gnn_critic.state_dict(), self.mappo_gnn_critic.checkpoint_path)
        
    def load_models(self):
        
        """ load weights """
        
        # load weights for each actor, target_actor, critic, target_critic model
        self.mappo_gnn_actor.load_state_dict(T.load(self.mappo_gnn_actor.checkpoint_path, map_location = T.device('cuda:0' if T.cuda.is_available() else 'cpu')))
        self.mappo_gnn_critic.load_state_dict(T.load(self.mappo_gnn_critic.checkpoint_path, map_location = T.device('cuda:0' if T.cuda.is_available() else 'cpu')))