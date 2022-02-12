# ==========================================================================================================================================================
# neural network (nn) module
# purpose: classes and functions to build a scalable neural network model
# ==========================================================================================================================================================

import os
import shutil
import torch as T
import torch.nn as nn
import torch_geometric.nn as gnn
from functools import partial
from utils.popart import popart

def activation_function(activation):
    
    """ function that returns ModuleDict of activation functions """
    
    return  nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['sigmoid', nn.Sigmoid()],
        ['softmax', nn.Softmax(1)],
        ['tanh', nn.Tanh()],
        ['none', nn.Identity()]
    ])[activation]

class conv_2d_auto_padding(nn.Conv2d):
    
    """ class to set padding dynamically based on kernel size to preserve dimensions of height and width after conv """
    
    def __init__(self, *args, **kwargs):
        
        """ class constructor for conv_2d_auto_padding to alter padding attributes of nn.Conv2d """
        
        # inherit class constructor attributes from nn.Conv2d
        super().__init__(*args, **kwargs)
        
        # dynamically adds padding based on the kernel_size
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) 

class fc_block(nn.Module):
    
    """ class to build basic fully connected block """
    
    def __init__(self, input_shape, output_shape, activation_func, dropout_p):
        
        """ class constructor that creates the layers attributes for fc_block """
        
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # input and output units for hidden layer 
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        # activation function for after batch norm
        self.activation_func = activation_func 
        
        # dropout probablity
        self.dropout_p = dropout_p
        
        # basic fc_block. inpuit --> linear --> batch norm --> activation function --> dropout 
        self.block = nn.Sequential(
            
            # linear hidden layer
            nn.Linear(self.input_shape, self.output_shape, bias = False),
            
            # batch norm
            nn.BatchNorm1d(self.output_shape),
            
            # activation func
            activation_function(self.activation_func),
            
            # dropout
            nn.Dropout(self.dropout_p),
            
        )
    
    def forward(self, x):
        
        """ function for forward pass of fc_block """
        
        x = self.block(x)
        
        return x

class vgg_block(nn.Module):
    
    """ class to build basic, vgg inspired block """
    
    def __init__(self, input_channels, output_channels, activation_func, conv, dropout_p, max_pool_kernel):
        
        """ class constructor that creates the layers attributes for vgg_block """
        
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # input and output channels for conv (num of filters)
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # activation function for after batch norm
        self.activation_func = activation_func
        
        # class of conv
        self.conv = conv
        
        # dropout probablity
        self.dropout_p = dropout_p
        
        # size of kernel for maxpooling
        self.max_pool_kernel = max_pool_kernel
        
        # basic vgg_block. input --> conv --> batch norm --> activation func --> dropout --> max pool
        self.block = nn.Sequential(
            
            # conv
            self.conv(self.input_channels, self.output_channels),
            
            # batch norm
            nn.BatchNorm2d(self.output_channels),
            
            # activation func
            activation_function(self.activation_func),
            
            # dropout
            nn.Dropout2d(self.dropout_p),
            
            # maxpooling
            # ceil mode to True to ensure that odd dimensions accounted for
            nn.MaxPool2d(self.max_pool_kernel, ceil_mode = True)
            
        )
    
    def forward(self, x):
        
        """ function for forward pass of vgg_block """
        
        x = self.block(x)
        
        return x

class gatv2_block(nn.Module):

    """ class to build gatv2_block """

    def __init__(self, input_channels, output_channels, num_heads, concat, dropout_p, activation_func):

        """ class constructor for attributes of the gatv2_block """

        # inherit class constructor attributes from nn.Module
        super().__init__()

        # input and output channels for gatv2 (embedding dimension for each node)
        self.input_channels = input_channels
        self.output_channels = output_channels

        # number of heads for gatv2
        self.num_heads = num_heads

        # boolean that when set to false, the multi-head attentions are averaged instead of concatenated
        self.concat = concat

        # dropout probablity
        self.dropout_p = dropout_p

        # activation function for after GATv2Conv 
        self.activation_func = activation_func

        # basic gatv2_block. input --> GATv2Conv --> GraphNorm --> activation func
        self.block = gnn.Sequential('x, edge_index', 
                                    [

                                        # GATv2Conv 
                                        (gnn.GATv2Conv(in_channels = self.input_channels, out_channels = self.output_channels, heads = self.num_heads, concat = concat, dropout = dropout_p), 
                                         'x, edge_index -> x'), 

                                        # graph norm
                                        (gnn.GraphNorm(self.output_channels * self.num_heads if concat == True else self.output_channels), 'x -> x'),

                                        # activation func
                                        activation_function(self.activation_func)

                                    ]
        )

    def forward(self, x, edge_index):
        
        """ function for forward pass of gatv2_block """
        
        x = self.block(x, edge_index)
        
        return x

class nn_layers(nn.Module):
    
    """ class to build layers of blocks (e.g. fc_block) """
    
    def __init__(self, input_channels, block, output_channels, *args, **kwargs):
        
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # input channels/shape
        self.input_channels = input_channels
        
        # class of block
        self.block = block
        
        # output channels/shape
        self.output_channels = output_channels
        self.input_output_list = list(zip(output_channels[:], output_channels[1:]))
        
        # module list of layers with same args and kwargs
        self.blocks = nn.ModuleList([
            
            self.block(self.input_channels, self.output_channels[0], *args, **kwargs),
            *[self.block(input_channels, output_channels, *args, **kwargs) for (input_channels, output_channels) in self.input_output_list]   
            
        ])
    
    def get_flat_output_shape(self, input_shape):
        
        """ function to obatain number of features after flattening after convolution layers """
        
        # assert that this function must be utilised on a convulution block
        assert hasattr(self.block, 'conv') == True, "Cannot execute get_flat_output_shape on non-convulution block"

        # initialise dummy tensor of ones with input shape
        x = T.ones(1, *input_shape)
        
        # feed dummy tensor to blocks by iterating over each block
        for block in self.blocks:
            
            x = block(x)
        
        # flatten resulting tensor and obtain features after flattening
        n_size = x.view(1, -1).size(1)
        
        return n_size

    def forward(self, x, *args, **kwargs):
        
        """ function for forward pass of layers """
        
        # iterate over each block
        for block in self.blocks:
            
            x = block(x, *args, **kwargs)
            
        return x 

class maddpg_actor_model(nn.Module):
    
    """ class to build model for MADDPG """
    
    def __init__(self, model, model_name, mode, training_name, learning_rate, num_agents, num_opp, dropout_p, fc_input_dims, fc_output_dims, tanh_actions_dims, sig_actions_dims):
        
        """ class constructor for attributes for the actor model """
        
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # model
        self.model = model
        
        # model name
        self.model_name = model_name
        
        # checkpoint filepath 
        self.checkpoint_path = None
        
        # if training model
        if mode != 'test' and mode != 'load_n_train':

            try:
                
                # create directory for saving models if it does not exist
                os.mkdir("saved_models/" + training_name + "_" + "best_models/")
                
            except:
                
                # remove existing directory and create new directory
                shutil.rmtree("saved_models/" + training_name + "_" + "best_models/")
                os.mkdir("saved_models/" + training_name + "_" + "best_models/")

        # checkpoint directory
        self.checkpoint_dir = "saved_models/" + training_name + "_" + "best_models/"
        
        # learning rate
        self.learning_rate = learning_rate
        
        # number of agents
        self.num_agents = num_agents

        # number of adverserial opponents
        self.num_opp = num_opp

        # model architecture for maddpg actor

        # hidden fc layers post gatv2 layers
        # input channels are the dimensions of observation from agent
        # fc_output_dims is the list of sizes of output channels fc_block
        self.actor_fc_layers = nn_layers(input_channels = fc_input_dims, block = fc_block, output_channels = fc_output_dims, activation_func = 'relu', dropout_p = dropout_p)

        # final fc_blocks for actions with tanh activation function
        self.tanh_actions_layer = fc_block(input_shape = fc_output_dims[-1], output_shape = tanh_actions_dims, activation_func = "tanh", dropout_p = dropout_p)

        # final fc_blocks for actions with sigmoid activation function
        self.sig_actions_layer = fc_block(input_shape = fc_output_dims[-1], output_shape = sig_actions_dims, activation_func = "sigmoid", dropout_p = dropout_p)
             
        # adam optimizer 
        self.optimizer = T.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        # device for training (cpu/gpu)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        # cast module to device
        self.to(self.device)
    
    def forward(self, x):
            
        """ function for forward pass through actor model """
            
        # actor_gmt_layer --> actor_fc_layers
        x = self.actor_fc_layers(x)

        # actor_fc_layers --> tanh_actions_layer
        tanh_actions = self.tanh_actions_layer(x)

        # actor_fc_layers --> sig_actions_layer
        sig_actions = self.sig_actions_layer(x)
        
        return tanh_actions, sig_actions

class maddpg_critic_model(nn.Module):
    
    """ class to build model for MADDPG """
    
    def __init__(self, model, model_name, mode, training_name, learning_rate, num_agents, num_opp, num_heads, dropout_p, bool_concat, gnn_input_dims, gnn_output_dims, gmt_hidden_dims, 
                 gmt_output_dims, u_actions_fc_input_dims, u_actions_fc_output_dims, c_actions_fc_input_dims, c_actions_fc_output_dims, concat_fc_output_dims):
        
        """ class constructor for attributes for the model """
        
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # model
        self.model = model
        
        # model name
        self.model_name = model_name
        
        # checkpoint filepath 
        self.checkpoint_path = None
        
        # if training model
        if mode != 'test' and mode != 'load_n_train':

            try:
                                # create directory for saving models if it does not exist
                os.mkdir("saved_models/" + training_name + "_" + "best_models/")
                
            except:
                
                # remove existing directory and create new directory
                shutil.rmtree("saved_models/" + training_name + "_" + "best_models/")
                os.mkdir("saved_models/" + training_name + "_" + "best_models/")

        # checkpoint directory
        self.checkpoint_dir = "saved_models/" + training_name + "_" + "best_models/"
        
        # learning rate
        self.learning_rate = learning_rate
        
        # number of agents
        self.num_agents = num_agents

        # number of adverserial opponents
        self.num_opp = num_opp
            
        # model architecture for maddpg_critic
            
        # gatv2 layers for state inputs 
        # gnn_input_dims are the dimensions of the initial node embeddings 
        # gnn_output_dims are the list of dimensions of the the output embeddings of each layer of gatv2 
        self.critic_state_gatv2_layer = nn_layers(input_channels = gnn_input_dims, block = gatv2_block, output_channels = gnn_output_dims, num_heads = num_heads, concat = bool_concat, 
                                                  activation_func = 'relu', dropout_p = dropout_p)
        
        # graph multiset transformer (gmt) for state inputs
        # in_channels are the dimensions of node embeddings after gatv2 layers
        # gmt_hidden_dims are the dimensions of the node embeddings post 1 initial linear layer in gmt 
        # gmt_output_dims are the dimensions of the sole remaining node embedding that represents the entire graph
        # uses GATv2Conv as Conv block for GMPool_G
        # remaining inputs are defaults 
        self.critic_state_gmt_layer = gnn.GraphMultisetTransformer(in_channels = gnn_output_dims[-1] * num_heads if bool_concat == True else gnn_output_dims[-1], hidden_channels = gmt_hidden_dims, 
                                                                   out_channels = gmt_output_dims , Conv = gnn.GATv2Conv, num_nodes = 300, pooling_ratio = 0.25, 
                                                                   pool_sequences = ['GMPool_G', 'SelfAtt', 'GMPool_I'], num_heads = 4, layer_norm = False)

        # hidden fc layers for motor actions inputs
        # u_actions_fc_input_dims are the dimensions of concatenated motor actions for all agents
        # u_actions_fc_output_dims is the list of sizes of output channels fc_block
        self.critic_u_actions_fc_layers = nn_layers(input_channels = u_actions_fc_input_dims, block = fc_block, output_channels = u_actions_fc_output_dims, activation_func = 'relu', 
                                                    dropout_p = dropout_p)

        # hidden fc layers for communication actions inputs
        # c_actions_fc_input_dims are the dimensions of concatenated communication actions for all agents
        # c_actions_fc_output_dims is the list of sizes of output channels fc_block
        self.critic_c_actions_fc_layers = nn_layers(input_channels = c_actions_fc_input_dims, block = fc_block, output_channels = c_actions_fc_output_dims, activation_func = 'relu', 
                                                    dropout_p = dropout_p)

        # hidden fc layers post gmt and actions layers
        # input channels are the dimensions of node embeddings of the one node from gmt and outputs from actions fc layers
        # concat_fc_output_dims is the list of sizes of output channels fc_block
        self.critic_concat_fc_layers = nn_layers(input_channels = gmt_output_dims + u_actions_fc_output_dims[-1] + c_actions_fc_output_dims[-1], block = fc_block, 
                                                 output_channels = concat_fc_output_dims, activation_func = 'relu', dropout_p = dropout_p)

        # final fc_block for Q value output w/o activation function
        self.q_layer = fc_block(input_shape = concat_fc_output_dims[-1], output_shape = 1, activation_func = "none", dropout_p = dropout_p)
            
        # adam optimizer 
        self.optimizer = T.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        # device for training (cpu/gpu)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        # cast module to device
        self.to(self.device)
    
    def forward(self, data, batch, u_actions, c_actions):
            
        """ function for forward pass through critic model """
        
        # obtain node embeddings and edge index from data
        x, edge_index = data.x, data.edge_index
       
        # x (graph of critic's state representation) --> critic_state_gatv2_layer
        x = self.critic_state_gatv2_layer(x = x, edge_index = edge_index)
        
        # critic_state_gatv2_layer --> critic_state_gmt_layer
        x = self.critic_state_gmt_layer(x = x, edge_index = edge_index, batch = batch)

        # u_actions --> critic_u_actions_fc_layers
        y = self.critic_u_actions_fc_layers(u_actions)

        # c_actions --> critic_c_actions_fc_layers
        z = self.critic_c_actions_fc_layers(c_actions)

        # concatenate node embeddings of the one node from gmt and outputs from actions fc layers
        conc = T.cat((x, y, z), 1)

        # critic_gmt_layer || critic_u_actions_fc_layers || critic_c_actions_fc_layers --> critic_concat_fc_layers
        conc = self.critic_concat_fc_layers(x = conc)

        # critic_concat_fc_layers --> q value
        q = self.q_layer(conc)
        
        return q

class mappo_actor_model(nn.Module):
    
    """ class to build model for MAPPO """
    
    def __init__(self, model, model_name, mode, training_name, learning_rate, num_agents, num_opp, u_range, u_noise, c_noise, is_adversary, dropout_p, fc_input_dims, fc_output_dims, 
                 tanh_actions_dims, sig_actions_dims):
        
        """ class constructor for attributes for the actor model """
        
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # model
        self.model = model
        
        # model name
        self.model_name = model_name
        
        # checkpoint filepath 
        self.checkpoint_path = None
        
        # if training model
        if mode != 'test' and mode != 'load_n_train':

            try:
                
                # create directory for saving models if it does not exist
                os.mkdir("saved_models/" + training_name + "_" + "best_models/")
                
            except:
                
                # remove existing directory and create new directory
                shutil.rmtree("saved_models/" + training_name + "_" + "best_models/")
                os.mkdir("saved_models/" + training_name + "_" + "best_models/")

        # checkpoint directory
        self.checkpoint_dir = "saved_models/" + training_name + "_" + "best_models/"
        
        # learning rate
        self.learning_rate = learning_rate
        
        # number of agents
        self.num_agents = num_agents

        # number of adverserial opponents
        self.num_opp = num_opp

        # range of motor actions
        self.u_range = u_range

        # motor and communication noise
        self.u_noise = u_noise
        self.c_noise = c_noise

        # boolean that states if model is for adversarial drones or not
        self.is_adversary = is_adversary

        # model architecture for mappo actor

        # hidden fc layers post gatv2 layers
        # input channels are the dimensions of observation from agent
        # fc_output_dims is the list of sizes of output channels fc_block
        self.actor_fc_layers = nn_layers(input_channels = fc_input_dims, block = fc_block, output_channels = fc_output_dims, activation_func = 'relu', dropout_p = dropout_p)

        # final fc_blocks for actions with tanh activation function
        self.tanh_actions_layer = fc_block(input_shape = fc_output_dims[-1], output_shape = tanh_actions_dims, activation_func = "tanh", dropout_p = dropout_p)

        # final fc_blocks for actions with sigmoid activation function
        self.sig_actions_layer = fc_block(input_shape = fc_output_dims[-1], output_shape = sig_actions_dims, activation_func = "sigmoid", dropout_p = dropout_p)
             
        # adam optimizer 
        self.optimizer = T.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        # device for training (cpu/gpu)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        # cast module to device
        self.to(self.device)
    
    def forward(self, x):
            
        """ function for forward pass through actor model """
            
        # actor_gmt_layer --> actor_fc_layers
        x = self.actor_fc_layers(x)

        # actor_fc_layers --> tanh_actions_layer
        tanh_actions = self.tanh_actions_layer(x)

        # actor_fc_layers --> sig_actions_layer
        sig_actions = self.sig_actions_layer(x)
        
        # obtain normal distribution of tanh_actions
        tanh_actions_norm_dist = T.distributions.normal.Normal(T.mul(tanh_actions, self.u_range), self.u_noise)

        # adversarial
        if self.is_adversary == True:

            sig_actions_norm_dist = T.distributions.normal.Normal(T.mul(sig_actions, self.num_agents), self.c_noise)

        else:

            sig_actions_norm_dist = T.distributions.normal.Normal(T.mul(sig_actions, self.num_opp), self.c_noise)

        return tanh_actions_norm_dist, sig_actions_norm_dist 

class mappo_critic_model(nn.Module):
    
    """ class to build model for MAPPO """
    
    def __init__(self, model, model_name, mode, training_name, learning_rate, num_agents, num_opp, num_heads, dropout_p, bool_concat, gnn_input_dims, gnn_output_dims, gmt_hidden_dims, 
                 gmt_output_dims, fc_output_dims):
        
        """ class constructor for attributes for the model """
        
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # model
        self.model = model
        
        # model name
        self.model_name = model_name
        
        # checkpoint filepath 
        self.checkpoint_path = None
        
        # if training model
        if mode != 'test' and mode != 'load_n_train':

            try:
                                # create directory for saving models if it does not exist
                os.mkdir("saved_models/" + training_name + "_" + "best_models/")
                
            except:
                
                # remove existing directory and create new directory
                shutil.rmtree("saved_models/" + training_name + "_" + "best_models/")
                os.mkdir("saved_models/" + training_name + "_" + "best_models/")

        # checkpoint directory
        self.checkpoint_dir = "saved_models/" + training_name + "_" + "best_models/"
        
        # learning rate
        self.learning_rate = learning_rate
        
        # number of agents
        self.num_agents = num_agents

        # number of adverserial opponents
        self.num_opp = num_opp
            
        # model architecture for mappo critic
            
        # gatv2 layers for state inputs 
        # gnn_input_dims are the dimensions of the initial node embeddings 
        # gnn_output_dims are the list of dimensions of the the output embeddings of each layer of gatv2 
        self.critic_state_gatv2_layer = nn_layers(input_channels = gnn_input_dims, block = gatv2_block, output_channels = gnn_output_dims, num_heads = num_heads, concat = bool_concat, 
                                                  activation_func = 'relu', dropout_p = dropout_p)
        
        # graph multiset transformer (gmt) for state inputs
        # in_channels are the dimensions of node embeddings after gatv2 layers
        # gmt_hidden_dims are the dimensions of the node embeddings post 1 initial linear layer in gmt 
        # gmt_output_dims are the dimensions of the sole remaining node embedding that represents the entire graph
        # uses GATv2Conv as Conv block for GMPool_G
        # remaining inputs are defaults 
        self.critic_state_gmt_layer = gnn.GraphMultisetTransformer(in_channels = gnn_output_dims[-1] * num_heads if bool_concat == True else gnn_output_dims[-1], hidden_channels = gmt_hidden_dims, 
                                                                   out_channels = gmt_output_dims , Conv = gnn.GATv2Conv, num_nodes = 300, pooling_ratio = 0.25, 
                                                                   pool_sequences = ['GMPool_G', 'SelfAtt', 'GMPool_I'], num_heads = 4, layer_norm = False)

        # hidden fc layers post gmt layer
        # input channels are the dimensions of node embeddings of the one node from gmt
        # fc_output_dims is the list of sizes of output channels fc_block
        self.critic_fc_layers = nn_layers(input_channels = gmt_output_dims, block = fc_block, output_channels = fc_output_dims, activation_func = 'relu', dropout_p = dropout_p)

        # final layer using popart for value normalisation
        self.popart = popart(input_shape = fc_output_dims[-1], output_shape = 1)
            
        # adam optimizer 
        self.optimizer = T.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        # device for training (cpu/gpu)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # cast module to device
        self.to(self.device)
    
    def forward(self, data, batch):
            
        """ function for forward pass through critic model """
        
        # obtain node embeddings and edge index from data
        x, edge_index = data.x, data.edge_index
       
        # x (graph of critic's state representation) --> critic_state_gatv2_layer
        x = self.critic_state_gatv2_layer(x = x, edge_index = edge_index)
        
        # critic_state_gatv2_layer --> critic_state_gmt_layer
        x = self.critic_state_gmt_layer(x = x, edge_index = edge_index, batch = batch)

        # critic_gmt_layer --> critic_fc_layers
        x = self.critic_fc_layers(x = x)

        # critic_fc_layers --> v value
        v = self.popart(x)
        
        return v