# ==========================================================================================================================================================
# utils
# purpose: utility functions
# ==========================================================================================================================================================

import math
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class gnn_goal_replay_buffer:
    
    def __init__(self, mem_size, num_agents, u_actions_dims, c_actions_dims, actor_input_dims, goal_dims):
        
        """ class constructor that initialises memory states attributes """
        
        # bound for memory log
        self.mem_size = mem_size
        
        # counter for memory logged
        self.mem_counter = 0 
        
        # track start and end index (non-inclusive) of latest episode
        self.ep_start_index = 0
        self.ep_end_index = 0

        # boolean to track if latest logged experience tuple is terminal
        self.is_ep_terminal = False

        # number of agents
        self.num_agents = num_agents
        
        # dimension of goal of an agent
        self.goal_dims = goal_dims

        # dimensions of action for motor and communications
        self.u_action_dims = u_actions_dims
        self.c_action_dims = c_actions_dims

        # dimensions of action space
        self.actions_dims = self.u_action_dims + self.c_action_dims
        
        # reward_log is list of reward from num_agents of actors
        # terminal_log indicates if episode is terminated
        self.rewards_log = np.zeros((self.mem_size, self.num_agents)) 
        self.terminal_log = np.zeros((self.mem_size, self.num_agents), dtype = bool)
        
        # list to store num_agents of each actor log of state, state_prime, actions and goals 
        self.actor_state_log_list = []
        self.actor_state_prime_log_list = []
        self.actor_u_action_log_list = []
        self.actor_c_action_log_list = []
        self.actor_goals_log_list = []

        # list to store graph data representation of critic state, state prime 
        self.critic_state_log_list = [0 for i in range(self.mem_size)]
        self.critic_state_prime_log_list = [0 for i in range(self.mem_size)]

        # list to store goals of agents
        self.critic_goals_log_list = np.zeros((self.mem_size, self.goal_dims * self.num_agents)) 
        
        # iterate over num_agents
        for actor_index in range(self.num_agents):
            
            # append each actor log to list
            # actor_state and actor_state_prime are local observations of environment by each actor
            self.actor_state_log_list.append(np.zeros((self.mem_size, actor_input_dims[actor_index])))
            self.actor_state_prime_log_list.append(np.zeros((self.mem_size, actor_input_dims[actor_index])))
            self.actor_u_action_log_list.append(np.zeros((self.mem_size, u_actions_dims)))
            self.actor_c_action_log_list.append(np.zeros((self.mem_size, c_actions_dims)))
            self.actor_goals_log_list.append(np.zeros((self.mem_size, self.goal_dims)))            
    
    def log(self, actor_state, actor_state_prime, actor_goals, critic_state, critic_state_prime, critic_goals, u_action, c_action, rewards, is_done):
        
        """ function to log memory """
        
        # index for logging. based on first in first out
        index = self.mem_counter % self.mem_size
        
        # iterate over num_agents
        for actor_index in range(self.num_agents):
            
            # log actor_state, actor_state_prime, motor and communication action and goal for each actor
            self.actor_state_log_list[actor_index][index] = actor_state[actor_index]
            self.actor_state_prime_log_list[actor_index][index] = actor_state_prime[actor_index]
            self.actor_u_action_log_list[actor_index][index] = u_action[actor_index]
            self.actor_c_action_log_list[actor_index][index] = c_action[actor_index]
            self.actor_goals_log_list[actor_index][index] = actor_goals[actor_index]

        # log critic_fc_state, critic_fc_state_prime, rewards and terminal flag
        self.critic_state_log_list[index] = critic_state
        self.critic_state_prime_log_list[index] = critic_state_prime
        self.critic_goals_log_list[index] = critic_goals
        self.rewards_log[index] = rewards
        self.terminal_log[index] = is_done
        
        # increment counter
        self.mem_counter += 1
        self.ep_end_index = (self.ep_end_index + 1) % self.mem_size

        # check if logged episode is terminal
        if np.any(is_done) == True:

            # update is_ep_terminal
            self.is_ep_terminal = True

        else: 

            # update is_ep_terminal
            self.is_ep_terminal = False

        # calculate ep_start_index
        if np.any(self.terminal_log[index - 1]) == True:

            self.ep_start_index = index
    
    def sample_log(self, batch_size, rng = np.random.default_rng(69)):
        
        """ function to randomly sample a batch of memory """
        
        # select amongst memory logs that is filled
        max_mem = min(self.mem_counter, self.mem_size)
        
        # randomly select memory from logs
        batch = rng.choice(max_mem, batch_size, replace = False)
        
        # initialise list for actor_state, actor_state_prime, actions, critic_state, critic_state_prime, critic_goals
        actor_state_log_list = []
        actor_state_prime_log_list = []
        actor_u_action_log_list = []
        actor_c_action_log_list = []
        actor_goals_log_list = []
        critic_state_log_list = []
        critic_state_prime_log_list = []
        critic_goals_log_list = []

        # iterate over num_agents
        for actor_index in range(self.num_agents):
            
            # obtain corresponding actor_state, actor_state_prime and actions
            actor_state_log_list.append(self.actor_state_log_list[actor_index][batch])
            actor_state_prime_log_list.append(self.actor_state_prime_log_list[actor_index][batch])
            actor_u_action_log_list.append(self.actor_u_action_log_list[actor_index][batch])
            actor_c_action_log_list.append(self.actor_c_action_log_list[actor_index][batch])
            actor_goals_log_list.append(self.actor_goals_log_list[actor_index][batch])
        
        # obtain corresponding rewards, terminal flag
        rewards_log = self.rewards_log[batch]
        terminal_log = self.terminal_log[batch]
        critic_goals_log_list = self.critic_goals_log_list[batch]

        # iterate over batch for gnn data in critic state
        for index in batch:

            # append relevant state, 
            critic_state_log_list.append(self.critic_state_log_list[index]) 
            critic_state_prime_log_list.append(self.critic_state_prime_log_list[index])
        
        return actor_state_log_list, actor_state_prime_log_list, actor_u_action_log_list, actor_c_action_log_list, actor_goals_log_list, critic_state_log_list, critic_state_prime_log_list, \
               critic_goals_log_list, rewards_log, terminal_log

def degree_to_radians(angle_in_degrees):

    """ function to convert angle in degrees to radians """

    return 2 * math.pi * (angle_in_degrees / 360.0)

def random_cartesian_from_polar(r_low, r_high):

    """ function that returns a random cartesian coordinates gives within the radius limits from polar coordinates """

    # generate the polar angle between 0 <= x < 360 degree 
    polar_angle = np.random.uniform(0, 360, 1)[0] 

    # generate the radius between r_low <= x < r_high 
    radius = np.random.uniform(r_low, r_high, 1)[0] 

    # generate (x, y) position tuple
    x_cord = radius * math.sin(degree_to_radians(polar_angle))
    y_cord = radius * math.cos(degree_to_radians(polar_angle))

    # return random generated position in np array
    return np.array([x_cord, y_cord])

def within_drone_radius(agent, entity):

    """ function that returns a boolean that determines if entity is within agent's drone radius """

    # obtain difference in position in terms of vector
    delta_pos = agent.state.p_pos - entity.state.p_pos

    # obtain difference in position in terms of magnitude
    dist = np.sqrt(np.sum(np.square(delta_pos)))

    # return True if entity is within agent's drone radius
    return True if dist < agent.drone_radius else False

def make_env(scenario_name, dim_c, num_good_agents, num_adversaries, num_landmarks, r_rad, i_rad, r_noise_pos, r_noise_vel, big_rew_cnst, rew_multiplier_cnst, ep_time_step_limit, drone_radius, 
             agent_size, agent_density, agent_initial_mass, agent_accel, agent_max_speed, agent_collide, agent_silent, agent_u_noise, agent_c_noise, agent_u_range, landmark_size, benchmark = False):

    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    
    from multiagent.environment import MultiAgentGoalEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()

    # for zone_def scenario
    if scenario_name == "zone_def_push" or scenario_name == "zone_def_tag":

        # create world
        world = scenario.make_world(dim_c = dim_c, num_good_agents = num_good_agents, num_adversaries = num_adversaries, num_landmarks = num_landmarks, r_rad = r_rad, i_rad = i_rad, 
                                    r_noise_pos = r_noise_pos, r_noise_vel = r_noise_vel, big_rew_cnst = big_rew_cnst, rew_multiplier_cnst = rew_multiplier_cnst, 
                                    ep_time_step_limit = ep_time_step_limit, drone_radius = drone_radius, agent_size = agent_size, agent_density = agent_density, 
                                    agent_initial_mass = agent_initial_mass, agent_accel = agent_accel, agent_max_speed = agent_max_speed, agent_collide = agent_collide, agent_silent = agent_silent, 
                                    agent_u_noise = agent_u_noise, agent_c_noise = agent_c_noise, agent_u_range = agent_u_range, landmark_size = landmark_size)

        # create multiagent environment
        if benchmark:  

            env = MultiAgentGoalEnv(world = world, reset_callback = scenario.reset_world, reward_callback = scenario.reward_goal, observation_callback = scenario.observation, 
                                    info_callback = scenario.benchmark_data, done_callback = scenario.is_terminal_goal)

        else:

            env = MultiAgentGoalEnv(world = world, reset_callback = scenario.reset_world, reward_callback = scenario.reward_goal, observation_callback = scenario.observation, 
                                    done_callback = scenario.is_terminal_goal)

    return env

def complete_graph_edge_index(num_nodes):

    """ function to generate the edge index of a complete graph given the number of nodes """

    # empty list to store edge index
    edge_index = []

    # iterate over num_nodes
    for i in range(num_nodes):

        # iterate over num_nodes
        for j in range(num_nodes):

            # append edge index
            edge_index.append([i, j])

    return np.array(edge_index)

def update_noise_exponential_decay(env, expo_decay_cnst, num_adver, eps_timestep, agent_u_noise_cnst, agent_c_noise_cnst, adver_u_noise_cnst, adver_c_noise_cnst):

    """ function to update the exploration noise of all agents following exponential decay """ 

    # iterate over all agents
    for i, agent in enumerate(env.agents):

        # check if adversarial drone
        if i < num_adver:

            # update adversarial drone noise
            agent.u_noise = adver_u_noise_cnst * math.exp(-expo_decay_cnst * eps_timestep)
            agent.c_noise = adver_c_noise_cnst * math.exp(-expo_decay_cnst * eps_timestep)

        # check if agent drone
        elif i >= num_adver:

            # update agent drone noise
            agent.u_noise = agent_u_noise_cnst * math.exp(-expo_decay_cnst * eps_timestep)
            agent.c_noise = agent_c_noise_cnst * math.exp(-expo_decay_cnst * eps_timestep)

def calculate_elo_rating(agent_curr_elo, adver_curr_elo, k_agent, k_adver, d, results_list, results_reward_dict):

    """ 
    function to calculate the elo rating of agent and adversarial drones based on results 

    results legend:

    win = 1
    loss = 0

    if there exit screen is considered: 

        exit screen --> 0 for agent that exceed screen, 0.5 for agent that didn't 

    """
    
    # store agent and adver current elo
    agent_elo = agent_curr_elo
    adver_elo = adver_curr_elo

    # iterate over results in results list
    for i in range(len(results_list)):

        # calculate expected probability of agent and adver to win
        q_agent = math.pow(10, agent_elo / float(d))
        q_adver = math.pow(10, adver_elo / float(d))
        e_agent = q_agent / (q_agent + q_adver)
        e_adver = q_adver / (q_agent + q_adver)

        # update elos based on outcome accordingly
        agent_elo = agent_elo + k_agent * (results_reward_dict[str(results_list[i])][0] - e_agent)
        adver_elo = adver_elo + k_adver * (results_reward_dict[str(results_list[i])][1] - e_adver)

    return agent_elo, adver_elo

def update_agent_goals_softmax_weights(agent_goals_softmax_weights, agent_goal_distribution, agent_elo, adver_elo, agent_goal, terminal_condition):

    """ function to update the softmax weights based on elo for agent """

    # check if its exit screen
    if terminal_condition == 3 or terminal_condition == 4:

        # no change to weights
        return agent_goals_softmax_weights

    # iterate over agent_goal_distribution
    for i in range(len(agent_goal_distribution)):

        # check if goal is agent_goal
        if agent_goal_distribution[i] == agent_goal:

            # obtain index of goal
            agent_goal_index = i

    # check if its adver win
    if terminal_condition == 1:

        # iterate over agent_goals_softmax_weights
        for i in range(len(agent_goals_softmax_weights)):

            # increase weights for easier goals
            if i <= agent_goal_index:

                # increase weight proportional to max of elo ratio
                agent_goals_softmax_weights[i] += max(agent_elo / adver_elo, adver_elo / agent_elo)

            # decrease weights for harder goals
            elif i > agent_goal_index:

                # decrease weight proportional to max of elo ratio
                agent_goals_softmax_weights[i] -= max(agent_elo / adver_elo, adver_elo / agent_elo)

    # check if its agent win
    elif terminal_condition == 2:

        # iterate over agent_goals_softmax_weights
        for i in range(len(agent_goals_softmax_weights)):

            # decrease weights for easier goals
            if i <= agent_goal_index:

                # decrease weight proportional to agent strength over adversarial
                agent_goals_softmax_weights[i] -= agent_elo / adver_elo 

            # increase weights for harder goals
            elif i > agent_goal_index:

                # increase weight proportional to agent strength over adversarial
                agent_goals_softmax_weights[i] += agent_elo / adver_elo 

    # obtain scaler 
    scaler = MinMaxScaler(feature_range = (-5, 5))
    agent_goals_softmax_weights = np.squeeze(scaler.fit_transform(agent_goals_softmax_weights.reshape(-1, 1)))

    # replace nan is any with small value for numerical stability
    agent_goals_softmax_weights = np.nan_to_num(agent_goals_softmax_weights, nan = 10**-5)

    return agent_goals_softmax_weights

def update_adver_goals_softmax_weights(adver_goals_softmax_weights, adver_goal_distribution, agent_elo, adver_elo, adver_goal, terminal_condition):

    """ function to update the softmax weights based on elo for adversarial """

    # check if its exit screen
    if terminal_condition == 3 or terminal_condition == 4:

        # no change to weights
        return adver_goals_softmax_weights

    # iterate over adver_goal_distribution
    for i in range(len(adver_goal_distribution)):

        # check if goal is agent_goal
        if adver_goal_distribution[i] == adver_goal:

            # obtain index of goal
            adver_goal_index = i

    # check if its adver win
    if terminal_condition == 1:

        # iterate over adver_goals_softmax_weights
        for i in range(len(adver_goals_softmax_weights)):

            # increase weights for harder goals
            if i <= adver_goal_index:

                # increase weight proportional to adversarial strength over agent
                adver_goals_softmax_weights[i] += adver_elo / agent_elo 

            # decrease weights for easier goals
            elif i > adver_goal_index:

                # decrease weight proportional to adversarial strength over agent
                adver_goals_softmax_weights[i] -= adver_elo / agent_elo 

    # check if its agent win
    elif terminal_condition == 2:

        # iterate over adver_goals_softmax_weights
        for i in range(len(adver_goals_softmax_weights)):

            # decrease weights for harder goals
            if i <= adver_goal_index:

                # decrease weight proportional to max of elo ratio
                adver_goals_softmax_weights[i] -= max(agent_elo / adver_elo, adver_elo / agent_elo)

            # increase weights for easier goals
            elif i > adver_goal_index:

                # increase weight proportional to max of elo ratio
                adver_goals_softmax_weights[i] += max(agent_elo / adver_elo, adver_elo / agent_elo) 

    # obtain scaler 
    scaler = MinMaxScaler(feature_range = (-5, 5))
    adver_goals_softmax_weights = np.squeeze(scaler.fit_transform(adver_goals_softmax_weights.reshape(-1, 1))) 

    # replace nan is any with small value for numerical stability
    adver_goals_softmax_weights = np.nan_to_num(adver_goals_softmax_weights, nan = 10**-5)
    
    return adver_goals_softmax_weights