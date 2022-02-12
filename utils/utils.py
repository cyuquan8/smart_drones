# ==========================================================================================================================================================
# utils
# purpose: utility functions
# ==========================================================================================================================================================

import math
import time
import numpy as np

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

def make_env(scenario_name, dim_c, num_good_agents, num_adversaries, num_landmarks, r_rad, i_rad, r_noise_pos, r_noise_vel, big_rew_cnst, rew_multiplier_cnst, ep_time_limit, drone_radius, 
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
    
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()

    # create world
    world = scenario.make_world(dim_c = dim_c, num_good_agents = num_good_agents, num_adversaries = num_adversaries, num_landmarks = num_landmarks, r_rad = r_rad, i_rad = i_rad, 
                                r_noise_pos = r_noise_pos, r_noise_vel = r_noise_vel, big_rew_cnst = big_rew_cnst, rew_multiplier_cnst = rew_multiplier_cnst, ep_time_limit = ep_time_limit, 
                                drone_radius = drone_radius, agent_size = agent_size, agent_density = agent_density, agent_initial_mass = agent_initial_mass, agent_accel = agent_accel, 
                                agent_max_speed = agent_max_speed, agent_collide = agent_collide, agent_silent = agent_silent, agent_u_noise = agent_u_noise, agent_c_noise = agent_c_noise, 
                                agent_u_range = agent_u_range, landmark_size = landmark_size)

    # create multiagent environment
    if benchmark:  

        env = MultiAgentEnv(world = world, reset_callback = scenario.reset_world, reward_callback = scenario.reward, observation_callback = scenario.observation, 
                            info_callback = scenario.benchmark_data, done_callback = scenario.is_terminal)

    else:

        env = MultiAgentEnv(world = world, reset_callback = scenario.reset_world, reward_callback = scenario.reward, observation_callback = scenario.observation, 
                            done_callback = scenario.is_terminal)

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