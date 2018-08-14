from ppo_tf.policy import Policy
from ppo_tf.value_function import NNValueFunction
from ppo_tf.utils import Logger, Scaler

import gym
import numpy as np
from gym import wrappers
import scipy.signal
from datetime import datetime
import os
import argparse
import signal

import hzd.params as params
import hzd.trajectory as hzdtrajectory

"""Hyperparameters"""
class Settings():
    env_name="Rabbit-v0"
    txt_log = "log/" + "/train.txt"
    policy_path = "log/"+ "/policy/"
    backend = "multiprocessing"
    n_jobs = 20
    frequency = 20.
    total_threshold = 1e8
    num_episode = 5
    max_episode_length=3000
    batch_mode="mean"
    state_size = 14
    action_size = 4
    action_min = -4.
    action_max = 4.
    control_kp = 150.
    control_kd = 10.
    desired_v_low = 1.
    desired_v_up = 1.
    conditions_dim = 3
    theta_dim = 20
    nn_units=[16,16]
    nn_activations=["relu", "relu", "passthru"]
    population = 20
    sigma_init=0.1
    sigma_decay=0.999
    sigma_limit=1e-4
    learning_rate=0.01
    learning_rate_decay=0.999 
    learning_rate_limit=1e-6
    aux = 0
    upplim_jthigh = 250*(np.pi/180)
    lowlim_jthigh = 90*(np.pi/180)
    upplim_jleg = 120*(np.pi/180)
    lowlim_jleg = 0*(np.pi/180)
    plot_mode = False
    sigmoid_slope = 1
    init_vel = 0.7743
    size_vel_buffer = 200
    def __init__(self):
        pass
settings = Settings()

"""PID controller"""
MIN_NUM = float('-inf')
MAX_NUM = float('inf')
class PID(object):
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx

    def step(self, qd, qdotd, q, qdot):
        error_pos = qd - q
        error_vel = qdotd - qdot
        output = self.kp*error_pos + self.kd*error_vel
        # print([self.kp, self.kd])
        return np.clip(output, self.min, self.max)

"""Customized Reward Func"""

def get_reward(velocity=0., desired_vel=0, mode="vel"):
    if mode == "vel":   # encourage vel only
        threshold = 0.1 
        if abs(velocity-desired_vel) <= threshold:
            if velocity <= desired_vel:
                velocity_reward = (velocity / desired_vel)**2
            else:
                velocity_reward = (desired_vel / velocity)**2
        else:
            velocity_reward = 0
        reward = velocity_reward
    return reward


class RabbitPolicy():
    def __init__(self, theta=None,
                 action_size=4,
                 action_min=-4.,
                 action_max=4.,
                 kp=120., kd=2., feq=20., 
                 mode="hzdrl"):

        self.theta = theta
        self.make_theta()

        self.action_size = action_size
        self.action_min = action_min
        self.action_max = action_max
        self.sample_time = 1/feq
        
        self.pid = PID(kp, 0., kd, mn=np.full((action_size,),action_min), 
                                   mx=np.full((action_size,),action_max))
        
        self.p = params.p
        self.mode = mode

    def make_theta(self):
        self.bound_theta_tanh()
        self.a_rightS = np.append(self.theta, [self.theta[2], self.theta[3], self.theta[0], self.theta[1]])
        # print("a_rightS = {}" .format(self.a_rightS))
        self.a_leftS = np.array([self.a_rightS[2], self.a_rightS[3], self.a_rightS[0], self.a_rightS[1],
                        self.a_rightS[6], self.a_rightS[7], self.a_rightS[4], self.a_rightS[5],
                        self.a_rightS[10], self.a_rightS[11], self.a_rightS[8], self.a_rightS[9],
                        self.a_rightS[14], self.a_rightS[15], self.a_rightS[12], self.a_rightS[13],
                        self.a_rightS[18], self.a_rightS[19], self.a_rightS[16], self.a_rightS[17],
                        self.a_rightS[22], self.a_rightS[23], self.a_rightS[20], self.a_rightS[21]])

    def get_action_star(self, state):
        pass

    def bound_theta_tanh(self): #Add offset and restrict to range corresponding to each joint. The input is assumed to be bounded as tanh, with range -1,1
        theta = self.theta

        upplim_jthigh = 250*(np.pi/180)
        lowlim_jthigh = 90*(np.pi/180)
        upplim_jleg = 120*(np.pi/180)
        lowlim_jleg = 0*(np.pi/180)

        theta_thighR = np.array([theta[0], theta[4], theta[8], theta[12], theta[16]])
        theta_legR = np.array([theta[1], theta[5], theta[9], theta[13], theta[17]])
        theta_thighL = np.array([theta[2], theta[6], theta[10], theta[14], theta[18]])
        theta_legL = np.array([theta[3], theta[7], theta[11], theta[15], theta[19]])
        theta_thighR = (((upplim_jthigh - lowlim_jthigh)/2)*theta_thighR) + ((upplim_jthigh + lowlim_jthigh)/2)
        theta_legR = upplim_jleg/2*(theta_legR + 1)
        theta_thighL = (((upplim_jthigh - lowlim_jthigh)/2)*theta_thighL) + ((upplim_jthigh + lowlim_jthigh)/2)
        theta_legL = upplim_jleg/2*(theta_legL + 1)
        [theta[0], theta[4], theta[8], theta[12], theta[16]] = theta_thighR
        [theta[1], theta[5], theta[9], theta[13], theta[17]] = theta_legR
        [theta[2], theta[6], theta[10], theta[14], theta[18]] = theta_thighL
        [theta[3], theta[7], theta[11], theta[15], theta[19]] = theta_legL
        
        self.theta = theta


    def get_action(self, state, plot_mode = False):
        pos, vel = state[0:7], state[7:14]

        tau_right = np.clip(hzdtrajectory.tau_Right(pos,self.p), 0, 1.05)
        tau_left = np.clip(hzdtrajectory.tau_Left(pos,self.p), 0, 1.05)
        
        if self.mode == "hzd": 
            reward_tau = 0
            reward_step = 0
            if tau_right > 1.0 and settings.aux == 0:
                settings.aux = 1
                reward_step = 10
            if settings.aux == 0:
                qd, tau = hzdtrajectory.yd_time_RightStance(pos,params.a_rightS,params.p)    #Compute the desired position for the actuated joints using the current measured state, the control parameters and bezier polynomials
                qdotd = hzdtrajectory.d1yd_time_RightStance(pos,vel,params.a_rightS,params.p)  #Compute the desired velocity for the actuated joints using the current measured state, the control parameters and bezier polynomials
                reward_tau = tau_right
            else:
                qd = hzdtrajectory.yd_time_LeftStance(pos,params.a_leftS,params.p)    #Compute the desired position for the actuated joints using the current measured state, the control parameters and bezier polynomials
                qdotd = hzdtrajectory.d1yd_time_LeftStance(pos,vel,params.a_leftS,params.p)  #Compute the desired velocity for the actuated joints using the current measured state, the control parameters and bezier poly
                reward_tau = tau_left
                if tau_left > 1.0 and settings.aux == 1:
                    settings.aux = 0
                    reward_step = 10
            reward_tau +=reward_step 
            # print(reward_tau)
                        
        if self.mode == "hzdrl": 
            reward_tau = 0
            reward_step = 0
            if tau_right > 1.0 and settings.aux == 0:
                settings.aux = 1
                reward_step = 10
            if settings.aux == 0:
                qd, tau = hzdtrajectory.yd_time_RightStance(pos,self.a_rightS,self.p)    #Compute the desired position for the actuated joints using the current measured state, the control parameters and bezier polynomials
                qdotd = hzdtrajectory.d1yd_time_RightStance(pos,vel,self.a_rightS,self.p)  #Compute the desired velocity for the actuated joints using the current measured state, the control parameters and bezier polynomials
                reward_tau = tau_right
            else:
                qd = hzdtrajectory.yd_time_LeftStance(pos,self.a_leftS,self.p)    #Compute the desired position for the actuated joints using the current measured state, the control parameters and bezier polynomials
                qdotd = hzdtrajectory.d1yd_time_LeftStance(pos,vel,self.a_leftS,self.p)  #Compute the desired velocity for the actuated joints using the current measured state, the control parameters and bezier poly
                reward_tau = tau_left
                if tau_left > 1.0 and settings.aux == 1:
                    settings.aux = 0
                    reward_step = 10
            reward_tau +=reward_step 
                # print(reward_tau)

        q = np.array([pos[3], pos[4], pos[5], pos[6]])    #Take the current position state of the actuated joints and assign them to vector which will be used to compute the error
        qdot = np.array([vel[3], vel[4], vel[5], vel[6]]) #Take the current velocity state of the actuated joints and assign them to vector which will be used to compute the error
        action = self.pid.step(qd, qdotd, q, qdot)
        # print([qd, qdotd, q, qdot, action])

        return action # , reward_tau


def make_env(env_name, seed=np.random.seed(None), render_mode=False, desired_velocity=None):
    env = gym.make(env_name)
    # env.assign_desired_vel(desired_velocity)
    env.reset()
    if render_mode: 
        env.render("human")
    #if (seed >= 0):
    #   env.seed(seed)
    return env

class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def init_gym(env_name):
    """
    Initialize gym environment, return dimension of observation
    and action spaces.
    Args:
        env_name: str environment name (e.g. "Humanoid-v1")
    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    # env = gym.make(env_name)
    env = make_env(env_name)
    # obs_dim = env.observation_space.shape[0]
    # act_dim = env.action_space.shape[0]
    obs_dim = 3
    act_dim = 20

    return env, obs_dim, act_dim


def run_episode(env, policy, scaler, animate=False):
    """ Run single episode with option to animate
    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode
    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs_raw = env.reset()
    obs = np.zeros(3)
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    
    desired_velocity = np.random.uniform(low=0.5, high=1.5)
    vel_list = []
    err_list = []

    while not done:
        if animate:
            env.render()
        obs = obs.astype(np.float32).reshape((1, -1))
        # obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        # obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        
        pi = RabbitPolicy(theta=np.squeeze(action, axis=0), 
                          action_size=4, 
                          action_min=-4, 
                          action_max=4,
                          kp=150, kd=6, feq=20., mode = "hzdrl")
        a = pi.get_action(obs_raw)

        obs_raw, reward_terms, done, _ = env.step(a)
        
        vel = reward_terms[3]
        err = (vel-desired_velocity)**2
        vel_list += [vel]
        err_list += [err]
        while len(vel_list) > settings.size_vel_buffer:
            del vel_list[0]
            del err_list[0]

        obs = np.array([np.mean(np.array(vel_list)), desired_velocity, np.mean(np.array(err_list))])

        reward = get_reward(velocity = reward_terms[3], 
                            desired_vel = desired_velocity)

        # if not isinstance(reward, float):
        #     reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, logger, episodes):
    """ Run policy and collect data for a minimum of min_steps and min_episodes
    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run
    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})

    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories
    Args:
        trajectories: as returned by run_policy()
        gamma: discount
    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories
    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value
    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf
    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)
    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """
    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()
    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)


    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode
                })

def make_log(txt_log, policy_path, log_init):
    if not os.path.exists(policy_path):
        os.makedirs(policy_path)
    with open(txt_log, "w") as text_file:
        text_file.write(str(log_init)+"\n")

def main(env_name, num_episodes, gamma, lam, kl_targ, batch_size, hid1_mult, policy_logvar):
    """ Main training loop
    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
        hid1_mult: hid1 size for policy and value_f (mutliplier of obs dimension)
        policy_logvar: natural log of initial policy variance
    """
    log_path = "log/ppo_tf/"+env_name
    text_path = log_path+"/train.txt"
    policy_path = log_path+"/policy/"
    make_log(text_path, policy_path, None)

    killer = GracefulKiller()
    env, obs_dim, act_dim = init_gym(env_name)
    # obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    logger = Logger(logname=env_name, now=now)
    aigym_path = os.path.join(log_path, now)
    env = wrappers.Monitor(env, aigym_path, force=True)
    scaler = Scaler(obs_dim)
    val_func = NNValueFunction(obs_dim, hid1_mult)
    policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar, [0.2,0.2])
    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, policy, scaler, logger, episodes=5)
    episode = 0
    current_timesteps = 0
    while episode < num_episodes:
        trajectories = run_policy(env, policy, scaler, logger, episodes=batch_size)
        episode += len(trajectories)
        add_value(trajectories, val_func)  # add estimated values to episodes
        add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
        add_gae(trajectories, gamma, lam)  # calculate advantage
        # concatenate all episodes into single NumPy arrays
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        # add various stats to training log:
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
        
        log_rewards = []
        for trajectory in trajectories:
            log_rewards += [np.sum(np.array(trajectory["rewards"]))]
            current_timesteps += len(trajectory["rewards"]) 
        log_string = (" ave_R ", int(np.mean(np.array(log_rewards))*100)/100., 
                      " std_R ", int(np.std(np.array(log_rewards))*100)/100.,
                      " min_R ", int(np.min(np.array(log_rewards))*100)/100.,
                      " max_R ", int(np.max(np.array(log_rewards))*100)/100.,
                      " times ", int(current_timesteps))
        with open(text_path, "a") as text_file:
            text_file.write(str(log_string) + "\n")

        policy.update(observes, actions, advantages, logger)  # update policy
        val_func.fit(observes, disc_sum_rew, logger)  # update value function
        logger.write(display=True)  # write logger results to file and stdout
        if killer.kill_now:
            if input('Terminate training (y/[n])? ') == 'y':
                break
            killer.kill_now = False
    logger.close()
    policy.close_sess()
    val_func.close_sess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=30000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.99)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.95)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)
    parser.add_argument('-m', '--hid1_mult', type=int,
                        help='Size of first hidden layer for value and policy NNs'
                             '(integer multiplier of observation dimension)',
                        default=8)
    parser.add_argument('-v', '--policy_logvar', type=float,
                        help='Initial policy log-variance (natural log of variance)',
                        default=-2.)

    args = parser.parse_args()
    main(**vars(args))