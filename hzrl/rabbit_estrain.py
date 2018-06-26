#	Train rabbit walk
#	by Bowen, June, 2018
#
################################
from __future__ import division

from es.nn import NeuralNetwork 
from es.oes import OpenES

import gym
from joblib import Parallel, delayed
import json
import os
import numpy as np


"""Hyperparameters"""
class Settings():
	env_name="BipedalWalker-v2"
	txt_log = "log/" + "/train.txt"
	policy_path = "log/"+ "/policy/"
	backend = "multiprocessing"
	n_jobs = 4
	frequency = 20.
	total_threshold = 1e8
	num_episode = 5
	max_episode_length=1600
	batch_mode="mean"
	state_size = 24
	action_size = 4
	action_min = -1.
	action_max = 1.
	control_kp = 10.
	control_kd = 1.
	desired_v_low = 0.
	desired_v_up = 10.
	conditions_dim = 1
	theta_dim = 12
	nn_units=[20,20,20]
	nn_activations=["relu", "relu", "passthru", "passthru"]
	population = 8
	sigma_init=0.1
	sigma_decay=0.9999 
	sigma_limit=1e-4
	learning_rate=0.01
	learning_rate_decay=0.9999 
	learning_rate_limit=1e-6
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

        self.int_val = self.last_int_val = self.last_error = 0.

    def reset(self):
        self.int_val = 0.0
        self.last_int_val = 0.0

    def step(self, error, sample_time):
        self.last_int_val = self.int_val

        integral = self.int_val + error * sample_time;
        derivative = (error - self.last_error) / sample_time;

        y = self.kp * error + self.ki * self.int_val + self.kd * derivative;
        if np.all(y>self.max):
            val = self.max
        elif np.all(y<self.min):
            val = self.min
        else:
            self.int_val = integral
        val = np.clip(y, self.min, self.max)

        
        self.last_error = error

        return val

"""The policy defined by theta. Theta is calculated through a NN given desired conditions."""
class Policy():

	def __init__(self, theta=None,
				 action_size=4,
				 action_min=-1.,
				 action_max=1.,
				 kp=0, kd=0, feq=20.):
		self.theta = theta
		self.action_size = action_size
		self.action_min = action_min
		self.action_max = action_max
		self.sample_time = 1/feq
		self.pid = PID(kp, 0., kd, mn=np.full((action_size,),action_min), 
								   mx=np.full((action_size,),action_max))

	def get_action(self, state):
		err = np.zeros(self.action_size,) + np.sum(state)*(np.sum(self.theta)/10000.)
		action = self.pid.step(err, self.sample_time)
		return action

def make_log(txt_log, policy_path):
	if not os.path.exists(policy_path):
		os.makedirs(policy_path)
	with open(txt_log, "w") as text_file:
		text_file.write("blahblah"+"\n")

def save_policy(path, solution):
	with open(path, 'wt') as out:
		json.dump([np.array(solution).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))

def make_env(env_name, seed=np.random.seed(None), render_mode=False):
	env = gym.make(env_name)
	env.reset()
	if render_mode:	
		env.render("human")
	if (seed >= 0):
		env.seed(seed)

	return env

def simulate(model, solution, settings, desired_velocity):
	
	model.set_weights(solution)
	theta = model.predict(desired_velocity)
	pi = Policy(theta=theta, action_size=settings.action_size, 
				action_min=settings.action_min, action_max=settings.action_max,
				kp=settings.control_kp, kd=settings.control_kd, feq=settings.frequency)
	env = make_env(settings.env_name)
	
	total_reward_list = []
	timesteps = 0
	state_list = []
	action_list = []
	reward_list = []
	termination_list = []
	for episode in range(settings.num_episode):	
		state = env.reset()
		if state is None:
			state = np.zeros(settings.state_size)
		state = state.reshape(1,-1)
		total_reward = 0
		for t in range(settings.max_episode_length):
			timesteps += 1
			action = pi.get_action(state)
			observation, reward, done, info = env.step(action)
			state = observation.reshape(1,-1)
			total_reward += reward
			state_list += [state.flatten()]
			action_list += [action]
			reward_list += [reward]
			termination_list += [done]

		total_reward_list += [np.array([total_reward]).flatten()]
	state = env.reset()

	total_rewards = np.array(total_reward_list)
	if settings.batch_mode == "min":
		rewards = np.min(total_rewards)
	else:
		rewards = np.mean(total_rewards)

	return [rewards, timesteps]



if __name__ == "__main__":

	make_log(settings.txt_log, settings.policy_path)

	model = NeuralNetwork(input_dim=settings.conditions_dim,
					 	  output_dim=settings.theta_dim,
					 	  units=settings.nn_units,
					 	  activations=settings.nn_activations)
	escls = OpenES(model.parameter_count, 
				   sigma_init=settings.sigma_init, 
				   sigma_decay=settings.sigma_decay, 
				   sigma_limit=settings.sigma_limit, 
				   learning_rate=settings.learning_rate,
				   learning_rate_decay=settings.learning_rate_decay, 
				   learning_rate_limit=settings.learning_rate_limit,
				   popsize=settings.population, 
				   antithetic=True)

	step = 0
	total_timesteps = 0
	while total_timesteps < settings.total_threshold:
		solutions = escls.ask()
		models = []
		for _ in range(settings.population):
			m = NeuralNetwork(input_dim=settings.conditions_dim,
					 	  	  output_dim=settings.theta_dim,
					 	  	  units=settings.nn_units,
					 	  	  activations=settings.nn_activations)
			models += [m]
		desired_velocities = np.random.uniform(low=settings.desired_v_low, 
											   high=settings.desired_v_up, 
											   size=(settings.population,))
		result = Parallel(n_jobs=settings.n_jobs, backend=settings.backend)(delayed(simulate)(models[i],solutions[i],settings,desired_velocities[i],) for i in range(len(models)))
		rewards_list = []
		timesteps_list = []
		for r in result:
			rewards_list += [r[0]]
			timesteps_list += [r[1]]
		rewards = np.array(rewards_list)
		total_timesteps = np.sum(np.array(timesteps_list))
		escls.tell(rewards, solutions)

		log_string = (" ave_R ", int(np.mean(np.array(rewards))*100)/100., 
					  " std_R ", int(np.std(np.array(rewards))*100)/100.,
					  " min_R ", int(np.min(np.array(rewards))*100)/100.,
					  " max_R ", int(np.max(np.array(rewards))*100)/100.,
					  " times ", int(total_timesteps))
		with open(settings.txt_log, "a") as text_file:
			text_file.write(str(log_string) + "\n")


		best_solution = escls.best_param()
		save_policy(settings.policy_path+str(step)+".json", best_solution)

		step += 1



 