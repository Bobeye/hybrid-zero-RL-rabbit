#	Train rabbit walk
#	by Bowen, June, 2018
#
################################
from __future__ import division

from es.nn import NeuralNetwork 
from es.oes import OpenES
from es.cmaes import CMAES
import hzd.trajectory as trajectory
import hzd.params as params

import gym
from joblib import Parallel, delayed
import json
import os
import numpy as np


"""Hyperparameters"""
class Settings():
	env_name="Rabbit-v0"
	txt_log = "log/" + "/nn_es_train.txt"
	policy_path = "log/"+ "/nn_es_policy/"
	backend = "multiprocessing"
	n_jobs = 8
	frequency = 20.
	total_threshold = 1e8
	num_episode = 5
	max_episode_length=1000
	batch_mode="mean"
	state_size = 14
	action_size = 4
	action_min = -4.
	action_max = 4.
	control_kp = 120.
	control_kd = 2.
	desired_v_low = 0.6
	desired_v_up = 1
	conditions_dim = 2
	theta_dim = 24
	nn_units=[16,16]
	nn_activations=["relu", "relu", "tanh"]
	population = 24
	sigma_init=0.1
	sigma_decay=0.9999 
	sigma_limit=1e-4
	learning_rate=0.01
	learning_rate_decay=0.9999 
	learning_rate_limit=1e-6
	aux = 0
	upplim_jthigh = 250*(np.pi/180)
	lowlim_jthigh = 90*(np.pi/180)
	upplim_jleg = 120*(np.pi/180)
	lowlim_jleg = 0*(np.pi/180)
	def __init__(self):
		pass
settings = Settings()

def make_log(txt_log, policy_path):
	if not os.path.exists(policy_path):
		os.makedirs(policy_path)
	with open(txt_log, "w") as text_file:
		text_file.write("Summary of the training: "+"\n")

def save_policy(path, solution):
	with open(path, 'wt') as out:
		json.dump([np.array(solution).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))

def make_env(env_name, seed=np.random.seed(None), render_mode=False, desired_velocity=None):
	env = gym.make(env_name)
	env.assign_desired_vel(desired_velocity)
	env.reset()
	if render_mode:	
		env.render("human")
	#if (seed >= 0):
	#	env.seed(seed)
	return env

def simulate(model, solution, settings, desired_velocity, render_mode):	
	model.set_weights(solution) #FIX FOR EACH PAIR model-solution 
	env = make_env(settings.env_name, render_mode=render_mode, desired_velocity = desired_velocity)
	
	total_reward_list = []
	timesteps = 0
	state_list = []
	action_list = []
	reward_list = []
	termination_list = []

	last_speed = None

	for episode in range(settings.num_episode):	
		# print (episode)
		state = env.reset()

		if state is None:
			state = np.zeros(settings.state_size)
		total_reward = 0

		for t in range(settings.max_episode_length):
			action = model.predict(state) * settings.action_max
			#print(action)
			observation, reward, done, info = env.step(action)
			if render_mode:
				env.render("human")
			state = observation
			total_reward += reward
			state_list += [state.flatten()]
			action_list += [action]
			reward_list += [reward]
			termination_list += [done]
			if done:
				break

		total_reward_list += [np.array([total_reward]).flatten()]
	state = env.reset()

	total_rewards = np.array(total_reward_list)
	if settings.batch_mode == "min":
		rewards = np.min(total_rewards)
	else:
		rewards = np.mean(total_rewards)
	# if render_mode:
	# 	env.close()
	
	return [rewards, timesteps]


 # SIGMOID``
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

if __name__ == "__main__":

	make_log(settings.txt_log, settings.policy_path)

	model = NeuralNetwork(input_dim=settings.state_size,
					 	  output_dim=settings.action_size,
					 	  units=settings.nn_units,
					 	  activations=settings.nn_activations)
	# # Adopt OpenAI ES
	# escls = OpenES(model.parameter_count, 
	# 			   sigma_init=settings.sigma_init, 
	# 			   sigma_decay=settings.sigma_decay, 
	# 			   sigma_limit=settings.sigma_limit, 
	# 			   learning_rate=settings.learning_rate,
	# 			   learning_rate_decay=settings.learning_rate_decay, 
	# 			   learning_rate_limit=settings.learning_rate_limit,
	# 			   popsize=settings.population, 
	# 			   antithetic=True)

	# Adopt CMA-ES
	escls = CMAES(model.parameter_count,
				  sigma_init=settings.sigma_init,
				  popsize=settings.population,
				  weight_decay=settings.sigma_decay)

	step = 0
	total_timesteps = 0
	while total_timesteps < settings.total_threshold:
		print("======== step {} ========" .format(step)) ###DEBUG MESSAGE	
		solutions = escls.ask()
		#print(solutions.shape)
		models = []
		for _ in range(settings.population):
			m = NeuralNetwork(input_dim=settings.state_size,
					 	  	  output_dim=settings.action_size,
					 	  	  units=settings.nn_units,
					 	  	  activations=settings.nn_activations)
			models += [m]
		#print (models)	
		desired_velocities = np.random.uniform(low=settings.desired_v_low, 
											   high=settings.desired_v_up, 
											   size=(settings.population,))
		
		#print(desired_velocities)

		# desired_velocities = np.zeros(settings.population,) + 2.
		render_mode = False
		result = Parallel(n_jobs=settings.n_jobs, backend=settings.backend)(delayed(simulate)(models[i],solutions[i],settings,desired_velocities[i],render_mode,) for i in range(len(models)))

		rewards_list = []
		timesteps_list = []
		for r in result:
			rewards_list += [r[0]]
			timesteps_list += [r[1]]
		rewards = np.array(rewards_list)
		total_timesteps += np.sum(np.array(timesteps_list))
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

		# print("total_timesteps = {}" .format(total_timesteps))
	
		step += 1