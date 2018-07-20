#	Evaluate rabbit walk
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

from rabbit_estrain import settings, Policy, make_env, sigmoid, bound_theta_tanh, bound_theta_sigmoid, init_plot, save_plot_2, get_reward
# from gym import wrappers

def load_model(filename):
    with open(filename) as f:    
    	data = json.load(f)
    print('loading file %s' % (filename))
    model_params = np.array(data[0]) # assuming other stuff is in data
    return model_params	

if __name__ == "__main__":
	policy_path = "log/"+"/policy/20.json"

	render_mode = True
	plot_mode = False
	record_mode = True
	
	if plot_mode:
		init_plot()

	desired_velocity = 0.6
	settings.control_kd = 7
	current_speed = settings.init_vel
	
	model = NeuralNetwork(input_dim=settings.conditions_dim,
					 	  output_dim=settings.theta_dim,
					 	  units=settings.nn_units,
					 	  activations=settings.nn_activations)
	#Obtain theta from .json file's data
	model_params = load_model(policy_path)
	model.set_weights(model_params)
	theta = model.predict(np.array([desired_velocity, current_speed]))
	# theta = bound_theta_tanh(theta)
	theta = bound_theta_sigmoid(theta)	
	print(theta)

	pi = Policy(theta=theta, action_size=settings.action_size, 
				action_min=settings.action_min, action_max=settings.action_max,
				kp=settings.control_kp, kd=settings.control_kd, feq=settings.frequency)
	env = make_env(settings.env_name, render_mode=render_mode, desired_velocity = desired_velocity)
	
	total_reward_list = []
	velocity_list = []
	for episode in range(1):	#for episode in range(settings.num_episode):
		# if record_mode:
		# 	env = wrappers.Monitor(env, '/tmp/rabbit', force=True)

		last_speed = None
		state = env.reset()
		timesteps = 0
		if state is None:
			state = np.zeros(settings.state_size)
		total_reward = 0
		for t in range(settings.max_episode_length*2):
			timesteps += 1
			if render_mode:
				env.render()
				
			current_speed = abs(state[7])	#current velocity of hip
			# print(current_speed)
			#print(current_speed)
			if last_speed is None or (current_speed - last_speed) < 1e-2: 
				theta = model.predict(np.array([desired_velocity, current_speed]))
				# theta = bound_theta_tanh(theta)
				theta = bound_theta_sigmoid(theta)
				#print(theta)
				last_speed = current_speed
			# else:
			# 	print("Exception")
				
			pi = Policy(theta=theta, action_size=settings.action_size, 
						action_min=settings.action_min, action_max=settings.action_max,
						kp=settings.control_kp, kd=settings.control_kd, feq=settings.frequency, mode="hzdrl")

			action, reward_tau = pi.get_action(state, settings.plot_mode)
			observation, reward_params, done, info = env.step(action)
			reward = get_reward(reward_params, reward_tau, desired_velocity, mode="quadratic")

			state = observation
			velocity_list += [state[7]]
			total_reward += reward

			if settings.plot_mode:
				save_plot_2(state)

			if done:
				break

		total_reward_list += [np.array([total_reward]).flatten()]
	
		# if record_mode:
		# 	env.close()				

	state = env.reset()
	total_rewards = np.array(total_reward_list)

	if settings.batch_mode == "min":
		rewards = np.min(total_rewards)
	else:
		rewards = np.mean(total_rewards)
	print (total_reward_list, rewards)
	velocity_list_new = velocity_list[500:1000]
	# velocity_list_new = velocity_list
	vel_mean = np.sum(velocity_list_new)/np.size(velocity_list_new)
	print(vel_mean, np.amin(velocity_list), np.amax(velocity_list))
