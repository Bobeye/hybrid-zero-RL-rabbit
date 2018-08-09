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

import matplotlib.pyplot as plt

from rabbit_estrain import settings, Policy, make_env, sigmoid, bound_theta_tanh, bound_theta_sigmoid, init_plot, save_plot_2, get_reward
from gym import wrappers

def load_model(filename):
    with open(filename) as f:    
    	data = json.load(f)
    print('loading file %s' % (filename))
    model_params = np.array(data[0]) # assuming other stuff is in data
    return model_params	

if __name__ == "__main__":
	policy_path = "log/"+"/policy/449.json"
	video_path = None
	video_path = policy_path.split(".json")[0]

	render_mode = True
	plot_mode = False
	record_mode = False
	settings.plot_mode = False
	
	if plot_mode:
		init_plot()

	desired_velocity = 1.
	# settings.control_kp = 150
	# settings.control_kd = 7
	current_speed = settings.init_vel
	model = NeuralNetwork(input_dim=settings.conditions_dim,
					 	  output_dim=settings.theta_dim+1,
					 	  units=settings.nn_units,
					 	  activations=settings.nn_activations)
	#Obtain theta from .json file's data
	model_params = load_model(policy_path)
	model.set_weights(model_params)

	current_speed_list, error = [], []

	# current_speed = settings.init_vel
	# current_speed_list = np.zeros(settings.size_vel_buffer)
	# error = np.zeros(settings.size_vel_buffer)
	# error[0] = desired_velocity - current_speed 
	# inputs_nn = np.array([current_speed, desired_velocity, sum(error)])

	# theta = model.predict(inputs_nn)
	# # theta = bound_theta_tanh(theta)
	# theta = bound_theta_sigmoid(theta)	
	# print(theta)

	# pi = Policy(theta=theta, action_size=settings.action_size, 
	# 			action_min=settings.action_min, action_max=settings.action_max,
	# 			kp=settings.control_kp, kd=settings.control_kd, feq=settings.frequency)
	env = make_env(settings.env_name, render_mode=render_mode, desired_velocity = desired_velocity)
	if video_path is not None:
		env = wrappers.Monitor(env, video_path, video_callable=lambda episode_id: True, force=True)
	
	# for desired_velocity in [0.8, 1.0, 1.2, 1.5]:
	for desired_velocity in range(1):
		total_reward_list = []
		velocity_list = []

		last_speed = None
		state = env.reset()
		timesteps = 0
		if state is None:
			state = np.zeros(settings.state_size)
		total_reward = 0
		for t in range(min(4000,settings.max_episode_length)):
			if t < 1500:
				desired_velocity = 0.8
			else:
				desired_velocity = 1.2



			timesteps += 1
			if render_mode:
				env.render()
				
			current_speed = state[7]
			current_speed_list += [current_speed]
			while len(current_speed_list) > settings.size_vel_buffer:
				del current_speed_list[0]
			current_speed_av = np.mean(np.array(current_speed_list))

			error += [desired_velocity - current_speed]
			while len(error) > settings.size_vel_buffer:
				del error[0]

			inputs_nn = np.array([current_speed_av, 
								  desired_velocity, 
								  np.mean(np.array(error))])	

			theta_kd = model.predict(inputs_nn)
			theta = bound_theta_sigmoid(theta_kd[0:settings.theta_dim])
			kd = sigmoid(theta_kd[-1]) * settings.control_kd

			last_speed = current_speed
			# else:
			# 	print("Exception")
				
			pi = Policy(theta=theta, action_size=settings.action_size, 
						action_min=settings.action_min, action_max=settings.action_max,
						kp=settings.control_kp, kd=kd, feq=settings.frequency, mode="hzdrl") #Use mode="hzd" to run the hzd controller, and mode="hzdrl" to run the learned controller

			action, reward_tau = pi.get_action(state, settings.plot_mode)
			observation, reward_params, done, info = env.step(action)
			reward = get_reward(reward_params, reward_tau, desired_velocity, current_speed_av, sum(error), mode="linear_bowen")
			# print(action)
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

		# state = env.reset()
		total_rewards = np.array(total_reward_list)

		if settings.batch_mode == "min":
			rewards = np.min(total_rewards)
		else:
			rewards = np.mean(total_rewards)
		print (total_reward_list, rewards)
		velocity_list_new = velocity_list#[500:1500]
		# print(np.size(velocity_list))
		# velocity_list_new = velocity_list
		vel_mean = np.sum(velocity_list_new)/np.size(velocity_list_new)
		print(desired_velocity, vel_mean, np.amin(velocity_list), np.amax(velocity_list))


		plt.plot(velocity_list)
		plt.savefig("log/"+"449_change_"+str(desired_velocity)+".png")