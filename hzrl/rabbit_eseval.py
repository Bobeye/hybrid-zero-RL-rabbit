#	Evaluate rabbit walk
#	by Guillermo, June, 2018
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
# from gym import wrappers
import matplotlib.pyplot as plt
def load_model(filename):
    with open(filename) as f:    
    	data = json.load(f)
    print('loading file %s' % (filename))
    model_params = np.array(data[0]) # assuming other stuff is in data
    return model_params	

if __name__ == "__main__":
	policy_path = "log/"+"/policy/449.json"

	render_mode = False
	plot_mode = True
	record_mode = False
	settings.plot_mode = True
	
	if plot_mode:
		init_plot()

	#SET DESIRED VELOCITIES
	# desired_vel_list = np.array([0.8, 1.0, 1.2, 0.8, 0.8, 0.8])
	desired_vel_list = np.array([1])
	desired_velocity = desired_vel_list[0]


	# settings.control_kp = 150
	# settings.control_kd = 7
	model = NeuralNetwork(input_dim=settings.conditions_dim,
						output_dim=settings.theta_dim+1,
						units=settings.nn_units,
						activations=settings.nn_activations)
	#Obtain theta from .json file's data
	model_params = load_model(policy_path)
	model.set_weights(model_params) #FIX FOR EACH PAIR model-solution		

	env = make_env(settings.env_name, render_mode=render_mode, desired_velocity = desired_velocity)	
	state = env.reset()

	current_speed_list, error = [], []
	velocity_list = []

	for episode in range(np.size(desired_vel_list)):	#for episode in range(settings.num_episode): 
	### FOR EVALUATE MORE THAN 1 EPISODE DISABLE done break FLAG	
		desired_velocity = desired_vel_list[episode]
		print("==========================")
		print("desired velocity = {}" .format(desired_velocity))

		if episode == 0:
			current_speed = settings.init_vel
		else:
			current_speed = settings.init_vel = state[7]

		total_reward_list = []
		# velocity_list = []
		
		# if record_mode:
		# 	env = wrappers.Monitor(env, '/tmp/rabbit', force=True)

		timesteps = 0
		if state is None:
			state = np.zeros(settings.state_size)
		total_reward = 0

		for t in range(settings.max_episode_length):
			timesteps += 1

			if t < 1000:
				desired_velocity = 0.8
			elif t < 2000:
				desired_velocity = 1.6
			else:
				desired_velocity = 1.6

			if render_mode:
				env.render()
				
			current_speed_list += [current_speed]
			while len(current_speed_list) > settings.size_vel_buffer:
				del current_speed_list[0]
			current_speed_av = np.mean(np.array(current_speed_list))

			error += [desired_velocity - current_speed]
			while len(error) > settings.size_vel_buffer:
				del error[0]

			#if last_speed is None or (current_speed - last_speed) < 1e-1: 
			inputs_nn = np.array([current_speed_av, desired_velocity, np.mean(np.array(error))])
			# inputs_nn = np.append(state[0:14], [desired_velocity, sum(error)])
			theta_kd = model.predict(inputs_nn)
			theta = bound_theta_sigmoid(theta_kd[0:settings.theta_dim])
			kd = sigmoid(theta_kd[-1]) * settings.control_kd
			print(kd)
			# print(theta)
				
			pi = Policy(theta=theta, action_size=settings.action_size, 
						action_min=settings.action_min, action_max=settings.action_max,
						kp=settings.control_kp, kd=kd, feq=settings.frequency, mode="hzdrl") #Use mode="hzd" to run the hzd controller, and mode="hzdrl" to run the learned controller

			action, reward_tau = pi.get_action(state, settings.plot_mode)
			observation, reward_params, done, info = env.step(action)
			reward = get_reward(reward_params, reward_tau, desired_velocity, current_speed_av, sum(error), mode="quadratic")
			# print(action)
			state = observation
			velocity_list += [state[7]]
			total_reward += reward

			if settings.plot_mode:
				save_plot_2(state)

			if done:
				break


		total_reward_list += [np.array([total_reward]).flatten()]
		print(total_reward)
		
		# velocity_list_new = velocity_list[400:1600]
		velocity_list_new = velocity_list
		print(np.size(velocity_list))
		vel_mean = np.sum(velocity_list_new)/np.size(velocity_list_new)
		print(vel_mean, np.amin(velocity_list), np.amax(velocity_list))
	
		# if record_mode:
		# 	env.close()				

	# state = env.reset()
	total_rewards = np.array(total_reward_list)

	if settings.batch_mode == "min":
		rewards = np.min(total_rewards)
	else:
		rewards = np.mean(total_rewards)
	print (total_reward_list, rewards)
	plt.plot(velocity_list)
	plt.show()
	
