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

from rabbit_estrain import settings, Policy, make_env, sigmoid, bound_theta_tanh, bound_theta_sigmoid, init_plot, get_reward


def load_model(filename):
    with open(filename) as f:    
    	data = json.load(f)
    print('loading file %s' % (filename))
    model_params = np.array(data[0]) # assuming other stuff is in data
    return model_params	

def eval_reward(reward_params, reward_tau, desired_vel):
	alive_bonus, posafter, posbefore, velocity, a, w = reward_params[0], reward_params[1], reward_params[2], reward_params[3], reward_params[4], reward_params[5]
	scale = 0.1
	if velocity < 0:
		velocity_reward = 0
		# velocity_reward = velocity
	elif velocity <= desired_vel:
		velocity_reward = (velocity)/ desired_vel    
		#print(velocity_reward) 
	else:
		velocity_reward = - abs(velocity - desired_vel)
		# print(velocity_reward)

	angular_reward = w	
	action_reward = -1e-2 * np.sum(a**2)
	displacement_reward  = posafter
	reward = alive_bonus + 10*velocity_reward + 1*action_reward + 2*displacement_reward + 1*reward_tau + 1*angular_reward
	reward = scale*reward
	# print("reward = {}" .format(reward))

	
	# print (alive_bonus , velocity_reward , height_reward , action_reward , displacement_reward)
	return reward


if __name__ == "__main__":
	policy_path = "log/"+"/policy/120.json" #24 better than 58

	render_mode = True
	eval_mode = True	
	init_plot()

	desired_velocity = 1
	settings.control_kd = 5
	current_speed = settings.init_vel
	
	model = NeuralNetwork(input_dim=settings.conditions_dim,
					 	  output_dim=settings.theta_dim,
					 	  units=settings.nn_units,
					 	  activations=settings.nn_activations)
	#Obtain theta from .json file's data
	model_params = load_model(policy_path)
	model.set_weights(model_params)
	# theta = model.predict(np.array([desired_velocity, current_speed])) #When usign 2 inputs for the NN
	theta = model.predict(current_speed)
	# theta = bound_theta_tanh(theta)	
	theta = bound_theta_sigmoid(theta)
	print (theta)

	pi = Policy(theta=theta, action_size=settings.action_size, 
				action_min=settings.action_min, action_max=settings.action_max,
				kp=settings.control_kp, kd=settings.control_kd, feq=settings.frequency)
	env = make_env(settings.env_name, render_mode=render_mode, desired_velocity = desired_velocity)
	
	total_reward_list = []
	velocity_list = []
	for episode in range(1):	#for episode in range(settings.num_episode):
		last_speed = None
		state = env.reset()
		for k in range(500):
			env.render()
		if state is None:
			state = np.zeros(settings.state_size)
		total_reward = 0
		for t in range(settings.max_episode_length*2):
			#timesteps += 1
			current_speed = abs(state[7])	#current velocity of hip
			# print(current_speed)
			#print(current_speed)
			if last_speed is None or (current_speed - last_speed) < 1e-2: 
				# theta = model.predict(np.array([desired_velocity, current_speed])) #When usign 2 inputs for the NN
				theta = model.predict(current_speed)
				# theta = bound_theta_tanh(theta)
				theta = bound_theta_sigmoid(theta)
				#print(theta)
				last_speed = current_speed
			# else:
			# 	print("Exception")
				
			pi = Policy(theta=theta, action_size=settings.action_size, 
						action_min=settings.action_min, action_max=settings.action_max,
						kp=settings.control_kp, kd=settings.control_kd, feq=settings.frequency)

			action, tau_reward = pi.get_action(state, eval_mode)
			observation, reward_params, done, info = env.step(action)
			# reward = get_reward(reward_params, tau_reward, desired_velocity, mode="tau")
			reward = eval_reward(reward_params, tau_reward, desired_velocity)



			state = observation
			velocity_list += [state[7]]
			total_reward += reward


			if render_mode:
				env.render()
			if done:
				break
		total_reward_list += [np.array([total_reward]).flatten()]
	state = env.reset()
	total_rewards = np.array(total_reward_list)

	if settings.batch_mode == "min":
		rewards = np.min(total_rewards)
	else:
		rewards = np.mean(total_rewards)
	print (total_reward_list, rewards)
	vel_mean = np.sum(velocity_list)/np.size(velocity_list)
	print(vel_mean)
