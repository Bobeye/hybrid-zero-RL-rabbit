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

from rabbit_estrain import settings, Policy, make_env


def load_model(filename):
    with open(filename) as f:    
    	data = json.load(f)
    print('loading file %s' % (filename))
    model_params = np.array(data[0]) # assuming other stuff is in data
    return model_params

if __name__ == "__main__":
	policy_path = "log/"+"Rabbit-v0" +"/policy/0.json"
	render_mode = True
	desired_velocity = 2.

	model = NeuralNetwork(input_dim=settings.conditions_dim,
					 	  output_dim=settings.theta_dim,
					 	  units=settings.nn_units,
					 	  activations=settings.nn_activations)
	model_params = load_model(policy_path)
	model.set_weights(model_params)
	theta = model.predict(desired_velocity)
	pi = Policy(theta=theta, action_size=settings.action_size, 
				action_min=settings.action_min, action_max=settings.action_max,
				kp=settings.control_kp, kd=settings.control_kd, feq=settings.frequency)
	env = make_env(settings.env_name, render_mode=render_mode, desired_velocity = desired_velocity)
	
	total_reward_list = []
	for episode in range(settings.num_episode):	
		state = env.reset()
		if state is None:
			state = np.zeros(settings.state_size)
		state = state
		total_reward = 0
		for t in range(settings.max_episode_length):
			timesteps += 1
			action = pi.get_action(state)
			observation, reward, done, info = env.step(action)
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

	print total_reward_list, rewards