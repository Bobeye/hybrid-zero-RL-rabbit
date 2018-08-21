from config import Settings
settings = Settings()

from rabbit_tools.policy import Policy
from rabbit_tools.reward import get_reward

from rl_tools.nn import NeuralNetwork
from rl_tools.oes import OpenES

import gym
from gym import wrappers
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import pickle

def make_env(env_name, render_mode=False):
	env = gym.make(env_name)
	env.reset()
	if render_mode:	
		env.render("human")
	return env

def load_model(filename):
    with open(filename) as f:    
    	data = json.load(f)
    print('loading file %s' % (filename))
    model_params = np.array(data[0]) # assuming other stuff is in data
    return model_params	



if __name__ == "__main__":
	policy_path = "log/policy/1999.json"
	render_mode = True

	current_speed = 0.
	model = NeuralNetwork(input_dim=settings.conditions_dim,
					 	  output_dim=settings.theta_dim+1,
					 	  units=settings.nn_units,
					 	  activations=settings.nn_activations)
	model_params = load_model(policy_path)
	model.set_weights(model_params)

	current_speed_list, error = [], []

	env = make_env(settings.env_name, render_mode=render_mode)
	for desired_velocity in [0.8, 1.0, 1.2, 1.5]:
		total_reward_list = []
		velocity_list = []

		state = env.reset()
		timesteps = 0
		if state is None:
			state = np.zeros(settings.state_size)
		total_reward = 0
		done = False
		# for t in range(min(3000,settings.max_episode_length)):
		while not done:
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
			theta = theta_kd[0:settings.theta_dim]
			kd = (1 / (1 + np.exp(-theta_kd[-1]))) * settings.control_kd

				
			pi = Policy(theta=theta, action_size=settings.action_size, 
						action_min=settings.action_min, action_max=settings.action_max,
						kp=settings.control_kp, kd=kd, feq=settings.frequency, mode="hzdrl") #Use mode="hzd" to run the hzd controller, and mode="hzdrl" to run the learned controller

			action = pi.get_action(state)
			observation, reward_params, done, info = env.step(action)
			reward = get_reward(observation, desired_velocity, current_speed_av)
	
			state = observation
			velocity_list += [state[7]]

			if done:
				break

		plt.plot(velocity_list)
		plt.show()

		# # plt.show()
		# plt.savefig(video_path+"/"+str(desired_velocity)+".png")
		# plt.plot(kds)
		# plt.savefig(video_path+"/"+"kds.png")

