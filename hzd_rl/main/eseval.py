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

class Evaluation():

	def __init__(self, policy_path=None, 
				 video_path=None, 
				 figure_path=None,
				 desired_velocity=[],
				 episode_length=3000):
		self.model = NeuralNetwork(input_dim=settings.conditions_dim,
					 	  		   output_dim=settings.theta_dim+1,
					 	  		   units=settings.nn_units,
					 	  		   activations=settings.nn_activations)
		model_params = load_model(policy_path)
		self.model.set_weights(model_params)

		self.video_path = video_path
		self.figure_path = figure_path
		self.desired_velocity = desired_velocity
		self.episode_length = episode_length

		if len(desired_velocity) == 1:
			file = figure_path+"/fix_"
			for v in self.desired_velocity:
				file += str(v)
				file += "_"
			self.figure_path = file
			self.video_path = file
			if not os.path.exists(self.figure_path):
				os.makedirs(self.figure_path)

			self.run_fix()
		elif len(desired_velocity) > 1:
			file = figure_path+"/vary_"
			for v in self.desired_velocity:
				file += str(v)
				file += "_"
			self.figure_path = file
			self.video_path = file
			if not os.path.exists(self.figure_path):
				os.makedirs(self.figure_path)


			self.run_vary()
		else:
			raise ValueError("desired velocity cannot be empty")

	def run_fix(self):
		render_mode = True
		dv = self.desired_velocity[0]
		current_speed = 0.
		current_speed_list, error = [], []
		env = make_env(settings.env_name, render_mode=render_mode)
		env = wrappers.Monitor(env, self.video_path, video_callable=lambda episode_id: True, force=True)
		
		kds = []
		velocity_list = []
		dvs = []
		states = []

		state = env.reset()
		if state is None:
			state = np.zeros(settings.state_size)
		done = False
		while done is False:
			if render_mode:
				env.render()
				
			current_speed = state[7]
			current_speed_list += [current_speed]
			while len(current_speed_list) > settings.size_vel_buffer:
				del current_speed_list[0]
			current_speed_av = np.mean(np.array(current_speed_list))

			error += [dv - current_speed]
			while len(error) > settings.size_vel_buffer:
				del error[0]

			inputs_nn = np.array([current_speed_av, 
								  dv, 
								  np.mean(np.array(error))])	

			theta_kd = self.model.predict(inputs_nn)
			theta = theta_kd[0:settings.theta_dim]
			kd = (1 / (1 + np.exp(-theta_kd[-1]))) * settings.control_kd
			kds += [kd]
			velocity_list += [state[7]]
			dvs += [dv]
			states += [state]
				
			pi = Policy(theta=theta, action_size=settings.action_size, 
						action_min=settings.action_min, action_max=settings.action_max,
						kp=settings.control_kp, kd=kd, feq=settings.frequency, mode="hzdrl") #Use mode="hzd" to run the hzd controller, and mode="hzdrl" to run the learned controller

			action = pi.get_action(state)
			observation, reward_params, done, info = env.step(action)
			reward = get_reward(observation, dv, current_speed_av)
			state = observation

		env.close()

		data = dict()
		data["kds"] = np.array(kds)
		data["vels"] = np.array(velocity_list)
		data["dvs"] = np.array(dvs)
		data["states"] = np.array(states)

		with open(self.figure_path+"/fix_"+str(dv)+".pkl", "wb") as p:
			pickle.dump(data, p)

		plt.plot(velocity_list)
		plt.savefig(self.figure_path+"/fix_"+str(dv)+".png")
		plt.clf()

	def run_vary(self):
		render_mode = True
		
		current_speed = 0.
		current_speed_list, error = [], []
		env = make_env(settings.env_name, render_mode=render_mode)
		env = wrappers.Monitor(env, self.video_path, video_callable=lambda episode_id: True, force=True)
		
		kds = []
		velocity_list = []
		dvs = []
		states = []

		state = env.reset()
		if state is None:
			state = np.zeros(settings.state_size)
		done = False
		times = 0
		i = 0
		while done is False:
			times += 1
			if times % int(self.episode_length/len(self.desired_velocity)) == 0:
				i += 1
			if i >= len(self.desired_velocity):
				pass
			else:
				dv = self.desired_velocity[i]
		
			if render_mode:
				env.render()
				
			current_speed = state[7]
			current_speed_list += [current_speed]
			while len(current_speed_list) > settings.size_vel_buffer:
				del current_speed_list[0]
			current_speed_av = np.mean(np.array(current_speed_list))

			error += [dv - current_speed]
			while len(error) > settings.size_vel_buffer:
				del error[0]

			inputs_nn = np.array([current_speed_av, 
								  dv, 
								  np.mean(np.array(error))])	

			theta_kd = self.model.predict(inputs_nn)
			theta = theta_kd[0:settings.theta_dim]
			kd = (1 / (1 + np.exp(-theta_kd[-1]))) * settings.control_kd
			kds += [kd]
			velocity_list += [state[7]]
			dvs += [dv]
			states += [state]
				
			pi = Policy(theta=theta, action_size=settings.action_size, 
						action_min=settings.action_min, action_max=settings.action_max,
						kp=settings.control_kp, kd=kd, feq=settings.frequency, mode="hzdrl") #Use mode="hzd" to run the hzd controller, and mode="hzdrl" to run the learned controller

			action = pi.get_action(state)
			observation, reward_params, done, info = env.step(action)
			reward = get_reward(observation, dv, current_speed_av)
			state = observation

		env.close()

		data = dict()
		data["kds"] = np.array(kds)
		data["vels"] = np.array(velocity_list)
		data["dvs"] = np.array(dvs)
		data["states"] = np.array(states)

		file = self.figure_path+"/vary_"
		for v in self.desired_velocity:
			file += str(v)
			file += "_"
		with open(file+".pkl", "wb") as p:
			pickle.dump(data, p)

		plt.plot(velocity_list)
		plt.savefig(file+".png")
		plt.clf()





if __name__ == "__main__":
	# candidates = [0.8, 1.0, 1.3]
	# for c in candidates:
	# 	test = Evaluation(policy_path="log/policy/2300.json",
	# 					  video_path="log/2300/video",
	# 					  figure_path="log/2300/data",
	# 					  desired_velocity=[c],
	# 					  episode_length=10000)

	# candidates = [[0.8, 1.4], [1.4, 0.8], [1.0, 0.8, 1.3], [0.9, 1.2, 0.9, 1.4]]
	# for c in candidates:
	# 	test = Evaluation(policy_path="log/policy/2300.json",
	# 					  video_path="log/2300/video",
	# 					  figure_path="log/2300/data",
	# 					  desired_velocity=c,
	# 					  episode_length=10000)


	# for i in range(2000):
	# 	if (i % 50 == 0 and i < 1000):
	# 		test = Evaluation(policy_path="log/policy/"+str(i)+".json",
	# 					  	  video_path="log/"+str(i)+"/video",
	# 					  	  figure_path="log/"+str(i)+"/data",
	# 					  	  desired_velocity=[0.9, 1.2, 0.9, 1.4],
	# 					  	  episode_length=10000)
	# 	if i>1000 and i<2000 and i%200 == 0:
	# 		test = Evaluation(policy_path="log/policy/"+str(i)+".json",
	# 					  	  video_path="log/"+str(i)+"/video",
	# 					  	  figure_path="log/"+str(i)+"/data",
	# 					  	  desired_velocity=[0.9, 1.2, 0.9, 1.4],
	# 					  	  episode_length=10000)

	for i in range(200):
		if i % 10 == 0:
			test = Evaluation(policy_path="log/policy/"+str(i)+".json",
						  	  video_path="log/"+str(i)+"/video",
						  	  figure_path="log/"+str(i)+"/data",
						  	  desired_velocity=[0.9, 1.2, 0.9, 1.4],
						  	  episode_length=10000)
		


	# policy_path = "demo/demo.json"
	# video_path = "demo"
	# render_mode = True

	# desired_velocity = 1.
	# current_speed = 0.
	# model = NeuralNetwork(input_dim=settings.conditions_dim,
	# 				 	  output_dim=settings.theta_dim+1,
	# 				 	  units=settings.nn_units,
	# 				 	  activations=settings.nn_activations)
	# #Obtain theta from .json file's data
	# model_params = load_model(policy_path)
	# model.set_weights(model_params)

	# current_speed_list, error = [], []

	# env = make_env(settings.env_name, render_mode=render_mode)
	# if video_path is not None:
	# 	env = wrappers.Monitor(env, video_path, video_callable=lambda episode_id: True, force=True)
	# kds = []
	# for desired_velocity in [0.8, 1.0, 1.2, 1.5]:
	# 	total_reward_list = []
	# 	velocity_list = []

	# 	state = env.reset()
	# 	timesteps = 0
	# 	if state is None:
	# 		state = np.zeros(settings.state_size)
	# 	total_reward = 0
	# 	for t in range(min(4000,settings.max_episode_length)):
	# 		# if t < 1200:
	# 		# 	desired_velocity = 0.8
	# 		# else:
	# 		# 	desired_velocity = 1.3

	# 		timesteps += 1
	# 		if render_mode:
	# 			env.render()
				
	# 		current_speed = state[7]
	# 		current_speed_list += [current_speed]
	# 		while len(current_speed_list) > settings.size_vel_buffer:
	# 			del current_speed_list[0]
	# 		current_speed_av = np.mean(np.array(current_speed_list))

	# 		error += [desired_velocity - current_speed]
	# 		while len(error) > settings.size_vel_buffer:
	# 			del error[0]

	# 		inputs_nn = np.array([current_speed_av, 
	# 							  desired_velocity, 
	# 							  np.mean(np.array(error))])	

	# 		theta_kd = model.predict(inputs_nn)
	# 		theta = theta_kd[0:settings.theta_dim]
	# 		kd = (1 / (1 + np.exp(-theta_kd[-1]))) * settings.control_kd
	# 		kds += [kd]
				
	# 		pi = Policy(theta=theta, action_size=settings.action_size, 
	# 					action_min=settings.action_min, action_max=settings.action_max,
	# 					kp=settings.control_kp, kd=kd, feq=settings.frequency, mode="hzdrl") #Use mode="hzd" to run the hzd controller, and mode="hzdrl" to run the learned controller

	# 		action = pi.get_action(state)
	# 		observation, reward_params, done, info = env.step(action)
	# 		reward = get_reward(observation, desired_velocity, current_speed_av)
	
	# 		state = observation
	# 		velocity_list += [state[7]]
	# 		total_reward += reward

	# 		if done:
	# 			break

	# 	total_reward_list += [np.array([total_reward]).flatten()]
	# 	total_rewards = np.array(total_reward_list)

	# 	if settings.batch_mode == "min":
	# 		rewards = np.min(total_rewards)
	# 	else:
	# 		rewards = np.mean(total_rewards)
		
	# 	velocity_list_new = velocity_list#[500:1500]
	# 	vel_mean = np.sum(velocity_list_new)/np.size(velocity_list_new)
	# 	print(desired_velocity, vel_mean, np.amin(velocity_list), np.amax(velocity_list))


	# 	plt.plot(velocity_list)
	# 	plt.savefig(video_path+"/"+str(desired_velocity)+".png")

	# 	# # plt.show()
	# 	# plt.savefig(video_path+"/"+str(desired_velocity)+".png")
	# 	# plt.plot(kds)
	# 	# plt.savefig(video_path+"/"+"kds.png")

