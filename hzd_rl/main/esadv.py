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
				 episode_length=3000,
				 policy_mode="hzdrl",
				 adversary_mode="forward"):
		self.model = NeuralNetwork(input_dim=settings.conditions_dim,
					 	  		   output_dim=settings.theta_dim+1,
					 	  		   units=settings.nn_units,
					 	  		   activations=settings.nn_activations)
		model_params = load_model(policy_path)
		self.model.set_weights(model_params)

		self.policy_mode = policy_mode
		self.adversary_mode = adversary_mode

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
		env = make_env("RabbitAdv-v0", render_mode=render_mode)
		env.update_adversary(6)
		env = wrappers.Monitor(env, self.video_path, video_callable=lambda episode_id: True, force=True)
		
		kds = []
		velocity_list = []
		dvs = []
		states = []
		adversaries = []

		state = env.reset()
		if state is None:
			state = np.zeros(settings.state_size)
		done = False
		t = 0
		u = np.zeros(2)
		while done is False and t < 5000:
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
			adversaries += [u]
				
			pi = Policy(theta=theta, action_size=settings.action_size, 
						action_min=settings.action_min, action_max=settings.action_max,
						kp=settings.control_kp, kd=kd, feq=settings.frequency, mode=self.policy_mode) #Use mode="hzd" to run the hzd controller, and mode="hzdrl" to run the learned controller

			action = pi.get_action(state)

			if t > 2000 and t % 1000 < 500:
				# u = env.adv_action_space.sample()
				if self.adversary_mode == "forward":
					u = np.array([4, 0])
				if self.adversary_mode == "backward":
					u = np.array([-5, 0])
				if self.adversary_mode == "backward_hard":
					u = np.array([-8, 0])
				observation, reward_params, done, info = env.step_adv(action, u)
			else:
				u = np.zeros(2)
				observation, reward_params, done, info = env.step(action)
			reward = get_reward(observation, dv, current_speed_av)
			state = observation

			t += 1

		env.close()

		data = dict()
		data["kds"] = np.array(kds)
		data["vels"] = np.array(velocity_list)
		data["dvs"] = np.array(dvs)
		data["states"] = np.array(states)
		data["adversaries"] = np.array(adversaries)

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
	candidates = [1.0]
	for c in candidates:
		test = Evaluation(policy_path="log/policy/2000.json",
						  video_path="log/2000/adv/hzdrl_backward_hard",
						  figure_path="log/2000/adv/hzdrl_backward_hard",
						  desired_velocity=[c],
						  episode_length=10000,
						  policy_mode="hzdrl",
						  adversary_mode="backward_hard")
	# policy_path = "log/policy/1900.json"
	# render_mode = True

	# current_speed = 0.
	# model = NeuralNetwork(input_dim=settings.conditions_dim,
	# 				 	  output_dim=settings.theta_dim+1,
	# 				 	  units=settings.nn_units,
	# 				 	  activations=settings.nn_activations)
	# model_params = load_model(policy_path)
	# model.set_weights(model_params)

	# current_speed_list, error = [], []

	# env = make_env("RabbitAdv-v0", render_mode=render_mode)
	# env.update_adversary(6)
	# for desired_velocity in [0.8, 1.0, 1.2, 1.4]:
	# 	total_reward_list = []
	# 	velocity_list = []

	# 	state = env.reset()
	# 	timesteps = 0
	# 	if state is None:
	# 		state = np.zeros(settings.state_size)
	# 	total_reward = 0
	# 	done = False
	# 	for t in range(min(10000,8000)):
	# 	# while not done:
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

				
	# 		pi = Policy(theta=theta, action_size=settings.action_size, 
	# 					action_min=settings.action_min, action_max=settings.action_max,
	# 					kp=settings.control_kp, kd=kd, feq=settings.frequency, mode="hzd") #Use mode="hzd" to run the hzd controller, and mode="hzdrl" to run the learned controller

	# 		action = pi.get_action(state)
			

	# 		# if timesteps % 10 == 0:
	# 		if timesteps % 200 < 50:
	# 			# u = env.adv_action_space.sample()
	# 			u = np.array([4,0])
	# 			observation, reward_params, done, info = env.step_adv(action, u)
	# 		else:
	# 			observation, reward_params, done, info = env.step(action)
	# 		reward = get_reward(observation, desired_velocity, current_speed_av)
	
	# 		state = observation
	# 		velocity_list += [state[7]]

	# 		if done:
	# 			break

	# 	plt.plot(velocity_list)
	# 	plt.show()

