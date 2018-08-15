from config import Settings
settings = Settings()

from rabbit_tools.policy import Policy
from rabbit_tools.reward import get_reward

from rl_tools.nn import NeuralNetwork
from rl_tools.oes import OpenES

import gym
from joblib import Parallel, delayed
import json
import os
import numpy as np
import time


def make_log(txt_log, policy_path):
	if not os.path.exists(policy_path):
		os.makedirs(policy_path)
	with open(txt_log, "w") as text_file:
		text_file.write("Summary of the training: "+"\n")

def save_policy(path, solution):
	with open(path, 'wt') as out:
		json.dump([np.array(solution).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))

def make_env(env_name, render_mode=False):
	env = gym.make(env_name)
	env.reset()
	if render_mode:	
		env.render("human")
	return env

def simulate(model, solution, settings, render_mode):

	model.set_weights(solution) #FIX FOR EACH PAIR model-solution
	
	current_speed = 0.
	desired_velocity = np.random.uniform(low=settings.desired_v_low, 
										 high=settings.desired_v_up)
	current_speed_list = []
	error = []
	env = make_env(settings.env_name, render_mode=render_mode)

	total_reward_list = [];	timesteps = 0;	reward_list = [];
	for episode in range(settings.num_episode):	
		state = env.reset()
		if state is None:
			state = np.zeros(settings.state_size)
		total_reward = 0	
		for t in range(settings.max_episode_length):
			if t == settings.max_episode_length//2:
				desired_velocity = np.random.uniform(low=settings.desired_v_low, 
										 			 high=settings.desired_v_up)

			timesteps += 1
			if render_mode:
				env.render("human")
			
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
						kp=settings.control_kp, kd=kd, feq=settings.frequency, mode = "hzdrl")

			action = pi.get_action(state)
			observation, reward_params, done, info = env.step(action)
			reward = get_reward(observation, desired_velocity, current_speed_av, mode="default")
			state = observation
				
			total_reward += reward
			reward_list += [reward]

			if done:
				break

		total_reward_list += [np.array([total_reward]).flatten()]

	total_rewards = np.array(total_reward_list)
	if settings.batch_mode == "min":
		rewards = np.min(total_rewards)
	else:
		rewards = np.mean(total_rewards)
	if render_mode:
		env.close()

	return [rewards, timesteps]


if __name__ == "__main__":
	make_log(settings.txt_log, settings.policy_path)
	model = NeuralNetwork(input_dim=settings.conditions_dim,
					 	  output_dim=settings.theta_dim+1,
					 	  units=settings.nn_units,
					 	  activations=settings.nn_activations)
	# Adopt OpenAI ES
	escls = OpenES(model.parameter_count, 
				   sigma_init=settings.sigma_init, 
				   sigma_decay=settings.sigma_decay, 
				   sigma_limit=settings.sigma_limit, 
				   learning_rate=settings.learning_rate,
				   learning_rate_decay=settings.learning_rate_decay, 
				   learning_rate_limit=settings.learning_rate_limit,
				   popsize=settings.population, 
				   antithetic=True)

	rewards = []
	step = 0
	total_timesteps = 0
	current_time = float(time.time())
	training_time = 0.
	episodes = 0
	while episodes < settings.total_episodes:
	# while total_timesteps < settings.total_threshold:
		print("======== step {} ========" .format(step)) ###DEBUG MESSAGE	
		solutions = escls.ask()
		models = []
		for _ in range(settings.population):
			m = NeuralNetwork(input_dim=settings.conditions_dim,
					 	  	  output_dim=settings.theta_dim+1,
					 	  	  units=settings.nn_units,
					 	  	  activations=settings.nn_activations)
			models += [m]

		render_mode = False
		result = Parallel(n_jobs=settings.n_jobs, backend=settings.backend)(delayed(simulate)(models[i],solutions[i],settings,render_mode,) for i in range(len(models)))

		rewards_list = []
		timesteps_list = []
		for r in result:
			rewards_list += [r[0]]
			timesteps_list += [r[1]]
		
		rewards += rewards_list
		while len(rewards) > 200:
			del rewards [0]
		total_timesteps += np.sum(np.array(timesteps_list))
		escls.tell(np.array(rewards_list), solutions)

		training_time += abs(float(time.time())-current_time)
		current_time = float(time.time())

		episodes += settings.population * settings.num_episode

		log_string = (" ave_R ", int(np.mean(np.array(rewards))*100)/100., 
					  " std_R ", int(np.std(np.array(rewards))*100)/100.,
					  " min_R ", int(np.min(np.array(rewards))*100)/100.,
					  " max_R ", int(np.max(np.array(rewards))*100)/100.,
					  " times ", int(total_timesteps),
					  " epi", int(episodes),
					  " clock ", int((training_time/60.)*100)/100., " min")
		with open(settings.txt_log, "a") as text_file:
			text_file.write(str(log_string) + "\n")

		best_solution = escls.best_param()
		save_policy(settings.policy_path+str(step)+".json", best_solution)
		


		step += 1

