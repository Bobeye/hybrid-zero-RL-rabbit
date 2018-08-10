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
from gym import wrappers
from joblib import Parallel, delayed
import json
import os
import numpy as np


"""Hyperparameters"""
class Settings():
	env_name="Rabbit-v0"
	txt_log = "log/" + "/train.txt"
	policy_path = "log/"+ "/policy/"
	backend = "multiprocessing"
	n_jobs = 20
	frequency = 20.
	total_threshold = 1e8
	num_episode = 5
	max_episode_length=3000
	batch_mode="mean"
	state_size = 14
	action_size = 4
	action_min = -4.
	action_max = 4.
	control_kp = 150.
	control_kd = 10.
	desired_v_low = 1.
	desired_v_up = 1.
	conditions_dim = 3	#3
	theta_dim = 20	#22 to include Kp and Ki
	nn_units=[16,16]
	nn_activations=["relu", "relu", "passthru"]
	population = 20
	sigma_init=0.1
	sigma_decay=0.999
	sigma_limit=1e-4
	learning_rate=0.01
	learning_rate_decay=0.999 
	learning_rate_limit=1e-6
	aux = 0
	upplim_jthigh = 250*(np.pi/180)
	lowlim_jthigh = 90*(np.pi/180)
	upplim_jleg = 120*(np.pi/180)
	lowlim_jleg = 0*(np.pi/180)
	upplim_kp = 150
	lowlim_kp = 150
	upplim_kd = 8
	lowlim_kd = 6

	plot_mode = False
	sigmoid_slope = 1
	init_vel = 0.7743
	size_vel_buffer = 200
	def __init__(self):
		pass
settings = Settings()

"""Customized Reward Func"""
def get_reward(reward_params, reward_tau, desired_vel, current_vel_av, sum_error, mode="quadratic"):
	alive_bonus, posafter, posbefore, velocity, a, w = reward_params[0], reward_params[1], reward_params[2], reward_params[3], reward_params[4], reward_params[5]
	# print (desired_vel)
	# print (velocity)
	if mode == "linear_bowen":	#Works better so far, but can not follow desired speed
		scale = 0.1
		# if velocity < 0:
		# 	velocity_reward = 0
		
		if abs(velocity-desired_vel) <= 0.1:
			if velocity <= desired_vel:
				velocity_reward = (velocity / desired_vel)**2
			else:
				velocity_reward = (desired_vel / velocity)**2
		else:
			velocity_reward = 0




		# elif abs(velocity-desired_vel) <= 0.1:
		# 	if velocity < desired_vel:
		# 		velocity_reward = (desirevelocity)/0.1

		# elif velocity <= desired_vel-0.1:
		# 	velocity_reward = (velocity/desired_vel)
		# elif velocity >= desired_vel+0.1:
		# 	velocity_reward = 0. #(desired_vel/velocity)
		# else:
		# 	velocity_reward = 1.

		action_reward = -1e-2 * np.sum(a**2)
		displacement_reward  = posafter
		reward = 10*velocity_reward # + 1*action_reward + 1*displacement_reward # + 0.5*sum_error
		reward = scale*reward	


	if mode == "linear1":	#Works better so far, but can not follow desired speed
		scale = 0.1
		if velocity < 0:
			velocity_reward = 0
		elif velocity <= desired_vel:
			velocity_reward = ((velocity/desired_vel)**2)
		else:
			velocity_reward = ((desired_vel/velocity)**2)

		action_reward = -1e-2 * np.sum(a**2)
		displacement_reward  = posafter
		reward = alive_bonus + 10*velocity_reward + 1*action_reward + 1*displacement_reward # + 0.5*sum_error
		reward = scale*reward	

	if mode == "quadraticB":
		scale = 0.1
		if abs(velocity - desired_vel) <= 0.1:
			if velocity <= desired_vel:
				velocity_reward = ((velocity/desired_vel)**2)
			else:
				velocity_reward = ((desired_vel/velocity)**2)
		else:
			velocity_reward = 0

		# action_reward = -1e-2 * np.sum(a**2)
		# displacement_reward  = posafter
		# error_reward =  - (desired_vel - velocity)**2
		# print(error_reward)
		reward = 10*velocity_reward # + 1*action_reward + 1*displacement_reward + 1*error_reward
		reward = scale*reward	

	return reward


"""PID controller"""
MIN_NUM = float('-inf')
MAX_NUM = float('inf')
class PID(object):
	def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
		self.kp = kp
		self.ki = ki
		self.kd = kd
		self.min = mn
		self.max = mx

	def step(self, qd, qdotd, q, qdot):
		error_pos = qd - q
		error_vel = qdotd - qdot
		output = self.kp*error_pos + self.kd*error_vel
		# print([self.kp, self.kd])
		return np.clip(output, self.min, self.max)

"""The policy defined by theta. Theta is calculated through a NN given desired conditions."""
class Policy():
	def __init__(self, theta=None,
				 action_size=4,
				 action_min=-4.,
				 action_max=4.,
				 kp=120., kd=2., feq=20., mode="hzdrl"):
		self.theta = theta
		self.make_theta()
		self.action_size = action_size
		self.action_min = action_min
		self.action_max = action_max
		self.sample_time = 1/feq
		self.pid = PID(kp, 0., kd, mn=np.full((action_size,),action_min), 
								   mx=np.full((action_size,),action_max))
		self.p = params.p
		self.mode = mode

	def make_theta(self):
		self.a_rightS = np.append(self.theta, [self.theta[2], self.theta[3], self.theta[0], self.theta[1]])
		# print("a_rightS = {}" .format(self.a_rightS))
		self.a_leftS = np.array([self.a_rightS[2], self.a_rightS[3], self.a_rightS[0], self.a_rightS[1],
						self.a_rightS[6], self.a_rightS[7], self.a_rightS[4], self.a_rightS[5],
						self.a_rightS[10], self.a_rightS[11], self.a_rightS[8], self.a_rightS[9],
						self.a_rightS[14], self.a_rightS[15], self.a_rightS[12], self.a_rightS[13],
						self.a_rightS[18], self.a_rightS[19], self.a_rightS[16], self.a_rightS[17],
						self.a_rightS[22], self.a_rightS[23], self.a_rightS[20], self.a_rightS[21]])

	def get_action_star(self, state):
		pass

	def get_action(self, state, plot_mode = False):
		pos, vel = state[0:7], state[7:14]
		tau_right = np.clip(trajectory.tau_Right(pos,self.p), 0, 1.05)
		tau_left = np.clip(trajectory.tau_Left(pos,self.p), 0, 1.05)
		
		if self.mode == "hzd": 
			reward_tau = 0
			reward_step = 0
			if tau_right > 1.0 and settings.aux == 0:
				settings.aux = 1
				reward_step = 10
			if settings.aux == 0:
				qd, tau = trajectory.yd_time_RightStance(pos,params.a_rightS,params.p)    #Compute the desired position for the actuated joints using the current measured state, the control parameters and bezier polynomials
				qdotd = trajectory.d1yd_time_RightStance(pos,vel,params.a_rightS,params.p)  #Compute the desired velocity for the actuated joints using the current measured state, the control parameters and bezier polynomials
				reward_tau = tau_right
			else:
				qd = trajectory.yd_time_LeftStance(pos,params.a_leftS,params.p)    #Compute the desired position for the actuated joints using the current measured state, the control parameters and bezier polynomials
				qdotd = trajectory.d1yd_time_LeftStance(pos,vel,params.a_leftS,params.p)  #Compute the desired velocity for the actuated joints using the current measured state, the control parameters and bezier poly
				reward_tau = tau_left
				if tau_left > 1.0 and settings.aux == 1:
					settings.aux = 0
					reward_step = 10
			reward_tau +=reward_step 
			# print(reward_tau)
						
		if self.mode == "hzdrl": 
			reward_tau = 0
			reward_step = 0
			if tau_right > 1.0 and settings.aux == 0:
				settings.aux = 1
				reward_step = 10
			if settings.aux == 0:
				qd, tau = trajectory.yd_time_RightStance(pos,self.a_rightS,self.p)    #Compute the desired position for the actuated joints using the current measured state, the control parameters and bezier polynomials
				qdotd = trajectory.d1yd_time_RightStance(pos,vel,self.a_rightS,self.p)  #Compute the desired velocity for the actuated joints using the current measured state, the control parameters and bezier polynomials
				reward_tau = tau_right
			else:
				qd = trajectory.yd_time_LeftStance(pos,self.a_leftS,self.p)    #Compute the desired position for the actuated joints using the current measured state, the control parameters and bezier polynomials
				qdotd = trajectory.d1yd_time_LeftStance(pos,vel,self.a_leftS,self.p)  #Compute the desired velocity for the actuated joints using the current measured state, the control parameters and bezier poly
				reward_tau = tau_left
				if tau_left > 1.0 and settings.aux == 1:
					settings.aux = 0
					reward_step = 10
			reward_tau +=reward_step 
				# print(reward_tau)
		
		if plot_mode:
			save_plot_1(tau_right, tau_left, qd, qdotd)			

		q = np.array([pos[3], pos[4], pos[5], pos[6]])    #Take the current position state of the actuated joints and assign them to vector which will be used to compute the error
		qdot = np.array([vel[3], vel[4], vel[5], vel[6]]) #Take the current velocity state of the actuated joints and assign them to vector which will be used to compute the error
		action = self.pid.step(qd, qdotd, q, qdot)
		# print([qd, qdotd, q, qdot, action])

		return action, reward_tau

def init_plot():
	tau_R = open("plots/tauR_data.txt","w+")     #Create text files to save the data for q and qdot desired (output of the trajectory functions)
	tau_L = open("plots/tauL_data.txt","w+")
	j1d = open("plots/j1d_data.txt","w+")			 
	j2d = open("plots/j2d_data.txt","w+")
	j3d = open("plots/j3d_data.txt","w+")
	j4d = open("plots/j4d_data.txt","w+")
	j1dotd = open("plots/j1dotd_data.txt","w+")	
	j2dotd = open("plots/j2dotd_data.txt","w+")
	j3dotd = open("plots/j3dotd_data.txt","w+")
	j4dotd = open("plots/j4dotd_data.txt","w+")
	hip_pos = open("plots/hip_pos_data.txt","w+")	#Create text files to save the data for the current state (hip  + joints)
	hip_vel = open("plots/hip_vel_data.txt","w+")
	j1pos = open("plots/j1_pos_data.txt","w+")
	j2pos = open("plots/j2_pos_data.txt","w+")
	j3pos = open("plots/j3_pos_data.txt","w+")
	j4pos = open("plots/j4_pos_data.txt","w+")

def save_plot_1(tau_right, tau_left, qd, qdotd):
	tau_R = open("plots/tauR_data.txt","a")     #Save the data in each iteration
	tau_L = open("plots/tauL_data.txt","a")
	j1d = open("plots/j1d_data.txt","a")
	j2d = open("plots/j2d_data.txt","a")
	j3d = open("plots/j3d_data.txt","a")
	j4d = open("plots/j4d_data.txt","a")
	j1dotd = open("plots/j1dotd_data.txt","a")
	j2dotd = open("plots/j2dotd_data.txt","a")
	j3dotd = open("plots/j3dotd_data.txt","a")
	j4dotd = open("plots/j4dotd_data.txt","a")

	tau_R.write("%.2f\r\n" %(tau_right))
	tau_L.write("%.2f\r\n" %(tau_left))
	j1d.write("%.2f\r\n" %(qd[0]))
	j2d.write("%.2f\r\n" %(qd[1]))
	j3d.write("%.2f\r\n" %(qd[2]))
	j4d.write("%.2f\r\n" %(qd[3]))
	j1dotd.write("%.2f\r\n" %(qdotd[0]))
	j2dotd.write("%.2f\r\n" %(qdotd[1]))
	j3dotd.write("%.2f\r\n" %(qdotd[2]))
	j4dotd.write("%.2f\r\n" %(qdotd[3]))

def save_plot_2(state):
	hip_pos = open("plots/hip_pos_data.txt","a")	#Create text files to save the data for the current state (hip  + joints)
	hip_vel = open("plots/hip_vel_data.txt","a")
	j1pos = open("plots/j1_pos_data.txt","a")
	j2pos = open("plots/j2_pos_data.txt","a")
	j3pos = open("plots/j3_pos_data.txt","a")
	j4pos = open("plots/j4_pos_data.txt","a")

	hip_pos.write("%.2f\r\n" %(state[0]))
	hip_vel.write("%.2f\r\n" %(state[7]))
	j1pos.write("%.2f\r\n" %(state[3]))
	j2pos.write("%.2f\r\n" %(state[4]))
	j3pos.write("%.2f\r\n" %(state[5]))
	j4pos.write("%.2f\r\n" %(state[6]))

	
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
	# env.assign_desired_vel(desired_velocity)
	env.reset()
	if render_mode:	
		env.render("human")
	#if (seed >= 0):
	#	env.seed(seed)
	return env

# BOUND THETA USING TANH FUNCTION
def bound_theta_tanh(theta):		#Add offset and restrict to range corresponding to each joint. The input is assumed to be bounded as tanh, with range -1,1
	theta_thighR = np.array([theta[0], theta[4], theta[8], theta[12], theta[16]])
	theta_legR = np.array([theta[1], theta[5], theta[9], theta[13], theta[17]])
	theta_thighL = np.array([theta[2], theta[6], theta[10], theta[14], theta[18]])
	theta_legL = np.array([theta[3], theta[7], theta[11], theta[15], theta[19]])
	theta_thighR = (((settings.upplim_jthigh - settings.lowlim_jthigh)/2)*theta_thighR) + ((settings.upplim_jthigh + settings.lowlim_jthigh)/2)
	theta_legR = settings.upplim_jleg/2*(theta_legR + 1)
	theta_thighL = (((settings.upplim_jthigh - settings.lowlim_jthigh)/2)*theta_thighL) + ((settings.upplim_jthigh + settings.lowlim_jthigh)/2)
	theta_legL = settings.upplim_jleg/2*(theta_legL + 1)
	[theta[0], theta[4], theta[8], theta[12], theta[16]] = theta_thighR
	[theta[1], theta[5], theta[9], theta[13], theta[17]] = theta_legR
	[theta[2], theta[6], theta[10], theta[14], theta[18]] = theta_thighL
	[theta[3], theta[7], theta[11], theta[15], theta[19]] = theta_legL
	return theta

# BOUND THETA USING SIGMOID FUNCTION
def sigmoid(x):
	return 1 / (1 + np.exp(-settings.sigmoid_slope*x))
def bound_theta_sigmoid(theta):		#Add offset and restrict to range corresponding to each joint. The input is assumed to be bounded as tanh, with range -1,1
	theta = sigmoid(theta)
	theta_thighR = np.array([theta[0], theta[4], theta[8], theta[12], theta[16]])
	theta_legR = np.array([theta[1], theta[5], theta[9], theta[13], theta[17]])
	theta_thighL = np.array([theta[2], theta[6], theta[10], theta[14], theta[18]])
	theta_legL = np.array([theta[3], theta[7], theta[11], theta[15], theta[19]])
	theta_thighR = ((settings.upplim_jthigh - settings.lowlim_jthigh)*theta_thighR) + settings.lowlim_jthigh
	theta_legR = settings.upplim_jleg*(theta_legR)
	theta_thighL = ((settings.upplim_jthigh - settings.lowlim_jthigh)*theta_thighL) + settings.lowlim_jthigh
	theta_legL = settings.upplim_jleg*(theta_legL)
	# theta_kp = ((settings.upplim_kp - settings.lowlim_kp)*theta[20]) + settings.lowlim_kp
	# theta_kd = ((settings.upplim_kd - settings.lowlim_kd)*theta[21]) + settings.lowlim_kd
	[theta[0], theta[4], theta[8], theta[12], theta[16]] = theta_thighR
	[theta[1], theta[5], theta[9], theta[13], theta[17]] = theta_legR
	[theta[2], theta[6], theta[10], theta[14], theta[18]] = theta_thighL
	[theta[3], theta[7], theta[11], theta[15], theta[19]] = theta_legL
	# theta[20] = theta_kp
	# theta[21] = theta_kd
	return theta


def simulate(model, solution, settings, 
			 dv_min, dv_max, render_mode):

	model.set_weights(solution) #FIX FOR EACH PAIR model-solution
	
	current_speed = settings.init_vel
	desired_velocity = np.random.uniform(low=dv_min, high=dv_max)
	current_speed_list = []
	error = []

	env = make_env(settings.env_name, render_mode=render_mode, desired_velocity = desired_velocity)


	total_reward_list = [];	timesteps = 0;	state_list = [];	action_list = [];	reward_list = [];	termination_list = [];	last_speed = None

	for episode in range(settings.num_episode):	
		state = env.reset()
		if state is None:
			state = np.zeros(settings.state_size)
		total_reward = 0
		
		for t in range(settings.max_episode_length):
			if t == settings.max_episode_length//2:
				desired_velocity = np.random.uniform(low=dv_min, high=dv_max)
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

			# if last_speed is None or (current_speed - last_speed) < 1e-1: # 1e-2
			inputs_nn = np.array([current_speed_av, 
								  desired_velocity, 
								  np.mean(np.array(error))])	
			# print current_speed, current_speed_av, desired_velocity
			theta_kd = model.predict(inputs_nn)
			theta = bound_theta_sigmoid(theta_kd[0:settings.theta_dim])
			kd = sigmoid(theta_kd[-1]) * settings.control_kd
			pi = Policy(theta=theta, action_size=settings.action_size, 
						action_min=settings.action_min, action_max=settings.action_max,
						kp=settings.control_kp, kd=kd, feq=settings.frequency, mode = "hzdrl")

			action, reward_tau = pi.get_action(state, settings.plot_mode)
			observation, reward_params, done, info = env.step(action)
			reward = get_reward(reward_params, reward_tau, desired_velocity, current_speed_av, sum(error), mode="linear_bowen")
			state = observation
			if settings.plot_mode:
				save_plot_2(state)			
			total_reward += reward
			# state_list += [state.flatten()]
			# action_list += [action]
			# reward_list += [reward]
			# termination_list += [done]
			if done:
				break

		total_reward_list += [np.array([total_reward]).flatten()]

	total_rewards = np.array(total_reward_list)
	if settings.batch_mode == "min":
		rewards = np.min(total_rewards)
	else:
		rewards = np.mean(total_rewards)
	# if render_mode:
	# 	env.close()

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

	# # Adopt CMA-ES
	# escls = CMAES(model.parameter_count,
	# 			  sigma_init=settings.sigma_init,
	# 			  popsize=settings.population,
	# 			  weight_decay=settings.sigma_decay)

	step = 0
	total_timesteps = 0
	while total_timesteps < settings.total_threshold:
		print("======== step {} ========" .format(step)) ###DEBUG MESSAGE	
		solutions = escls.ask()
		#print(solutions.shape)
		models = []
		for _ in range(settings.population):
			m = NeuralNetwork(input_dim=settings.conditions_dim,
					 	  	  output_dim=settings.theta_dim+1,
					 	  	  units=settings.nn_units,
					 	  	  activations=settings.nn_activations)
			models += [m]

		
		# desired_velocities = np.random.uniform(low=settings.desired_v_low, 
		# 									   high=settings.desired_v_up, 
		# 									   size=(settings.population,))
		
		# desired_velocities = np.array([0.8, 1.2, 1.2, 1.2, 0.8, 0.8, 1.2,1.2, 0.8, 0.8, 1.2, 0.8, 0.8, 0.8, 1.2, 1.2, 0.8, 0.8, 0.8, 1.2,0.8, 1.2, 0.8, 0.8])
		# desired_velocities = np.random.uniform(low=0.5, high=1.5, size=(settings.population,2))

		up_lim_vel = 1.6
		down_lim_vel = 0.6
		# desired_velocities = np.zeros(settings.population,) + 2.
		render_mode = False
		result = Parallel(n_jobs=settings.n_jobs, backend=settings.backend)(delayed(simulate)(models[i],solutions[i],settings,0.7, 1.6,render_mode,) for i in range(len(models)))

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