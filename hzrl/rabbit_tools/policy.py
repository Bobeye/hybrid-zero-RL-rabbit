import numpy as np

import hzd.params as params

class Policy():
	def __init__(self, theta=None,
				 action_size=4,
				 action_min=-4.,
				 action_max=4.,
				 kp=120., kd=2., feq=20., 
				 mode="hzdrl"):

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
		self.bound_theta_tanh()
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

	def bound_theta_tanh(self):	#Add offset and restrict to range corresponding to each joint. The input is assumed to be bounded as tanh, with range -1,1
		theta = self.theta

		upplim_jthigh = 250*(np.pi/180)
		lowlim_jthigh = 90*(np.pi/180)
		upplim_jleg = 120*(np.pi/180)
		lowlim_jleg = 0*(np.pi/180)

		theta_thighR = np.array([theta[0], theta[4], theta[8], theta[12], theta[16]])
		theta_legR = np.array([theta[1], theta[5], theta[9], theta[13], theta[17]])
		theta_thighL = np.array([theta[2], theta[6], theta[10], theta[14], theta[18]])
		theta_legL = np.array([theta[3], theta[7], theta[11], theta[15], theta[19]])
		theta_thighR = (((upplim_jthigh - lowlim_jthigh)/2)*theta_thighR) + ((upplim_jthigh + lowlim_jthigh)/2)
		theta_legR = upplim_jleg/2*(theta_legR + 1)
		theta_thighL = (((upplim_jthigh - lowlim_jthigh)/2)*theta_thighL) + ((upplim_jthigh + lowlim_jthigh)/2)
		theta_legL = upplim_jleg/2*(theta_legL + 1)
		[theta[0], theta[4], theta[8], theta[12], theta[16]] = theta_thighR
		[theta[1], theta[5], theta[9], theta[13], theta[17]] = theta_legR
		[theta[2], theta[6], theta[10], theta[14], theta[18]] = theta_thighL
		[theta[3], theta[7], theta[11], theta[15], theta[19]] = theta_legL
		
		self.theta = theta


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

		q = np.array([pos[3], pos[4], pos[5], pos[6]])    #Take the current position state of the actuated joints and assign them to vector which will be used to compute the error
		qdot = np.array([vel[3], vel[4], vel[5], vel[6]]) #Take the current velocity state of the actuated joints and assign them to vector which will be used to compute the error
		action = self.pid.step(qd, qdotd, q, qdot)
		# print([qd, qdotd, q, qdot, action])

		return action, reward_tau