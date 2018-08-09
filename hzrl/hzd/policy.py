import trajectory
import params
import numpy as np

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
    	return np.clip(output, self.min, self.max)

"""The policy defined by theta. Theta is calculated through a NN given desired conditions."""
class Policy():
	def __init__(self, theta=None,
				 action_size=4,
				 action_min=-4.,
				 action_max=4.,
				 kp=50, kd=1, feq=20.):
		self.theta = theta
		self.make_theta()
		self.action_size = action_size
		self.action_min = action_min
		self.action_max = action_max
		self.sample_time = 1/feq
		self.pid = PID(kp, 0., kd, mn=np.full((action_size,),action_min), 
								   mx=np.full((action_size,),action_max))
		self.p = np.array([0.2517, -0.2])

	def make_theta(self):
		self.a_rightS = np.append(self.theta[0:20], [self.theta[2], self.theta[3], self.theta[0], self.theta[1]])
		# print("a_rightS = {}" .format(self.a_rightS))
		self.a_leftS = np.array([self.a_rightS[2], self.a_rightS[3], self.a_rightS[0], self.a_rightS[1],
						self.a_rightS[6], self.a_rightS[7], self.a_rightS[4], self.a_rightS[5],
						self.a_rightS[10], self.a_rightS[11], self.a_rightS[8], self.a_rightS[9],
						self.a_rightS[14], self.a_rightS[15], self.a_rightS[12], self.a_rightS[13],
						self.a_rightS[18], self.a_rightS[19], self.a_rightS[16], self.a_rightS[17],
						self.a_rightS[22], self.a_rightS[23], self.a_rightS[20], self.a_rightS[21]])

	def get_action_star(self, state):
		pass

	def get_action(self, state, eval_mode = False):
		pos, vel = state[0:7], state[7:14]
		tau_right = np.clip(trajectory.tau_Right(pos,self.p), 0, 1.05)
		tau_left = np.clip(trajectory.tau_Left(pos,self.p), 0, 1.05)	
		
		# if tau_right > 1.0:
		# 	settings.aux = 1
		# if settings.aux == 0:
		# 	qd, tau = trajectory.yd_time_RightStance(pos,params.a_rightS,params.p)    #Compute the desired position for the actuated joints using the current measured state, the control parameters and bezier polynomials
		# 	qdotd = trajectory.d1yd_time_RightStance(pos,vel,params.a_rightS,params.p)  #Compute the desired velocity for the actuated joints using the current measured state, the control parameters and bezier polynomials
		# else:
		# 	qd = trajectory.yd_time_LeftStance(pos,params.a_leftS,params.p)    #Compute the desired position for the actuated joints using the current measured state, the control parameters and bezier polynomials
		# 	qdotd = trajectory.d1yd_time_LeftStance(pos,vel,params.a_leftS,params.p)  #Compute the desired velocity for the actuated joints using the current measured state, the control parameters and bezier poly
		# 	if tau_left > 1.0:
		# 		settings.aux = 0		

		#print ([tau_right, tau_left])
		

		if tau_right > 1.0:
			settings.aux = 1
		if settings.aux == 0:
			qd, tau = trajectory.yd_time_RightStance(pos,self.a_rightS,self.p)    #Compute the desired position for the actuated joints using the current measured state, the control parameters and bezier polynomials
			qdotd = trajectory.d1yd_time_RightStance(pos,vel,self.a_rightS,self.p)  #Compute the desired velocity for the actuated joints using the current measured state, the control parameters and bezier polynomials
		else:
			qd = trajectory.yd_time_LeftStance(pos,self.a_leftS,self.p)    #Compute the desired position for the actuated joints using the current measured state, the control parameters and bezier polynomials
			qdotd = trajectory.d1yd_time_LeftStance(pos,vel,self.a_leftS,self.p)  #Compute the desired velocity for the actuated joints using the current measured state, the control parameters and bezier poly
			if tau_left > 1.0:
				settings.aux = 0
		
		if eval_mode:
			save_plot(tau_right, tau_left, qd, qdotd)			

		q = np.array([pos[3], pos[4], pos[5], pos[6]])    #Take the current position state of the actuated joints and assign them to vector which will be used to compute the error
		qdot = np.array([vel[3], vel[4], vel[5], vel[6]]) #Take the current velocity state of the actuated joints and assign them to vector which will be used to compute the error
		action = self.pid.step(qd, qdotd, q, qdot)

		return action