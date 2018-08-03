import gym
import time
import numpy as np
import trajectory
import params

#THIS SIMULATION USES THE rabbit_new.xml MODEL, IN WHICH ACTUATOR IS DEFINED AS 
#MOTOR, MAKING THE INPUT OF THE ACTUATOR TO BE TORQUE. THEN A PID CONTROLLER IS IMPLEMENTED
#IN THIS FILE SO THE JOINTS GET TO THE DESIRED ANGLE WHICH IS THE ANGLE OBTAINED FROM THE TRAJECTORIES

class Settings():
    env_name="Rabbit-v0"
    max_episode_length=1600
    batch_mode="mean"
    state_size = 14
    action_size = 4
    action_min = -4.
    action_max = 4.
    control_kp = 150.
    control_kd = 6.
    frequency = 20.
    aux = 0
    eval_mode = True
    def __init__(self):
        pass
settings = Settings()



def init_plot():
    tau_R = open("plots/tauR_data.txt","w+")     #Create text files to save the data o
    tau_L = open("plots/tauL_data.txt","w+")
    j1 = open("plots/j1_data.txt","w+")
    j2 = open("plots/j2_data.txt","w+")
    j3 = open("plots/j3_data.txt","w+")
    j4 = open("plots/j4_data.txt","w+")
    j1d = open("plots/j1d_data.txt","w+")
    j2d = open("plots/j2d_data.txt","w+")
    j3d = open("plots/j3d_data.txt","w+")
    j4d = open("plots/j4d_data.txt","w+")

def save_plot(tau_right, tau_left, qd, qdotd):
    tau_R = open("plots/tauR_data.txt","a")     #Create text files to save the data o
    tau_L = open("plots/tauL_data.txt","a")
    j1 = open("plots/j1_data.txt","a")
    j2 = open("plots/j2_data.txt","a")
    j3 = open("plots/j3_data.txt","a")
    j4 = open("plots/j4_data.txt","a")
    j1d = open("plots/j1d_data.txt","a")
    j2d = open("plots/j2d_data.txt","a")
    j3d = open("plots/j3d_data.txt","a")
    j4d = open("plots/j4d_data.txt","a")

    tau_R.write("%.2f\r\n" %(tau_right))
    tau_L.write("%.2f\r\n" %(tau_left))
    j1.write("%.2f\r\n" %(qd[0]))
    j2.write("%.2f\r\n" %(qd[1]))
    j3.write("%.2f\r\n" %(qd[2]))
    j4.write("%.2f\r\n" %(qd[3]))
    j1d.write("%.2f\r\n" %(qdotd[0]))
    j2d.write("%.2f\r\n" %(qdotd[1]))
    j3d.write("%.2f\r\n" %(qdotd[2]))
    j4d.write("%.2f\r\n" %(qdotd[3]))

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
				 kp=120, kd=2, feq=20.):
		self.theta = theta
		self.make_theta()
		self.action_size = action_size
		self.action_min = action_min
		self.action_max = action_max
		self.sample_time = 1/feq
		self.pid = PID(kp, 0., kd, mn=np.full((action_size,),action_min), 
								   mx=np.full((action_size,),action_max))
		self.p = params.p

	def make_theta(self):
		self.a_rightS = np.append(self.theta[0:20], [self.theta[2], self.theta[3], self.theta[0], self.theta[1]])
		# print("a_rightS = {}" .format(self.a_rightS))
		self.a_leftS = np.array([self.a_rightS[2], self.a_rightS[3], self.a_rightS[0], self.a_rightS[1],
						self.a_rightS[6], self.a_rightS[7], self.a_rightS[4], self.a_rightS[5],
						self.a_rightS[10], self.a_rightS[11], self.a_rightS[8], self.a_rightS[9],
						self.a_rightS[14], self.a_rightS[15], self.a_rightS[12], self.a_rightS[13],
						self.a_rightS[18], self.a_rightS[19], self.a_rightS[16], self.a_rightS[17],
						self.a_rightS[22], self.a_rightS[23], self.a_rightS[20], self.a_rightS[21]])


	def get_action(self, state, eval_mode = False):
		pos, vel = state[0:7], state[7:14]
		tau_right = np.clip(trajectory.tau_Right(pos,self.p), 0, 1.05)
		tau_left = np.clip(trajectory.tau_Left(pos,self.p), 0, 1.05)	
		
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
						

		q = np.array([pos[3], pos[4], pos[5], pos[6]])    #Take the current position state of the actuated joints and assign them to vector which will be used to compute the error
		qdot = np.array([vel[3], vel[4], vel[5], vel[6]]) #Take the current velocity state of the actuated joints and assign them to vector which will be used to compute the error
		action = self.pid.step(qd, qdotd, q, qdot)
		# print([qd, qdotd, q, qdot, action])

		return action




def make_env(env_name, seed=np.random.seed(None), render_mode=False, desired_velocity=None):
	env = gym.make(env_name)
	env.assign_desired_vel(desired_velocity)
	env.reset()
	if render_mode:	
		env.render("human")
	#if (seed >= 0):
	#	env.seed(seed)
	return env

# a_rightS = params.a_rightS
# a_leftS = params.a_leftS
# p = params.p
# Kp, Kd = pid_init() #Initialize PID parameters for njoints = 4 (4 actuators)

# for i in range(1000):
#     env.unwrapped.set_state(params.position[0,:],params.velocity[0,:])
#    # env.render()
aux = 0
bnd_ctrl_act = 4
plot_mode = False
render_mode = True

if plot_mode:
    init_plot()


desired_velocity = 1
max_episode_length = 1600
iter = 1


for k in range(iter):


    velocity_list = []
    total_reward = 0

    theta = np.zeros(20)

    pi = Policy(theta=theta, action_size=settings.action_size, 
				action_min=settings.action_min, action_max=settings.action_max,
				kp=settings.control_kp, kd=settings.control_kd, feq=settings.frequency)

    env = make_env(settings.env_name, render_mode=render_mode, desired_velocity = desired_velocity)    

    start_time_iter = time.time()
    # speed = np.zeros(200)
    state = env.reset()
    if state is None:
        state = np.zeros(settings.state_size)


    for i in range(max_episode_length*2):
        # pos, vel = env.get_state()
        # speed[i] = vel[0]
        # tau_right = trajectory.tau_Right(pos,p)
        # tau_left = trajectory.tau_Left(pos,p)
    
        # timesteps += 1

        if render_mode:
            env.render()   

        action = pi.get_action(state, settings.eval_mode)

     
        observation, reward, done, _info = env.step(action)
 
        #print(observation)

        state = observation
        velocity_list += [state[7]]
        total_reward += reward

        # touch_sensor1 = env.get_sensor_data("s_t1")
        # touch_sensor2 = env.get_sensor_data("s_t2")
        # print("sensor 1 = {}" .format(touch_sensor1))
        # print("sensor 2 = {}" .format(touch_sensor2))
        


    # aver_speed = np.mean(speed)
    # print(aver_speed)
    #break

        # if done:
        #     break    

    print (total_reward)
    velocity_list_new = velocity_list[500:1000]
    vel_mean = np.sum(velocity_list_new)/np.size(velocity_list_new)
    print(vel_mean, np.amin(velocity_list), np.amax(velocity_list))

    # elapsed_time_iter = time.time() - start_time_iter
    # print("+++++ITERATION {} +++++++" .format(k))    
    # print("elapsed_time_iter = {}" .format(elapsed_time_iter))
