import gym
import time
import numpy as np
import params
import trajectory

#THIS SIMULATION USES THE rabbit_new.xml MODEL, IN WHICH ACTUATOR IS DEFINED AS 
#MOTOR, MAKING THE INPUT OF THE ACTUATOR TO BE TORQUE. THEN A PID CONTROLLER IS IMPLEMENTED
#IN THIS FILE SO THE JOINTS GET TO THE DESIRED ANGLE WHICH IS THE ANGLE OBTAINED FROM THE TRAJECTORIES

def pid_init():
    Kp = 200
    Kd = 20
    return Kp, Kd

def controller_update(qd, qdotd, q, qdot, Kp, Kd):
    error_pos = qd - q
    error_vel = qdotd - qdot
    output = Kp*error_pos + Kd*error_vel
    return output
    
env = gym.make('Rabbit-v0')
env.reset()

action = np.array([0.0, 0.0, 0.0, 0.0])

a_rightS = params.a_rightS
a_leftS = params.a_leftS
p = params.p
Kp, Kd = pid_init() #Initialize PID parameters for njoints = 4 (4 actuators)

for i in range(500):
    env.unwrapped.set_state(params.position[0,:],params.velocity[0,:])
  #  env.render()

aux = 0   

iter = 10000
for k in range(iter):
    start_time_iter = time.time()
 

    for i in range(20):
        pos, vel = env.get_state()
        
        tau_right = trajectory.tau_Right(pos,p)
        tau_left = trajectory.tau_Left(pos,p)
    
        if tau_right > 1.0:
            aux = 1
        
        if aux == 0:
            qd, tau = trajectory.yd_time_RightStance(pos,a_rightS,p)    #Compute the desired position for the actuated joints using the current measured state, the control parameters and bezier polynomials
            qdotd = trajectory.d1yd_time_RightStance(pos,vel,a_rightS,p)  #Compute the desired velocity for the actuated joints using the current measured state, the control parameters and bezier polynomials
            
        else:
            a = params.a_leftS
            qd = trajectory.yd_time_LeftStance(pos,a_leftS,p)    #Compute the desired position for the actuated joints using the current measured state, the control parameters and bezier polynomials
            qdotd = trajectory.d1yd_time_LeftStance(pos,vel,a_leftS,p)  #Compute the desired velocity for the actuated joints using the current measured state, the control parameters and bezier poly
            if tau_left > 1.0:
                aux = 0


        # print("Iter {}" .format(i))
        # print("q {}" .format(pos))
        # print("qd {}" .format(qd))
        # print("qdot {}" .format(vel))
        # print("qdotd {}" .format(qdotd))

        q = np.array([pos[3], pos[4], pos[5], pos[6]])    #Take the current position state of the actuated joints and assign them to vector which will be used to compute the error
        qdot = np.array([vel[3], vel[4], vel[5], vel[6]]) #Take the current velocity state of the actuated joints and assign them to vector which will be used to compute the error   

        action = controller_update(qd, qdotd, q, qdot, Kp, Kd)
        action = np.clip(action,-4,4)    
        observation, _reward, done, _info = env.step(action)
        env.render()

    # elapsed_time_iter = time.time() - start_time_iter
    # print("+++++ITERATION {} +++++++" .format(k))    
    # print("elapsed_time_iter = {}" .format(elapsed_time_iter))
