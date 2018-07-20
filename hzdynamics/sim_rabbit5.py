import gym
import time
import numpy as np
import params
import trajectory

#THIS SIMULATION USES THE rabbit_new.xml MODEL, IN WHICH ACTUATOR IS DEFINED AS 
#MOTOR, MAKING THE INPUT OF THE ACTUATOR TO BE TORQUE. THEN A PID CONTROLLER IS IMPLEMENTED
#IN THIS FILE SO THE JOINTS GET TO THE DESIRED ANGLE WHICH IS THE ANGLE OBTAINED FROM THE TRAJECTORIES


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

def pid_init():
    Kp = 150        # 5 for 0.75m/s ; 180 for 0.0.85m/s ;  ;180 for 1 m/s
    Kd = 5          # 5 for 0.75m/s ; 7 for 0.0.85m/s ;  ;9 for 1 m/s
    return Kp, Kd

def controller_update(qd, qdotd, q, qdot, Kp, Kd):
    error_pos = qd - q
    error_vel = qdotd - qdot
    output = Kp*error_pos + Kd*error_vel
    return output
    
env = gym.make('Rabbit-v0')
env.assign_desired_vel(desired_vel = 1)
#print(env.desired_vel)
env.reset()

action = np.array([0.0, 0.0, 0.0, 0.0])

a_rightS = params.a_rightS
a_leftS = params.a_leftS

p = params.p
Kp, Kd = pid_init() #Initialize PID parameters for njoints = 4 (4 actuators)

for i in range(1000):
    env.unwrapped.set_state(params.position[0,:],params.velocity[0,:])
   # env.render()

aux = 0
bnd_ctrl_act = 4
init_plot()




iter = 20
for k in range(iter):
    start_time_iter = time.time()
    speed = np.zeros(200)

    for i in range(200):
        pos, vel = env.get_state()
        speed[i] = vel[0]
        # tau_right = trajectory.tau_Right(pos,p)
        # tau_left = trajectory.tau_Left(pos,p)

        tau_right = np.clip(trajectory.tau_Right(pos,p), 0, 1.05)
        tau_left = np.clip(trajectory.tau_Left(pos,p), 0, 1.05)	

        if tau_right > 1.0 and aux == 0:
            aux = 1
        
        if aux == 0:
            qd, tau = trajectory.yd_time_RightStance(pos,a_rightS,p)    #Compute the desired position for the actuated joints using the current measured state, the control parameters and bezier polynomials
            qdotd = trajectory.d1yd_time_RightStance(pos,vel,a_rightS,p)  #Compute the desired velocity for the actuated joints using the current measured state, the control parameters and bezier polynomials
            
        else:
            qd = trajectory.yd_time_LeftStance(pos,a_leftS,p)    #Compute the desired position for the actuated joints using the current measured state, the control parameters and bezier polynomials
            qdotd = trajectory.d1yd_time_LeftStance(pos,vel,a_leftS,p)  #Compute the desired velocity for the actuated joints using the current measured state, the control parameters and bezier poly
            if tau_left > 1.0 and aux ==1:
                aux = 0

        save_plot(tau_right, tau_left, qd, qdotd)

        # print("Iter {}" .format(i))
        # print("q {}" .format(pos))
        # print("qd {}" .format(qd))
        # print("qdot {}" .format(vel))
        # print("qdotd {}" .format(qdotd))

        q = np.array([pos[3], pos[4], pos[5], pos[6]])    #Take the current position state of the actuated joints and assign them to vector which will be used to compute the error
        qdot = np.array([vel[3], vel[4], vel[5], vel[6]]) #Take the current velocity state of the actuated joints and assign them to vector which will be used to compute the error   

        action = controller_update(qd, qdotd, q, qdot, Kp, Kd)
        action = np.clip(action,-bnd_ctrl_act,bnd_ctrl_act)
        # print(action)
        observation, _reward, done, _info = env.step(action)
        #print(observation)


        # touch_sensor1 = env.get_sensor_data("s_t1")
        # touch_sensor2 = env.get_sensor_data("s_t2")
        # print("sensor 1 = {}" .format(touch_sensor1))
        # print("sensor 2 = {}" .format(touch_sensor2))
        
        env.render()

    aver_speed = np.mean(speed)
    print(aver_speed)
    #break

    # elapsed_time_iter = time.time() - start_time_iter
    # print("+++++ITERATION {} +++++++" .format(k))    
    # print("elapsed_time_iter = {}" .format(elapsed_time_iter))
