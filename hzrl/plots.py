import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def load_plot(file):
    var_data = open(file,"r")
    if var_data.mode == "r":
        var_output = var_data.read()
        var_output = var_output.split('\n')
        del var_output[-1]
        var_output = list(map(float, var_output))
        return var_output

tau_R = load_plot("plots/tauR_data.txt")     #Load data for q and qdot desired (output of the trajectory functions)
tau_L = load_plot("plots/tauL_data.txt")
j1d = load_plot("plots/j1d_data.txt")			 
j2d = load_plot("plots/j2d_data.txt")
j3d = load_plot("plots/j3d_data.txt")
j4d = load_plot("plots/j4d_data.txt")
j1dotd = load_plot("plots/j1dotd_data.txt")	
j2dotd = load_plot("plots/j2dotd_data.txt")
j3dotd = load_plot("plots/j3dotd_data.txt")
j4dotd = load_plot("plots/j4dotd_data.txt")
hip_pos = load_plot("plots/hip_pos_data.txt")	#Load data for the current state (hip  + joints)
hip_vel = load_plot("plots/hip_vel_data.txt")
j1pos = load_plot("plots/j1_pos_data.txt")
j2pos = load_plot("plots/j2_pos_data.txt")
j3pos = load_plot("plots/j3_pos_data.txt")
j4pos = load_plot("plots/j4_pos_data.txt")


### PLOT DESIRED AND OBTAINED TRAJECTORY AND THE ERROR BETWEEN THEM

# plt.plot(j1d[0:np.size(j1d)],color="blue", linewidth=1.5, linestyle="-", label="$Desired$")
# plt.plot(j1pos[0:np.size(j1pos)],color="red", linewidth=1.5, linestyle="-", label="$Obtained$")

# #Plot legend
# plt.legend(loc='upper right')
# #Axis format
# ax = plt.gca()  # gca stands for 'get current axis'
# #Axis labels
# ax.set_xlabel('Simulation steps')
# ax.set_ylabel('Angle[rad]')
# plt.savefig("Joint1 trajectories")
# plt.show()


# j1_error = j1d - j1pos

# plt.plot(j1_error[0:np.size(j1d)],color="red", linewidth=1.5, linestyle="-", label="$ErrorJoint1$")
# plt.show()

# #Plot legend
# plt.legend(loc='upper right')
# #Axis format
# ax = plt.gca()  # gca stands for 'get current axis'
# #Axis labels
# ax.set_xlabel('Simulation steps')
# ax.set_ylabel('Error')
# plt.savefig("Joint1 trajectories")
# plt.show()

plt.plot(hip_vel[0:np.size(hip_vel)],color="blue", linewidth=1.5, linestyle="-", label="$Hip_velocity$")   
#Plot legend
plt.legend(loc='upper right')
#Axis format
ax = plt.gca()  # gca stands for 'get current axis'
#Axis labels
ax.set_xlabel('Simlation steps')
ax.set_ylabel('Hip velocity')
plt.savefig("Hip_velocity")
plt.show()


# plt.plot(tau_R[0:np.size(tau_R)],color="blue", linewidth=1.5, linestyle="-", label="$Tau RightS$")   
# plt.plot(tau_L[0:np.size(tau_L)],color="red", linewidth=1.5, linestyle="-", label="$Tau LeftS$")
# #Plot legend
# plt.legend(loc='upper right')
# #Axis format
# ax = plt.gca()  # gca stands for 'get current axis'
# #Axis labels
# ax.set_xlabel('Simlation steps')
# ax.set_ylabel('Tau')
# plt.savefig("Tau")
# plt.show()

# # Note that using plt.subplots below is equivalent to using
# # fig = plt.figure() and then ax = fig.add_subplot(111)
# fig, ax = plt.subplots()
# ax.plot(tau_R[0:430], j4[0:430])
# #ax.plot(tau_L[451:700], j2[451:700])

# ax.set(xlabel='tau', ylabel='desired joint 1 angle',
#        title='theta_1 vs tau_R')
# ax.grid()

# fig.savefig("test.png")
# plt.show()
# ###
# fig, ax = plt.subplots()
# ax.plot(tau_L[431:700], j4[431:700])

# ax.set(xlabel='tau', ylabel='desired joint 1 angle',
#        title='theta_1 vs tau_L')
# ax.grid()

# fig.savefig("test.png")
# plt.show()
