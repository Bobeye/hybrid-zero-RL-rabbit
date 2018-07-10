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

tau_R = load_plot("plots/tauR_data.txt")
tau_L = load_plot("plots/tauL_data.txt")
j1 = load_plot("plots/j1_data.txt")
j2 = load_plot("plots/j2_data.txt")
j3 = load_plot("plots/j3_data.txt")
j4 = load_plot("plots/j4_data.txt")
j1d = load_plot("plots/j1d_data.txt")
j2d = load_plot("plots/j2d_data.txt")
j3d = load_plot("plots/j3d_data.txt")
j4d = load_plot("plots/j4d_data.txt")


plt.plot(tau_R[0:np.size(tau_R)],color="blue", linewidth=1.5, linestyle="-", label="$Tau RightS$")   
plt.plot(tau_L[0:np.size(tau_L)],color="red", linewidth=1.5, linestyle="-", label="$Tau LeftS$")
#Plot legend
plt.legend(loc='upper right')
#Axis format
ax = plt.gca()  # gca stands for 'get current axis'
#Axis labels
ax.set_xlabel('Simlation steps')
ax.set_ylabel('Tau')
plt.savefig("Tau")
plt.show()

# Note that using plt.subplots below is equivalent to using
# fig = plt.figure() and then ax = fig.add_subplot(111)
fig, ax = plt.subplots()
ax.plot(tau_R[0:430], j4[0:430])
#ax.plot(tau_L[451:700], j2[451:700])

ax.set(xlabel='tau', ylabel='desired joint 1 angle',
       title='theta_1 vs tau_R')
ax.grid()

fig.savefig("test.png")
plt.show()
###
fig, ax = plt.subplots()
ax.plot(tau_L[431:700], j4[431:700])

ax.set(xlabel='tau', ylabel='desired joint 1 angle',
       title='theta_1 vs tau_L')
ax.grid()

fig.savefig("test.png")
plt.show()
