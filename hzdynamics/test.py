import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import numpy as np


"""Hyperparameters"""
class Settings():
	env_name="Rabbit-v0"
	txt_log = "log/" + "/train.txt"
	policy_path = "log/"+ "/policy/"
	backend = "multiprocessing"
	n_jobs = 8
	frequency = 20.
	total_threshold = 1e8
	num_episode = 5
	max_episode_length=1000
	batch_mode="mean"
	state_size = 14
	action_size = 4
	action_min = -4.
	action_max = 4.
	control_kp = 80.
	control_kd = 2.
	desired_v_low = 0.6
	desired_v_up = 1
	conditions_dim = 2
	theta_dim = 24
	nn_units=[16,16]
	nn_activations=["relu", "relu", "tanh"]
	population = 24
	sigma_init=0.1
	sigma_decay=0.9999 
	sigma_limit=1e-4
	learning_rate=0.01
	learning_rate_decay=0.9999 
	learning_rate_limit=1e-6
	aux = 0
	upplim_jthigh = 250*(np.pi/180)
	lowlim_jthigh = 90*(np.pi/180)
	upplim_jleg = 150*(np.pi/180)
	lowlim_jleg = 0*(np.pi/180)
	sigmoid_slope = 1
	init_vel = 0.3769
	def __init__(self):
		pass
settings = Settings()


def bound_theta(theta):		#Add offset and restrict to range corresponding to each joint
	theta_thighR = theta

	theta_thighR = (((settings.upplim_jthigh - settings.lowlim_jthigh)/2)*theta_thighR) + ((settings.upplim_jthigh + settings.lowlim_jthigh)/2)
	return theta_thighR

def bound_theta_sigmoid(theta):		#Add offset and restrict to range corresponding to each joint. The input is assumed to be bounded as tanh, with range -1,1
	theta = sigmoid(theta)
	theta_thighR = theta
	theta_legR = theta
	theta_thighR = ((settings.upplim_jthigh - settings.lowlim_jthigh)*theta_thighR) + settings.lowlim_jthigh
	theta_legR = settings.upplim_jleg*(theta_legR)
	return theta_legR

 # SIGMOID``
def sigmoid(x):
	return 1 / (1 + np.exp(-settings.sigmoid_slope*x))

if __name__ == "__main__":
	steps = 100
	t = np.linspace(-5.0, 5.0, num=steps)
	#theta = np.tanh(t)
	theta = t
	theta_old = sigmoid(t)
	
	theta_new = bound_theta_sigmoid(theta)

	plt.plot(theta_old[0:steps],color="blue", linewidth=1.5, linestyle="-", label="$\Theta_old$")   
	plt.plot(theta_new[0:steps],color="red", linewidth=1.5, linestyle="-", label="$\Theta_new$")
	#Plot legend
	plt.legend(loc='upper right')
	#Axis format
	ax = plt.gca()  # gca stands for 'get current axis'
	#Axis labels
	ax.set_xlabel('Steps')
	ax.set_ylabel('Theta')
	plt.show()