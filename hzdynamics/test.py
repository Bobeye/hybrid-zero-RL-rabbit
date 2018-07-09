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
	upplim_jleg = 90*(np.pi/180)
	lowlim_jleg = 0*(np.pi/180)
	def __init__(self):
		pass
settings = Settings()


def bound_theta(theta):		#Add offset and restrict to range corresponding to each joint
	theta_thighR = theta

	theta_thighR = (((settings.upplim_jthigh - settings.lowlim_jthigh)/2)*theta_thighR) + ((settings.upplim_jthigh + settings.lowlim_jthigh)/2)
	return theta_thighR

 # SIGMOID``
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
	t = np.linspace(-5.0, 5.0, num=100)
	theta = np.tanh(t)
	theta_new = bound_theta(theta)

	plt.plot(theta[0:450],color="blue", linewidth=1.5, linestyle="-", label="$\Theta_1$")   
	plt.plot(theta_new[0:450],color="red", linewidth=1.5, linestyle="-", label="$\Theta_2$")
	#Plot legend
	plt.legend(loc='upper right')
	#Axis format
	ax = plt.gca()  # gca stands for 'get current axis'
	#Axis labels
	ax.set_xlabel('Steps')
	ax.set_ylabel('Joint angle [rad]')
	plt.show()