



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
	conditions_dim = 3
	theta_dim = 20
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
	plot_mode = False
	sigmoid_slope = 1
	init_vel = 0.7743
	size_vel_buffer = 200
	def __init__(self):
		pass
settings = Settings()