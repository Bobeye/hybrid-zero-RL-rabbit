from rabbit_tools import trajectory

"""Customized Reward Func"""
def get_reward(observation, desired_vel, velocity, mode="default"):
	# velocity = observation[7]
	angle = observation[2]
	if mode == "default":	
		if abs(velocity-desired_vel) <= 1:
			if velocity <= desired_vel:
				velocity_reward = (velocity / desired_vel)**2
			else:
				velocity_reward = (desired_vel / velocity)**2
		else:
			velocity_reward = 0

		lf_clr = trajectory.left_foot_height(observation[0:7])
		rf_clr = trajectory.right_foot_height(observation[0:7])
		clearance_reward = 0.5*(lf_clr-rf_clr)**2
		# clearance_reward = abs(lf_clr-rf_clr)
		#step_reward = 0.5*(observation[3]-observation[5])**2

		reward = 0.1 * (10 * velocity_reward + 1*clearance_reward# + 1*step_reward #+ 0.25*angle 

		#reward = 0.1 * (10 * velocity_reward + 2*clearance_reward + 0.25*angle)   Did not work with abs(velocity-desired_vel) <= 1:  

		
	else:
		raise NotImplementedError

	return reward

