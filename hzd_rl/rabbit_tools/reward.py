import trajectory

"""Customized Reward Func"""
def get_reward(observation, desired_vel, velocity, mode="default"):
	if mode == "default":	
		if abs(velocity-desired_vel) <= 0.05:
			if velocity <= desired_vel:
				velocity_reward = (velocity / desired_vel)**2
			else:
				velocity_reward = (desired_vel / velocity)**2
		else:
			velocity_reward = 0

		lf_clr = trajectory.left_foot_height(observation[0:7])
		rf_clr = trajectory.right_foot_height(observation[0:7])
		clearance_reward = 0.5*(lf_clr-rf_clr)**2
		
		reward = 0.1 * (10 * velocity_reward + clearance_reward)
	else:
		raise NotImplementedError

	return reward

