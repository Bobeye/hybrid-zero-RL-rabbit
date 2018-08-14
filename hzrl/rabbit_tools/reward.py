

"""Customized Reward Func"""

def get_reward(velocity=0., desired_vel=0, mode="vel"):
	if mode == "vel":	# encourage vel only
		threshold = 0.1	
		if abs(velocity-desired_vel) <= threshold:
			if velocity <= desired_vel:
				velocity_reward = (velocity / desired_vel)**2
			else:
				velocity_reward = (desired_vel / velocity)**2
		else:
			velocity_reward = 0
		reward = velocity_reward
	return reward

