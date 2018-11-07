from rabbit_tools import trajectory

"""Customized Reward Func"""
def get_reward(observation, desired_vel, velocity, mode="default"):
	if mode == "default":	
		if abs(velocity-desired_vel) <= 0.1:
			if velocity <= desired_vel:
				velocity_reward = (velocity / desired_vel)**2
			else:
				velocity_reward = (desired_vel / velocity)**2
		else:
			velocity_reward = 0

		# lf_clr = trajectory.left_foot_height(observation[0:7])
		# rf_clr = trajectory.right_foot_height(observation[0:7])
		# clearance_reward = 0.02*(lf_clr-rf_clr)**2
		clearance_reward = 0.

		ang_reward = 0.
		# ang = observation[2]
		# if abs(ang - 0.3) < 0.2:
		# 	ang_reward = 0.1
		# else:
		# 	ang_reward = 0
		# if ang < 0:
		# 	ang_reward = 0.1*ang
		# else:
		# 	ang_reward = 0
		
		reward = 0.1 * (10 * velocity_reward + clearance_reward + ang_reward)
	else:
		raise NotImplementedError

	return reward

