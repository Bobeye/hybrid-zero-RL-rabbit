



"""Customized Reward Func"""
def get_reward(reward_params, 
			   reward_tau, 
			   desired_vel, 
			   current_vel_av, 
			   sum_error, 
			   mode="linear1"):
	alive_bonus, posafter, posbefore, velocity, a, w = reward_params[0], reward_params[1], reward_params[2], reward_params[3], reward_params[4], reward_params[5]
	# print (desired_vel)
	# print (velocity)
	if mode == "linear_bowen":	#Works better so far, but can not follow desired speed
		scale = 0.1
		# if velocity < 0:
		# 	velocity_reward = 0
		
		if abs(velocity-desired_vel) <= 0.1:
			if velocity <= desired_vel:
				velocity_reward = (velocity / desired_vel)**2
			else:
				velocity_reward = (desired_vel / velocity)**2
		else:
			velocity_reward = 0




		# elif abs(velocity-desired_vel) <= 0.1:
		# 	if velocity < desired_vel:
		# 		velocity_reward = (desirevelocity)/0.1

		# elif velocity <= desired_vel-0.1:
		# 	velocity_reward = (velocity/desired_vel)
		# elif velocity >= desired_vel+0.1:
		# 	velocity_reward = 0. #(desired_vel/velocity)
		# else:
		# 	velocity_reward = 1.

		action_reward = -1e-2 * np.sum(a**2)
		displacement_reward  = posafter
		reward = 10*velocity_reward # + 1*action_reward + 1*displacement_reward # + 0.5*sum_error
		reward = scale*reward	


	if mode == "linear1":	#Works better so far, but can not follow desired speed
		scale = 0.1
		if velocity < 0:
			velocity_reward = 0
		elif velocity <= desired_vel:
			velocity_reward = (velocity/desired_vel)**2
		else:
			velocity_reward = (desired_vel/velocity)**2

		action_reward = -1e-2 * np.sum(a**2)
		displacement_reward  = posafter
		reward = alive_bonus + 10*velocity_reward + 1*action_reward + 1*displacement_reward # + 0.5*sum_error
		reward = scale*reward	

	if mode == "linear2":
		scale = 0.1
		if current_vel_av < 0:
			velocity_reward = 0
		elif current_vel_av <= desired_vel:
			velocity_reward = (current_vel_av/desired_vel)**2
		else:
			velocity_reward = (desired_vel/current_vel_av)**2

		action_reward = -1e-2 * np.sum(a**2)
		displacement_reward  = posafter
		reward = alive_bonus + 10*velocity_reward + 1*action_reward + 1*displacement_reward
		reward = scale*reward	

		action_reward = -1e-2 * np.sum(a**2)
		displacement_reward  = posafter
		reward = alive_bonus + 1*velocity_reward + 1*action_reward + 1*displacement_reward
		reward = scale*reward

	if mode == "angular":
		scale = 0.5
		# print(velocity)
		if velocity < 0:
			velocity_reward = velocity
			# velocity_reward = velocity
		elif velocity <= desired_vel:
			velocity_reward = (velocity)/ desired_vel    
			#print(velocity_reward) 
		else:
			velocity_reward = desired_vel/velocity
			# print(velocity_reward)

		angular_reward = w
		action_reward = -1e-2 * np.sum(a**2)
		displacement_reward  = posafter
		reward = alive_bonus + 1*velocity_reward + 1*action_reward + 1*displacement_reward + angular_reward
		reward = scale*reward		

	if mode == "quadratic":
		#print("yes")
		scale = 0.5
		if velocity < 0:
			velocity_reward = 0
			# velocity_reward = velocity
		elif velocity <= desired_vel:
			velocity_reward = (desired_vel-velocity)**2    
			#print(velocity_reward) 
		else:
			velocity_reward = - (desired_vel-velocity)**2
			# print(velocity_reward)	
		action_reward = -1e-2 * np.sum(a**2)
		displacement_reward  = posafter
		reward = alive_bonus + 1*velocity_reward + 1*action_reward + 1*displacement_reward
		reward = scale*reward


	if mode == "tau": # Consider phase variable Tau in the reward function and add penalization when speed is higher than the desired one
		#Also use angular velocity to try to impose torso position
		scale = 0.1
		if velocity < 0:
			velocity_reward = 0
			# velocity_reward = -abs(velocity)
		elif velocity <= desired_vel:
			velocity_reward = (velocity)/ desired_vel    
			#print(velocity_reward) 
		else:
			velocity_reward = - abs(velocity - desired_vel)
			# print(velocity_reward)

		angular_reward = w	
		action_reward = -1e-2 * np.sum(a**2)
		displacement_reward  = posafter
		reward = alive_bonus + 10*velocity_reward + 1*action_reward + 2*displacement_reward + 1*reward_tau + 1*angular_reward
		reward = scale*reward
		# print("reward = {}" .format(reward))
	
	# print(reward)
	# print (alive_bonus , velocity_reward , height_reward , action_reward , displacement_reward)
	return reward