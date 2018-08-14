#	Evaluate rabbit walk
#	by Bowen, June, 2018
#
################################
from __future__ import division

from es.nn import NeuralNetwork 
from es.oes import OpenES
from es.cmaes import CMAES
import hzd.trajectory as trajectory
import hzd.params as params

import gym
from joblib import Parallel, delayed
import json
import os
import numpy as np

from rabbit_nn_estrain import settings, make_env


def load_model(filename):
    with open(filename) as f:    
    	data = json.load(f)
    print('loading file %s' % (filename))
    model_params = np.array(data[0]) # assuming other stuff is in data
    return model_params	

if __name__ == "__main__":
    policy_path = "log/"+"/nn_es_policy/50.json"

    render_mode = True
    eval_mode = False	

    desired_velocity = 1

    model = NeuralNetwork(input_dim=settings.state_size,
                            output_dim=settings.action_size,
                            units=settings.nn_units,
                            activations=settings.nn_activations)
	#Obtain theta from .json file's data
    model_params = load_model(policy_path)
    model.set_weights(model_params)
    env = make_env(settings.env_name, render_mode=render_mode, desired_velocity = desired_velocity)

    total_reward_list = []
    velocity_list = []

    for episode in range(1):	#for episode in range(settings.num_episode):
        last_speed = None
        state = env.reset()
        # for k in range(500):
        # 	env.render()
        if state is None:
            state = np.zeros(settings.state_size)
        total_reward = 0
        for t in range(settings.max_episode_length*2):
            #timesteps += 1

            action = model.predict(state) * settings.action_max
            # print(action)
            observation, reward, done, info = env.step(action)
            state = observation
            velocity_list += [state[7]]
            total_reward += reward

            if eval_mode:
                save_plot_2(state)

            if render_mode:
                env.render()
            # if done:
            # 	break
        total_reward_list += [np.array([total_reward]).flatten()]
    state = env.reset()
    total_rewards = np.array(total_reward_list)

    if settings.batch_mode == "min":
        rewards = np.min(total_rewards)
    else:
        rewards = np.mean(total_rewards)
    print (total_reward_list, rewards)
    velocity_list_new = velocity_list[500:3000]
    vel_mean = np.sum(velocity_list_new)/np.size(velocity_list_new)
    print(vel_mean, np.amin(velocity_list), np.amax(velocity_list))
