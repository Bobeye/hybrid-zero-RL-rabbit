import gym
from gym import spaces
 


class HZDENV(object):

    def __init__(self, env_id):
        self.env = gym.make(env_id)          #HOW TO MAKE THE ARGUMENT SOMETHING THAT CAN BE CALLED WITH THE NAME OF THE ENVIRONMENT DEFINED IN run_hzd.py
        print("ENVIRONMENT = ", env_id)
        # n_actions = self.env.action_space.shape[0]
        # self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')


    # def reset(self):
    #     self.env.reset()
    
    # def step(self, a):
    #     obs, rwd, done, info = self.env.step(a)
    #     R = obs[0]**2

    #     return obs, R, done, info



