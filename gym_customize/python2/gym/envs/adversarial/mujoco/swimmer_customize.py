import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import random

class SwimmerCustomizeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.direction = np.array(random.sample([0, 1], 1))
        mujoco_env.MujocoEnv.__init__(self, 'swimmer.xml', 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        ctrl_cost_coeff = 0.0001
        yposbefore = self.model.data.qpos[1, 0]
        xposbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.model.data.qpos[0, 0]
        yposafter = self.model.data.qpos[1, 0]
        if self.direction == 0:
            reward_fwd = (xposafter - xposbefore) / self.dt
        else:
            reward_fwd = (yposafter - yposbefore) / self.dt 

        # print np.arctan2(xposafter-xposbefore, yposafter-yposbefore)

        # else:
        #     reward_fwd = (xposbefore - xposafter) / self.dt
        # reward_fwd = (xposbefore - xposafter) / self.dt
        # reward_fwd = (yposafter - yposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat, self.direction])

    def reset_model(self):
        self.direction = np.array(random.sample([0, 1], 1))
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-1., high=1., size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-1., high=1., size=self.model.nv)
        )
        return self._get_obs()
