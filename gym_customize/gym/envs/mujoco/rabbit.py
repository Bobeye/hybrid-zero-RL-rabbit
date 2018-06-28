import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class RabbitEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, desired_vel=10.):
        self.desired_vel = desired_vel
        mujoco_env.MujocoEnv.__init__(self, 'rabbit_new.xml', 1)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        velocity = (posafter - posbefore) / self.dt
        if velocity < 0:
            velocity_reward = 0.
        elif velocity > self.desired_vel:
            velocity_reward = 1
        else:
            velocity_reward = (self.desired_vel - velocity) / self.desired_vel
        action_reward = -1e-3 * np.sum(a**2)
        height_reward = height
        reward = alive_bonus + velocity_reward + height_reward + action_reward

        done = False
        s = self.state_vector()
        if not np.isfinite(s).all() or not (np.abs(s[2:]) < 100).all():
            done = True
        if height < 0.3:
            done = True
        if abs(ang%(2*np.pi)) < 1.:
            done = True 

        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[0:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        position = np.array([[-0.2000, 0.7546, 0.1675, 2.4948, 0.4405, 3.1894, 0.2132],
                            [-0.1484, 0.7460, 0.1734, 2.4568, 0.6307, 2.9492, 0.6271],
                            [-0.1299, 0.7518, 0.1767, 2.4873, 0.6133, 2.9434, 0.6740],
                            [-0.1121, 0.7567, 0.1794, 2.5177, 0.5954, 2.9297, 0.7256]])
        velocity = np.array([[0.7743, 0.2891, 0.3796, 1.1377, -0.9273, -0.1285, 1.6298],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]])
        qpos = position[0,:] + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = velocity[0,:] + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20



    def get_state(self):
        return self.sim.data.qpos, self.sim.data.qvel
