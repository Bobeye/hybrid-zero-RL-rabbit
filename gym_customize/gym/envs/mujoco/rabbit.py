import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class RabbitEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'rabbit_new.xml', 1)
        utils.EzPickle.__init__(self)
        self.count = 0

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
        # print("heigth = {}" .format(height))
        # print("ang = {}" .format(ang))
        # print("velocity = {}" .format(velocity))
        # print("reward = {}" .format(reward))

        done = False
        s = self.state_vector()
        #print(s)
        if not np.isfinite(s).all():
            done = True
            print("done 1")
        if height < 0.3:
            done = True
            print("done 2")
        if abs(ang) > 1.5:
            done = True
            print("done 3")

        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[0:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        init_pos = np.array([-0.2000, 0.7546, 0.1675, 2.4948, 0.4405, 3.1894, 0.2132])
        init_vel = np.array([0.7743, 0.2891, 0.3796, 1.1377, -0.9273, -0.1285, 1.6298])
        qpos = init_pos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = init_vel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    #============= This are new added methods. For using them you have to declare them in "mujoco_env.py" and "core.py"
    #============= Otherwise you have to call them from the main .py file by using "env.unwrapped.new_method_name()"
    def get_state(self):
        return self.sim.data.qpos, self.sim.data.qvel

    def get_sensor_data(self,sensor_name):
        return self.sim.data.get_sensor(sensor_name)    

    def assign_desired_vel(self,desired_vel):
        self.desired_vel = desired_vel
    #=====================================================================================================================
