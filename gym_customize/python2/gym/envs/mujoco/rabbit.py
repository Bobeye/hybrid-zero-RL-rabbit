import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class RabbitEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.freeze = 0
        mujoco_env.MujocoEnv.__init__(self, 'rabbit.xml', 1)
        utils.EzPickle.__init__(self)
        self.count = 0

    def _step(self, a):
        scale = 0.1     
        posbefore = self.model.data.qpos[0]
        self.do_simulation(a, self.frame_skip)

        posafter, height, ang = self.model.data.qpos[0:3]
        alive_bonus = 1.0
        # velocity = (posafter - posbefore) / self.dt
        velocity = self.model.data.qvel[0]    #hip velocity
        w = self.model.data.qvel[2]           # hip angular velocity

        done = False
        s = self.state_vector()
        if not np.isfinite(s).all():
            done = True
            #print("done 1")
        if height < 0.6 or height > 1:
            done = True
            #print("done 2")
        # if abs(ang) > 0.5: 
        # if ang > 1 or ang < -0.5:   
        if ang < -0.2 or ang > 0.5:
            done = True
            #print("done 3")

        # TODO: detecting stuck:
        if abs(velocity) <= 1e-3:
            self.freeze += 1
        else:
            self.freeze = 0     
        if self.freeze > 100:
            done = True
            #print("done 4")
            self.freeze = 0

        ob = self._get_obs()
        reward_params = [alive_bonus, posafter, posbefore, velocity, a, w]
        return ob, reward_params, done, {}



    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[0:],
            np.clip(self.model.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        init_pos = np.array([-0.2000, 0.7546, 0.1675, 2.4948, 0.4405, 3.1894, 0.2132])+self.np_random.uniform(low=-.001, high=.001, size=self.model.nq)
        init_vel = np.array([0.7743, 0.2891, 0.3796, 1.1377, -0.9273, -0.1285, 1.6298])+self.np_random.uniform(low=-.001, high=.001, size=self.model.nv)
        #qpos = init_pos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        #qvel = init_vel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qpos = init_pos
        qvel = init_vel
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
        return self.model.data.qpos, self.model.data.qvel

    def get_sensor_data(self,sensor_name):
        return self.model.data.get_sensor(sensor_name)    

    def assign_desired_vel(self,desired_vel):
        self.desired_vel = desired_vel
    #=====================================================================================================================