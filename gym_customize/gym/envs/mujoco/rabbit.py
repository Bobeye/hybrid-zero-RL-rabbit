import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class RabbitEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'rabbit_new.xml', 1)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):

        #added by GC
        #print("qpos = {}".format(self.sim.data.qpos))
        #print(type(self.sim.data.qpos))
        #> <class 'numpy.ndarray'>
        #print("qvel = {}".format(self.sim.data.qvel))
        #print(type(self.sim.data.qvel))
        #> <class 'numpy.ndarray'>
        #added by GC

        return np.concatenate([
            self.sim.data.qpos.flat[0:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def get_state(self):
        #print("Printing initial state....")
        #s = self.state_vector()
        #print(s)
        #init_pos = self.sim.data.qpos
        #init_vel = self.sim.data.qvel
        #return init_pos, init_vel
        #return self.init_qpos, self.init_qvel
        return self.sim.data.qpos, self.sim.data.qvel

    #def get_state

    #def set_state(self,pos,vel):
    #    pass

# class joint:
#     def __init__(self, joint_name, Kp, Ki,Kd):
#         self.name = joint_name   
#         self.eT0 = 0
#         self.iT0 = 0
#         self.Kp = Kp
#         self.Kd = Ki
#         self.Ki = Kd


#     def pid_update(self, Kp, Ki, Kd, desired_value, actual_value, dt):
#         self.actual_value = RabbitEnv.sim.data.qpos[1]
#         self.error = desired_value - actual_value
#         integral = integral + (error*dt)
#         derivative = (error - error_prior)/dt
#         output = Kp*error + Ki*integral + Kd*derivative
#         error_prior = error

#         print("this is working")



