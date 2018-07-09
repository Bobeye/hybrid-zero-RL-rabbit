import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class RabbitEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.freeze = 0
        mujoco_env.MujocoEnv.__init__(self, 'rabbit_new.xml', 1)
        utils.EzPickle.__init__(self)
        self.count = 0

    def step(self, a):
             
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)

        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        # velocity = (posafter - posbefore) / self.dt
        velocity = self.sim.data.qvel[0]    #hip velocity
        # print("vel_1{}",velocity)
        # print("vel_2{}",velocity2)
        if velocity < 0:
            velocity_reward = 0.
        elif velocity > self.desired_vel:
            velocity_reward = 1        
        else:
            velocity_reward = 10*((self.desired_vel - velocity) / self.desired_vel)
            
        action_reward = -1e-2 * np.sum(a**2)
        height_reward = 1*height
        displacement_reward  = 20*(posafter)

        # print("vel_reward = {}" .format(velocity_reward))
        # print("height_reward = {}" .format(height_reward))
        # print("displ_reward = {}" .format(displacement_reward))
        # print("vel_reward = {}" .format(velocity_reward))

        reward = alive_bonus + 2*velocity_reward + 0.1*height_reward + action_reward + displacement_reward
        print (alive_bonus , velocity_reward , height_reward , action_reward , displacement_reward)

        #ADD  COEFFICIENTS to set weigth importance to each reward

        # print("heigth = {}" .format(height))
        # print("ang = {}" .format(ang))
        # print("velocity = {}" .format(velocity))
        # print("reward = {}" .format(reward))
        done = False
        s = self.state_vector()
        if not np.isfinite(s).all():
            done = True
            reward -= 10
            print("done 1")
        if height < 0.6:
            done = True
            reward -= 20
            print("done 2")
        #if ang < 0 or ang > 1:
        if abs(ang) > 1:    
            done = True
            reward -= 10
            print("done 3")

        # TODO: detecting stuck:
        if abs(velocity) <= 1e-3:
            self.freeze += 1
        else:
            self.freeze = 0
        
        if self.freeze > 100:
            done = True
            print("done 4")
            self.freeze = 0


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
        return self.sim.data.qpos, self.sim.data.qvel

    def get_sensor_data(self,sensor_name):
        return self.sim.data.get_sensor(sensor_name)    

    def assign_desired_vel(self,desired_vel):
        self.desired_vel = desired_vel
    #=====================================================================================================================