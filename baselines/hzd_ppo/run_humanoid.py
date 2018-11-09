#!/usr/bin/env python3
import os
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger

import gym
from gym import wrappers

def train(num_timesteps, seed, save_path, model_path=None):
    env_id = 'Humanoid-v2'
    from baselines.hzd_ppo import mlp_policy, pposgd_humanoid
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = make_mujoco_env(env_id, seed)
    # initalize env for evaluation and rendering
    eval_env = gym.make(env_id)
    eval_env.reset()
    eval_env.render()
    video_path = save_path + "video/"
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    eval_env = wrappers.Monitor(eval_env, video_path, force=True, video_callable=lambda episode_id: episode_id%1==0)

    # parameters below were the best found in a simple random search
    # these are good enough to make humanoid walk, but whether those are
    # an absolute best or not is not certain
    env = RewScale(env, 0.1)
    pi = pposgd_humanoid.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10,
            optim_stepsize=3e-4,
            optim_batchsize=64,
            gamma=0.99,
            lam=0.95,
            schedule='linear',
            save_path=save_path,
            eval_env=eval_env
        )
    env.close()
    if model_path:
        U.save_state(model_path)

    return pi

class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale
    def reward(self, r):
        return r * self.scale

def main():
    
    parser = mujoco_arg_parser()
    # parser.add_argument('--model-path', default=os.path.join(logger.get_dir(), 'humanoid_policy'))
    parser.set_defaults(num_timesteps=int(1e8))

    args = parser.parse_args()
    logger.configure(args.save_path)
    if not args.play:
        # train the model
        train(num_timesteps=args.num_timesteps, seed=args.seed, save_path=args.save_path, model_path=None)
    else:
        # construct the model object, load pre-trained model and render
        pi = train(num_timesteps=1, seed=args.seed)
        U.load_state(args.model_path)
        env = make_mujoco_env('Humanoid-v2', seed=0)

        ob = env.reset()
        while True:
            action = pi.act(stochastic=False, ob=ob)[0]
            ob, _, done, _ =  env.step(action)
            env.render()
            if done:
                ob = env.reset()




if __name__ == '__main__':
    main()
