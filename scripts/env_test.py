import IPython
import ipdb
import numpy as np
import random
import argparse
import copy

from marl_envs.particle_envs.make_env import make_env
from marl_envs.my_env.capture_target import CaptureTarget as CT

from marl_envs.my_env.small_box_pushing import SmallBoxPushing as SBP
from marl_envs.my_env.cmotp_wrapper import CMOTP_WRAPPER as CMOTP

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', action='store', type=str, default='pomdp_simple_coop_tag_v0')
    parser.add_argument('--obsr', action='store', type=float, default=0.4)
    parser.add_argument('--nag', action='store', type=int, default=2)
    parser.add_argument('--seed', action='store', type=int, default=1)
    parser.add_argument('--grid_dim', action='store', type=int, nargs=2, default=[8,8])
    parser.add_argument('-r', '--render', action='store_true')
    params = parser.parse_args()

    #main(params.env_name, obs_r=params.obsr)

    #env = make_env(params.env_name, discrete_action_input=True, benchmark=False, obs_resolution=3, flick_p=0.0, enable_boundary=False, obs_r=params.obsr, discrete_mul=2) 
    #env = CT(1, 2, (12,12))
    #env = BP(tuple(params.grid_dim), terminate_step=300, small_box_only=True, big_box_only=False, terminal_reward_only=True, big_box_reward=1000, small_box_reward=100)
    #env = SBP(tuple(params.grid_dim), n_agent=params.nag, terminate_step=300, terminal_reward_only=True, small_box_reward=100)
    env = CMOTP(version=1, local_obs_shape=[5,5])

    #env.seed(11)
    np.random.seed(params.seed)
    random.seed(params.seed)
    dists = []

    R = 0
    for i in range(1000):
        t = [False, False]
        obs = env.reset()
        env.render()
        step=0
        while not all(t) and step<5000:
            actions = [env.action_space_sample(a) for a in range(env.n_agent)]
            a, obs, r, t, v, _  = env.step(actions)
            env.render()
            if r[0] != 0:
                ipdb.set_trace()
            R += sum(r)
            step += 1
            print(step)
 
