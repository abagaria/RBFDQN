# This will be the only experiment

import sys

import gym
from tqdm import trange
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np

from RBFDQN.RBFDQN import Q_class
from RBFDQN import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="LunarLanderContinuous-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0,
                        type=int)  # Sets Gym, PyTorch and Numpy seeds

    parser.add_argument("--experiment_name", type=str, help="Experiment Name", required=True)
    parser.add_argument("--run_title", type=str, required=True) # This is the subdir that we'll be saving in.
    parser.add_argument("--render", action="store_true", default=False)

    args = parser.parse_args()

    full_experiment_name = os.path.join(args.experiment_name, args.run_title)
    utils.create_log_dir(os.path.join(full_experiment_name, "scores"))

    env_dic = {}
    env_dic['Pendulum-v0'] = 0
    env_dic['LunarLanderContinuous-v2'] = 10
    env_dic['BipedalWalker-v2'] = 20
    env_dic['Ant-v1'] = 30
    env_dic['HalfCheetah-v1'] = 40
    env_dic['Hopper-v1'] = 50
    env_dic['InvertedDoublePendulum-v1'] = 60
    env_dic['InvertedPendulum-v1'] = 70
    env_dic['Reacher-v1'] = 80
    if args.env not in env_dic.keys():
        print(
            "environment not recognized ... use one of the following environments"
        )
        print(env_dic.keys())
        assert False
    hyper_parameter_name = str(env_dic[args.env])
    alg = 'rbf'
    params = utils.get_hyper_parameters(hyper_parameter_name, alg)
    params['hyper_parameters_name'] = hyper_parameter_name
    env = gym.make(params['env_name'])
    params['env'] = env
    params['seed_number'] = args.seed
    utils.set_random_seed(params)
    s0 = env.reset()
    utils.action_checker(env)
    Q_object = Q_class(params,
                       env,
                       state_size=len(s0),
                       action_size=len(env.action_space.low))
    G_li = []
    for episode in range(params['max_episode']):
        #train policy with exploration
        s, done = env.reset(), False
        while done == False:
            a = Q_object.e_greedy_policy(s, episode + 1, 'train')
            sp, r, done, _ = env.step(np.array(a))
            Q_object.buffer_object.append(s, a, r, done, sp)
            s = sp

        #now update the Q network
        for i in trange(params['updates_per_episode'],
                        file=sys.stdout,
                        desc='training'):
            Q_object.update()
        #test the learned policy, without performing any exploration
        s, t, G, done = env.reset(), 0, 0, False
        while done == False:
            a = Q_object.e_greedy_policy(s, episode + 1, 'test')
            sp, r, done, _ = env.step(np.array(a))
            if episode % 10 == 0 and args.render:
                env.render()
            s, t, G = sp, t + 1, G + r
        print("in episode {} we collected return {} in {} timesteps".format(
            episode, G, t))
        G_li.append(G)
        if episode % 10 == 0 and episode > 0:
            utils.save(G_li, params, alg)

        utils.save_scores(G_li, full_experiment_name, args.seed)

    utils.save(G_li, params, alg)
    utils.save_scores(G_li, full_experiment_name, args.seed)
