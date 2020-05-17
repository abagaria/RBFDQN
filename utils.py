import numpy
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym
import dmc2gym


def create_env(domain_name, task_name, seed, dmc):
	if dmc:
		env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=seed)
	else:
		env = gym.make(domain_name)
		env.seed(seed)
	return env


def action_checker(env):
	for l,h in zip(env.action_space.low,env.action_space.high):
		if l!=-h:
			print("asymetric action space")
			print("don't know how to deal with it")
			assert False
	if numpy.max(env.action_space.low)!=numpy.min(env.action_space.low):
		print("different action range per dimension")
		assert False
	if numpy.max(env.action_space.high)!=numpy.min(env.action_space.high):
		print("different action range per dimension")
		assert False


def get_hyper_parameters(name,alg):
	meta_params={}
	with open(alg+"_hyper_parameters/"+name+".hyper") as f:
		lines = [line.rstrip('\n') for line in f]
		for l in lines:
			parameter_name,parameter_value,parameter_type=(l.split(','))
			if parameter_type=='string':
				meta_params[parameter_name]=str(parameter_value)
			elif parameter_type=='integer':
				meta_params[parameter_name]=int(parameter_value)
			elif parameter_type=='float':
				meta_params[parameter_name]=float(parameter_value)
			else:
				print("unknown parameter type ... aborting")
				print(l)
				sys.exit(1)
	return meta_params

def save(li_returns,params,alg):
	directory=alg+"_results/"+params['hyper_parameters_name']+'/'
	if not os.path.exists(directory):
		os.makedirs(directory)
	numpy.savetxt(directory+str(params['seed_number'])+".txt",li_returns)

def set_random_seed(meta_params):
	seed_number=meta_params['seed_number']
	import numpy
	numpy.random.seed(seed_number)
	import random
	random.seed(seed_number)

	import tensorflow as tf
	session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
	from keras import backend as K
	tf.set_random_seed(seed_number)
	sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

	K.set_session(sess)
	meta_params['env'].seed(seed_number)
	#meta_params['env'].action_space.seed(seed_number)
	meta_params['env'].reset()
	#print("set the random seed to be able to reproduce the result ...")


def get_scores_from_file(text_file_name):
	with open(text_file_name, "r") as f:
		data = [float(line) for line in f.readlines()]
	return data


def moving_average(a, n=25):
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n-1:] / n


def plot_one_class_classifier(X, clf, episode, env_name):
	if X.shape[1] == 3:
		plot_3dim_one_class_classifier(X, clf, episode, env_name)
	else:
		plot_2dim_one_class_classifier(X, clf, episode, env_name)


def plot_3dim_one_class_classifier(X, clf, episode, env_name):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	X0, X1, X2 = X[:, 0], X[:, 1], X[:, 2]
	predictions = clf.predict(X)
	p = ax.scatter(X0, X1, X2, c=predictions)
	fig.colorbar(p)
	plt.savefig(f"{env_name}_ocsvm_episode_{episode}.png")
	plt.close()


# def plot_2dim_one_class_classifier(X, clf, episode, env_name):
# 	fig = plt.figure()
# 	ax = fig.add_subplot(111)
# 	X0, X1 = X[:, 0], X[:, 1]
# 	predictions = clf.predict(X)
# 	p = ax.scatter(X0, X1, c=predictions)
# 	fig.colorbar(p)
# 	plt.savefig(f"{env_name}_family_of_ocsvms_episode_{episode}.png")
# 	plt.close()


def plot_2dim_one_class_classifier(X, clf, episode, env_name, dir_path="mountain-car-ocsvm-plots", chunk_size=1000):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	# Chunk up the inputs so as to conserve GPU memory
	num_chunks = int(np.ceil(X.shape[0] / chunk_size))
	state_chunks = np.array_split(X, num_chunks, axis=0)
	inlier_probabilities = np.zeros((X.shape[0],))

	current_idx = 0

	for chunk_number, state_chunk in tqdm(enumerate(state_chunks), desc="Making OCSVM plot"):  # type: (int, np.ndarray)
		chunk_probabilities = clf.predict(state_chunk)
		current_chunk_size = len(state_chunk)
		inlier_probabilities[current_idx:current_idx + current_chunk_size] = chunk_probabilities
		current_idx += current_chunk_size

	p = ax.scatter(X[:, 0], X[:, 1], c=inlier_probabilities)
	fig.colorbar(p)
	plt.savefig(f"{dir_path}/{env_name}_family_of_ocsvms_episode_{episode}.png")
	plt.close()


def make_chunked_value_function_plot(q_agent, episode, seed, dir_path, chunk_size=1000):
	states = np.array([exp["sp"] for exp in q_agent.buffer_object.storage])

	# Chunk up the inputs so as to conserve GPU memory
	num_chunks = int(np.ceil(states.shape[0] / chunk_size))
	state_chunks = np.array_split(states, num_chunks, axis=0)
	values = np.zeros((states.shape[0],))

	current_idx = 0

	for chunk_number, state_chunk in tqdm(enumerate(state_chunks), desc="Making VF plot"):  # type: (int, np.ndarray)
		chunk_qvalues = q_agent.qRef_li.predict(state_chunk)
		chunk_values = numpy.max(chunk_qvalues, axis=1, keepdims=True)
		current_chunk_size = len(state_chunk)
		values[current_idx:current_idx + current_chunk_size] = chunk_values
		current_idx += current_chunk_size

	plt.scatter(states[:, 0], states[:, 1], c=values)
	plt.colorbar()

	file_name = f"{q_agent.name}_value_function_seed_{seed}_episode_{episode}"
	file_name = f"{dir_path}/{file_name}" if dir_path != "" else file_name
	plt.savefig(f"{file_name}.png")
	plt.close()
