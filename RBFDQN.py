import gym, sys, keras
from keras.models import Model
from keras.layers import Dense, Input, Concatenate, Lambda, Activation, dot
from keras.initializers import RandomUniform
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras import constraints
import numpy, random
import utils, buffer_class
from tqdm import trange
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from thundersvm import OneClassSVM
import numpy as np
import ipdb
from novelty_classifier import NoveltyDetectionClassifier


class Q_class:

	def __init__(self,params,env,state_size,action_size):
		self.env=env
		self.params=params
		self.state_size,self.action_size=state_size,action_size
		self.network,self.qRef_li=self.create_network()
		self.target_network,self.target_qRef_li=self.create_network()
		self.target_network.set_weights(self.network.get_weights())
		self.buffer_object=buffer_class.buffer_class(max_length=self.params['max_buffer_size'], state_size=state_size, action_size=action_size)
		self.one_class_classifier = None

	def func_L2(self,tensors):
		return -K.sqrt(K.sum(K.square(tensors[0]-tensors[1]),
						axis=1,
						keepdims=True)+ self.params['norm_smoothing'])

	def create_network(self):
		state_input,action_input=Input(shape=(self.state_size,)), Input(shape=(self.action_size,))
		h = state_input
		for _ in range(self.params['num_layers']):	
			h = Dense(self.params['layer_size'])(h)
			h = Activation("relu")(h)				
		q_output = Dense(self.params['num_points'])(h)# value of anchor points
		ha = Dense(512)(state_input)
		ha = Activation("relu")(ha)
		ha = keras.layers.Dropout(rate=0.4)(ha)
		#define anchor point locations
		a_negative_distance_li,a_li,L2_layer=[],[],Lambda(self.func_L2,output_shape=(1,))
		for a_index in range(self.params['num_points']):
			temp = Dense(self.action_size,
							activation='tanh',#to ensure output is between -1 and 1
							kernel_initializer=RandomUniform(minval=-.1, maxval=+.1, seed=None),
							bias_initializer=RandomUniform(minval=-1, maxval=+1, seed=None))(ha)
			# now ensure output is between -high and high
			temp = Lambda(lambda x: x*self.env.action_space.high[0],(self.action_size,))(temp)
			a_li.append(temp)
			#get negative distance between a and each anchor point
			temp = L2_layer([temp,action_input])
			a_negative_distance_li.append(temp)
		a_negative_distance_cat=Concatenate(axis=-1)(a_negative_distance_li)
		#pass negative distance from softmax (with temperature)
		a_negative_distance_cat=Lambda(lambda x: x * self.params['temperature'],output_shape=(self.params['num_points'],))(a_negative_distance_cat)
		softmax=Activation('softmax')(a_negative_distance_cat)

		final_q=dot([q_output,softmax],axes=1, normalize=False)
		model = Model(inputs=[state_input, action_input], outputs=final_q)
		opt = optimizers.RMSprop(lr=self.params['learning_rate'],clipnorm=2.5)
		model.compile(loss='mse',optimizer=opt)


		qRef_li=[]
		for j in range(self.params['num_points']):
			each_qRef=[]
			for i in range(self.params['num_points']):
				each_qRef.append(L2_layer([a_li[i],a_li[j]]))
			each_qRef = Concatenate(axis=-1)(each_qRef)
			each_qRef = Lambda(lambda x: x * self.params['temperature'],output_shape=(self.params['num_points'],))(each_qRef)
			each_qRef = Activation('softmax')(each_qRef)
			test_final_q = dot([q_output,each_qRef],axes=1, normalize=False)
			qRef_li.append(test_final_q)
		# given a state, qRef_li gives anchor locations and their corresponding values ...
		qRef_li = Model(inputs=state_input,
					  outputs=[Concatenate(axis=1)(a_li),
							   Concatenate(axis=-1)(qRef_li)])

		return model,qRef_li

	def e_greedy_policy(self,s,episode,train_or_test):
		epsilon=1./numpy.power(episode,1./self.params['policy_parameter'])
		if train_or_test=='train' and random.random() < epsilon:
			a = self.env.action_space.sample()
			return a.tolist()

		else:
			s_matrix = numpy.array(s).reshape(1,self.state_size)
			aRef_li,qRef_li = self.qRef_li.predict(s_matrix)
			max_index = numpy.argmax(qRef_li)
			aRef_begin,aRef_end = max_index*self.action_size,(max_index+1)*self.action_size
			a = aRef_li[0,aRef_begin:aRef_end]
			return np.clip(a.tolist(), -1, 1)

	def get_q_value_target(self, rewards, dones, s_prime, q_prime, q_max):
		gamma = self.params["gamma"]
		if self.one_class_classifier is None:
			target = rewards + (gamma * (1 - dones) * q_prime)
		elif isinstance(self.one_class_classifier, OneClassSVM):
			novelty_predictions = self.one_class_classifier.predict(s_prime)
			novel_idx = np.where(novelty_predictions != 1)[0]
			q_prime[novel_idx, :] = args.qmax
			target = rewards + (gamma * (1 - dones) * q_prime)
		elif isinstance(self.one_class_classifier, NoveltyDetectionClassifier):
			known_probabilities = self.one_class_classifier.predict(s_prime)
			# known_probabilities = np.clip(known_probabilities, a_min=0., a_max=0.5)  # TODO: Trying clipping the intrinsic rewards
			known_target = rewards + (gamma * (1 - dones) * q_prime)
			known_target = known_probabilities * known_target.squeeze()
			unknown_target = (1. - known_probabilities) * q_max
			target = known_target + unknown_target
		else:
			raise NotImplementedError(self.one_class_classifier)
		return target

	def update(self):
		'''
		1-samples a bunch of tuples from the buffer
		2-to compute a*, randomly initializes some a, then does gradien ascent to improve them
		3-gets Q corresponding with best action fro previous step
		4-then performs Q-learning update
		5-from time to time, syncs target network
		'''
		if len(self.buffer_object.storage)<params['batch_size']:
			return
		else:
			pass

		s_matrix, a_matrix, r_matrix, sp_matrix, done_matrix = self.buffer_object.sample(batch_size=params["batch_size"],
																						 reward_clip=params["reward_clip"])

		next_aRef_li,next_qRef_li=self.target_qRef_li.predict(sp_matrix)
		next_qRef_star_matrix=numpy.max(next_qRef_li,axis=1,keepdims=True)

		# label=r_matrix+self.params['gamma']*(1-done_matrix)*next_qRef_star_matrix
		label = self.get_q_value_target(r_matrix, done_matrix, sp_matrix, next_qRef_star_matrix, q_max=args.qmax)
		self.network.fit(x=[s_matrix,a_matrix],
						y=label,
						epochs=self.params['updates_per_batch'],
						batch_size=params['batch_size'],
						verbose=0)
		self.update_target_net()

	def update_target_net(self):
		''' 
		given the online and the target network, update target to online, by using
		an exponential weighted average
		'''
		network_weights=self.network.get_weights()
		target_weights=self.target_network.get_weights()
		new_target_weights=[]
		for n,t in zip(network_weights,target_weights):
			temp=self.params['target_network_learning_rate']*n+(1-self.params['target_network_learning_rate'])*t
			new_target_weights.append(temp)
		self.target_network.set_weights(new_target_weights)

	def train_novelty_detector(self, visited_states, episode):
		self.one_class_classifier = NoveltyDetectionClassifier(nu_high=0.5, nu_low=0.01, nu_resolution=0.05)
		self.one_class_classifier.fit(visited_states)
		plot_name_prefix = f"{args.env}_ocsvm_"

		utils.plot_one_class_classifier(visited_states, self.one_class_classifier, episode, plot_name_prefix)

	@staticmethod
	def get_nu(current_episode, max_episode):
		nu_initial = 0.2
		nu_final = 0.
		m = (nu_final - nu_initial) / max_episode
		nu = (m * current_episode) + nu_initial
		return nu


if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="LunarLanderContinuous-v2")# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--sparse", action="store_true", default=False)
	parser.add_argument("--qmax", type=float, default=10.)
	parser.add_argument("--use_ocsvm", action="store_true", default=False)
	parser.add_argument("--max_steps", default="")
	args = parser.parse_args()

	env_dic = {}
	env_dic['Pendulum-v0']=0
	env_dic['LunarLanderContinuous-v2']=10
	env_dic['BipedalWalker-v2']=20
	env_dic['Ant-v1']=30
	env_dic['HalfCheetah-v1']=40
	env_dic['Hopper-v1']=50
	env_dic['InvertedDoublePendulum-v1']=60
	env_dic['InvertedPendulum-v1']=70
	env_dic['MountainCarContinuous-v0'] = 90
	if args.env not in env_dic.keys():
		print("environment not recognized ... use one of the following environments")
		print(env_dic.keys())
		assert False
	hyper_parameter_name=str(env_dic[args.env])
	alg='rbf'
	params=utils.get_hyper_parameters(hyper_parameter_name,alg)
	params['hyper_parameters_name']=hyper_parameter_name
	env=utils.create_env("pendulum", "swingup", args.seed, dmc=True)
	params['env']=env
	params['seed_number']=args.seed
	utils.set_random_seed(params)
	s0=env.reset()
	utils.action_checker(env)
	Q_object=Q_class(params,env,state_size=len(s0),action_size=len(env.action_space.low))
	G_li=[]

	visited_states = []
	max_steps = int(args.max_steps) if args.max_steps != "" else np.inf

	for episode in range(params['max_episode']):
		#train policy with exploration
		s,done=env.reset(),False
		step_number = 0
		while done==False and step_number < max_steps:
			visited_states.append(s)
			a=Q_object.e_greedy_policy(s,episode+1,'train')
			sp,r,_,info=env.step(numpy.array(a))

			# TODO: This is dmc specific
			r *= 100.
			done = r > 0

			if args.sparse:
				time_limit_truncated = info.get('TimeLimit.truncated', False)
				is_terminal = done and not time_limit_truncated
				r = +100. if is_terminal and r > 0. else 0.

				Q_object.buffer_object.append(s,a,r,is_terminal,sp)
			else:
				Q_object.buffer_object.append(s, a, r, done, sp)

			s=sp
			step_number += 1

		#now update the Q network
		for i in trange(params['updates_per_episode'],file=sys.stdout, desc='training'):
			Q_object.update()

		if args.use_ocsvm:
			Q_object.train_novelty_detector(np.array(visited_states), episode)

		#test the learned policy, without performing any exploration
		s,t,G,done=env.reset(),0,0,False
		step_number = 0
		while done==False and step_number < max_steps:
			a=Q_object.e_greedy_policy(s,episode+1,'test')
			sp,r,_,_=env.step(numpy.array(a))
			# if episode % 10 == 0:
			# 	env.render()
			s,t,G=sp,t+1,G+r
			step_number += 1
			done = r > 0
		print("in episode {} we collected return {} in {} timesteps".format(episode,G,t))
		G_li.append(G)
		if episode % 10 == 0 and episode>0:	
			utils.save(G_li,params,alg)

	utils.save(G_li,params,alg)