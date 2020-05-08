import numpy
import numpy as np
import random
from collections import deque

class buffer_class:
	def __init__(self,max_length,state_size,action_size,prioritize=True):
		self.storage=deque(maxlen=max_length)
		self.state_size=state_size
		self.action_size=action_size
		self.prioritize=prioritize
		self.pos_transitions=[]

	def append(self,s,a,r,done,sp):
		dic={}
		dic['s']=s
		dic['a']=a
		dic['r']=r
		if done==True:
			dic['done']=1
		else:
			dic['done']=0
		dic['sp']=sp
		self.storage.append(dic)

		if done and r>0:
			self.pos_transitions.append((s,a,r,sp,done))

	def sample(self, batch_size, reward_clip):
		sample_batch_size = batch_size - 1 if self.prioritize and len(self.pos_transitions) > 0 else batch_size
		batch = random.sample(self.storage, sample_batch_size)
		s_li = [b['s'] for b in batch]
		sp_li = [b['sp'] for b in batch]
		r_li = [b['r'] for b in batch]
		done_li = [b['done'] for b in batch]
		a_li = [b['a'] for b in batch]
		s_matrix = numpy.array(s_li).reshape(sample_batch_size, self.state_size)
		a_matrix = numpy.array(a_li).reshape(sample_batch_size, self.action_size)
		r_matrix = numpy.array(r_li).reshape(sample_batch_size, 1)
		r_matrix = numpy.clip(r_matrix, a_min=-reward_clip, a_max=reward_clip)
		sp_matrix = numpy.array(sp_li).reshape(sample_batch_size, self.state_size)
		done_matrix = numpy.array(done_li).reshape(sample_batch_size, 1)

		if self.prioritize:
			s_matrix, a_matrix, r_matrix, sp_matrix, done_matrix = self._append_positive_transition(states=s_matrix,
																									actions=a_matrix,
																									rewards=r_matrix,
																									next_states=sp_matrix,
																									dones=done_matrix,
																									pos_transitions=self.pos_transitions)

		return s_matrix, a_matrix, r_matrix, sp_matrix, done_matrix

	def _append_positive_transition(self, *, states, actions, rewards, next_states, dones, pos_transitions):
		def _unsqueeze(array):
			return array[None, ...]

		if len(pos_transitions) > 0:
			# Create tensors corresponding to the sampled positive transition
			pos_transition = random.sample(pos_transitions, k=1)[0]
			pos_state = _unsqueeze(pos_transition[0])
			pos_action = np.array([pos_transition[1]])
			pos_reward = np.array([[pos_transition[2]]])
			pos_next_state = _unsqueeze(pos_transition[3])
			pos_done = np.array([[float(pos_transition[4])]])
			assert pos_done == 1, pos_done

			# Add the positive transition tensor to the mini-batch
			states = np.concatenate((states, pos_state), axis=0)
			actions = np.concatenate((actions, pos_action), axis=0)
			rewards = np.concatenate((rewards, pos_reward), axis=0)
			next_states = np.concatenate((next_states, pos_next_state), axis=0)
			dones = np.concatenate((dones, pos_done), axis=0)

			# Shuffle the mini-batch to maintain the IID property
			idx = np.random.permutation(states.shape[0])
			states = states[idx, :]
			actions = actions[idx, :]
			rewards = rewards[idx]
			next_states = next_states[idx, :]
			dones = dones[idx]

		return states, actions, rewards, next_states, dones
