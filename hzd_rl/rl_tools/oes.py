import numpy as np 

class Adam():
	def __init__(self, pi, dim, stepsize=1e-2, beta1=0.99, beta2=0.999, epsilon=1e-8):
		self.pi = pi
		self.dim = dim
		self.stepsize = stepsize
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.m = np.zeros(self.dim, dtype=np.float32)
		self.v = np.zeros(self.dim, dtype=np.float32)
		self.t = 0

	def compute_step(self, globalg):
		self.t += 1
		a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
		self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
		self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
		step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
		return step

def compute_ranks(x):
	"""
	Returns ranks in [0, len(x))
	Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
	(https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
	"""
	assert x.ndim == 1
	ranks = np.empty(len(x), dtype=int)
	ranks[x.argsort()] = np.arange(len(x))
	return ranks

def compute_centered_ranks(x):
	"""
	https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
	"""
	y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
	y /= (x.size - 1)
	y -= .5
	return y

def compute_weight_decay(weight_decay, model_param_list):
	model_param_grid = np.array(model_param_list)
	return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


class OpenES:
	''' Basic Version of OpenAI Evolution Strategies.'''
	def __init__(self, num_params,             # number of model parameters
				 sigma_init=0.1,               # initial standard deviation
				 sigma_decay=0.999,            # anneal standard deviation
				 sigma_limit=0.01,             # stop annealing if less than this
				 learning_rate=0.01,           # learning rate for standard deviation
				 learning_rate_decay = 0.9999, # annealing the learning rate
				 learning_rate_limit = 0.001,  # stop annealing learning rate
				 popsize=256,                  # population size
				 antithetic=False,             # whether to use antithetic sampling
				 weight_decay=0.01,            # weight decay coefficient
				 rank_fitness=True,            # use rank rather than fitness numbers
				 forget_best=True):            # forget historical best

		self.num_params = num_params
		self.sigma_decay = sigma_decay
		self.sigma = sigma_init
		self.sigma_init = sigma_init
		self.sigma_limit = sigma_limit
		self.learning_rate = learning_rate
		self.learning_rate_decay = learning_rate_decay
		self.learning_rate_limit = learning_rate_limit
		self.popsize = popsize
		self.antithetic = antithetic
		if self.antithetic:
			assert (self.popsize % 2 == 0), "Population size must be even"
			self.half_popsize = int(self.popsize / 2)

		self.reward = np.zeros(self.popsize)
		self.mu = np.zeros(self.num_params)
		self.best_mu = np.zeros(self.num_params)
		self.best_reward = 0
		self.first_interation = True
		self.forget_best = forget_best
		self.weight_decay = weight_decay
		self.rank_fitness = rank_fitness
		if self.rank_fitness:
			self.forget_best = True # always forget the best one if we rank
		self.optimizer = Adam(self.mu, self.num_params, stepsize=self.learning_rate)

	def rms_stdev(self):
		sigma = self.sigma
		return np.mean(np.sqrt(sigma*sigma))

	def ask(self):
		'''returns a list of parameters'''
		# antithetic sampling
		if self.antithetic:
			self.epsilon_half = np.random.randn(self.half_popsize, self.num_params)
			self.epsilon = np.concatenate([self.epsilon_half, - self.epsilon_half])
		else:
			self.epsilon = np.random.randn(self.popsize, self.num_params)

		self.solutions = self.mu.reshape(1, self.num_params) + self.epsilon * self.sigma

		return self.solutions

	def tell(self, reward_table_result, update_solutions):
		if update_solutions is None:
			us_mode = False
		else:
			us_mode = True

		# input must be a numpy float array
		assert(len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."
		
		reward = np.array(reward_table_result)
		
		if self.rank_fitness:
			reward = compute_centered_ranks(reward)

		if self.weight_decay > 0:
			l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
			reward += l2_decay

		idx = np.argsort(reward)[::-1]

		best_reward = reward[idx[0]]
		if us_mode:
			best_mu = update_solutions[idx[0]]
		else:
			best_mu = self.solutions[idx[0]]

		self.curr_best_reward = best_reward
		self.curr_best_mu = best_mu

		if self.first_interation:
			self.first_interation = False
			self.best_reward = self.curr_best_reward
			self.best_mu = best_mu
		else:
			if self.forget_best or (self.curr_best_reward > self.best_reward):
				self.best_mu = best_mu
				self.best_reward = self.curr_best_reward

		if us_mode:
			# change_mu = update_solutions.mean(axis=0) - self.solutions.mean(axis=0)
			# self.mu += self.learning_rate * change_mu
			self.epsilon = (update_solutions-self.mu)/self.sigma
			normalized_reward = (reward - np.mean(reward)) / np.std(reward)
			change_mu = 1./(self.popsize*self.sigma)*np.dot(self.epsilon.T, normalized_reward)
			# step = self.optimizer.compute_step(-change_mu)
			# self.mu += step
			self.mu += self.learning_rate * change_mu
		else:
			normalized_reward = (reward - np.mean(reward)) / np.std(reward)
			change_mu = 1./(self.popsize*self.sigma)*np.dot(self.epsilon.T, normalized_reward)
			self.mu += self.learning_rate * change_mu

		# adjust sigma according to the adaptive sigma calculation
		if (self.sigma > self.sigma_limit):
			self.sigma *= self.sigma_decay

		if (self.learning_rate > self.learning_rate_limit):
			self.learning_rate *= self.learning_rate_decay

	def current_param(self):
		return self.curr_best_mu

	def set_mu(self, mu):
		self.mu = np.array(mu)

	def get_mu(self):
		return self.mu

	def best_param(self):
		return self.best_mu

	def result(self): # return best params so far, along with historically best reward, curr reward, sigma
		return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)
