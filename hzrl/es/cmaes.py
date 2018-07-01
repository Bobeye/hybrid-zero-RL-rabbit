import numpy as np 

class CMAES:
	'''CMA-ES wrapper.'''
	def __init__(self, num_params,      # number of model parameters
				 sigma_init=0.10,       # initial standard deviation
				 popsize=255,           # population size
				 weight_decay=0.01):    # weight decay coefficient

		self.num_params = num_params
		self.sigma_init = sigma_init
		self.popsize = popsize
		self.weight_decay = weight_decay
		self.solutions = None

		self.best_param = np.zeros(self.num_params)
		self.best_reward = None 
		self.current_param = np.zeros(self.num_params)
		self.curr_best_reward = None

		import cma
		self.es = cma.CMAEvolutionStrategy( self.num_params * [0],
											self.sigma_init,
											{'popsize': self.popsize,
											})

	def rms_stdev(self):
		sigma = self.es.result[6]
		return np.mean(np.sqrt(sigma*sigma))

	def ask(self):
		'''returns a list of parameters'''
		self.solutions = np.array(self.es.ask())
		return self.solutions

	def tell(self, reward_table_result, update_solutions):
		reward_table = -np.array(reward_table_result)
		if self.weight_decay > 0:
			l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
			reward_table += l2_decay
		self.es.tell(update_solutions, (reward_table).tolist())

	def current_param(self):
		return self.es.result[5] # mean solution, presumably better with noise

	def set_mu(self, mu):
		pass

	def best_param(self):
		return self.es.result[0] # best evaluated solution

	def get_mu(self):
		return self.es.mean

	def result(self): # return best params so far, along with historically best reward, curr reward, sigma
		r = self.es.result
		return (self.es.mean, -r[1], -r[1], r[6])
		# return (r[0], -r[1], -r[1], r[6])
		# return (self.best_param, self.best_reward, self.curr_best_reward, self.sigma)
