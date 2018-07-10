import autograd.numpy as np
from autograd import elementwise_grad, grad

class GeneralizedAdvantageEstimation():

	def __init__(self, Lambda=0.95, Gamma=0.99):
		self.Lambda = Lambda
		self.Gamma = Gamma

	def get(self, states, rewards, values, steps=200):
		if steps > states.shape[0]:
			raise ValueError("Number of steps cannot be smaller than states dimension")
		err = 0
		for t in range(steps-1):
			advantage_t = 0
			for i in range(t+1):	
				advantage_t = advantage_t + rewards[t]*(self.Gamma**i)
			advantage_t = advantage_t + (self.Gamma**(t+1)) * values[t+1]
			advantage_t = advantage_t - values[0]
			err = err + (self.Lambda**t) * advantage_t
		err = err * (1 - self.Lambda)
		return err

def get_prob(k, cov, mean, sample):
	prob = (1/(np.sqrt((2*np.pi)**k)*np.linalg.det(cov))) * np.exp((-.5)*(np.dot(np.dot((sample-mean).T, np.linalg.inv(cov)), (sample-mean))))
	return prob


class PolicyGradient():

	def __init__(self, solution_new, solution_old, model,
				 states=None,
				 actions=None,
				 rewards=None,
				 values=None,
				 sigma_new=0.1,
				 sigma_old=0.09,
				 n_steps=200,
				 n_batch=5,
				 Lambda=0.99,
				 Gamma=0.95,
				 Epsilon=0.2):

		self.sigma_old = sigma_old
		self.sigma_new = sigma_new
		self.n_steps = n_steps
		self.n_batch = n_batch
		self.Lambda = Lambda
		self.Gamma = Gamma
		self.Epsilon = Epsilon

		self.sigma_old = sigma_old
		self.model_old = model_old
		self.model_old.set_weights(solution_old)

		self.gae = GeneralizedAdvantageEstimation(Lambda=Lambda, Gamma=Gamma)

		self.gradient = self.calc_gradient(solution_new, model, states, actions, rewards, values)

	def calc_gradient(self, solution_new, model, states, actions, rewards, values):
		g = elementwise_grad(self.clip_loss, argnum=0)(solution_new, model, states, actions, rewards, values)
		g = np.nan_to_num(g)
		return g

	def clip_loss(self, solution_new, model, states, actions, rewards, values):
		model.set_weights(solution_new)

		loss = 0
		total_steps = 0
		for b in range(self.n_batch):
			n_steps = min(states[b].shape[0],self.n_steps)
			s_batch = states[b]
			a_batch = actions[b]
			r_batch = rewards[b]
			v_batch = values[b]
			action_size = a_batch[0].shape[0]
			a_star_old, _ = self.model_old.predict(s_batch[0])
			prob_old = get_prob(action_size, 
								np.diag(np.full((action_size,),sigma_old)), 
								a_star_old, a_batch[0])
			a_star_new, _ = self.model.predict(s_batch[0])
			prob_new = get_prob(action_size,
								np.diag(np.full((action_size,),sigma_new)), 
								a_star_new, a_batch[0])
			r_prob = prob_new / prob_old
			loss = loss - np.log(min(r_prob,np.clip(r_prob, 1-self.Epsilon, 1+self.Epsilon))) * self.gae.get(s_batch, r_batch, v_batch, steps=n_steps)
			total_steps += n_steps

		loss = loss / total_steps

		return loss


