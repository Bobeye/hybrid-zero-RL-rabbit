import numpy as np

def relu(x):
	return np.maximum(x, 0)

def passthru(x):
	return x

def tanh(x):
	return np.tanh(x)

class NeuralNetwork():

	def __init__(self, input_dim=1,
				 output_dim=24,
				 units=[20,20,20],
				 activations=["relu", "relu", "passthru", "passthru"]):

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.units = units
		self.num_layers = len(units)
		
		self.activations = []
		for act in activations:
			if act == "relu":
				self.activations += [relu]
			elif act == "passthru":
				self.activations += [passthru]
			else:
				self.activations += [tanh]

		unit_list = [input_dim]
		for u in self.units:
			unit_list += [u, u]
		unit_list += [output_dim]
		self.shapes = []
		for l in range(self.num_layers+1):
			self.shapes += [(unit_list[2*l], unit_list[2*l+1])]
		
		self.weight = []
		self.bias = []
		self.parameter_count = 0
		idx = 0
		for shape in self.shapes:
			self.weight.append(np.zeros(shape=shape))	
			self.bias.append(np.zeros(shape=shape[1]))
			self.parameter_count += (np.product(shape) + shape[1])
			idx += 1

	def set_weights(self, model_params):
		pointer = 0
		for i in range(len(self.shapes)):
			w_shape = self.shapes[i]
			b_shape = self.shapes[i][1]
			s_w = np.product(w_shape)
			s = s_w + b_shape
			chunk = np.array(model_params[pointer:pointer+s])
			self.weight[i] = chunk[:s_w].reshape(w_shape)
			self.bias[i] = chunk[s_w:].reshape(b_shape)
			pointer += s

	def predict(self, X):
		h = np.array([X]).flatten()
		num_layers= len(self.weight)
		for i in range(num_layers):
			w = self.weight[i]
			b = self.bias[i]
			h = np.matmul(h, w) + b
			h = self.activations[i](h)
		return h
