import autograd.numpy as np 

class NeuralNetwork():

	def __init__(self, input_dim=1,
				 output_dim=20,
				 units=[20,20,20],
				 activations=["relu", "relu", "passthru"]):
	
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.units = units
		self.num_layers = len(units)


	def get_params(self):
		pass

	def set_params(self):
		pass

	def predict(self):
		return None

