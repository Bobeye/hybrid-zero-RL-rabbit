from nn import NeuralNetwork

class ActorNetwork():

	def __init__(self, state_dim=20, action_dim=4, 
				 nnunits=[20,20], nnactivations=["relu", "relu", "passthru"]):
		self.model = NeuralNetwork(input_dim=state_dim,
				 				   output_dim=action_dim,
				 				   units=nnunits,
				 				   activations=nnactivations)

	def set_weights(self, model_params):
		self.model.set_weights(model_params)

	def get_action(self, state):
		return self.model.predcit(state)
