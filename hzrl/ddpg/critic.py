from nn import NeuralNetwork

class CriticNetwork():

	def __init__(self, state_dim=20, action_dim=4, inter_state_dim=5, 
				 snn_units=[20,20], snn_activations=["relu", "relu", "relu"],
				 nn_units=[20], nn_activations=["relu", "passthru"]):

		self.state_model = NeuralNetwork(input_dim=state_dim,
				 				   		 output_dim=inter_state_dim,
				 				   		 units=snn_units,
				 				   		 activations=snn_activations)
		self.state_action_model = NeuralNetwork(input_dim=inter_state_dim+action_dim,
				 				   		 		output_dim=1,
				 				   		 		units=nn_units,
				 				   		 		activations=nn_activations)
		self.parameter_count = self.state_model.parameter_count + self.state_action_model.parameter_count

	def set_weights(self, model_params):
		self.state_model.set_weights(model_params[0:self.state_model.parameter_count])
		self.state_action_model.set_weights(model_params[-self.state_action_model.parameter_count:])

	def get_value(self, state, action):
		inter_state = self.state_model.predict(state)
		x = np.concatenate([inter_state, action], axis=0)
		return self.state_action_model.predict(x)