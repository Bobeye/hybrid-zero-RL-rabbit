# RNN in keras
# by Bowen, Aug 2018
######################################

from keras.models import Model
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, Input


class LSTM_NN():


	def __init__(self, num_steps=30, feature_size=10, 
				 hidden_units=[20, 20], output_size=7):
		self.num_steps = num_steps
		self.feature_size = feature_size
		self.hidden_units = hidden_units
		self.output_size = output_size

		self._build()

		self.parameter_count = 0
		for layer in self.model.layers:
			for part in layer.get_weights():
				self.parameter_count += part.size

		self.layer_sizes = []
		self.layer_shapes = []
		for layer in self.model.layers[1:]:
			layer_size = []
			layer_shape = []
			for part in layer.get_weights():
				layer_size += [part.size]
				layer_shape += [part.shape]
			self.layer_sizes += [layer_size]
			self.layer_shapes += [layer_shape]

		print self.layer_sizes, self.layer_shapes

	def _build(self):
		layer_in = Input(shape=(self.num_steps,self.feature_size))
		layer_lstm = layer_in
		for hidden_size in self.hidden_units:
			layer_lstm = LSTM(hidden_size, return_sequences=True)(layer_lstm)
		layer_out = TimeDistributed(Dense(1, activation="sigmoid"))(layer_lstm)
		self.model = Model(layer_in, layer_out)
		self.model.compile(loss='mse', optimizer='adam')
		print self.model.summary()

	def set_weights(self, model_params):
		weights = []
		pointer = 0
		for l in range(len(self.layer_sizes)):
			layer_param = []
			for s in range(len(self.layer_sizes[l])):
				layer_param += [model_params[pointer:pointer+self.layer_sizes[l][s]].reshape(self.layer_shapes[l][s])]
				pointer += self.layer_sizes[l][s]
			weights += [np.array(layer_param)]
		for i in range(len(self.model.layers)):
			if i == 0:
				pass
			else:
				self.model.layers[i].set_weights(weights[i-1])

	def predict(self, X):
		return self.model.predict(X)


if __name__ == "__main__":
	import numpy as np

	batchsize = 1
	hidden_size = 15
	num_steps = 30
	feature_size = 12

	test = LSTM_NN(num_steps=num_steps, feature_size=feature_size,
			hidden_units=[hidden_size, hidden_size, 10],
			output_size = 1)

	params = np.zeros(test.parameter_count)
	test.set_weights(params)

	x = np.zeros((batchsize, num_steps, feature_size))
	print test.predict(x).shape