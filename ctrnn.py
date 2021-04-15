from _typeshed import NoneType
import numpy as np

class CTRNN:
	def __init__(self, n_neurons, weights=None, biases=None, inputs=None):
		"""
		Continuous-Time Recurrent Neural Network. Used to simulate systems like
			multi-species (N > 2) ecosystems.

		n_neurons : int
			Number of neurons, species, etc. to simulate within the CTRNN
		weights : np.array[n_neurons, n_neurons]<number>
			Weight values between each neuron. weights[i,i] = 0 (or 1?) for all i.
		biases : np.array[n_neurons]<number>
			Bias weight offsets at each neuron.
		inputs : unknown so far; np.array[n_neurons]<number -> number>
			External input? Not sure. It seems to me like we could just have biases
			be a vector of 0's and have inputs' values at each point in time account
			for the constant offset of biases.
		"""
		# Validate weights data; None is ok since we have CTRNN._default_weights()
		if type(weights) not in {type(None), np.ndarray} or \
			np.shape(weights) != (n_neurons, n_neurons):
			raise ValueError(
				f"Your value given for weights is not an n_neurons x n_neurons " +
				f"(n_neurons={n_neurons}) np.ndarray. Given {weights}, of type " +
				f"{type(weights)}, shape {np.shape(weights)}."
			)
		# Validate biases data; None is ok since we have CTRNN._default_weights()
		if type(biases) not in {type(None), np.ndarray} or \
			np.shape(biases) != n_neurons:
			raise ValueError(
				f"Your value given for biases is not an n_neurons-long" +
				f"(n_neurons={n_neurons}) np.ndarray. Given {biases}, of type " +
				f"{type(biases)}, shape {np.shape(biases)}."
			)

		self.n_neurons = n_neurons
		# After this, self.weights is def
		self.weights = weights if weights is not None \
			else CTRNN._default_weights((self.n_neurons, self.n_neurons))
		self.biases = biases if biases is not None \
			else CTRNN._default_weights(self.n_neurons)
		
		# Array of functions?
		self.inputs = inputs
	
	def _default_weights(shape):
		"""
		Default value given to weights between neurons if no weights are provided
		from instantiation.

		shape : int or Tuple<int>
		"""
		return np.zeros(shape)
	
	def _sigmoid(x):
		"""
		This is a (static) class method to keep all the functionality within one
		class. From the standpoint of a typical project, this is poor design, but
		we'll only be using this function within the class, so I see no need to have
		it be importable to other files.

		x : number
		"""
		return 1.0 / (1.0 + np.exp(-x))
