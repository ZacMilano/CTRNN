import numpy as np

class CTRNN:
	def __init__(self, n_neurons, initial_activations=None, weights=None,
							 biases=None, time_constants=None, inputs=None, dt=0.01):
		"""
		Continuous-Time Recurrent Neural Network. Used to simulate systems like
			multi-species (N >= 2) ecosystems.

		n_neurons : int
			Number of neurons, species, etc. to simulate within the CTRNN
		initial_activations : np.array[n_neurons]<number>
			Initial activation values for each neuron.
		weights : np.array[n_neurons, n_neurons]<number>
			Weight values between each neuron. weights[i,i] = 0 (or 1? TODO) for all i.
		biases : np.array[n_neurons]<number>
			Bias weight offsets at each neuron.
		time_constants : np.array[n_neurons]<number>
			Time constants of each post neuron. Not sure what this is.
		inputs : unknown so far; np.array[n_neurons]<number -> number>
			External input? Not sure. It seems to me like we could just have biases
			be a vector of 0's and have inputs' values at each point in time account
			for the constant offset of biases.
		dt : number
			Length of time between each step.
		"""
		# Validate initial activation, weights, biases, time constants; None is ok
		# since we have CTRNN._default_weights().
		# Not sure how to get name of variable from variable itself; usually this is
		# tricky or impossible.
		def validate(to_validate, name, shape):
			if type(to_validate) not in {type(None), np.ndarray} or \
				np.shape(to_validate) != shape:
				# (2,3) --> "2x3", (5,) --> "5", (6,7,2) --> "6x7x2"
				shape_string = f"{'x'.join(map(str, shape))}" if type(shape) is tuple \
					else f"{shape}-length"
				raise ValueError(
					f"Your value given for {name} is not a(n) {shape_string}" +
					f"(n_neurons={n_neurons}) np.ndarray. Given {to_validate}, of type " +
					f"{type(to_validate)}, shape {np.shape(to_validate)}."
				)

		validate(initial_activations, "initial_activations", n_neurons)
		validate(weights, "weights", (n_neurons, n_neurons))
		validate(biases, "biases", n_neurons)
		validate(time_constants, "time_constants", n_neurons)

		self.n_neurons = n_neurons
		# After these initializations, activations, weights, and biases are all
		# np.ndarrays of the proper shape
		self.activations = initial_activations if initial_activations is not None \
			else CTRNN._default_weights(self.n_neurons)
		self.weights = weights if weights is not None \
			else CTRNN._default_weights((self.n_neurons, self.n_neurons))
		self.biases = biases if biases is not None \
			else CTRNN._default_weights(self.n_neurons)
		# Array of functions?
		self.inputs = inputs
		self.dt = dt
	
	def step(self, t):
		# TODO: This. Use the discrete differential equation-type algorithm given in
		# lecture.
		pass
	
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
