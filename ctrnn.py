import numpy as np

class CTRNN:
	def __init__(self, N, dt=0.01, initial_activations=None, weights=None,
							 biases=None, time_constants=None, inputs=None,):
		"""
		Continuous-Time Recurrent Neural Network. Used to simulate systems like
			multi-species (N >= 2) ecosystems.

		N : int
			Number of neurons, species, etc. to simulate within the CTRNN
		dt : float
			Length of time between each step.
		initial_activations : np.array[N]<float>
			Initial activation values for each neuron, at t=0.
		weights : np.array[N, N]<float>
			Weight strength values between each neuron.
		biases : np.array[N]<float>
			Bias weight offsets at each neuron.
		time_constants : np.array[N]<float>
			Time constants of each post-synapse neuron.
		inputs : np.array[N]<float -> float>
			External input. In this implementation, this will be a vector of 0's; we
			are only considering the neural network in an isolated vacuum.
		"""
		# Validate initial activation, weights, biases, time constants; None is ok
		# since we have CTRNN._default_weights().
		# Not sure how to get name of variable from variable itself; usually this is
		# tricky or impossible, so we use the `name` parameter instead.
		def validate(to_validate, name, shape):
			if type(to_validate) not in {type(None), np.ndarray}:
				if np.shape(to_validate) != shape:
					# (2,3) --> "2x3", (5,) --> "5", (6,7,2) --> "6x7x2"
					shape_string = f"{'x'.join(map(str, shape))}" if type(shape) is tuple \
						else f"{shape}-length"
					raise ValueError(
						f"Your value given for {name} is not a(n) {shape_string} " +
						f"(N={N}) np.ndarray. Given {to_validate}, of type " +
						f"{type(to_validate)}, shape {np.shape(to_validate)}."
					)

		validate(initial_activations, "initial_activations", N)
		validate(weights, "weights", (N, N))
		validate(biases, "biases", N)
		validate(time_constants, "time_constants", N)

		self.N = N
		self.dt = dt
		# After these initializations, all are np.ndarray's of the proper shape
		self.activations = initial_activations if initial_activations is not None \
			else np.random.random(self.N)
		self.weights = weights if weights is not None \
			else 4.0 * np.random.random((self.N, self.N))
		self.biases = biases if biases is not None \
			else np.random.random(self.N)
		# self.time_constants[i] is between [10dt, 100dt)
		self.time_constants = np.clip(time_constants, 10*self.dt, 100*self.dt) \
			if time_constants is not None \
			else self.dt * np.exp(2*np.log(10) * np.random.random(self.N))
			# else CTRNN._default_weights(self.N)
		self.inputs = inputs if inputs is not None \
			else np.zeros(self.N)
			# else CTRNN._default_weights(self.N)
	
	def step(self):
		"""
		Go through 1/self.dt length of time through the CTRNN governing
		differential equations.
		"""
		previous_activations = self.activations
		biased = CTRNN._sigmoid(previous_activations + self.biases)
		with_weights = np.matmul(self.weights, biased)
		self.activations += self.dt * (-previous_activations + with_weights + \
			self.inputs) / self.time_constants
	
	def _default_weights(shape):
		"""
		Default value given to weights between neurons if no weights are provided
		from instantiation.

		shape : int or Tuple<int>
		"""
		return 4.0 * np.random.random(shape)
		# return np.random.normal(loc=1.0, scale=1.0, size=shape)
		# return np.zeros(shape)
	
	def _sigmoid(x):
		"""
		This is a (static) class method to keep all the functionality within one
		class. From the standpoint of a typical project, this is poor design, but
		we'll only be using this function within the class, so I see no need to have
		it be importable to other files.

		x : float or np.ndarray<float>
		"""
		# np.clip() to prevent overflow errors
		return 1.0 / (1.0 + np.exp(np.clip(-x, -100, 100)))
