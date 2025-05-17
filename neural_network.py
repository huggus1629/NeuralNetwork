import numpy as np


class NeuralNetwork:
	def __init__(self, layer_sizes):
		self.layer_sizes = layer_sizes
		self.parameters = self.create_network()

	def create_network(self):
		self.bias_counter = 0
		self.weights_counter = 0

		# Weights und Biases generieren und einen zufälligen Wert geben.
		parameters = {}
		for i in range(1, len(self.layer_sizes)):
			w_shape = (self.layer_sizes[i], self.layer_sizes[i - 1])
			b_shape = (self.layer_sizes[i], 1)

			parameters[f"W{i}"] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i - 1]) * np.sqrt(2 / self.layer_sizes[i - 1])
			parameters[f"b{i}"] = np.zeros((self.layer_sizes[i], 1))

			self.weights_counter += w_shape[0] * w_shape[1]
			self.bias_counter += b_shape[0]

		return parameters

	def save(self, filename):
		# Datei Speichern als .npz file
		if not filename.endswith('.npz'):
			filename += '.npz'
		np.savez(filename, **self.parameters, layer_sizes=np.array(self.layer_sizes))

	def load(self, filename):
		# .npz Datei Herunterladen
		if not filename.endswith('.npz'):
			filename += '.npz'
		data = np.load(filename, allow_pickle=True)
		self.parameters = {key: data[key] for key in data if key != 'layer_sizes'}
		self.layer_sizes = data['layer_sizes'].tolist()

	@staticmethod
	def relu(Z):
		return np.maximum(0, Z)

	@staticmethod
	def softmax(Z):
		# Softmax Funktion mit numerischer Stabilisierung
		expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
		return expZ / (np.sum(expZ, axis=0, keepdims=True) + 1e-8)

	def forward(self, X):
		if isinstance(X, list):
			X = np.array(X)
		if len(X.shape) == 1:
			# Array in Spaltenvektor umformen
			X = X.reshape(-1, 1)
		if X.shape[0] != self.layer_sizes[0]:
			# Überprüfung, ob die Liste die vorgegebene Größe hat
			raise ValueError(
				f"Falsche Input-Größe: erwartet {self.layer_sizes[0]} Eingänge, aber erhalten {X.shape[0]}.")

		A = X
		num_layers = len(self.parameters) // 2

		# Forwardpropagation, Berechnung der Aktivierung A bzw. Z nach der ReLU Funktion für jedes Layer
		# Theoretisch wird jedes Aktivierungswert a berechnet als Summe von jedem Aktivierungswert vom vorherigen Layer mal ein weight w und einem bias
		# Man kann alle diese Operationen vereinfacht als Matrixprodukt von der Matrix W mal dem Vektor A + dem Vektor b schreiben
		for i in range(1, num_layers):
			W = self.parameters[f"W{i}"]
			b = self.parameters[f"b{i}"]
			Z = np.dot(W, A) + b
			A = self.relu(Z)

		# Letzter Forwardpropagationsschritt mit Softmax, um eine Wahrscheinlichkeit für eine bestimmte Zahl zu geben
		W = self.parameters[f"W{num_layers}"]
		b = self.parameters[f"b{num_layers}"]
		Z = np.dot(W, A) + b
		A = self.softmax(Z)

		return A

	def compute_loss(self, A, Y):
		# Verlust berechnen mit der Cross Entropy Loss Funktion
		m = Y.shape[1]  # Anzahl Spalten im Ergebnisvektor Y
		return -np.sum(Y * np.log(A + 1e-8)) / m

	def backward(self, X, Y):
		grads = {}
		A = X
		caches = {"A0": X}
		num_layers = len(self.parameters) // 2

		# Forwardpropagation speichern
		for i in range(1, num_layers + 1):
			W = self.parameters[f"W{i}"]
			b = self.parameters[f"b{i}"]
			Z = np.dot(W, A) + b
			A = self.relu(Z) if i != num_layers else self.softmax(Z)
			caches[f"Z{i}"] = Z
			caches[f"A{i}"] = A

		# Initialisierung: Ableitung vom Loss bezüglich letztem Output
		dZ = caches[f"A{num_layers}"] - Y  # Softmax + CrossEntropy zusammen

		# Backpropagation rückwärts durch die Schichten
		for i in reversed(range(1, num_layers + 1)):
			A_prev = caches[f"A{i - 1}"]
			W = self.parameters[f"W{i}"]

			# Ableitung dW mit Kettenregel
			grads[f"dW{i}"] = np.dot(dZ, A_prev.T)  # Ableitung dL/dW
			grads[f"db{i}"] = np.sum(dZ, axis=1,
									 keepdims=True)  # Ableitung db, Summe der Fehler auf der gleichen Spalte
			for key in grads:
				grads[key] = np.clip(grads[key], -1.0, 1.0)
			if i > 1:
				dA_prev = np.dot(W.T, dZ)
				# ReLU Ableitung korrekt anwenden (nur da wo Z > 0)
				dZ = dA_prev * (caches[f"Z{i - 1}"] > 0)

		self.grads = grads

	def update_parameters(self, learning_rate=0.01):
		num_layers = len(self.parameters) // 2
		# Zu jedem Weight und Bias die in der Backpropagation berechneten Ableitungen mal einem Lernrate subtrahieren
		for i in range(1, num_layers + 1):
			self.parameters[f"W{i}"] -= learning_rate * self.grads[f"dW{i}"]
			self.parameters[f"b{i}"] -= learning_rate * self.grads[f"db{i}"]



	def train(self, X, Y, epochs=100, learning_rate=0.01, print_loss=False, step_decay_parameter=1.68):
		step_decay_learnrate = learning_rate
		for epoch in range(epochs):
			# Shuffle der Daten zu Beginn jedes Epochs
			perm = np.random.permutation(X.shape[1])
			X = X[:, perm]
			Y = Y[:, perm]

			A = self.forward(X)
			loss = self.compute_loss(A, Y)
			self.backward(X, Y)
			self.update_parameters(step_decay_learnrate)
			if print_loss and epoch % 5 == 0:
				predictions = np.argmax(A, axis=0)
				labels = np.argmax(Y, axis=0)
				acc = np.mean(predictions == labels)
				print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc:.2%}")
			if epoch % 10 == 0 and epoch != 0:
				step_decay_learnrate /= step_decay_parameter



