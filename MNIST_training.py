import numpy as np
from neural_network import NeuralNetwork

# --- Load MNIST dataset from .npz ---
data = np.load("mnist.npz")
X_train = data["x_train"].reshape(60000, 784).T / 255.0  # Normalize to [0,1], shape: (784, 60000)
Y_train_labels = data["y_train"]

X_test = data["x_test"].reshape(10000, 784).T / 255.0
Y_test_labels = data["y_test"]

# --- One-hot encode labels ---
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y].T

Y_train = one_hot(Y_train_labels)
Y_test = one_hot(Y_test_labels)

# --- Shuffle training data ---
perm = np.random.permutation(X_train.shape[1])
X_train = X_train[:, perm]
Y_train = Y_train[:, perm]
Y_train_labels = Y_train_labels[perm]

# --- Create a simple and trainable network ---
model = NeuralNetwork([784, 128, 10])
model.load("network.npz")
model.train(X_train[:, :60000], Y_train[:, :60000], epochs=20, learning_rate=0.001, print_loss=True)

perm_test = np.random.permutation(X_test.shape[1])

# Apply the same permutation to both X_test and Y_test_labels
X_test = X_test[:, perm_test]
Y_test_labels = Y_test_labels[perm_test]

# --- Test 10 predictions ---
score = 0
print("\nTesting 10 predictions:")
for i in range(10000):
    x = X_test[:, i].reshape(-1, 1)
    output = model.forward(x)
    predicted = np.argmax(output)
    actual = Y_test_labels[i]
    if predicted == actual:
        score += 1
print(score)
model.save("network.npz")