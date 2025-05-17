import numpy as np
from neural_network import NeuralNetwork


data = np.load("mnist.npz")
X_train = data["x_train"].reshape(60000, 784).T / 255.0  # (784, 60000)
Y_train_labels = data["y_train"]

X_test = data["x_test"].reshape(10000, 784).T / 255.0
Y_test_labels = data["y_test"]

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y].T

Y_train = one_hot(Y_train_labels)
Y_test = one_hot(Y_test_labels)

perm = np.random.permutation(X_train.shape[1])
X_train = X_train[:, perm]
Y_train = Y_train[:, perm]
Y_train_labels = Y_train_labels[perm]

def train_neural_network(layer_sizes, X_train, Y_train, Y_labels, X_test, Y_test_labels, learning_rate=0.001, epochs=50, train_samples=10000, step_decay_parameter=2):
    model = NeuralNetwork(layer_sizes)

    #Model Trainieren mit Trainingsdaten X_Train und die jeweiligen Lösungen
    model.train(X_train[:, :train_samples], Y_train[:, :train_samples], epochs=epochs, learning_rate=learning_rate, print_loss=False, step_decay_parameter=step_decay_parameter)

    #Test ausführen mit Testdaten X_Test und Y_Test
    score = 0
    for i in range(10000):
        x = X_test[:, i].reshape(-1, 1)
        output = model.forward(x)
        predicted = np.argmax(output)
        actual = Y_test_labels[i]
        if predicted == actual:
            score += 1

    accuracy = score / 10000
    params = model.weights_counter + model.bias_counter
    efficiency = accuracy / params

    entry = (
        f"Architektur: {layer_sizes}\n"
        f"Genauigkeit: {accuracy:.4f}\n"
        f"Anzahl Parameter: {params}\n"
        f"Parametereffizienz: {efficiency:.8e}\n"
        f"Learning Rate: {learning_rate}\n"
        f"Epochs: {epochs}\n"
        f"Train Samples: {train_samples}\n"
        f"step_decay_parameter: {step_decay_parameter}\n"
        f"{'-'*40}\n"
    )

    # Append results to report file
    with open("parameter_report.txt", "a") as f:
        f.write(entry)

    # Save model
    filename = f"network_{'_'.join(map(str, layer_sizes))}_ep{epochs}_lr{learning_rate:.4f}.npz"
    model.save(filename)
    model.save(filename)

#architectures = [
    #[784, 32, 10],
    #[784, 64, 10],
    #[784, 128, 10],
    #[784, 32, 32, 10],
    #[784, 64, 32, 10],
    #[784, 32, 32, 32, 10]
#]


epochs = 100
learning_rate = 0.01
step_decay_parameter = 1.68
#for arch in architectures:
    #train_neural_network(arch, X_train, Y_train, Y_train_labels, X_test, Y_test_labels, epochs=epochs, learning_rate=learning_rate)
#for i in range(10):
    #train_neural_network([784, 32, 10], X_train, Y_train, Y_train_labels, X_test, Y_test_labels, epochs=epochs, learning_rate=learning_rate, step_decay_parameter=step_decay_parameter)
    #step_decay_parameter -= 0.02

train_neural_network([784, 64, 10], X_train, Y_train, Y_train_labels, X_test, Y_test_labels, epochs=epochs, learning_rate=learning_rate, step_decay_parameter=step_decay_parameter)