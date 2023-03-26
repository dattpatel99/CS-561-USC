import numpy as np
from matplotlib import pyplot as plt

EPOCH = 1500
LEARNING_RATE = 0.003
BATCH_SIZE = 100
LAMBD = 0.02

NN_ARCH = [
    {"input": 4, "output": 20,"activation": "relu"},
    {"input": 20, "output": 10, "activation": "relu"},
    {"input": 10, "output": 1, "activation": "sigmoid"},
]

np.random.seed(20)

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
   return 1/(1 + np.exp(-x))

def sigmoid_back(dA, x):
    sig = sigmoid(x)
    return  dA * sig * (1 - sig)

def relu_back(dA, x):
    dX = np.array(dA, copy = True)
    dX[x <= 0] = 0
    return dX

class Layer:
    def __init__(self):
        self.weights = []
        self.bias = []
        self.activation = None
        self.der_activation = None
    
    def _set_activation(self, s):
        if s == "relu":
            self.activation = relu
            self.der_activation = relu_back
        elif s == 'sigmoid':
            self.activation = sigmoid
            self.der_activation = sigmoid_back

    def set_layer_data(self, data):
        num_input = data["input"]
        num_output = data['output']
        self._set_activation(data["activation"])
        self.weights = np.random.randn(num_output, num_input) * 0.01
        self.bias = np.random.randn(num_output, 1) * 0.01
        
class Network:
    def __init__(self):
        self.epochs = EPOCH
        self.lr = LEARNING_RATE
        self.bs = BATCH_SIZE
        self.layers = {}
        self.forward_memory = {}
        self.grads = {}
    
    def create_layers(self):
        for id, layer in enumerate(NN_ARCH):
            layer_data = Layer()
            layer_data.set_layer_data(layer)
            self.layers[id] = layer_data
    
    def cost_function(self, pred, real):
        n = real.shape[1]
        logProbs = np.multiply(real, np.log(pred)) + np.multiply(1-real, np.log(1-pred))
        cost = (-1.0) * np.sum(logProbs)
        regularization = (np.sum(np.square(self.layers[0].weights)) + np.sum(np.square(self.layers[1].weights)) + np.sum(np.square(self.layers[2].weights))) * (LAMBD/(2*n))
        cost = cost + regularization
        cost = np.mean(np.squeeze(cost))
        return cost
    
    def convert_prob_into_class(self, probs):
        probs_ = np.copy(probs)
        probs_[probs_ > 0.5] = 1
        probs_[probs_ <= 0.5] = 0
        return probs_

    def get_accuracy_value(self, pred, real):
        pred_ = self.convert_prob_into_class(pred)
        return (pred_ == real).all(axis=0).mean()

    # Propogate through network once
    def forward_prop(self, X):
        data = X.T
        for key in self.layers:
            cur_layer = self.layers[key]
            self.forward_memory["D"+str(key)] = data # Data we input
            self.forward_memory["F"+str(key)] = np.dot(cur_layer.weights, data) + cur_layer.bias # Data before Activiation
            self.forward_memory["A"+str(key)] = cur_layer.activation(self.forward_memory["F"+str(key)]) # Data After Activiation
            data = self.forward_memory["A"+str(key)]
        return self.forward_memory["A"+str(list(self.layers.keys())[-1])]

    # Propogate through network once
    def forward_prop_test(self, X):
        data = X.T
        forward_memory = {}
        for key in self.layers:
            cur_layer = self.layers[key]
            forward_memory["D"+str(key)] = data # Data we input
            forward_memory["F"+str(key)] = np.dot(cur_layer.weights, data) + cur_layer.bias # Data before Activiation
            forward_memory["A"+str(key)] = cur_layer.activation(forward_memory["F"+str(key)]) # Data After Activiation
            data = forward_memory["A"+str(key)]
        return forward_memory["A"+str(list(self.layers.keys())[-1])]

    def backward_prop(self,pred,real):
        n = real.shape[1]
        L = len(self.layers)
        real = real.reshape(pred.shape)

        # Getting delta E and trying to remove the 0 division error
        delta_E = (pred-real)

        # backward prop
        for i in reversed(range(L)):
            cur_layer = self.layers[i]
            dZ = cur_layer.der_activation(delta_E, self.forward_memory['A'+str(i)])
            dW = (1/n) * (np.dot(dZ, self.forward_memory['D'+str(i)].T) + (LAMBD)*cur_layer.weights)
            db = (1/n) * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(cur_layer.weights.T,dZ)

            self.grads["dA"+str(i)],self.grads["dW"+str(i+1)],self.grads["db"+str(i+1)] = dA_prev,dW,db
            delta_E = self.grads["dA" + str(i)]

    def update(self):
        for i in range(len(self.layers)):
            cur_layer = self.layers[i]
            cur_layer.weights = cur_layer.weights - (self.lr * self.grads["dW" + str(i+1)]) 
            cur_layer.bias = cur_layer.bias - (self.lr * self.grads["db" + str(i+1)])

    def create_mini_batch(self,data, labels):
        batch = np.random.choice(data.shape[0], size=self.bs, replace=False)
        t_data, t_labels =  data[batch], labels[batch]
        return t_data, t_labels

    def train(self, data, labels, test_x, test_y, dataset):
        self.create_layers()
        cost_history = []
        accuracy_history = {}
        test_acc = []
        best_finder = {}
        num_batches = data.shape[0] // self.bs
        for e in range(num_batches):
            t_data, t_labels = self.create_mini_batch(data, labels)
            for i in range(self.epochs):
                pred = self.forward_prop(t_data)
                cost = self.cost_function(pred, t_labels.T)
                acc = self.get_accuracy_value(pred, t_labels.T)
                test_pred = self.forward_prop_test(test_x)
                test_acc_single = self.get_accuracy_value(test_pred, test_y.T)
                self.backward_prop(pred,t_labels)
                self.update()
                cost_history.append(cost)
                accuracy_history[acc] = self.layers
                test_acc.append(test_acc_single)
                best_finder[acc] = self.layers
        return cost_history, accuracy_history, test_acc, best_finder

def modify_coords(coords):
    sin_coords = np.sin(coords)
    coords = np.concatenate((coords, sin_coords), axis=1)
    return coords

def read_data(train_coords, train_label):
    coords, labels = None, None
    coords = np.genfromtxt(train_coords,delimiter=',')
    coords = modify_coords(coords)
    labels = np.genfromtxt(train_label, delimiter=',')
    labels = labels.reshape(-1,1)
    return coords, labels

files = [
['xor_train_data.csv','xor_train_label.csv','xor_test_data.csv','xor_test_label.csv'],
["circle_train_data.csv", "circle_train_label.csv", "circle_test_data.csv", "circle_test_label.csv"], 
["gaussian_train_data.csv", "gaussian_train_label.csv", "gaussian_test_data.csv", "gaussian_test_label.csv"], 
["spiral_train_data.csv",'spiral_train_label.csv','spiral_test_data.csv','spiral_test_label.csv'], 
]

def run_local():
    train_costs= []
    train_accuracies = []
    overall_test_acc = []
    final_test = []
    for i in range(len(files)):
        X, Y = read_data(files[i][0], files[i][1])
        test_X, test_Y = read_data(files[i][2], files[i][3])
        nn = Network()
        cost_history, acc_history, test_acc, best_finder = nn.train(X,Y, test_X, test_Y, files[i][0])
        train_costs.append(cost_history)
        train_accuracies.append(list(acc_history.keys()))
        test_result = nn.forward_prop(test_X)
        acc = nn.get_accuracy_value(test_result, test_Y.T)
        test_acc.append(acc)
        final_test.append(acc)
        overall_test_acc.append(test_acc)

    for each in final_test:
        print(each)

    # Initialise the subplot function using number of rows and columns
    figure, axis = plt.subplots(4, 2)
    axis[0, 0].plot(train_costs[0])
    axis[1, 0].set_title("XOR Cost")
    axis[0, 1].plot(overall_test_acc[0])
    axis[1, 1].set_title("XOR Accuracy")

    axis[1, 0].plot(train_costs[1])
    axis[3, 0].set_title("Circle Cost")
    axis[1, 1].plot(overall_test_acc[1])
    axis[3, 1].set_title("Circle Accuracy")

    axis[2, 0].plot(train_costs[2])
    axis[0, 0].set_title("Spiral Cost")
    axis[2, 1].plot(overall_test_acc[2])
    axis[0, 1].set_title("Spiral Accuracy")

    axis[3, 0].plot(train_costs[3])
    axis[2, 0].set_title("Gaussian Cost")
    axis[3, 1].plot(overall_test_acc[3])
    axis[2, 1].set_title("Gaussian Accuracy")
    plt.show()

run_local()