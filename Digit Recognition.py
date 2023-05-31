import numpy as np
import random
import Mnist_Loader as ml

class QuadCost():

    @staticmethod
    def fn(a,y):

        return 0.5*((a-y)**2)
    
    @staticmethod
    def deri(a,y,z):

        return (a-y) * sigmoid_prime(z)
    

class crossEntropy():

    @staticmethod
    def fn(a,y):

        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    @staticmethod
    def deri(a,y,z):

        return a-y


class Network():

    def __init__(self, layers, cost):
        
        self.layers = layers
        self.costf = cost
        self.t_data_size = 0

        ## biases setup, from layer2 to layer L
        self.biases = [np.random.randn(ls,1) for ls in layers[1:]]

        ## weight setup from layer1~2 to layer L-1 ~ L
        ## self.weight = [np.random.randn(layers[x+1],layers[x]) for x in range(len(layers)-1)] : The old way to initialize weights
        self.weight = [np.random.randn(layers[x+1],layers[x])/np.sqrt(layers[x]) for x in range(len(layers)-1)]

    def print_arg(self):
        print(self.biases)
        print(self.weight)

    def train(self, training_data, epochs, eta, mb_size, lmbda, test_data = None):

        #mb_size means mini_batches_size

        self.t_data_size = len(training_data)

        for i in range(epochs):

            #distributes training_data into mbs
            random.shuffle(training_data)
            mbs = [training_data[k:k+mb_size] for k in range(0,len(training_data),mb_size)]

            #SGD
            for mb in mbs:
                self.SGD(mb,eta,lmbda)

            #Evaluate if test_data exists, and print out the accuarcy of the current model
            if(test_data):
                count = self.evaluate(test_data)
                print(f'epoch {i+1} result: {count}/{len(test_data)}, accuracy: {round(count/len(test_data)*100,2)}%')
            else:
                print(f'epoch {i+1} training finished')

    def SGD(self, mb, eta, lmbda):
        
        mb_size = len(mb)

        #initialize gradient
        partial_b = [np.zeros(b.shape) for b in self.biases]
        partial_w = [np.zeros(w.shape) for w in self.weight]

        #sum up gradients from every training data
        for x,y in mb:
            delta_pb,delta_pw = self.BP(x,y)
            for i in range(len(self.layers)-1):
                partial_b[i] += delta_pb[i]
                partial_w[i] += delta_pw[i]

        #update weight and biases
        for i in range(len(self.layers)-1):
            self.biases[i] -= eta/mb_size*partial_b[i]
            self.weight[i] *= (1 - eta*lmbda/self.t_data_size)
            self.weight[i] -= eta/mb_size*partial_w[i]

        return
            

        


    def BP(self,x,y):
        
        n_layer = len(self.layers)
        
        #initialize gradient
        partial_b = [np.zeros(b.shape) for b in self.biases]
        partial_w = [np.zeros(w.shape) for w in self.weight]


        #cache for later caculation
        activation = x
        activations = [x]

        zs = []

        #FP
        for w,b in zip(self.weight, self.biases):

            z = np.dot(w,activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #BP
        error =  self.costf.deri(activations[-1], y, zs[-1]) #formula 1
        partial_b[-1] = error #formula 3
        partial_w[-1] = np.dot(error, activations[-2].transpose()) #formula 4

        #from -2 to -n_layer+1
        #formula 2,3,4
        for i in range(2,n_layer):
            error = np.dot(self.weight[-i+1].transpose(), error) * sigmoid_prime(zs[-i])
            partial_b[-i] = error
            partial_w[-i] = np.dot(error, activations[-i-1].transpose())

        return (partial_b, partial_w)


    def feedForward(self, input):
        a = input
        for w,b in zip(self.weight, self.biases):

            a = sigmoid(np.dot(w,a) + b)

        return a
    
    def evaluate(self, test_data):

        count = 0

        for x,y in test_data:

            if (np.argmax(self.feedForward(x)) == y):
                count+=1 

        return count
    
#functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


network = Network([784,30,30,10], cost=crossEntropy)
#network.print_arg()

training_data, validation_data, test_data = ml.load_data_wrapper()
network.train(
    training_data, 
    epochs=30, 
    eta=0.1, 
    mb_size=10, 
    lmbda=5.0, 
    test_data=test_data)


