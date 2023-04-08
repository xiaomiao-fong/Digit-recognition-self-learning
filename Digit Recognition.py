import numpy
import random
import Mnist_Loader as ml


class Network():

    def __init__(self, layers):
        
        self.layers = layers

        ##biases setup, from layer2 to layer L
        self.biases = [numpy.random.randn(ls,1) for ls in layers[1:]]

        ##weight setup from layer1~2 to layer L-1 ~ L
        self.weight = [numpy.random.randn(layers[x+1],layers[x]) for x in range(len(layers)-1)]

    def print_arg(self):
        print(self.biases)
        print(self.weight)

    def train(self, training_data, epochs, eta, mb_size, test_data = None):

        #mb_size means mini_batches_size

        for i in range(epochs):

            #distributes training_data into mbs
            random.shuffle(training_data)
            mbs = [training_data[k:k+mb_size] for k in range(0,len(training_data),mb_size)]

            #SGD
            for mb in mbs:
                self.SGD(mb,eta)

            #Evaluate if test_data exists, and print out the accuarcy of the current model
            if(test_data):
                count = self.evaluate(test_data)
                print(f'epoch {i+1} result: {count}/{len(test_data)}, accuracy: {round(count/len(test_data)*100,2)}%')
            else:
                print(f'epoch {i+1} training finished')

    def SGD(self, mb, eta):
        
        mb_size = len(mb)

        #initialize gradient
        partial_b = [numpy.zeros(b.shape) for b in self.biases]
        partial_w = [numpy.zeros(w.shape) for w in self.weight]

        #sum up gradients from every training data
        for x,y in mb:
            delta_pb,delta_pw = self.BP(x,y)
            for i in range(len(self.layers)-1):
                partial_b[i] += delta_pb[i]
                partial_w[i] += delta_pw[i]

        #update weight and biases
        for i in range(len(self.layers)-1):
            self.biases[i] -= eta/mb_size*partial_b[i]
            self.weight[i] -= eta/mb_size*partial_w[i]

        return
            

        


    def BP(self,x,y):
        
        n_layer = len(self.layers)
        
        #initialize gradient
        partial_b = [numpy.zeros(b.shape) for b in self.biases]
        partial_w = [numpy.zeros(w.shape) for w in self.weight]


        #cache for later caculation
        activation = x
        activations = [x]

        zs = []

        #FP
        for w,b in zip(self.weight, self.biases):

            z = numpy.dot(w,activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #BP
        error =  (activations[-1]-y)*sigmoid_prime(zs[-1]) #formula 1
        partial_b[-1] = error #formula 3
        partial_w[-1] = numpy.dot(error, activations[-2].transpose()) #formula 4

        #from -2 to -n_layer+1
        #formula 2,3,4
        for i in range(2,n_layer):
            error = numpy.dot(self.weight[-i+1].transpose(), error) * sigmoid_prime(zs[-i])
            partial_b[-i] = error
            partial_w[-i] = numpy.dot(error, activations[-i-1].transpose())

        return (partial_b, partial_w)


    def feedForward(self, input):
        a = input
        for w,b in zip(self.weight, self.biases):

            a = sigmoid(numpy.dot(w,a) + b)

        return a
    
    def evaluate(self, test_data):

        count = 0

        for x,y in test_data:

            if (numpy.argmax(self.feedForward(x)) == y):
                count+=1 

        return count
    
#functions
def sigmoid(z):
    return 1.0/(1.0+numpy.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


network = Network([784,30,10])
#network.print_arg()

training_data, validation_data, test_data = ml.load_data_wrapper()
network.train(training_data, 30, 10.0, 10, test_data=test_data)


