import numpy
import scipy.special

##################
# Neural Network Class
##################
class NeuralNetwork:    
    def __init__(
      self,
      input_nodes,
      hidden_nodes,
      output_nodes,
      learning_rate
    ):
        # number of nodes in each layer
        self.inodes = input_nodes;
        self.hnodes = hidden_nodes;
        self.onodes = output_nodes;

        # link weight matrices: wih, who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21 w31
        # w12 w22 w32
        # w13 w23 w33
        # self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        # self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        
        # learning rate
        self.lr = learning_rate;

        # activation function (sigmoid function)
        self.activation_function = lambda x: scipy.special.expit(x)

        pass
    
    def train(self, inputs_list, targets_list):
        # convert inputs to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass
    
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs);
        # calculate the signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs);

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs);
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs);

        return final_outputs
