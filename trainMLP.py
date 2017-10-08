"""
CSCI630 - Foundations of Intelligent Systems
Authors: Swapnil Kamat (snk6855@rit.edu)
         Pavan Bhat (pxb8715.rit.edu)
         Siddharth Tarey (st2476@rit.edu)
"""
import csv
import math
import matplotlib.pyplot as plot_obj
import random


class network:

    __slots__ = 'numHiddenNodes', 'inputNodes', 'outputNodes', 'hiddenNodes', 'biasNodes', 'v', 'layer', 'weights'

    def __init__(self, numHiddenNodes):
        '''

        :param numHiddenNodes:
        '''
        self.inputNodes = []
        self.outputNodes = [1, 2, 3, 4]
        self.hiddenNodes = []
        self.biasNodes = []
        self.layer = [[] for i in range(3)]
        self.v = [0 for i in range(numHiddenNodes + 9)]
        self.weights = [[0 for i in range(numHiddenNodes + 9)] for j in range(numHiddenNodes + 9)]
        self.setup(numHiddenNodes)

    def setup(self, numHiddenNodes):
        '''

        :param numHiddenNodes:
        :return:
        '''
        self.biasNodes.append(5)

        for i in range(numHiddenNodes):
            self.hiddenNodes.append(i + 6)
        self.biasNodes.append(6 + numHiddenNodes)
        self.inputNodes.append(7 + numHiddenNodes)
        self.inputNodes.append(8 + numHiddenNodes)

        # create layers and add nodes to them
        self.layer[0].extend(self.inputNodes)
        self.layer[0].append(self.biasNodes[1])
        self.layer[1].extend(self.hiddenNodes)
        self.layer[1].append(self.biasNodes[0])
        self.layer[2].extend(self.outputNodes)

        # assign values to bias nodes
        self.v[self.biasNodes[0]] = 1
        self.v[self.biasNodes[1]] = 1

        # assign initial weights as small float values in range [-1,1]
        self.assignInitialWeights()

    def assignInitialWeights(self):
        '''

        :return:
        '''
        # assign initial weights
        for i in self.inputNodes:
            for j in self.hiddenNodes:
                self.weights[i][j] = random.uniform(-1.0, 1.0)

        for j in self.hiddenNodes:
            self.weights[self.biasNodes[1]][j] = random.uniform(-1.0, 1.0)

        for i in self.hiddenNodes:
            for j in self.outputNodes:
                self.weights[i][j] = random.uniform(-1.0, 1.0)

        for j in self.outputNodes:
            self.weights[self.biasNodes[0]][j] = random.uniform(-1.0, 1.0)


def main():
    '''
    main method
    :return:
    '''
    name = 'train_data.csv'
    file = open(name)
    data_set = csv.reader(file)

    numHiddenNodes = int(input("Enter number of hidden layer nodes: (ideally = 5) "))
    n = network(numHiddenNodes)

    # input attributes vector
    x = []

    # output vector
    y = []

    # read the rows in dataset and store the values of attributes in arrays
    for line in data_set:
        x.append([float(line[0]), float(line[1])])
        y.append(float(line[2]))

    file.close()

    for i in [0, 10, 100, 1000, 10000]:

        filename = 'weights'+str(i)+'.csv'
        # open a csv file to write the weights
        write_weights = open(filename, "w")

        training(i, n, numHiddenNodes, x, y)

        write_row = csv.writer(write_weights, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

        for j in n.weights:
            write_row.writerow(j)

        write_weights.close()

def training(epochs, network, numHiddenNodes, x, y):
    '''

    :param epochs:
    :param network:
    :param numHiddenNodes:
    :param x:
    :param y:
    :return:
    '''
    # alpha
    alpha = 0.1

    # delta values
    delta = [0 for i in range(numHiddenNodes + 9)]

    # Sum of squared errors
    sse = []

    for epoch in range(epochs):
        # hypothesis
        hyp_in = [0 for i in range(numHiddenNodes + 9)]

        for i in range(len(x)):

            # Propagate the input forward to compute the outputs
            for inp, index in zip(network.inputNodes, range(2)):
                network.v[inp] = x[i][index]

            for l in range(1, len(network.layer)):
                for j in network.layer[l]:
                    if j not in network.biasNodes:
                        sum_in = 0.0
                        for k in network.layer[l-1]:
                            sum_in += (network.weights[k][j] * network.v[k])
                        hyp_in[j] = sum_in
                        network.v[j] = hypothesis(sum_in)

            # Propagate deltas backward from output layer to input layer
            for j in network.layer[len(network.layer)-1]:
                # delta[j] = hypothesis(hyp_in[j]) * (1 - hypothesis(hyp_in[j])) * (j - v[j])
                if y[i] == j:
                    delta[j] = network.v[j] * (1 - network.v[j]) * (1 - network.v[j])
                else:
                    delta[j] = network.v[j] * (1 - network.v[j]) * (0 - network.v[j])

            for l in range(len(network.layer)-2, -1, -1):
                for j in network.layer[l]:
                    if j not in network.biasNodes:
                        sum_err = 0.0
                        for k in network.layer[l+1]:
                            if k not in network.biasNodes:
                                sum_err += network.weights[j][k] * delta[k]
                        # delta[j] = hypothesis(hyp_in[j]) * (1 - hypothesis(hyp_in[j])) * sum_err
                        delta[j] = network.v[j] * (1 - network.v[j]) * sum_err

            # Update every weight in the network using deltas
            for s in network.inputNodes:
                for j in network.hiddenNodes:
                    network.weights[s][j] += (alpha * network.v[s] * delta[j])

            for j in network.hiddenNodes:
                network.weights[network.biasNodes[1]][j] += (alpha * network.v[network.biasNodes[1]] * delta[j])

            for s in network.hiddenNodes:
                for j in network.outputNodes:
                    network.weights[s][j] += (alpha * network.v[s] * delta[j])

            for j in network.outputNodes:
                network.weights[network.biasNodes[0]][j] += (alpha * network.v[network.biasNodes[0]] * delta[j])

        error = 0
        for k in range(1, 5, 1):
            error += math.pow(delta[k], 2)
        sse.append(error)

    display(sse)


def hypothesis(sum_in):
    '''
    Sigmoid activation function
    :param attr1: first attribute
    :param attr2: second attribute
    :param weights: weights associated with the attributes and bias
    :return: the current value of hypothesis calculated
    '''
    return 1 / (1 + math.exp(-sum_in))


def display(sse):
    '''
    Method to display the graph plots
    :param sse: list of sun of squared errors (SSE) for all the epochs
    :return:
    '''
    # plot and display graph for SSE v/s epoch
    plot_obj.plot(sse)
    plot_obj.show()

main()
