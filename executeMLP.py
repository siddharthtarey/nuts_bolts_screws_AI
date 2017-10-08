"""
CSCI630 - Foundations of Intelligent Systems
Authors: Swapnil Kamat (snk6855@rit.edu)
         Pavan Bhat (pxb8715.rit.edu)
         Siddharth Tarey (st2476@rit.edu)
"""
import csv
import math
import matplotlib.pyplot as mp
import matplotlib.patches as patch


class network:

    __slots__ = 'numHiddenNodes', 'inputNodes', 'outputNodes', 'hiddenNodes', 'biasNodes', 'v', 'layer', 'weights'

    def __init__(self, numHiddenNodes, filename):
        '''
        Constructor to initialize the neural net
        :param numHiddenNodes:
        '''
        self.inputNodes = []
        self.outputNodes = [1, 2, 3, 4]
        self.hiddenNodes = []
        self.biasNodes = []
        self.v = [0 for i in range(numHiddenNodes + 9)]
        self.layer = [[] for i in range(3)]
        self.weights = []
        self.setup(numHiddenNodes)
        self.initWeights(filename)

    def setup(self, numHiddenNodes):
        '''
        Set up the neural net architecture
        :param numHiddenNodes: number of hidden layer nodes
        :return:
        '''
        # set up node numbers in the layers

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

    def initWeights(self, filename):
        '''
        Initialize the weights
        :param filename: file to read the weights from
        :return:
        '''
        file = open(filename)
        dataset = csv.reader(file)

        for line in dataset:
            a = []
            for j in line:
                a.append(float(j))
            self.weights.append(a)

def test(weights, obj):
    '''
    Test function to iterate over the test samples and classify
    :param weights: weights in the neural net
    :param obj:
    :return:
    '''
    name = 'train_data.csv'
    file = open(name)
    data_set = csv.reader(file)

    # input attributes
    x = []

    # output vector
    y = []

    # confusion matrix
    matrix = [[0 for i in range(4)] for j in range(4)]

    class1x1 = []
    class1x2 = []
    class2x1 = []
    class2x2 = []
    class3x1 = []
    class3x2 = []
    class4x1 = []
    class4x2 = []
    bbox1x1 = []
    bbox1x2 = []
    bbox2x1 = []
    bbox2x2 = []
    bbox3x1 = []
    bbox3x2 = []
    bbox4x1 = []
    bbox4x2 = []

    # read the rows in dataset and store the values of attributes in arrays
    for line in data_set:
        if line != []:
            x.append([float(line[0]), float(line[1])])
            y.append(int(line[2]))
            if (int(line[2]) == 1):
                class1x1.append(float(line[0]))
                class1x2.append(float(line[1]))
            if (int(line[2]) == 2):
                class2x1.append(float(line[0]))
                class2x2.append(float(line[1]))
            if (int(line[2]) == 3):
                class3x1.append(float(line[0]))
                class3x2.append(float(line[1]))
            if (int(line[2]) == 4):
                class4x1.append(float(line[0]))
                class4x2.append(float(line[1]))

    classified = []

    # test the inputs
    for i in range(len(x)):

        maxval = 0
        # Propagate the input forward to compute the outputs
        for inp, index in zip(obj.inputNodes, range(2)):
            obj.v[inp] = x[i][index]

        for l in range(1, len(obj.layer)):
            for j in obj.layer[l]:
                if j not in obj.biasNodes:
                    sum_in = 0.0
                    for k in obj.layer[l - 1]:
                        sum_in += (weights[k][j] * obj.v[k])
                    obj.v[j] = hypothesis(sum_in)

        # assign the class to the test samples
        for k in obj.outputNodes:
            if maxval < obj.v[k]:
                maxval = obj.v[k]
                classifier = k
        classified.append(classifier)

    # Calculate the confusion matrix and compute the overall profit
    sum = 0
    profit = 0
    cost = [[20, -7, -7, -7], [-7, 15, -7, -7], [-7, -7, 5, -7], [-3, -3, -3, -3]]
    for i, j in zip(classified, range(0, len(y))):
        matrix[i-1][y[j]-1] += 1
        profit += cost[i-1][y[j]-1]
        if i-1 == y[j]-1:
            sum += 1

    print('Recognition rate (% correct) = ', ((sum/len(y))*100))
    print('Profit obtained = $', round(profit*0.01, 2))
    print('Confusion Matrix: ')
    for i in range(4):
        for j in range(4):
            print(matrix[i][j], ' ', end="")
        print()

    fig = mp.figure()
    ax = fig.add_subplot(111, aspect='equal')
    mp.scatter(class1x1, class1x2, color='green')
    mp.scatter(class2x1, class2x2, color='red')
    mp.scatter(class3x1, class3x2, color='blue')
    mp.scatter(class4x1, class4x2, color='yellow')

    xmin1 = 999
    xmin2 = 999
    xmin3 = 999
    xmin4 = 999
    xmin5 = 999
    xmin6 = 999
    xmin7 = 999
    xmin8 = 999

    xmax1 = 0
    xmax2 = 0
    xmax3 = 0
    xmax4 = 0
    xmax5 = 0
    xmax6 = 0
    xmax7 = 0
    xmax8 = 0
    for j in range(len(classified)):
        if (int(classified[j]) == 1):
            bbox1x1.append([x[j][0], x[j][1]])
        if (int(classified[j]) == 2):
            bbox2x1.append([x[j][0], x[j][1]])
        if (int(classified[j]) == 3):
            bbox3x1.append([x[j][0], x[j][1]])
        if (int(classified[j]) == 4):
            bbox4x1.append([x[j][0], x[j][1]])

    for i in range(len(bbox1x1)):
        if (bbox1x1[i][0] < xmin1):
            xmin1 = bbox1x1[i][0]
            xmin2 = bbox1x1[i][1]
        if (bbox1x1[i][0] > xmax1):
            xmax1 = bbox1x1[i][0]
            xmax2 = bbox1x1[i][1]

    for i in range(len(bbox2x1)):
        if (bbox2x1[i][0] < xmin3 and bbox2x1[i][1] < xmin4):
            xmin3 = bbox2x1[i][0]
            xmin4 = bbox2x1[i][1]
        if (bbox2x1[i][0] > xmax3 and bbox2x1[i][1] > xmax4):
            xmax3 = bbox2x1[i][0]
            xmax4 = bbox2x1[i][1]

    for i in range(len(bbox3x1)):
        if (bbox3x1[i][0] < xmin5 and bbox3x1[i][1] < xmin6):
            xmin5 = bbox3x1[i][0]
            xmin6 = bbox3x1[i][1]
        if (bbox3x1[i][0] > xmax5 and bbox3x1[i][1] > xmax6):
            xmax5 = bbox3x1[i][0]
            xmax6 = bbox3x1[i][1]

    for i in range(len(bbox4x1)):
        if (bbox4x1[i][0] < xmin7 and bbox4x1[i][1] < xmin8):
            xmin7 = bbox4x1[i][0]
            xmin8 = bbox4x1[i][1]
        if (bbox4x1[i][0] > xmax7 and bbox3x1[i][1] > xmax8):
            xmax7 = bbox4x1[i][0]
            xmax8 = bbox4x1[i][1]

    rect1 = patch.Rectangle((xmin1, xmin2), (xmax1 - xmin1), (xmax2 - xmin2), fill=False, edgecolor='green')
    rect2 = patch.Rectangle((xmin3, xmin4), (xmax3 - xmin3), (xmax4 - xmin4), fill=False, edgecolor='red')
    rect3 = patch.Rectangle((xmin5, xmin6), (xmax5 - xmin5), (xmax6 - xmin6), fill=False, edgecolor='blue')
    rect4 = patch.Rectangle((xmin7, xmin8), (xmax7 - xmin7), (xmax8 - xmin8), fill=False, edgecolor='yellow')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)
    mp.show()


def hypothesis(sum_in):
    '''
    Sigmoid activation function
    :param attr1: first attribute
    :param attr2: second attribute
    :param weights: weights associated with the attributes and bias
    :return: the current value of hypothesis calculated
    '''
    return 1 / (1 + math.exp(-sum_in))


def main():
    '''
    main method
    :return:
    '''
    numHiddenNodes = int(input("Enter number of hidden layer nodes: (ideally = 5) "))

    for i in [0, 10, 100, 1000, 10000]:
        name = 'weights'+str(i)+'.csv'
        print('\n\n ---- Epochs = ', i, ' ----')
        n = network(numHiddenNodes, name)
        test(n.weights, n)


main()
