"""
CSCI630 - Foundations of Intelligent Systems
Authors: Swapnil Kamat (snk6855@rit.edu)
         Pavan Bhat (pxb8715.rit.edu)
         Siddharth Tarey (st2476@rit.edu)
"""
import csv
import matplotlib.pyplot as mp
import matplotlib.patches as patch
from matplotlib.path import Path

def main():
    '''
    main method
    :return:
    '''
    # load decision tree generated before pruning
    print('\nBefore pruning -----------')
    tree = loadTree('tree.csv')
    executeDecisionTree(tree)

    # load decision tree generated after Chi Squared pruning
    print('\nAfter Chi Squared pruning -----------')
    prunedtree = loadTree('prunedtree.csv')
    executeDecisionTree(prunedtree)


def executeDecisionTree(tree):
    '''
    Execute the decision tree for each sample in the test file
    :param tree:
    :return:
    '''
    test = open('test_data.csv')
    data = csv.reader(test)
    #classification lists
    class1x1 = []
    class1x2 = []
    class2x1 = []
    class2x2 = []
    class3x1 = []
    class3x2 = []
    class4x1 = []
    class4x2 = []
    # c0-ordinates to generate the bounding boxes
    bbox1x1 = []
    bbox1x2 = []
    bbox2x1 = []
    bbox2x2 = []
    bbox3x1 = []
    bbox3x2 = []
    bbox4x1 = []
    bbox4x2 = []
    x = []
    classified = []

    # load the samples from the test data
    for sample in data:
        if sample != []:
            x.append([float(sample[0]), float(sample[1]), int(sample[2])])
            if (int(sample[2]) == 1):
                class1x1.append(float(sample[0]))
                class1x2.append(float(sample[1]))
            if (int(sample[2]) == 2):
                class2x1.append(float(sample[0]))
                class2x2.append(float(sample[1]))
            if (int(sample[2]) == 3):
                class3x1.append(float(sample[0]))
                class3x2.append(float(sample[1]))
            if (int(sample[2]) == 4):
                class4x1.append(float(sample[0]))
                class4x2.append(float(sample[1]))

    # traverse the decision tree for each sample
    for j in x:
        node = tree[0]
        while True:
            if len(node) == 2:
                # if leaf node is reached mark the sample according to the class of the leaf node
                classified.append(node[1])
                if (int(node[1]) == 1):
                    bbox1x1.append([float(j[0]), float(j[1])])
                if (int(node[1]) == 2):
                    bbox2x1.append([float(j[0]), float(j[1])])
                if (int(node[1]) == 3):
                    bbox3x1.append([float(j[0]), float(j[1])])
                if (int(node[1]) == 4):
                    bbox4x1.append([float(j[0]), float(j[1])])
                break
            else:
                if j[node[2]] <= node[1]:
                    node = tree[node[3]]
                else:
                    node = tree[node[4]]

    # confusion matrix
    matrix = [[0 for i in range(4)] for j in range(4)]

    # Calculate the confusion matrix and compute the overall profit
    sum = 0
    profit = 0
    cost = [[20, -7, -7, -7], [-7, 15, -7, -7], [-7, -7, 5, -7], [-3, -3, -3, -3]]
    for i, j in zip(classified, range(0, len(x))):
        matrix[i - 1][x[j][2] - 1] += 1
        profit += cost[i - 1][x[j][2] - 1]
        if i - 1 == x[j][2] - 1:
            sum += 1

    # print the command line statistics
    print('Recognition rate (% correct) = ', ((sum / len(x)) * 100))
    print('Profit obtained = $', round(profit * 0.01, 2))
    print('Confusion Matrix: ')
    print('\tActual -->\t\tBolt\tNut\t\tRing\tScrap')
    for i in range(4):
        if i == 0:
            print('Assigned as bolt  : ', end="")
        elif i == 1:
            print('Assigned as nut   : ', end="")
        elif i == 2:
            print('Assigned as ring  : ', end="")
        elif i == 3:
            print('Assigned as scrap : ', end="")
        for j in range(4):
            print(matrix[i][j], ' \t\t', end="")
        print()
    fig = mp.figure()
    ax = fig.add_subplot(111, aspect='equal')
    s1 = mp.scatter(class1x1, class1x2, color='green')
    #scatter plot the test data
    mp.scatter(class2x1, class2x2, color='red')
    mp.scatter(class3x1, class3x2, color='blue')
    mp.scatter(class4x1, class4x2, color='yellow')
    #define co-ordinates for bounding boxes
    xmin1 = 999
    xmin2 = 999
    ymin1 = 999
    ymin2 = 999

    xmin3 = 999
    xmin4 = 999
    ymin3 = 999
    ymin4 = 999

    xmin5 = 999
    xmin6 = 999
    ymin5 = 999
    ymin6 = 999

    xmin7 = 999
    xmin8 = 999
    ymin7 = 999
    ymin8 = 999

    xmax1 = 0
    xmax2 = 0
    ymax1 = 0
    ymax2 = 0

    xmax3 = 0
    xmax4 = 0
    ymax3 = 0
    ymax4 = 0

    xmax5 = 0
    xmax6 = 0
    ymax5 = 0
    ymax6 = 0

    xmax7 = 0
    xmax8 = 0
    ymax7 = 0
    ymax8 = 0

    # find dimesions for the bounding boxes
    for i in range(len(bbox1x1)):
        if (bbox1x1[i][0] < xmin1):
            xmin1 = bbox1x1[i][0]
            xmin2 = bbox1x1[i][1]
        if (bbox1x1[i][0] > xmax1):
            xmax1 = bbox1x1[i][0]
            xmax2 = bbox1x1[i][1]
        if(bbox1x1[i][1] < ymin1):
            ymin1 = bbox1x1[i][0]
            ymin2 = bbox1x1[i][1]
        if (bbox1x1[i][1] > ymax1):
            ymax1 = bbox1x1[i][0]
            ymax2 = bbox1x1[i][1]
    # find dimesions for the bounding boxes
    for i in range(len(bbox2x1)):
        if (bbox2x1[i][0] < xmin3):
            xmin3 = bbox2x1[i][0]
            xmin4 = bbox2x1[i][1]
        if (bbox2x1[i][0] > xmax3):
            xmax3 = bbox2x1[i][0]
            xmax4 = bbox2x1[i][1]
        if(bbox2x1[i][1] < ymin3):
            ymin3 = bbox2x1[i][0]
            ymin4 = bbox2x1[i][1]
        if (bbox2x1[i][1] > ymax3):
            ymax3 = bbox2x1[i][0]
            ymax4 = bbox2x1[i][1]

    for i in range(len(bbox3x1)):
        if (bbox3x1[i][0] < xmin5):
            xmin5 = bbox3x1[i][0]
            xmin6 = bbox3x1[i][1]
        if (bbox3x1[i][0] > xmax5):
            xmax5 = bbox3x1[i][0]
            xmax6 = bbox3x1[i][1]
        if(bbox3x1[i][1] < ymin5):
            ymin5 = bbox3x1[i][0]
            ymin6 = bbox3x1[i][1]
        if (bbox3x1[i][1] > ymax5):
            ymax5 = bbox3x1[i][0]
            ymax6 = bbox3x1[i][1]

    for i in range(len(bbox4x1)):
        if (bbox4x1[i][0] < xmin7):
            xmin7 = bbox4x1[i][0]
            xmin8 = bbox4x1[i][1]
        if (bbox4x1[i][0] > xmax7):
            xmax7 = bbox4x1[i][0]
            xmax8 = bbox4x1[i][1]
        if(bbox4x1[i][1] < ymin7):
            ymin7 = bbox4x1[i][0]
            ymin8 = bbox4x1[i][1]
        if (bbox4x1[i][1] > ymax7):
            ymax7 = bbox4x1[i][0]
            ymax8 = bbox4x1[i][1]
    # define the bounding boxes

    # plot the points and the bounding boxes
    v1 = [(xmin1,xmin2),(xmax1,xmax2),(ymax1,ymax2),(ymin1,ymin2),(xmin1,xmin2)]
    v2 = [(xmin3, xmin4),  (ymax3, ymax4),(xmax3, xmax4), (ymin4, ymin4), (xmin3, xmin4)]
    v3 = [(xmin5, xmin6), (ymax5, ymax6),(xmax5, xmax6),  (ymin5, ymin6), (xmin5, xmin6)]
    v4 = [(xmin7, xmin8),  (ymax7, ymax8),(xmax7, xmax8), (ymin7, ymin8), (xmax7, xmax8)]
    path1 = Path(v1)
    path2 = Path(v2)
    path3 = Path(v3)
    path4 = Path(v4)
    patch1 = patch.PathPatch(path1, edgecolor='green', lw=1,fill=False )
    patch2 = patch.PathPatch(path2, edgecolor='red', lw=1, fill=False)
    patch3 = patch.PathPatch(path3, edgecolor='blue', lw=1, fill=False)
    patch4 = patch.PathPatch(path4, edgecolor='yellow', lw=1, fill=False)
    ax.add_patch(patch1)
    ax.add_patch(patch2)
    ax.add_patch(patch3)
    ax.add_patch(patch4)

    mp.show()


def loadTree(filename):
    '''
    This function loads the decision tree from a file
    :param filename: file name of the decision tree to load
    :return:
    '''
    file = open(filename)
    dataset = csv.reader(file)
    tree = {}

    # read the file and load the decision tree
    for i in dataset:
        if len(i) != 2:
            # for internal nodes
            tree[int(i[0])] = [int(i[0]), float(i[1]), int(i[2]), int(i[3]), int(i[4])]
        else:
            # for leaf nodes
            tree[int(i[0])] = [int(i[0]), int(i[1])]

    return tree


main()
