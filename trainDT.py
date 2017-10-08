"""
CSCI630 - Foundations of Intelligent Systems
Authors: Swapnil Kamat (snk6855@rit.edu)
         Pavan Bhat (pxb8715.rit.edu)
         Siddharth Tarey (st2476@rit.edu)
"""
import csv
import math

counter = 0
totalnodecount = 0
leafnodecount = 0
ptotalnodecount = 0
pleafnodecount = 0


def main():
    '''
    main method
    :return:
    '''
    global totalnodecount, leafnodecount, ptotalnodecount, pleafnodecount
    # name = input('Enter the file name: ')
    name = 'train_data.csv'
    file = open(name)
    data_set = csv.reader(file)

    # input attributes
    x = []

    # output
    y = []

    # read the rows in dataset and store the values of attributes in arrays
    for line in data_set:
        x.append([float(line[0]), float(line[1]), int(line[2])])

    file.close()

    root = TreeNode()

    write_weights = open('tree.csv', "w")
    write_row = csv.writer(write_weights, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

    parentNum = 0

    buildTree(x, root, parentNum, write_row)

    write_weights.close()

    print('\nBefore pruning -')
    print('Total node count = ', totalnodecount)
    print('Leaf node count = ', leafnodecount)



    pruneTree(root)

    write_weights_p = open('prunedtree.csv', "w")
    write_row_p = csv.writer(write_weights_p, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

    writePrunedTree(root, write_row_p)

    print('\nAfter Chi Squared pruning -')
    print('Total node count = ', ptotalnodecount)
    print('Leaf node count = ', pleafnodecount)


def buildTree(x, node, parentNum, write_row):
    global counter
    global totalnodecount
    global leafnodecount

    if len(x) == 0:
        return

    sameValue = x[0][2]
    c = 0
    for i in x:
        if i[2] == sameValue:
            c += 1
    if c == len(x):
        node.value = sameValue

        totalCount = len(x)
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0

        if sameValue == 1:
            count1 = len(x)
        elif sameValue == 2:
            count2 = len(x)
        elif sameValue == 3:
            count3 = len(x)
        elif sameValue == 4:
            count4 = len(x)

        write_row.writerow([parentNum, sameValue])
        node.id = parentNum
        node.isleaf = True
        node.count1 = count1
        node.count2 = count2
        node.count3 = count3
        node.count4 = count4
        node.totalcount = totalCount
        node.maxclass = sameValue
        totalnodecount += 1
        leafnodecount += 1
        return

    leftNode = TreeNode()
    rightNode = TreeNode()

    attrgain = list()
    for i in range(2):
        sortX(x, i)
        gain = []
        for adj in range(len(x)-1):
            left = []
            right = []
            if x[adj][2] != x[adj+1][2]:
                mean = (x[adj][i] + x[adj+1][i])/2
                for j in x:
                    if j[i] < mean:
                        left.append(j)
                    else:
                        right.append(j)
                gain.append([ig(left, right, x), mean])
        attrgain.append(findMax(gain, i))

    leftlist = []
    rightlist = []

    if attrgain[0][0] > attrgain[1][0]:
        node.value = [attrgain[0][1], 0]
        for i in x:
            if i[0] < attrgain[0][1]:
                leftlist.append(i)
            else:
                rightlist.append(i)
        leftIndex = counter+1
        rightIndex = counter +2
        counter += 2
        # node.leftnode = leftNode
        # node.rightnode = rightNode

        totalCount = len(x)
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0

        for i in x:
            if i[2] == 1:
                count1 += 1
            if i[2] == 2:
                count2 += 1
            if i[2] == 3:
                count3 += 1
            if i[2] == 4:
                count4 += 1

        max = count1
        maxclass = 1
        if max<count2:
            max = count2
            maxclass = 2
        if max < count3:
            max = count3
            maxclass = 3
        if max < count4:
            maxclass = 4

        write_row.writerow([parentNum, attrgain[0][1], 0, leftIndex, rightIndex])
        node.rightid = rightIndex
        node.leftid = leftIndex
        node.id = parentNum

        node.count1 = count1
        node.count2 = count2
        node.count3 = count3
        node.count4 = count4
        node.totalcount = totalCount
        node.maxclass = maxclass
        totalnodecount += 1

        if len(leftlist) != 0:
            node.leftnode = leftNode
            leftNode.parent = node
            buildTree(leftlist, leftNode, leftIndex, write_row)

        if len(rightlist) != 0:
            node.rightnode = rightNode
            rightNode.parent = node
            buildTree(rightlist, rightNode, rightIndex, write_row)

    else:
        node.value = [attrgain[1][1], 1]
        for i in x:
            if i[1] < attrgain[1][1]:
                leftlist.append(i)
            else:
                rightlist.append(i)
        leftIndex = counter + 1
        rightIndex = counter + 2
        counter += 2
        # node.leftnode = leftNode
        # node.rightnode = rightNode
        totalCount = len(x)
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0

        for i in x:
            if i[2] == 1:
                count1 += 1
            if i[2] == 2:
                count2 += 1
            if i[2] == 3:
                count3 += 1
            if i[2] == 4:
                count4 += 1

        max = count1
        maxclass = 1
        if max < count2:
            max = count2
            maxclass = 2
        if max < count3:
            max = count3
            maxclass = 3
        if max < count4:
            maxclass = 4

        write_row.writerow([parentNum, attrgain[1][1], 1, leftIndex, rightIndex])
        node.rightid = rightIndex
        node.leftid = leftIndex
        node.id = parentNum

        node.count1 = count1
        node.count2 = count2
        node.count3 = count3
        node.count4 = count4
        node.totalcount = totalCount
        node.maxclass = maxclass
        totalnodecount += 1

        if len(leftlist) != 0:
            node.leftnode = leftNode
            leftNode.parent = node
            buildTree(leftlist, leftNode, leftIndex, write_row)

        if len(rightlist) != 0:
            node.rightnode = rightNode
            rightNode.parent = node
            buildTree(rightlist, rightNode, rightIndex, write_row)


def sortX(x, attr):
    for i in range(len(x)):
        for j in range(len(x)-1):
            if x[j][attr] > x[j+1][attr]:
                temp = x[j]
                x[j] = x[j+1]
                x[j+1] = temp


def findMax(x, index):
    max = x[0][index]
    temp = x[0]
    for i in range(len(x)):
        if max < x[i][index]:
            max = x[i][index]
            temp = x[i]
    return temp


def ig(left, right, x):
    return outputG(x) - remainder(left, right)


def outputG(x):
    total = len(x)

    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0

    for i in x:
        if i[2] == 1:
            count1 += 1
        elif i[2] == 2:
            count2 += 1
        elif i[2] == 3:
            count3 += 1
        elif i[2] == 4:
            count4 += 1

    return inf((count1/total), (count2/total), (count3/total), (count4/total))


def remainder(left, right):
    left1 = 0
    left2 = 0
    left3 = 0
    left4 = 0

    for i in left:
        if i[2] == 1:
            left1 += 1
        elif i[2] == 2:
            left2 += 1
        elif i[2] == 3:
            left3 += 1
        elif i[2] == 4:
            left4 += 1

    totalleft = len(left)
    totalright = len(right)
    total = totalleft + totalright

    right1 = 0
    right2 = 0
    right3 = 0
    right4 = 0

    for i in right:
        if i[2] == 1:
            right1 += 1
        elif i[2] == 2:
            right2 += 1
        elif i[2] == 3:
            right3 += 1
        elif i[2] == 4:
            right4 += 1

    result = (totalleft/total) * inf((left1/totalleft), (left2/totalleft), (left3/totalleft), (left4/totalleft)) + \
             (totalright / total) * inf((right1 / totalright), (right2 / totalright), (right3 / totalright), \
                                       (right4 / totalright))
    return result


def inf(a, b, c, d):
    sum = 0

    if a != 0:
        sum += - a*math.log(a)
    if b != 0:
        sum += - b * math.log(b)
    if c != 0:
        sum += - c * math.log(c)
    if d != 0:
        sum += - d * math.log(d)

    return sum


def pruneTree(node):
    stack = []
    stack.append(node)
    count = 0
    while len(stack) != 0:
        node = stack.pop()
        if node.leftnode is not None or node.rightnode is not None:
            if node.leftnode.isleaf and node.rightnode.isleaf:
                # check pruning condition
                if pruneCheck(node):
                    node.isleaf = True
                    node.rightnode = None
                    node.leftnode = None
                    stack.append(node.parent)
            else:
                if node.leftnode is not None and not node.leftnode.isleaf:
                    stack.append(node.leftnode)
                if node.rightnode is not None and not node.rightnode.isleaf:
                    stack.append(node.rightnode)



def calculatedel(child, parent):
    c1k = 0
    c2k = 0
    c3k = 0
    c4k = 0

    if parent.totalcount != 0:
        c1k = child.count1 * (child.totalcount / parent.totalcount)
        c2k = child.count2 * (child.totalcount / parent.totalcount)
        c3k = child.count3 * (child.totalcount / parent.totalcount)
        c4k = child.count4 * (child.totalcount / parent.totalcount)

    del1 = 0
    del2 = 0
    del3 = 0
    del4 = 0
    if c1k != 0:
        del1 = (math.pow(child.count1 - c1k, 2))/c1k
    if c2k != 0:
        del2 = (math.pow(child.count2 - c2k, 2))/c2k
    if c3k != 0:
        del3 = (math.pow(child.count3 - c3k, 2))/c3k
    if c4k != 0:
        del4 = (math.pow(child.count4 - c4k, 2))/c4k

    return (del1 + del2 + del3 + del4)


def pruneCheck(node):
    leftnode = node.leftnode
    rightnode = node.rightnode

    delta = calculatedel(leftnode, node) + calculatedel(rightnode, node)

    if delta < 11.34:
        return True
    return False


def writePrunedTree(node, write_row_p):
    global ptotalnodecount
    global pleafnodecount
    if node is not None:
        if node.isleaf:
            pleafnodecount += 1
            ptotalnodecount += 1
            write_row_p.writerow([node.id, node.maxclass])
        else:
            ptotalnodecount += 1
            write_row_p.writerow([node.id, node.value[0], node.value[1], node.leftid, node.rightid])
        writePrunedTree(node.leftnode, write_row_p)
        writePrunedTree(node.rightnode, write_row_p)


class TreeNode:

    __slots__ = 'id','value', 'leftnode', 'rightnode', 'count1', 'count2', 'count3', 'count4', 'totalcount', \
                'maxclass', 'parent', 'isleaf', 'leftid', 'rightid'

    def __init__(self, value = None, leftnode = None, rightnode = None):
        self.value = value
        self.leftnode = leftnode
        self.rightnode = rightnode
        self.isleaf = False
        self.parent = -1
        self.id = -1
        self.leftid = -1
        self.rightid = -1


if __name__ == '__main__':
    main()

