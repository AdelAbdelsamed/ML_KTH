import math
import random
import numpy as np


def create_subsets_compute_information_gain(dataset,attribute,v, inform_gain_wanted):
    subsets_list_level = []
    information_gain_level = {}
    for j in range(len(attribute[v].values)): # Loop all values in attribute a4 to create subsets
        # Create subset of the list
        subsets_list_level.append(select(dataset, attribute[v], j + 1))
        # Compute the information gain of the subset
        if inform_gain_wanted:
            temp_list = []
            for i in range(6):
                if i == v:
                    continue # Attribute no.
                temp_list.append(averageGain(subsets_list_level[j], attribute[i]))
            information_gain_level['a' + str(v +1) + ' == ' + str(j + 1)] = temp_list
    return  subsets_list_level, information_gain_level

def entropy(dataset):
    "Calculate the entropy of a dataset"
    n = len(dataset)
    nPos = len([x for x in dataset if x.positive])
    nNeg = n - nPos
    if nPos == 0 or nNeg == 0:
        return 0.0
    return -float(nPos)/n * log2(float(nPos)/n) + \
        -float(nNeg)/n * log2(float(nNeg)/n)


def averageGain(dataset, attribute):
    "Calculate the expected information gain when an attribute becomes known"
    weighted = 0.0
    for v in attribute.values:
        subset = select(dataset, attribute, v)
        weighted += entropy(subset) * len(subset)
    return entropy(dataset) - weighted/len(dataset)

def entropy_subsets(dataset, attribute):
    "Calculate the entropy of both subsets after the split"
    subsets_entropy = []
    for v in attribute.values:
        subset = select(dataset, attribute, v)
        subsets_entropy.append(entropy(subset))
    return subsets_entropy


def log2(x):
    "Logarithm, base 2"
    return math.log(x, 2)


def select(dataset, attribute, value):
    "Return subset of data samples where the attribute has the given value"
    return [x for x in dataset if x.attribute[attribute] == value]


def bestAttribute(dataset, attributes):
    "Attribute with highest expected information gain"
    gains = [(averageGain(dataset, a), a) for a in attributes]
    return max(gains, key=lambda x: x[0])[1]


def allPositive(dataset):
    "Check if all samples are positive"
    return all([x.positive for x in dataset])


def allNegative(dataset):
    "Check if all samples are negative"
    return not any([x.positive for x in dataset])


def mostCommon(dataset):
    "Majority class of the dataset"
    pCount = len([x for x in dataset if x.positive])
    nCount = len([x for x in dataset if not x.positive])
    return pCount > nCount

def partition(data, fraction):
    # Set data in a list
    ldata = list(data)
    # Randomly shuffle the data
    random.shuffle(ldata)
    # Define the breaking point of the list
    breakPoint = int(len(data)*fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def reduced_error_pruning(data, test_data, attributes, fractions ,no_instances):
    # Define the errors
    mean_error_test_set = np.zeros((len(fractions), 1))
    std_error_test_set = np.zeros((len(fractions), 1))
    errors_on_test_set = np.zeros((len(fractions),no_instances))

    for i in range(len(fractions)):

        for j in range(no_instances):
            # Partition the data
            monk1train, monk1val = partition(data, fractions[i])
            # Build the tree based on the training set
            tree = buildTree(monk1train, attributes)
            # Obtain all possible pruned trees
            pruned_trees = allPruned(tree)
            
            curr_acc = 0 # Current accuracy value
            max_acc = 0 # Max accuracy value
            best_pTree = 0 # Best pruned tree
            for k in range(len(pruned_trees)):
                curr_acc =  check(pruned_trees[k], monk1val)
                if(curr_acc > max_acc):
                    max_acc = curr_acc
                    bestTree = k
            # Evaluate best tree performance on test set
            errors_on_test_set[i,j] = 1 - check(pruned_trees[bestTree], test_data)

    mean_error_test_set = np.mean(errors_on_test_set, 1)  
    std_error_test_set = np.std(errors_on_test_set, 1)  

    return mean_error_test_set, std_error_test_set



class TreeNode:
    "Decision tree representation"

    def __init__(self, attribute, branches, default):
        self.attribute = attribute
        self.branches = branches
        self.default = default

    def __repr__(self):
        "Produce readable (string) representation of the tree"
        accum = str(self.attribute) + '('
        for x in sorted(self.branches):
            accum += str(self.branches[x])
        return accum + ')'


class TreeLeaf:
    "Decision tree representation for leaf nodes"

    def __init__(self, cvalue):
        self.cvalue = cvalue

    def __repr__(self):
        "Produce readable (string) representation of this leaf"
        if self.cvalue:
            return '+'
        return '-'


def buildTree(dataset, attributes, maxdepth=1000000):
    "Recursively build a decision tree"

    def buildBranch(dataset, default, attributes):
        if not dataset:
            return TreeLeaf(default)
        if allPositive(dataset):
            return TreeLeaf(True)
        if allNegative(dataset):
            return TreeLeaf(False)
        return buildTree(dataset, attributes, maxdepth-1)

    default = mostCommon(dataset)
    if maxdepth < 1:
        return TreeLeaf(default)
    a = bestAttribute(dataset, attributes)
    attributesLeft = [x for x in attributes if x != a]
    branches = [(v, buildBranch(select(dataset, a, v), default, attributesLeft))
                for v in a.values]
    return TreeNode(a, dict(branches), default)


def classify(tree, sample):
    "Classify a sample using the given decition tree"
    if isinstance(tree, TreeLeaf):
        return tree.cvalue
    return classify(tree.branches[sample.attribute[tree.attribute]], sample)


def check(tree, testdata):
    "Measure fraction of correctly classified samples"
    correct = 0
    for x in testdata:
        if classify(tree, x) == x.positive:
            correct += 1
    return float(correct)/len(testdata)


def allPruned(tree):
    "Return a list if trees, each with one node replaced by the corresponding default class"
    if isinstance(tree, TreeLeaf):
        return ()
    alternatives = (TreeLeaf(tree.default),)
    for v in tree.branches:
        for r in allPruned(tree.branches[v]):
            b = tree.branches.copy()
            b[v] = r
            alternatives += (TreeNode(tree.attribute, b, tree.default),)
    return alternatives
