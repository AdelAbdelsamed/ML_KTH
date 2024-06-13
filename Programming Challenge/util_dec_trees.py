from sklearn import tree
import numpy as np



# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)

class DecisionTreeClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = DecisionTreeClassifier()
        rtn.classifier = tree.DecisionTreeClassifier(max_depth= int(Xtr.shape[1]/2 + 1))
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)
    
# ## Boosting functions to implement
# 
# The lab descriptions state what each function should do.


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # TODO: Fill in the rest, construct the alphas etc.
        # ==========================
        # Obtain indices of correct classification
        #idx_c = np.where(vote == labels)[0]
        #epsilon_t =  np.sum(wCur[idx_c]) 

        epsilon_t = sum(wCur[i] * (1 - (1 if vote[i] == labels[i] else 0)) for i in range(Npts))
        alpha = 0.5*( np.log(1 - epsilon_t) - np.log(epsilon_t))

        new_w = wCur
        # Update the weights
        for i in range(Npts):
            if vote[i] == labels[i]:
                new_w[i] = new_w[i]*np.exp(-alpha)
            else:
                new_w[i] = new_w[i]*np.exp(alpha)

        # Normalize the new weights
        wCur = new_w/np.sum(new_w)

        alphas.append(alpha) # you will need to append the new alpha
        # ==========================
        
    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================

        for t in range(Ncomps):
            # Obtain classifiers for each trained classifier h
            c_X_t = classifiers[t].classify(X)

            # Aggregate the votes for all classes over the trained classfiers
            for c_i in range(Nclasses):

                for x_i, ct_i in enumerate(c_X_t):

                    votes[x_i, c_i] += alphas[t] * (1 if ct_i == c_i else 0)            
        
        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)
    
def train_boosted_dec_trees(model, X_train, X_test, y_train, y_test):
    # Train
    trained_classifier = model.trainClassifier(X_train, y_train)
    # Predict
    yPr = trained_classifier.classify(X_train)
    yPr_test = trained_classifier.classify(X_test)
    # Compute classification error
    means = np.mean((yPr==y_train).astype(float))
    means_test = np.mean((yPr_test==y_test).astype(float))
    print("Decision Trees using Adaboost [Lab3 implementation]")
    print("Training: " + str(means))
    print("Validation: " + str(means_test))
    return trained_classifier