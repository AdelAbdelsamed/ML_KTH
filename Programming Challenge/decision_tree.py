from sklearn.model_selection import train_test_split, cross_validate
from sklearn import tree
from preprocess_data import *
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from util_dec_trees import *


def trainTree(model, X_train, X_test, y_train, y_test, label):
    model.fit(X_train, y_train) # Fit the tree
    
    train_acc = model.score(X_train, y_train) # Obtain score on training set
    test_acc = (model.score(X_test, y_test)) # Obtain score on test set 

    print("Decision Tree" + label + " scores:")
    print("Training Set Accuracy: ", train_acc)
    print("Test Set Accuracy: ", test_acc)

    return model

def trainTree_wCV(model, X_train, X_test, y_train, y_test):
    cv_results = cross_validate(model, X_train, y_train,  cv= 5, return_estimator=True)

    test_acc = cv_results["test_score"] # Obtain score on validation sets from the splits
    print("Decision Trees with cross validation scores:") 
    print("Validation Sets Scores", test_acc)

    # Obtain the best performing tree
    best_cv_tree = cv_results["estimator"][np.argmax(cv_results["test_score"])]
    test_acc = best_cv_tree.score(X_test, y_test) # Obtain score on test set
    print("Test set accuracy:", test_acc)

    return best_cv_tree


# def trainTreewithAdaboost(model, X_train, X_test, y_train, y_test):
#     model.fit(X_train, y_train) # Fit the tree
    
#     test_acc = model.score(X_train, y_train) # Obtain score on training set
#     val_acc = (model.score(X_test, y_test)) # Obtain score on test set 

#     print("Decision Trees using Adaboost")
#     print("Training: " + str(test_acc))
#     print("Validation: " + str(val_acc))


# def trainXGBclassifier(model, X_train, X_test, y_train, y_test):
#     model.fit(X_train, y_train) # Fit the tree
    
#     test_acc = model.score(X_train, y_train) # Obtain score on training set
#     val_acc = (model.score(X_test, y_test)) # Obtain score on test set 

#     print("XGB Classifier")
#     print("Training: " + str(test_acc))
#     print("Validation: " + str(val_acc))

