from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate
import numpy as np




def svm(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train) # Obtain score on training set
    test_acc = (model.score(X_test, y_test)) # Obtain score on test set 

    print("SVM classifier scores:")
    print("Training Set Accuracy: ", train_acc)
    print("Test Set Accuracy: ", test_acc)
    return model

def trainSVM_wCV(model, X_train, X_test, y_train, y_test):
    cv_results = cross_validate(model, X_train, y_train,  cv= 3, return_estimator=True)

    test_acc = cv_results["test_score"] # Obtain score on validation sets from the splits
    print("SVM with cross validation scores:") 
    print("Validation Sets Scores", test_acc)

    # Obtain the best performing tree
    best_svm = cv_results["estimator"][np.argmax(cv_results["test_score"])]
    test_acc = best_svm.score(X_test, y_test) # Obtain score on test set

    return best_svm


