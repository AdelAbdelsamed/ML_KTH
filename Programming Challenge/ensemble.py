from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb



def adaboost(X_train, X_test, y_train):
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=8), n_estimators=100)
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels


def rfc(X_train, X_test, y_train):
    model = RandomForestClassifier(criterion='entropy', n_estimators=200, max_features='sqrt')
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels


def stacking(X_train, X_test, y_train, y_test):        
    classifiers = [('qda', QuadraticDiscriminantAnalysis()), ('lda', LinearDiscriminantAnalysis()), ('rf', RandomForestClassifier(criterion='entropy', n_estimators=250, max_features='sqrt'))]    
    model = StackingClassifier(classifiers, final_estimator=LinearDiscriminantAnalysis(), cv=10)
    score = model.fit(X_train, y_train).score(X_test, y_test)
    return score


def final_classifier(X_train, X_test, y_train, y_test):    
    classifiers = [('qda', QuadraticDiscriminantAnalysis()), ('lda', LinearDiscriminantAnalysis()), ('rf', RandomForestClassifier(criterion='entropy', n_estimators=250, max_features='sqrt')),('xgb', xgb.XGBClassifier(learning_rate=0.2, n_estimators=50, max_depth=6, gamma=0.5))]    
    model = StackingClassifier(classifiers, final_estimator=LinearDiscriminantAnalysis(), cv=10)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train) # Obtain score on training set
    test_acc = (model.score(X_test, y_test)) # Obtain score on test set 

    print("Stacking classifier scores:")
    print("Training Set Accuracy: ", train_acc)
    print("Test Set Accuracy: ", test_acc)
    return model

def final_classifier2(X_train, X_test, y_train, y_test):    
    classifiers = [('rf', RandomForestClassifier(criterion='entropy', n_estimators=200, max_features='sqrt')), ('xgb', xgb.XGBClassifier(learning_rate=0.2, n_estimators=100, max_depth=6, gamma=0.5))]    
    model = StackingClassifier(classifiers, final_estimator= LogisticRegression(), cv=10)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train) # Obtain score on training set
    test_acc = (model.score(X_test, y_test)) # Obtain score on test set 

    print("Stacking classifier scores:")
    print("Training Set Accuracy: ", train_acc)
    print("Test Set Accuracy: ", test_acc)
    return model





