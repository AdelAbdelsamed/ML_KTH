from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import CategoricalNB
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import StackingClassifier
from preprocess_data import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from main import write_to_txt
#from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from decision_tree import trainTree

estimators = []
estimators.append(('AdaBoostClassifier', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=8), n_estimators=150, random_state=13) ))
estimators.append(('Bagging Classifier', BaggingClassifier(random_state=13) ))
estimators.append(('Decision Tree Classifier', DecisionTreeClassifier(random_state = 13) ))
estimators.append(('Dummy Classifier', DummyClassifier(random_state=13) ))
estimators.append(('Extra Tree Classifier', ExtraTreeClassifier(random_state=13) ))
estimators.append(('Extra Trees Classifier', ExtraTreesClassifier(random_state=13) ))
estimators.append(('Gaussian Process Classifier', GaussianProcessClassifier(random_state=13) ))
estimators.append(('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=13) ))
estimators.append(('Hist Gradient Boosting Classifier', HistGradientBoostingClassifier(random_state=13) ))
estimators.append(('KNN', KNeighborsClassifier() ))
estimators.append(('LogisticRegression', LogisticRegression(max_iter=1000, random_state=13)))
estimators.append(('Logistic Regression CV', LogisticRegressionCV(max_iter=1000, random_state=13) ))
estimators.append(('MLPClassifier', MLPClassifier(max_iter=2000,random_state=13) ))
estimators.append(('Nearest Centroid', NearestCentroid() ))
estimators.append(('Passive Aggressive Classifier', PassiveAggressiveClassifier(random_state=13) ))
estimators.append(('Perceptron', Perceptron(random_state=13) ))
estimators.append(('RandomForest', RandomForestClassifier(max_depth= 8, criterion='entropy', n_estimators=150, max_features='sqrt', bootstrap=True, min_samples_leaf= 1, min_samples_split= 3, random_state=13) ))
estimators.append(('Ridge Classifier', RidgeClassifier(random_state=13) ))
estimators.append(('Ridge Classifier CV', RidgeClassifierCV() ))
estimators.append(('SGDClassifier', SGDClassifier(random_state=13) ))
estimators.append(('XGB', XGBClassifier(random_state=13,max_depth = 5, learning_rate = 0.1, n_estimators = 50)))
estimators.append(('CatBoost', CatBoostClassifier(logging_level='Silent') ))

XGB = XGBClassifier(random_state=13)

X, y = fetchTrainDataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

SC = StackingClassifier(estimators=estimators,final_estimator=XGB,cv=6)
SC.fit(X_train, y_train)
y_pred = SC.predict(X_test)

print(f"\nStacking classifier training Accuracy: {SC.score(X_train, y_train):0.2f}")
print(f"Stacking classifier test Accuracy: {SC.score(X_test, y_test):0.2f}")
precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_pred, average='macro', labels=[0, 1, 2]) # Further evaluation metrics
print("Precision of test samples:", precision)
print("Recall of test samples: ", recall)
print("fbeta_score of test samples: ", fbeta_score)

Xeval = fetchEvalDataset()

#Xeval = np.hstack((Xeval[:,0:5], Xeval[:, 6:12])) # Remove correlated variables
yPrEval = SC.predict(Xeval)
write_to_txt(yPrEval)

# Train the classifier
xgb_classifier = XGBClassifier(max_depth = 5, learning_rate = 0.1, n_estimators = 50)
trained_xgb_classifier = trainTree(xgb_classifier, X_train, X_test, y_train, y_test, ' XGB ')
y_pred_test = trained_xgb_classifier.predict(X_test) #Predicted values
precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_pred_test, average='macro', labels=[0, 1, 2]) # Further evaluation metrics
print("Precision of test samples:", precision)
print("Recall of test samples: ", recall)
print("fbeta_score of test samples: ", fbeta_score)