from preprocess_data import *
from decision_tree import *
from util_dec_trees import *
from sklearn.model_selection import cross_validate
from svm import *
from pca import *
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support
import ensemble
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, BaggingClassifier



def write_to_txt(yPrEval):
    # Define the filename for the output text file
    output_file = "final_stacking_predicted_labels2.txt"

    predicted_labels = [] # List containing the predicted labels

    for i in range(np.size(yPrEval)):
        if yPrEval[i] == 0.0:
            predicted_labels.append('Atsutobob')
        elif yPrEval[i] == 1.0:
            predicted_labels.append('Boborg')
        elif yPrEval[i] == 2.0:
            predicted_labels.append('Jorgsuto')


    # Open the file in write mode and write the labels to it
    with open(output_file, 'w') as file:
        for label in predicted_labels:
            file.write(label + '\n')

    print(f"Predicted labels have been written to {output_file}")

print("---------------------------- Data Preprocessing -----------------------------")

X, y = fetchTrainDataset()
# Remove x4 as it is highly correlated with x0
#X = np.hstack((X[:,0:5], X[:, 6:12])) # Remove correlated variables
Xeval = fetchEvalDataset()

print("----------------------------- Model Selection -----------------------------")

# Split data into training and validation data set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=0)

# Train Decision Tree and evaluate it on test set
#dec_tree = tree.DecisionTreeClassifier(max_depth= 12)
#trained_dec_tree = trainTree(dec_tree, X_train, X_test, y_train, y_test, '')
# Train tree using CV and evaluate it on test set
#trained_dec_tree_wCV = trainTree_wCV(dec_tree, X_train, X_test, y_train, y_test)

# Bagging with Decision Trees
#bagged_dec_tree = BaggingClassifier(estimator=dec_tree, n_estimators=50)
#trained_bagged_dec_tree_wCV = trainTree_wCV(bagged_dec_tree, X_train, X_test, y_train, y_test)

# Use Adaboost
adaboost_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=8), n_estimators=150, random_state= 10)
trained_adaboost_classifier = trainTree(adaboost_classifier, X_train, X_test, y_train, y_test, ' Adaboost ')

# Use Random forest
rf_clf = RandomForestClassifier(criterion='entropy', n_estimators=150, max_features='sqrt', bootstrap=True, max_depth= 8, random_state= 2)
trained_rf_classifier = trainTree(adaboost_classifier, X_train, X_test, y_train, y_test, ' Random Forest ')


# # Boosted decision trees [Implementation from Lab3]
# boos_dec_tree = BoostClassifier(DecisionTreeClassifier(), T=20)
# trained_boos_dec_tree = train_boosted_dec_trees(boos_dec_tree, X_train, X_test, y_train, y_test)
# yPr = trained_boos_dec_tree.classify(X_train)
# yPr_test = trained_boos_dec_tree.classify(X_test)
# # Compute classification error
# means = np.mean((yPr==y_train).astype(float))
# means_test = np.mean((yPr_test==y_test).astype(float))

# # Use XGB classifier
# xgb_classifier = xgb.XGBClassifier(learning_rate=0.2, n_estimators=50, max_depth=6, gamma=0.5)
# trained_xgb_classifier = trainTree(xgb_classifier, X_train, X_test, y_train, y_test, ' XGB ')
# trained_xgb_classifier_wCV = trainTree_wCV(xgb_classifier, X_train, X_test, y_train, y_test)

# #pca_X_train, pca_X_test = perform_pca(X_train, X_test, 8)

# # Use stacking classifier
# trained_stacked_classifiers = ensemble.final_classifier2(X_train, X_test, y_train, y_test)

# n_iter = 100
# test_acc = np.zeros((n_iter,1))
# classifiers = [] # Save the classifiers
# #for i in range(n_iter):

# # Split data into training and validation data set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# # Train the classifier
# xgb_classifier = xgb.XGBClassifier(max_depth = 5, learning_rate = 0.1, n_estimators = 50)
# trained_xgb_classifier = trainTree(xgb_classifier, X_train, X_test, y_train, y_test, ' XGB ')


    # Append accuracy on test set
    # test_acc[i] = (trained_xgb_classifier.score(X_test, y_test))
    # print("Iteration " + str(i+1))
    # print("Current Training Set Accuracy: ", test_acc[i])
    # classifiers.append(trained_xgb_classifier)


# print("Iterations finished!")
# print("Mean test accuracy is ", np.mean(test_acc))
# print("Min. test accuracy is ", np.min(test_acc))
# print("Max. test accuracy is ", np.max(test_acc))
# Best performing model on the test set
# best_clf = classifiers[np.argmax(test_acc)]
# y_pred_test = trained_xgb_classifier.predict(X_test) #Predicted values
# precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_pred_test, average=None, labels=[0, 1, 2]) # Further evaluation metrics
# print("Precision of test samples:", precision)
# print("Recall of test samples: ", recall)
# print("fbeta_score of test samples: ", fbeta_score)

# Use SVM
# model = SVC(kernel='rbf')
# trained_svm = svm(model, X_train, X_test, y_train, y_test)
# trained_svm_wCV = trainSVM_wCV(model, X_train, X_test, y_train, y_test)

# Write the results
#yPrEval = trained_xgb_classifier.predict(Xeval)
#yPrEval = trained_boos_dec_tree.classify(Xeval)
#write_to_txt(yPrEval)









