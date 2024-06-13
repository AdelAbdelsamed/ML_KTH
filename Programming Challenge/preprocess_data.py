# Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
from sklearn.utils import shuffle



def fetchTrainDataset():
    # Function fetches the dataset
    # y: Transforms the labels into numeric classes and checks for error in the data
    # x1-x6: Outlier detectiona and normalization using z-score
    # x7: Mapping into numerical values and checks for error in the data
    # x8-x12: Outlier detectiona and normalization using z-score
    print("Fetching training data set...")

    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv('TrainOnMe.csv', dtype={"y": np.str_, "x7": np.str_, "x1": np.float32,
                                              "x2": np.float32, "x3": np.float32, "x4": np.float32,
                                               "x5": np.float32, "x6": np.float32, "x9": np.float32,
                                                "x10": np.float32, "x11": np.float32, "x12": np.str_, "x13": np.float32 })

    # Remove the first column which is just an identifier
    df = df.drop(columns=['Unnamed: 0'])

    # Remove column 12 which is always true in the training and evaluation data set
    df = df.drop(columns=['x12'])

    # Convert the DataFrame to a Numpy array
    data = df.to_numpy()

    # Extract the target variable (y) and the feature variables (x1 to x13)
    y_str = data[:, 0]  # Assuming the target variable is in the first column
    X_num1 = data[:, 1:7]  # Extract the numerical values
    X_num2 = data[:, 8:13]
    x7_str = data[:, 7] # Extract feature x7 which contains strings

    y = np.zeros(y_str.shape)
    y_unident_idx = []

    # Create three classes for the output values
    # Class 0: Atsutobob
    # Class 1: Boborg
    # Class 2: Jorgsuto
    for i in range(np.size(y_str)):
        if y_str[i] == 'Atsutobob':
            y[i] = 0
        elif y_str[i] == 'Boborg':
            y[i] = 1
        elif y_str[i] == 'Jorgsuto':
            y[i] = 2
        else:
            y_unident_idx.append(i)

    if len(y_unident_idx) == 0:
        print('Labels y were mapped to the corresponding classes correctly!')
        # Assess the unidentified labels
        #print(y_unident_idx)
        #print(y_str[y_unident_idx])
    else:
        print('Labels y mapping contains errors!')
        exit()

    # # Encode the variable x7
    # # 0 -  Hambogris
    # # 1 -  Polkagris
    # # 2 -  Polskorgris
    # # 3 -  Schottisgris
    # # 4 -  Slängpolskorgris
    x7_labels = {
        'Hambogris': 0.0,
        'Polkagris': 0.1,
        'Polskorgris': 0.2,
        'Schottisgris': 0.3,
        'Slängpolskorgris': 0.4
    }
    x7_str = x7_str.astype(str)
    unique_var_x7 = np.unique(x7_str)
    # Are there any typos in class x7?
    #print(unique_var_x7)

    x7_mapped = np.ones(np.size(x7_str))*5
    # Map the Strings to their classes
    x7_mapped[x7_str == 'Hambogris'] = x7_labels['Hambogris']
    x7_mapped[x7_str == 'Polkagris'] = x7_labels['Polkagris']
    x7_mapped[x7_str == 'Polskorgris'] = x7_labels['Polskorgris']
    x7_mapped[x7_str == 'Schottisgris'] = x7_labels['Schottisgris']
    x7_mapped[x7_str == 'Slängpolskorgris'] = x7_labels['Slängpolskorgris']

    # Extract the allowed values from the dictionary
    allowed_x7 = list(x7_labels.values())
    # Check if any element in the vector is not in the list of allowed values
    elements_not_in_dict = ~np.isin(x7_mapped, allowed_x7)
    # Print the elements that are not in the dictionary
    if np.any(elements_not_in_dict):
        print("Following x7 values are not encoded:")
        print(x7_str[elements_not_in_dict])
        print('The occurrence is in index ', np.nonzero(elements_not_in_dict))
        exit()
    else:
        print("Encoding of the feature x7 is successful! ")

    print("Data is successfully fecthed! ")

    # Use one-hot encoding for x7
    # x7_labels_oh_encoding = {
    #     'Hambogris': np.array([1, 0, 0, 0, 0]),
    #     'Polkagris': np.array([0, 1, 0, 0, 0]),
    #     'Polskorgris': np.array([0, 0, 1, 0, 0]),
    #     'Schottisgris': np.array([0, 0, 0, 1, 0]),
    #     'Slängpolskorgris': np.array([0, 0, 0, 0, 1])
    # }

    # print(np.size(x7_str,0))
    # x7_mapped = np.ones((np.size(x7_str,0),5))
    # # Map the Strings to their classes
    # x7_mapped[x7_str == 'Hambogris', :] = x7_labels_oh_encoding['Hambogris']
    # x7_mapped[x7_str == 'Polkagris', :] = x7_labels_oh_encoding['Polkagris']
    # x7_mapped[x7_str == 'Polskorgris', :] = x7_labels_oh_encoding['Polskorgris']
    # x7_mapped[x7_str == 'Schottisgris', :] = x7_labels_oh_encoding['Schottisgris']
    # x7_mapped[x7_str == 'Slängpolskorgris', :] = x7_labels_oh_encoding['Slängpolskorgris']

    # if np.any(np.sum(x7_mapped, axis = 1) > 1):
    #     print("There is an error in the encoding of the x7 variables!")
    # else:
    #     print("One-hot encoding of x7 was successful!")


    # Transform to float
    X_num1 = X_num1.astype('float64')
    X_num2 = X_num2.astype('float64')

    X_num = np.hstack((X_num1, X_num2)) # Stack numerical data together

    # Check for nan values
    nan_mask = np.any(np.isnan(X_num), axis = 1)
    X_num = X_num[~nan_mask] # Remove rows with nan values
    x7_mapped = x7_mapped[~nan_mask]
    y = y[~nan_mask]

    if not np.any(np.isnan(x7_mapped)) and not np.any(np.isnan(y)) and not np.any(np.isnan(X_num)):
        print('Nan values were removed successfully!')

    # Normalize data using the z-score
    # z = (x - mu)/sigma
    X_num_norm = (X_num - np.mean(X_num, 0))/np.std(X_num, 0)

    # Detect outliers using the z-score threshold ( |z-score| > threshold)
    threshold =  5 # You can adjust this threshold based on your data and problem

    # Create a boolean mask to identify outliers
    outlier_mask = np.any(X_num_norm > threshold, axis=1)
    # Remove rows with outliers
    X_num_norm = X_num_norm[~outlier_mask]
    y = y[~outlier_mask]
    x7_mapped = x7_mapped[~outlier_mask]

    print("Outliers removed successfully!!")


    # Stack the data matrix
    X = np.hstack((X_num_norm[:, 0:6], x7_mapped.reshape(-1,1), X_num_norm[:,6:13]))

    return X, y


def fetchEvalDataset():
    # Function fetches the dataset
    # x1-x6: Outlier detectiona and normalization using z-score
    # x7: Mapping into numerical values and checks for error in the data
    # x8-x12: Outlier detectiona and normalization using z-score
    print("Fetching evaluation data set...")

    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv('EvaluateOnMe.csv', dtype={"x7": np.str_, "x1": np.float32,
                                              "x2": np.float32, "x3": np.float32, "x4": np.float32,
                                               "x5": np.float32, "x6": np.float32, "x9": np.float32,
                                                "x10": np.float32, "x11": np.float32, "x12": np.str_, "x13": np.float32 })

    # Remove the first column which is just an identifier
    df = df.drop(columns=['Unnamed: 0'])

    # Remove column 12 which is always true in the training and evaluation data set
    df = df.drop(columns=['x12'])

    # Convert the DataFrame to a Numpy array
    data = df.to_numpy()

    # Extract the target variable (y) and the feature variables (x1 to x13)
    X_num1 = data[:, 0:6]  # Extract the numerical values
    X_num2 = data[:, 7:12]
    x7_str = data[:, 6] # Extract feature x7 which contains strings

    # Encode the variable x7
    # 0 -  Hambogris
    # 1 -  Polkagris
    # 2 -  Polskorgris
    # 3 -  Schottisgris
    # 4 -  Slängpolskorgris
    x7_labels = {
        'Hambogris': 0.0,
        'Polkagris': 1.0,
        'Polskorgris': 2.0,
        'Schottisgris': 3.0,
        'Slängpolskorgris': 4.0
    }
    x7_str = x7_str.astype(str)
    unique_var_x7 = np.unique(x7_str)
    # Are there any typos in class x7?
    # print(unique_var_x7)

    x7_mapped = np.ones(np.size(x7_str))*5
    # Map the Strings to their classes
    x7_mapped[x7_str == 'Hambogris'] = x7_labels['Hambogris']
    x7_mapped[x7_str == 'Polkagris'] = x7_labels['Polkagris']
    x7_mapped[x7_str == 'Polskorgris'] = x7_labels['Polskorgris']
    x7_mapped[x7_str == 'Schottisgris'] = x7_labels['Schottisgris']
    x7_mapped[x7_str == 'Slängpolskorgris'] = x7_labels['Slängpolskorgris']

    # Extract the allowed values from the dictionary
    allowed_x7 = list(x7_labels.values())
    # Check if any element in the vector is not in the list of allowed values
    elements_not_in_dict = ~np.isin(x7_mapped, allowed_x7)
    # Print the elements that are not in the dictionary
    if np.any(elements_not_in_dict):
        print("Following x7 values are not encoded:")
        print(x7_str[elements_not_in_dict])
        print('The occurrence is in index ', np.nonzero(elements_not_in_dict))
        exit()
    else:
        print("Encoding of the feature x7 is successful! ")

    print("Evaluation data is successfully fecthed! ")

    # Use one-hot encoding for x7
    # x7_labels_oh_encoding = {
    #     'Hambogris': np.array([1, 0, 0, 0, 0]),
    #     'Polkagris': np.array([0, 1, 0, 0, 0]),
    #     'Polskorgris': np.array([0, 0, 1, 0, 0]),
    #     'Schottisgris': np.array([0, 0, 0, 1, 0]),
    #     'Slängpolskorgris': np.array([0, 0, 0, 0, 1])
    # }

    # print(np.size(x7_str,0))
    # x7_mapped = np.ones((np.size(x7_str,0),5))
    # # Map the Strings to their classes
    # x7_mapped[x7_str == 'Hambogris', :] = x7_labels_oh_encoding['Hambogris']
    # x7_mapped[x7_str == 'Polkagris', :] = x7_labels_oh_encoding['Polkagris']
    # x7_mapped[x7_str == 'Polskorgris', :] = x7_labels_oh_encoding['Polskorgris']
    # x7_mapped[x7_str == 'Schottisgris', :] = x7_labels_oh_encoding['Schottisgris']
    # x7_mapped[x7_str == 'Slängpolskorgris', :] = x7_labels_oh_encoding['Slängpolskorgris']

    # if np.any(np.sum(x7_mapped, axis = 1) > 1):
    #     print("There is an error in the encoding of the x7 variables!")
    # else:
    #     print("One-hot encoding of x7 was successful!")


    # Transform to float
    X_num1 = X_num1.astype('float64')
    X_num2 = X_num2.astype('float64')

    X_num = np.hstack((X_num1, X_num2)) # Stack numerical data together

    if not np.any(np.isnan(x7_mapped)) and not np.any(np.isnan(X_num)):
        print('Nan values were removed successfully!')

    # Normalize data using the z-score
    # z = (x - mu)/sigma
    X_num_norm = (X_num - np.mean(X_num, 0))/np.std(X_num, 0)


    # Stack the data matrix
    X = np.hstack((X_num_norm[:, 0:6], x7_mapped.reshape(-1,1), X_num_norm[:,6:13]))

    return X

def count_labels(y):
    # Slight imbalance between classes 1 and 2
    label_counts = []

    label_counts.append((y == 0).sum())
    label_counts.append((y == 1).sum())
    label_counts.append((y == 2).sum())

    Ny = y.shape[0]

    label_occ_perc = [label_counts[i]/Ny for i in range(len(label_counts))]
    print("Class label percentages", label_occ_perc)

    # Plot the no. of occurences of each label
    plt.bar(np.array([0, 1, 2]), np.array(label_counts))
    plt.xlabel("Class Label")
    plt.ylabel("No. of occurences")
    plt.title("Class Imbalance Check")
    plt.show()

def correlation_check(X):
    # Check for correlation betwen the variables 
    plt.figure(figsize=(12,10))
    cov_X = np.cov(X.T)
    sns.heatmap(cov_X, annot=True, cmap=plt.cm.CMRmap_r)
    plt.title("Correlation of the variables")
    plt.show()

def detect_correlated_variables(X, threshold):
    cov_X = np.cov(X.T)
    col_corr = []  # List of all the names of correlated columns
    for i in range(cov_X.shape[1]):
        for j in range(i):
            if abs(cov_X[i, j]) > threshold: # we are interested in absolute coeff value
                print("Feature x" + str(i) + " x" + str(j) + " are correlated!")
                col_corr.append("(" + str(i) + ", " + str(j) + ")" )
    return col_corr

#Xeval = fetchEvalDataset()
#X, y = fetchTrainDataset()

# Check class imbalance
#count_labels(y)

# Check for correlation
#correlation_check(X)
#detect_correlated_variables(X, 0.94) # Variables (x0 and x4), (x0, x11) and (x4, x11) are highly correlated  


