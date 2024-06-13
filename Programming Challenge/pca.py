from sklearn.decomposition import PCA
import numpy as np
import matplotlib_inline as plt
from preprocess_data import *
from sklearn.model_selection import train_test_split


# did not improve accuracy
def perform_pca(X_train, X_test, n_comp):
    
    pca = PCA(n_components= n_comp) # create a PCA object
    pca.fit(X_train) # fit the pca
    pca_X_train = pca.transform(X_train) # get PCA coordinates for scaled_data
    pca_X_test = pca.transform(X_test)
    

    
    # Plot the variation explained in each principal component
    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    print(np.sum(pca.explained_variance_ratio_))
    
    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()

    return pca_X_train, pca_X_test



# X, y = fetchTrainDataset()
# Xeval = fetchEvalDataset()

# # Split data into training and validation data set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# perform_pca(X_train, X_test, 8)