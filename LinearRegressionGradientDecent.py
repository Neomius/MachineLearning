import pandas as pd
import numpy as np
import warnings

def FeatureNormalization(X):
    m = np.size(X, axis=0)  # number of training examples
    n = np.size(X, axis=1)  # number of features

    mu = np.mean(X, axis=0)
    mu = np.reshape(mu, [1, n])
    print("Size of mu:", np.size(mu))
    sigma = np.std(X)
    mu_matrix = mu * (np.ones(m, 1))
    sigma_matrix = np.ones(m, 1) * mu_matrix
    X_norm = (X - mu_matrix) * sigma_matrix
    return mu, sigma, X_norm

### Import and clean train and test data
train = pd.read_csv('data/train.csv')
print('Shape of the train data with all features:', train.shape)

train = train.select_dtypes(exclude=['object']) # Excludes columns with strings
print("")
print('Shape of the train data with numerical features:', train.shape)
train.drop('Id',axis = 1, inplace = True) # Drop first column with name 'Id'
train.fillna(0,inplace=True) # Fill up NaN cells with (0)


print("")
print("List of features contained our dataset:",list(train.columns))





#warnings.filterwarnings('ignore') Dont know what this does

#This might be useful
#col_train = list(train.columns)
#col_train_bis = list(train.columns)
#col_train_bis.remove('SalePrice')

mat_train = np.matrix(train)
X = np.matrix(train.drop('SalePrice',axis = 1))

#mat_y = np.array(train.SalePrice).reshape((1314,1))
Y = np.matrix(train.SalePrice)

print("Shape of X: ", np.shape(X))
print("Shape of Y: ", np.shape(np.transpose(Y)))


#Feature Normatization

mu, sigma, X_norm = FeatureNormalization(X)
