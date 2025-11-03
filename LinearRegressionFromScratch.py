import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Asser Abdelgawad

def fit_LinRegr(X_train, y_train):
    '''Implements linear regression to find a w that minimizes in-sample training error'''

    # Find N (number of samples) and d (number of features)
    N, d = X_train.shape
 
    # Augment dataset to include the bias parameter
    X_train = np.hstack((np.ones((N,1)), X_train))
    
    # Implement closed form solution from linear algebra to find least-squares error w
    Square_matrix = np.dot(np.transpose(X_train), X_train)
    Projection_matrix = np.dot(np.linalg.pinv(Square_matrix), np.transpose(X_train))
    w = np.dot(Projection_matrix, y_train)

    return w

def mse(X_train,y_train,w):
    '''Computes mean squared error to be used to assess the performance of chosen parameters'''

    # Find N and d
    N, d = X_train.shape
    
    # Augment dataset
    X_train = np.hstack((np.ones((N,1)), X_train))

    error_sum = 0

    # Squared error = (y_actual - y_pred)^2
    for example, feature_set in enumerate(X_train):
      error_sum += (y_train[example] - pred(X_train[example], w)) ** 2
    
    # Mean
    return error_sum / N

def pred(X_i,w):
    '''Applies model (just a linear equation) to obtain a prediction'''
    
    # Linear regression is just Y_pred = X_train * w 
    return np.dot(X_i, w)


def test_SciKit(X_train, X_test, Y_train, Y_test):
    '''Implements linear regression from SciKit'''

    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    return mean_squared_error(Y_test, Y_pred)

def subtestFn():
    # This function tests if your solution is robust against singular matrix

    # X_train has two perfectly correlated features
    X_train = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]])
    y_train = np.asarray([1,2,3,4])
    
    try:
      w=fit_LinRegr(X_train, y_train)
      print ("weights: ", w)
      print ("NO ERROR")
    except:
      print ("ERROR")

def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
    
    w=fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e=mse(X_test,y_test,w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)

print ('------------------subtestFn----------------------')
subtestFn()

print ('------------------testFn_Part2-------------------')
testFn_Part2()

'''
MSE between my version of linear regression and SciKit's is practically identical
(differences arise only after 10 decimal places), meaning performance is 
very similar. This means our weights are very close.
'''