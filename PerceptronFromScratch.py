import numpy as np
from sklearn.externals._packaging.version import NegativeInfinity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix 

# Asser Abdelgawad

def fit_perceptron(X_train, y_train):
    '''Implements PLA to find a w that minimizes in-sample training error'''
    
    # Limits  
    max_epochs = 5000
    max_errorPer = np.inf

    # Find number of examples (N) and features (d) in the dataset
    N, d = X_train.shape
    
    # Augment dataset for training
    X_train = np.hstack((np.ones((N,1)), X_train))

    # Initalize w
    w = np.zeros((d+1,1))
    pocket_w = w

    # PLA
    for i in range(0, max_epochs):

      # Stop if perfect classifer found
      if (errorPer == 0):
        pocket_w = w

      # Update rule
      for example, feature_set in enumerate(X_train):

        # w = w + (y_n * x_n) if there is a misclassification (otherwise do nothing)
        if (pred(X_train[example], w) != y_train[example]):
          w = np.transpose(w) + (y_train[example] * X_train[example]) 
          w = w.T
          break

      # Check if the new weight vector has least error so far, and store if so
      if (errorPer(X_train, y_train, w) < max_errorPer):
        pocket_w = w

    return pocket_w

def errorPer(X_train,y_train,w):
    '''Computes in-sample training error with a zero-one loss function'''

    # Initialize
    N = (X_train.shape)[0] - 1 # Minus 1 to de-augment
    error_sum = 0

    # For every example, we apply our classifier model and obtain its response
    # Checking this response against true values gives us an error sum
    for example, feature_set in enumerate(X_train):
      if (pred(X_train[example], w) != y_train[example]):
        error_sum += 1

    # Error sum is normalized by the number of examples
    return error_sum / N

def confMatrix(X_train,y_train,w):
    '''Implements visual indicator of model performance on new data'''

    true_neg = 0
    false_neg = 0
    true_pos = 0
    false_pos = 0

    # Find N and d, then augment dataset so we can apply classifier
    N, d = X_train.shape
    X_train = np.hstack((np.ones((N,1)), X_train))

    for example, feature_set in enumerate(X_train):

      # Compare y_pred vs. y_actual for test dataset
      match (pred(X_train[example], w), y_train[example]):

        case (1,1):
          true_pos += 1

        case (1,-1):
          false_pos += 1

        case (-1,1):
          false_neg += 1

        case (-1,-1):
          true_neg += 1

    # Format
    confusion_matrix = [[true_neg, false_pos],
                        [false_neg, true_pos]]

    return confusion_matrix

  

def pred(X_i,w):
    '''Implement classifier's decision rule, a sign function'''
 
    # The rule is y_pred = sign(w * x_i)
    if (np.dot(X_i, w) > 0):
      return 1
    else: 
      return -1

def test_SciKit(X_train, X_test, Y_train, Y_test):
    '''Implements pocket PLA from SciKit for binary classifiaction'''

    model = Perceptron(random_state=42, max_iter=5000) 
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    return confusion_matrix(Y_test, Y_pred)

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)

    #Set the labels to +1 and -1
    y_train[y_train == 1] = 1
    y_train[y_train != 1] = -1
    y_test[y_test == 1] = 1
    y_test[y_test != 1] = -1

    #Pocket algorithm using Numpy
    w=fit_perceptron(X_train,y_train)
    cM=confMatrix(X_test,y_test,w)

    #Pocket algorithm using scikit-learn
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    #Print the result
    print ('--------------Test Result-------------------')
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
    

test_Part1()
