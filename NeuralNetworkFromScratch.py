import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


def fit_NeuralNetwork(X_train,y_train,alpha,hidden_layer_sizes,epochs):
    '''After some heuristic data pre-processing, normalizations, and weight initalizations, we simply want to
    run a forward prop, use that output to run a backward prop, update the weights via SGD using the gradients found in
    backprop, and repeat this for every training sample to complete an epoch. This whole process repeats for as many epochs. '''

    # Initialize the epoch errors
    err=np.zeros((epochs,1))

    # Initialize the architecture
    N, d = X_train.shape
    x0 = np.ones((N,1))
    X_train = np.hstack((x0,X_train))
    d=d+1
    L=len(hidden_layer_sizes)
    L=L+2

    #Initializing the weights for input layer
    weight_layer = np.random.normal(0, 0.1, (d,hidden_layer_sizes[0])) #np.ones((d,hidden_layer_sizes[0]))
    weights = []
    weights.append(weight_layer) #append(0.1*weight_layer)

    #Initializing the weights for hidden layers
    for l in range(L-3):
        weight_layer = np.random.normal(0, 0.1, (hidden_layer_sizes[l]+1,hidden_layer_sizes[l+1]))
        weights.append(weight_layer)

    #Initializing the weights for output layers
    weight_layer= np.random.normal(0, 0.1, (hidden_layer_sizes[l+1]+1,1))
    weights.append(weight_layer)

    # Begin training
    avg_err_per_epoch_list = []

    for epoch in range(epochs):

      # Shuffle datapoint traversal order
      traversal_queue = np.arange(N)
      np.random.shuffle(traversal_queue)

      err_for_all_samples = 0

      # Traverse all datapoints with a training round
      for sample in traversal_queue:

        # Transpose input vector for dot product compatibility with weights
        x_augmented = np.transpose(X_train[sample])

        # Apply forward prop (X for a given node in a given layer is the activated version of S for that same node and layer)
        neuron_values_X, intermediate_dot_products_S = forwardPropagation(x_augmented, weights)

        # Apply back prop
        gradients = backPropagation(neuron_values_X,y_train[sample],intermediate_dot_products_S,weights)

        # Update weights
        weights = updateWeights(weights,gradients,alpha)

        # Compute training error after SGD with this particular sample
        err_for_all_samples += errorPerSample(neuron_values_X,y_train[sample])

      # calculate average error per sample for this epoch
      avg_err_per_epoch_list.append(err_for_all_samples/N)

    return avg_err_per_epoch_list, weights


def forwardPropagation(x, weights):
    '''One forward pass involves starting with a sample x_n. We then run that through the dot product of each weight
    into the next node, then apply the activation function. This is repeated for all layers until the output.'''

    # Initialize the lists that will house each layer's outputs (s^(l)) and inputs to the next layer (x^(l))
    X = [x]
    S = []

    ### Programming the first step explicitly for clarity/learning, then looping ###

    # x is a vector of size d^(l=0)+1, and represents a single training sample

    # At the input layer, take this vector & multiply it by the weight matrix from l=0 to l=1 (first element in w list)
    s_1 = np.dot(X[0], weights[0])    # s_L is a vector with the dot product for each layer's node, so d^(l+1)
    S.append(s_1)

    # Activate this result => i.e: we want theta(s^L)
    for i in range(len(s_1)): s_1[i] = activation(s_1[i])

    # Add a bias node, and now we have the set of inputs for the next layer step (l=1 to l=2)
    x_1 = np.hstack((1,s_1))
    X.append(x_1)

    # Repeat until layer L-1
    L_minus_one = len(weights)
    for i in range(1, len(weights) - 1):

      # compute s_l: vector of dot products for all nodes in general layer l
      s_l = np.dot(X[i], weights[i])
      S.append(s_l)

      # activate, this becomes x_l
      for j in range(len(s_l)): s_l[j] = activation(s_l[j])

      # add bias node
      x_l = np.hstack((1,s_l))
      X.append(x_l)

    # At layer L, we switch up the activation because we want a probability output (logistic regression)
    s_L = np.dot(X[-1], weights[-1])
    S.append(s_L)

    x_L = outputf(s_L) # one output node so s_L must be a scalar
    X.append(x_L)

    # This completes a full forward pass for a single input sample x_n (repeat for the remaining N-1 samples for an epoch)
    return X,S

def errorPerSample(X,y_n):
    '''Isolate the output value and send it to errorf to calculate loss'''
    return errorf(X[-1][0], y_n)

def backPropagation(X,y_n,s,weights):
    '''To compute the gradient of the loss function with respect to each weight, we need to compute the backward
    messages multiplied by the input node. The backward message at the last layer is straightforward, so we can go
    recursively towards the input layer'''

    # g becomes the gradient of loss wrt weights
    g = []

    # to match fwd prop notation
    S = s

    ### Programming the first step explicitly for clarity/learning, then looping ###

    # backward message: delta_L = dLoss/ds_L = dLoss/dx_L * dx_L/ds_L
    # X[-1] is last output (sigmoid-activated), x_L. S[-1] is input to the sigmoid, s_L
    # 0-index is to extract it, but it's a scalar anyways
    delta_L = derivativeError(X[-1][0], y_n) * derivativeOutput(S[-1][0])

    # calculate loss gradients for L-1 to L weights: dLoss/dweight_i,j = delta_L * x_i^(L-1)
    grads = np.zeros(weights[-1].shape)
    for i in range(len(X[-2])):
        grads[i][0] = X[-2][i] * delta_L
    g.insert(0, grads) # insert this matrix to our list of weight gradients

    # before we move on, we need to calculate the backward messages for layer L-1
    # each node in L-1 needs a backward message, except the bias node
    w = weights[-1].copy()
    w = np.delete(w, 0, axis=0)

    constant = np.dot(w, delta_L)
    theta_prime_vector = np.zeros(constant.shape)
    for i in range(len(S[-2])):
      theta_prime_vector[i] = derivativeActivation(S[-2][i])

    # elementwise product between the (dot product of weights with next msg) and (activation derivative)
    # delta_Lminus1 = np.dot(delta, weights_from_L-1toL) x (theta_prime of s^(l-1))
    delta_Lminus1 = np.multiply(constant, theta_prime_vector) # elementwise product

    # calculate loss gradients for L-2 to L-1 weights
    # dLoss/dweight_i,j =  x_i^(L-1) * delta_Lminus1
    grads = np.zeros(weights[-2].shape)
    for i in range(len(X[-3])):
      for j in range(len(delta_Lminus1)):
        grads[i][j] = X[-3][i] * delta_Lminus1[j][0]
        # really it's delta_Lminus[j], the [j][0] is just to extract the element from the 1x1 array
        # to make it a compatible datatype with X[k][i]
    g.insert(0, grads) # insert this matrix to our list of weight gradients

    ### GENERALIZE (and repeat for entire list of weight matrices) ###
    delta_Lnext = delta_Lminus1
    for k in range(len(weights) - 2):

      w = weights[-k-2].copy()
      w = np.delete(w, 0, axis=0)
      constant = np.dot(w, delta_Lnext)

      theta_prime_vector = np.zeros(constant.shape)

      for i in range(len(S[-k-3])):
        theta_prime_vector[i] = derivativeActivation(S[-k-3][i])

      delta_Lcurr = np.multiply(constant, theta_prime_vector) # elementwise product

      # calculate loss gradients for L-2 to L-1 weights: dLoss/dweight_i,j =  x_i^(L-1) * delta_L
      grads = np.zeros(weights[-k-3].shape)
      for i in range(len(X[-k-4])):
        for j in range(len(delta_Lcurr)):
          grads[i][j] = X[-k-4][i] * delta_Lcurr[j][0] #see above note about delta_Lminus[j][0]
      g.insert(0, grads)

      delta_Lnext = delta_Lcurr

    return g


def updateWeights(weights,g,alpha):
    '''Implement stochastic gradient descent with the gradients found from backpropagation'''
    
    nW=[]

    # each weight matrix in the list has a different size depending on the layers it connects
    for i in range(len(weights)):
        rows, cols = weights[i].shape
        currWeight=weights[i]
        currG=g[i]
        for j in range(rows):
            for k in range(cols):

                # weight = weight - (alpha * g)
                currWeight[j,k]= currWeight[j,k] - alpha*currG[j,k]

        nW.append(currWeight)
    return nW





def activation(s):
    '''ReLu is the activation function that will be implemented upon every summation scalar s
    at every node, except at the very last layer where sigmoid activation is applied for logistic regression'''
    return max(0,s)

def derivativeActivation(s):
    '''Derivative of ReLu is needed to calculate backward messages (this is one of the two chain rules)'''
    if (s > 0):
        return 1
    else:
        return 0

def outputf(s):
    '''NN's are symmetric internally except for two aspects: the final activation into the output layer
    AND the loss function. In this case, this is a logistic regression model so the final activation is sigmoid'''
    return (1/(1+np.exp(-s)))

def derivativeOutput(s):
    '''The very first backward message (which is at the output layer) is straightforward.
    delta_L = dLoss/ds_L = dLoss/dx_L * dx_L/ds_L. This is dx_L/ds_L.'''

    # Derivative of sigmoid (dx_L/ds_L)
    return np.exp(-s)/((1+np.exp(-s))**2)

def errorf(x_L,y):
    '''errorf is the loss function for logistic regression, which is the log loss function
    error = -log(P(y|x)) where P(y|x) = sigmoid(s) for y = 1, and 1 - sigmoid(s) for y = -1'''

    # x_L = sigmoid(s) already, so we just need to apply the condition stated above
    if (y == 1):
        return -np.log(x_L)
    else:
        return -np.log(1-x_L)

def derivativeError(x_L,y):
    '''Recall the very first backward message (which is at the output layer) is straighforward.
    delta_L = dLoss/ds_L = dLoss/dx_L * dx_L/ds_L. This is dLoss/dx_L.'''

    # Derivative of log loss (dLoss/dx_L)
    if (y == 1):
        return -1/x_L
    else:
        return 1/(1-x_L)

def pred(x_n,weights):
    '''Simply use forward propagation on optimal weights, and check the output neuron'''

    X,S = forwardPropagation(x_n, weights)

    # 0.5 probability is threshold we belong to a certain class, so let's check the output neuron
    if X[-1][0] >= 0.5:
        return 1
    else:
        return -1


def confMatrix(X_train,y_train,w):
    '''Use our model to predict class outputs, and observe accuracy using confusion matrix'''

    ### X_train, y_train poorly named here. This can be X_test or y_test for example. ###
    #i.e: Confusion matrix can be used for any type of dataset 

    # Augment entire dataset
    N, d = X_train.shape
    X_augmented = np.hstack((np.ones((N,1)), X_train))

    # Apply predictions to entire dataset
    y_pred = np.zeros(N)

    # at this point, weights have been optimized
    for sample in range(len(X_augmented)):
        y_pred[sample] = pred(X_augmented[sample],w)

    # compare prediction with truth value and depict false and true positives and negatives
    cM = confusion_matrix(y_train, y_pred)

    return cM


def plotErr(e,epochs):
    '''Plot training error that we tabulated after each sample and each epoch of fit_NeuralNetwork()'''

    # x-axis is epochs, y-axis is error average error per sample for each epoch 
    plt.plot(range(epochs),e)
    plt.xlabel('Epoch')
    plt.ylabel('Training Error')
    plt.show()
    return

def test_SciKit(X_train, X_test, Y_train, Y_test):
    '''Constructing NN using SciKit's built-in implementation on the dataset. Will compare to ours'''

    # Initialize and train model (SciKit handles all biasing, data augmentation, etc.)
    nn=MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes=(30, 10), random_state=1)
    nn.fit(X_train, Y_train)

    # Conduct predictions on test set after the NN has been trained
    Y_pred = nn.predict(X_test)

    # Assess performance
    cM=confusion_matrix(Y_test, Y_pred)

    ### SEPARATELY INTERESTED IN SOME TRAINING AND TEST ACCURACIES AS WELL ###
    print(f"Training accuracy for (30, 10) is {nn.score(X_train, Y_train)}")
    print(f"Testing accuracy for (30, 10) is {nn.score(X_test, Y_test)}\n")

    # Repeat for hidden_layers = (5,5) and (10,10)
    hidden_layers = [(5,5), (10,10)]
    for hidden_layer_sizes in hidden_layers:
      nn=MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes=hidden_layer_sizes, random_state=1)
      nn.fit(X_train, Y_train)
      Y_pred = nn.predict(X_test)
      print(f"Training accuracy for {hidden_layer_sizes} is {nn.score(X_train, Y_train)}")
      print(f"Testing accuracy for {hidden_layer_sizes} is {nn.score(X_test, Y_test)}\n")

    return cM

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2, random_state=1)

    for i in range(80):
        if y_train[i]==1:
            y_train[i]=-1
        else:
            y_train[i]=1
    for j in range(20):
        if y_test[j]==1:
            y_test[j]=-1
        else:
            y_test[j]=1

    err,w=fit_NeuralNetwork(X_train,y_train,1e-2,[30, 10],100)

    plotErr(err,100)

    cM=confMatrix(X_test,y_test,w)

    sciKit=test_SciKit(X_train, X_test, y_train, y_test)

    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)

test_Part1()


### Asser's test functions ### (FINAL SUBMISSION)
'''
# input vector
input = np.array([1,1,2]) #d^(l=0) + 1 = 3

# assume all hidden layers have 2 nodes including bias, with 5 layers total. So 4 3x2 matrices (last matrix is 3x1 cuz 1 node at layer L)
weights = [[[1,1],[1,1],[1,1]],[[1,1],[1,1],[1,1]],[[1,1],[1,1],[1,1]],[[1],[1],[1]]]
fwd_output = forwardPropagation(input, weights.copy())
print(fwd_output)
print()


### TESTING BACK PROP ###
back_output = backPropagation(fwd_output[0], 1, fwd_output[1], weights.copy())
print(back_output)
print(weights)

newWeights = updateWeights(weights,back_output, 0.001)
print(newWeights)
'''