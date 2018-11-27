# *******************************************************
# hw6ec module
# Assignment 6 extra credit
# ENGI E1006
# *******************************************************

import numpy as np
import scipy.stats as stats
import scipy.cluster as sc
import scipy.spatial as spatial
import matplotlib.pyplot as plt

def practice():
    '''this function creates a set of synthetic data for testing. it
    returns an array of random data.'''    
    
    x = np.random.multivariate_normal([2, 5.5], [[1,1], [1,2.5]], 250)
    y = np.random.multivariate_normal([.5, 1], [[1, 0], [0, 1]], 250)
    
    x1 = x[:,0]
    x2 = x[:,1]
    x3 = np.zeros_like(x1)
    
    y1 = y[:,0]
    y2 = y[:,1]
    y3 = np.ones_like(y1)
    
    Col1 = np.vstack((x1, x2, x3))
    Col2 = np.vstack((y1, y2, y3))

    result = np.hstack((Col1, Col2))
    result = np.swapaxes(result, 0, 1)    
        
    return result
    
def readData(filename):
    '''this function loads data from a file into an array, and processes
    it for use. It returns an array.'''    
    
    data = np.genfromtxt(filename, dtype = str, delimiter = ',')

    data = data[:,1:]
    
    data_len = data.shape[0]
    data_wid = data.shape[1]
    
    for i in range(data_len):
        if data[i][0] == 'M':
            data[i][0] = 1
        elif data[i][0] == 'B':
            data[i][0] = 0 
    
    data = np.roll(data, data_wid - 1, axis = 1)
    
    new_data = data.astype(float, copy = False)

    return new_data    
    
def KNNclassifier(training, test, k, p):
    '''Here k is an integer and everything else is as before. When labeling
    the test data this classifier checks the k nearest neighbors to a test
    observation and uses the most common label among these k neighbors as 
    its guess for the test observation. Note that if k = 1 this classifier
    should be the same as NNclassifier.''' 

    #scipy function to find array of KNN    
    tree = spatial.KDTree(training)
    d, IDofKNN = tree.query(test, k, p)
        
    #get dimensions so I can flatten and reshape
    ID_wid = IDofKNN.shape[0]
    if k > 1:
        
        ID_len = IDofKNN.shape[1]

    else:
        
        ID_len = 1
        
    #flattened arrays to make loops and debugging easier
    flatID = IDofKNN.flatten()
    flatLabel = np.copy(flatID)

    for item in range((ID_wid * ID_len)):
        flatLabel[item] = training[int(flatID[item]), -1]
    
    #reshape labels    
    LabelofKNN = np.reshape(flatLabel, (ID_wid, ID_len))
    
    #find mode of each row
    test_ans, counts = stats.mode(LabelofKNN, axis = 1)
    
    return test_ans    
    
    
def NNclassifier(training, test):
    '''this function uses the training data to provide labels for a set
    of test data. This function takes as input an q x (n+1) array, training, 
    that consists of q rows of observation-label pairs. That is, each row is 
    an n-dimensional observation concatenated with an extra dimension for 
    the class label. The other parameter is a j x n array consisting of j 
    unlabeled, n-dimensional observations. This function will output a 
    1-dimensional array consisting of j labels for the test array 
    observations. It determines those labels in the following way: 
    For each observation (row) of the test array it will find the nearest 
    observation in the training array and use its label to label the test 
    observation. '''
    
    #for each element in test, find NN in training, put NN label in new 1D
    #array to return    
    test_ans = np.zeros(test.shape[0])
    test_shape = test.shape
    tng_shape = training.shape
    tng_size = tng_shape[0]    
    test_size = test_shape[0]
    ans_col = test_shape[1]    
    tng_labels = training[:,ans_col - 1]

    #actual finding NN
    IDofNN, dist = sc.vq.vq(test, training)    

    #get label from training, using ID, and put in test_Ans
    #slice labels into new array
    #use that slice to fill test_ans
    
    for item in range(test_size):
        IDgetting = IDofNN[item]
        test_ans[item] = tng_labels[IDgetting]
        
    return test_ans
    
    
def n_validator(data, p, classifier, *args):
    '''this function is to estimate the performance of a classifier in a 
    particular setting. This function takes as input an m x (n+1) array of 
    data, an integer p, the classifier it is checking, and any remaining 
    parameters that the classifier requires will be stored in args .  Here's 
    how it works: It will first randomly mix-up the rows of data. Then it 
    will partition data into p equal parts. If p equal parts are not possible 
    then as close to equal as is achievable (no cell should have more than 1 
    more observation than the other cells). Next it will test the classifier 
    on each part by passing the classifier p-1 of the parts of data in 
    training and the other part in test. It will check the labels the 
    classifier returns against the actual labels stored in data to provide a 
    score for that partition. It will sum the scores across all p partitions 
    and then divide this by m. This number is the estimate of the classifier's 
    performance on data from this source. It should return this number 
    (between 0 and 1).'''
    #declare some overhead
    successes = 0
    attempts = 0
    
    k = args[0]
    dist = args[1]
    
    # permute data array
    # mixed_data = data
    mixed_data = np.random.permutation(data)

    # loop to cycle folds and count success
    for i in range(p):
        
        # This splits data into list D of p folds    
        D = np.array_split(mixed_data, p)

        # B is the fold popped off, slice B into test and answers 
        B = D.pop(i - 1)
        
        B_ans = B[:,-1]
        #B = B[:,:-1]
        B = B[:,:B.shape[1]]
        
        # create ANS
        ANS = np.array(B.shape[0])
        
        # C is the remaining folds, stacked into TNG
        C = np.vstack(D)
        
        # ANS is array of answer labels
        ANS = classifier(C, B, k, dist)

        # loop to count successes and attempts
        for item in range(len(B_ans)):
            
            if ANS[item - 1] == B_ans[item - 1]:
                successes = successes + 1
            
            attempts = attempts + 1
        
    # accuracy determined and passed back
    accuracy = successes / attempts
        
    return accuracy