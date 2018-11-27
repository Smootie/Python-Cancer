# *******************************************************
# hw5b module
# Assignment 5 Part 2
# ENGI E1006
# *******************************************************
'''Write a main function that uses n_validator from part 1 to test your 
new classifier using odd k from 1 to 15. Report the best value of k you find 
for each of the data sets used in Part 1.
'''

import cancerpredicter as cp
import numpy as np

def main():

    # randomize    
    np.random.seed()
    
    # call function to build sythentic data
    A = cp.practice()
    
    # read real-world data into array
    D = cp.readData('wdbc.data')
    
    # p is the arg to change the method of finding distance
    # p = 1 uses 'manhattan distance'.
    p = 1    
    
    # test KNN
    best_Kpercent = 0
    best_Ki = 0
    for i in range(1, 15, 2):
        
        K_accuracy = cp.n_validator(A, 10, cp.KNNclassifier, i, p)
        if K_accuracy > best_Kpercent:
            best_Kpercent = K_accuracy
            best_Ki = i
        
    best_CKpercent = 0.00
    best_CKi = 0
    for i in range(1, 15, 2):
        
        CK_accuracy = cp.n_validator(D, 10, cp.KNNclassifier, i, p)
        if CK_accuracy > best_CKpercent:
            best_CKpercent = CK_accuracy
            best_CKi = i
        
    
    #CK_accuracy = cp.n_validator(D, 10, cp.KNNclassifier, 3)
    
    # print results
    #print ('The synthetic data accuracy is: ' + str(accuracy))
    #print ('The real data accuracy is: ' + str(C_accuracy))
    print ()
    print ("Using 'Manhattan distance'")
    print ('Best syth data K: ' + str(best_Ki) + ' and %: ' + str(best_Kpercent))
    print ('Best real data K: ' + str(best_CKi) + ' and %: ' + str(best_CKpercent))
    
    # test KNN again using a different distance method, p
    p = 3    
    
    best_Kpercent = 0
    best_Ki = 0
    for i in range(1, 15, 2):
        
        K_accuracy = cp.n_validator(A, 10, cp.KNNclassifier, i, p)
        if K_accuracy > best_Kpercent:
            best_Kpercent = K_accuracy
            best_Ki = i
        
    best_CKpercent = 0.00
    best_CKi = 0
    for i in range(1, 15, 2):
        
        CK_accuracy = cp.n_validator(D, 10, cp.KNNclassifier, i, p)
        if CK_accuracy > best_CKpercent:
            best_CKpercent = CK_accuracy
            best_CKi = i
        
    
    #CK_accuracy = cp.n_validator(D, 10, cp.KNNclassifier, 3)
    
    # print results
    #print ('The synthetic data accuracy is: ' + str(accuracy))
    #print ('The real data accuracy is: ' + str(C_accuracy))
    print ()
    print ("Using an undocumented distance formula native to KDtree")
    print ("   (here p = 3, but p could be as high as infinity)")
    print ('Best syth data K: ' + str(best_Ki) + ' and %: ' + str(best_Kpercent))
    print ('Best real data K: ' + str(best_CKi) + ' and %: ' + str(best_CKpercent))
     
main()