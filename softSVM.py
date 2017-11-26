# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:59:48 2017

@author: Andrei Baraian
@author: Lucian Epure
"""

import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix,solvers
from math import exp

##--------------------------------------------------------------------------
##
##  Compute the P matrix for this kernel-based classifier.
##
##          [                    ]   (n x n matrix with elements 
##      P = [ t_i t_j K(x_i x_j) ]    t_i t_j x_i x_j).
##          [                    ]

def makeP(x,t,K):
    N = len(x)
    assert N == len(t)
    P = matrix(0.0,(N,N),tc='d')
    for i in range(N):
        for j in range(N):
            P[i,j] = t[i] * t[j] * K(x[i],x[j])
    return P

##--------------------------------------------------------------------------
##
##  Compute the bias for this kernel-based classifier.
##
##  The bias can be generated from any support vector which satisfies the 
##  condition to be non zero and <= C, but it is better to average over 
##  all support vectors.
##
##      b = Ts[n] - sum_i Ls[i] * Ts[i] * K(Xs[i],Xs[n]), 0 < Ls[i] < C
##
    
def makeB(Xs,Ts,C,Ls,K,status):
    if status != "optimal" : raise Exception("Can't find Lambdas")
    sv_count = 0
    b_sum = 0.0
    for n in range(len(Ts)):
        if Ls[n] >= 1e-10 and Ls[n] < C:  ##  0 < Ls[i] < C
            sv_count += 1
            b_sum += Ts[n]
            for i in range(len(Ts)):
                b_sum -= Ls[i] * Ts[i]  * K(Xs[i],Xs[n]) 
    return b_sum/sv_count


##--------------------------------------------------------------------------
##
##  makeLambdas
##
##  We use the qp solver from the cvx package to find a list of Lagrange
##  multipliers (lambdas or L's) for an Xs/Ts problem, where Xs is a list
##  of input vectors (themselves represented as simple lists) and Ts a list
##  of desired outputs.
##
##
##  Note that we are trying to solve the problem:
##
##      Maximize:
##
##        W(L) =  \sum_i L_i
##                - 1/2 sum_i sum_j L_i * L_j * t_i * t_j * K(x_i,x_j)
##
##      subject to:  \sum_i t_i L_i  =  0   and   L_i >= 0 and L_i <= C.
##
##
##  but the "standard" quadratic programming problem is subtly different,
##  it attempts to *minimize* the following quadratic form:
##
##        f(y) = 1/2 y^t P y  +  q^t y
##
##  subject to:  G L <= h   and   A y = b, where P is an n x n 
##  symmetric matrix, G is an 2*n x n matrix, A is a 1 x n
##  (row) vector, q is an n x 1 (column) vector, as are h and y.
##  N.B., Vector y is the solution being searched for.
##
##
##          [-1.0]
##             .
##             .
##      q = [-1.0]      (n element column vector).
##          [-1.0]
##             .
##             .
##          [-1.0]
##
##
##          [-1.0,  0.0  ....  0.0]
##          [ 0.0, -1.0           ]
##          [             .       ]   
##          [             .       ]    
##          [            0.0,     ]   
##          [           -1.0,  0.0]
##          [            0.0, -1.0]
##      G = [ 1.0,  0.0  ....  0.0]
##          [ 0.0,  1.0           ]
##          [             .       ]   
##          [             .       ]   
##          [            0.0,     ]    
##          [            1.0,  0.0]
##          [            0.0,  1.0]
##
##
##
##      A = [ t_1, t_2, t_3, ... t_n],  a row vector with n elements
##                                      made using the t list input.
##
##
##          [0]
##           .
##           .
##      h = [0]      (2n element column vector of n 0's and n C's).
##          [C]
##           .
##           .
##          [C]
##    
##
##      b = [0.0], i.e., a 1 x 1 matrix containing 0.0.
##
##
##          [                    ]   (n x n matrix with elements 
##      P = [ t_i t_j K(x_i x_j) ]    t_i t_j x_i x_j).
##          [                    ]
##
##
##  The solution (if one exists) is returned by the "qp" solver as
##  a vector of elements.  The solver actually returns a dictionary
##  of values, this contains lots of information about the solution,
##  quality, etc.  But, from the point of view of this routine the
##  important part is the vector of "l" values, which is accessed
##  under the 'x' key in the returned dictionary.
##
    
def makeLambdas(Xs,Ts,C,K):
    P = makeP(Xs,Ts,K)
    n = len(Ts)
    q = matrix(-1.0,(n,1))      ## n-element column vector of -1.0
    h = matrix(0.0,(2*n,1))     ## Build the first half of h with 0 elements
    h[n:2*n,0] = C              ## Set the other half of h with C elements
    G = matrix(0.0,(2*n,n))     ## 2*n * n matrix
    G[::(2*n+1)] = -1.0         ## -1 on the "first" main diagonal
    G[n::(2*n+1)] = 1           ## 1 on the "second" main diagonal
    A = matrix(Ts,(1,n),tc='d') ## n-element row vector of training outputs
    r = solvers.qp(P,q,G,h,A,matrix(0.0))  ## call qp to solve the dual function
    Ls = [round(l,6) for l in list(r['x'])]
    ## return results
    ## First element is a string, which will be "optimal" if a solution
    ## has been found. The second element is a list of Lagrange multipliers
    ## for the problem, rounded to six decimal digits to remove algorithm noise.
    ## 
    return (r['status'],Ls)

    
##--------------------------------------------------------------------------
##
##  rbfKernel
##
##  Return the radial basis function kernel exp(-||x-y||^2/2*sigma^2).
##  By default, sigma2 is 0.25
##
##
def rbfKernel(v1,v2,sigma2=0.25):
    assert len(v1) == len(v2)
    assert sigma2 >= 0.0
    mag2 = sum(map(lambda x,y: (x-y)*(x-y),v1,v2))
    return  exp(-mag2/(2.0 * sigma2))

##--------------------------------------------------------------------------
##
## Classify a single input vector using a trained nonlinear, kernel based SVM
## 
def classify(x,Xs,Ts,Ls,b,K,verbose=True):
    y = b
    for n in range(len(Ts)):
        if Ls[n] >= 1e-10:
            y += Ls[n] * Ts[n] * K(Xs[n],x)
    if verbose:
        print "%s %8.5f --> " %(x,y),
        if y > 0.0: print "+1"
        elif y < 0.0: print "-1"
        else: print "0 (ERROR)" 
    if y > 0.0:
        return +1
    elif y < 0.0:
        return -1
    else:
        return 0
    
##--------------------------------------------------------------------------
##
## Test a non linear, kernel based SVM
## As parameters, we need to supply the training data (Xs and Ts), the testing
## data (Xtest, Ttest), the Lagrange multipliers and the bias obtained by 
## training the classifier and the kernel
##
        
def testClassifier(Xs,Xtest,Ts,Ttest,Ls,b,K,verbose=True):
    assert len(Xs) == len(Ts)
    misclassification = 0
    for i in range(len(Xtest)):
        c = classify(Xtest[i],Xs,Ts,Ls,b,K,verbose)
        if c != Ttest[i]:
            if verbose:
                print "Misclassification: input %s, output %d, expected %d" %(Xtest[i],c,Ttest[i])
            misclassification += 1
    return misclassification   

##--------------------------------------------------------------------------
##
## Predict the values of a test set  using a trained SVMand return them as an array
## The parameters are:  d - the testing set
##                      Xs, Ts - training data
##                      Ls, b - the Lagrange multipliers and the bias
##    

def predict(d,Xs,Ts,Ls,b):
    y_predict = np.zeros(len(d))
    for i in range(len(d)):
        s = 0   ## will be the prediction for each input vector in d
        for j in range(len(Ts)):
            if Ls[j] >= 1e-10:
                s += Ls[j] * Ts[j] * K(Xs[j],d[i])
        y_predict[i] = s
    return y_predict + b  ## add the bias to every prediction

##--------------------------------------------------------------------------
##
## Plot the decision boundary and the margins of the trained SVM
## The parameters are the training data, Lagrange multipliers and the bias
## We generate the whole space of the grid and for each point in the grid
## we calculate the its prediction. As a contour, we plot only the points
## whose prediction are on the decision boundary, +1 margin or -1 margin
##
## A point xc is on the decision boundary if sum_r Ls[r] * Ts[r] * K(Xs[r],xc) = 0
## If it is on the +1 margin, then sum_r Ls[r] * Ts[r] * K(Xs[r],xc) = 1 and
## if it is on the -1 margin, then sum_r Ls[r] * Ts[r] * K(Xs[r],xc) = -1 and
    
def plotBoundaries(Xs,Ts,Ls,b):
    X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))  ## generate the grid space
    X = np.array([[x1, x2] for x1,x2 in zip(np.ravel(X1),np.ravel(X2))]) ## convert the grid space in an np.array
    d = X[:,[0,1]] ## convert the grid space in an array
    Z = predict(d,Xs,Ts,Ls,b).reshape(X1.shape) ## get the predictions for every point
    plt.contour(X1, X2, Z, [0.0], colors='green', linewidths=2, origin='lower')  ## plot the decision boundary
    plt.contour(X1, X2, Z + 1, [0.0], colors='blue', linewidths=2, origin='lower') ## plot the -1 margin
    plt.contour(X1, X2, Z - 1, [0.0], colors='red', linewidths=2, origin='lower') ## plot the +1 margin
    
    #plt.savefig("trainingSmallC.png")
    #plt.savefig("trainingLargeC.png")
    #plt.savefig("testingSmallC.png")
    #plt.savefig("testingLargeC.png")
    
##--------------------------------------------------------------------------
    
## Plot the data points
## Points from class 1 (expected classifier output: +1) - Red
## Points form class 2 (expected classifier output: -1) - Blue
## For the training set plot the point with a circle and for the testing
## set plot the points with an x (test parameter)
##
    
def plotData(x,y,test=False):
    pos = np.where(y == 1)
    neg = np.where(y == -1)
    if test:
        posPoint = 'rx'
        negPoint = 'bx'
    else:
        posPoint = 'ro'
        negPoint = 'bo'
    plt.plot(x[pos,0],x[pos,1],posPoint)
    plt.plot(x[neg,0],x[neg,1],negPoint)
##--------------------------------------------------------------------------
 
data = np.loadtxt("training-dataset-aut-2017.txt")   ## Read the training data set

## Split the training data in feature(x) and desired output(y)
Xs = data[:,[0,1]]
Ts = data[:,2]


#plotData(Xs,Ts)   ## Plot the training data

K = rbfKernel       ## Define the radial basis function kernel

## Initialize the regularization parameter, C, which controls the amount of 
## allowed soft error in the solution. Small C allows misclassification 
## but give wider margins, large C causes hard-margin behaviour, no 
## misclassifications but potentially narrow margins and overfitting
## small C = 1
## high C = 1000000

C = 1
#C = 1000000

stat, Ls = makeLambdas(Xs,Ts,C,K)   ## Compute the Lagrange multipliers (lambdas)
b = makeB(Xs,Ts,C,Ls,K,stat)        ## Find the bias, no explicit w vector needed

## Test classifier using the training data 
##
## When using C = 1, there are 7 misclassifications, since we don't penalize 
## so much the points situated on the wrong side of the decision boundary. 
##
## When setting C = 1000000, there are 0 misclassifications, because we
## generate a complex boundary
## 
## It is not relevant testing our classifier on the training data, but we 
## should test it against previously unseen data sets
miss = testClassifier(Xs,Xs,Ts,Ts,Ls,b,K,False)
print "There were %d misclassications when using the training data as testing set with C=%d!" %(miss,C)
print "--------------------------------------------------------"

dataTest = np.loadtxt("testing-dataset-aut-2017.txt")    # Read testing data

## Split the testing data in feature(x) and desired output(y)
Xtest = dataTest[:,[0,1]]
Ttest = dataTest[:,2]

plotData(Xtest,Ttest,True)      ## Plot the testing data

## Test classifier using the testing data
##
## When setting C = 1, we get 159 misclassifications, out of 1000 data points,
## resulting in a performance of 15,9% misclassifications.
##
## When setting C = 1000000, we get 248 misclassifications, put of 1000 data points,
## resulting in a performance of 24,8% misclassifications, performing much 
## worse than when we used C = 1
##
## Conclusion
## 
## When setting C very high, we get 0 misclassifications on the training set, but that
## does not define its performance. When testing on previously unseen data points,
## we get a poor performance of misclassficiations, since we obtained an 
## overfitted classifier. It is better to set C to a lower value (1), even though
## it does have some misclassifications in the training data, but performs
## better on previously unseen data points.
miss = testClassifier(Xs,Xtest,Ts,Ttest,Ls,b,K,False)
print "There were %d misclassications when using the testing data as testing set with C=%d!" %(miss,C)
print "--------------------------------------------------------"

## plot the decision boundary and the +1 and -1 margins
plotBoundaries(Xs,Ts,Ls,b)