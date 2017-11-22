# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:59:48 2017

@author: Andrei Baraian
"""

import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix,solvers
from math import exp

def makeP(x,t,K):
    N = len(x)
    assert N == len(t)
    P = matrix(0.0,(N,N),tc='d')
    for i in range(N):
        for j in range(N):
            P[i,j] = t[i] * t[j] * K(x[i],x[j])
    return P

def makeB(x,y,C,Ls,K,status):
    if status != "optimal" : raise Exception("Can't find Lambdas")
    sv_count = 0
    b_sum = 0.0
    for n in range(len(y)):
        if Ls[n] >= 1e-10 and Ls[n] < C:
            sv_count += 1
            b_sum += y[n]
            for i in range(len(y)):
                #if Ls[i] >= 1e-10 and Ls[n] < C:
                b_sum -= Ls[i] * y[i]  * K(x[i],x[n]) 
    return b_sum/sv_count

def makeLambdas(X,T,C,K):
    P = makeP(X,T,K)
    n = len(T)
    q = matrix(-1.0,(n,1))
    h = matrix(0.0,(2*n,1))
    h[n:2*n,0] = C
    G = matrix(0.0,(2*n,n))
    G[::(2*n+1)] = -1.0
    G[n::(2*n+1)] = 1
    A = matrix(T,(1,n),tc='d')
    r = solvers.qp(P,q,G,h,A,matrix(0.0))
    a=np.ravel(r['x'])
    Ls = [round(l,6) for l in list(r['x'])]
    return (r['status'],Ls,a)
    

def rbfKernel(v1,v2,sigma2=0.25):
    assert len(v1) == len(v2)
    assert sigma2 >= 0.0
    mag2 = sum(map(lambda x,y: (x-y)*(x-y),v1,v2))
    return  exp(-mag2/(2.0 * sigma2))

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
 
def testClassifier(Xs,Ts,Ls,b,K,verbose=True):
    assert len(Xs) == len(Ts)
    good = True
    misclassification = 0
    for i in range(len(Xs)):
        c = classify(Xs[i],Xs,Ts,Ls,b,K)
        if c != Ts[i]:
            if verbose:
                print "Misclassification: input %s, output %d, expected %d" %(Xs[i],c,Ts[i])
                #plt.plot(Xs[i,0],Xs[i,1],'go')
            good = False
            misclassification += 1
    return misclassification   

def project(d,b,Ls,Ts,K,a1,svx,svy):
    y_predict = np.zeros(len(d))
    for l in range(len(d)):
        s = 0
        for i,j,k in zip(a1,svy,svx):
            print "here is d"
            print d[l]
            print "here is svx"
            print svx
            s += i * j * K(d[l],svx)
        y_predict[l] = s
    return y_predict + b

def plotBoundaries(b,Ls,Ts,K,a1,svx,svy):
    X1, X2 = np.meshgrid(np.linspace(-8,8,20), np.linspace(-6,6,20))
    X = np.array([[x1, x2] for x1,x2 in zip(np.ravel(X1),np.ravel(X2))])
    #d = X[:,[0,1]]
    Z = project(X,b,Ls,Ts,K,a1,svx,svy).reshape(X1.shape)
    plt.contour(X1, X2, Z, [0.0], colors='green', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z + 1, [0.0], colors='red', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z - 1, [0.0], colors='blue', linewidths=1, origin='lower')
    
    plt.axis("tight")
    plt.show()
    
def plotSV(Ls,Ts,b,K):
    X1, X2 = np.meshgrid(np.linspace(-8,8,50), np.linspace(-6,6,50))
    X = np.array([[x1, x2] for x1,x2 in zip(np.ravel(X1),np.ravel(X2))])
    #d = X[:,[0,1]]
    d = np.array([[2.045, 0.258]])
    print d
    #print d
    #print len(d)
    for i in range(len(d)):
        s = b
        for j in range(len(Ts)):
            if Ls[j] >= 1e-10:
                s += Ls[j] * Ts[j] * K(d[j],d[i])
        if s >= 0.0:
            plt.plot(d[i,0],d[i,1],'go')
        if s <= 0.0:
            plt.plot(d[i,0],d[i,1],'yo')
        
    

data = np.loadtxt("training-dataset.txt")
#data = np.loadtxt("train1.txt")
x = data[:,[0,1]]
y = data[:,2]
pos = np.where(y == 1)
neg = np.where(y == -1)
plt.plot(x[pos,0],x[pos,1],'bo')
plt.plot(x[neg,0],x[neg,1],'ro')
K = rbfKernel
C = 1
stat, Ls, a = makeLambdas(x,y,C,K)
b = makeB(x,y,C,Ls,K,stat)
#miss = testClassifier(x,y,Ls,b,K)
print "There were %d misclassications!" %miss
print "--------------------------------------------------------"
print stat
print b
#plotSV(Ls,y,b,K)
sv = a > 1e-5
a1 = a[sv]
svx = x[sv]
svy = y[sv]
plotBoundaries(b,Ls,y,K,a1,svx,svy)
#pca, predicate logic, logic under uncertainty