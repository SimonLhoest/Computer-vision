# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 20:36:13 2021

@author: lhoes
"""
import numpy as np

def householder(v):
    x=v.shape[0]
    if v.any()==np.zeros((x,1)).any():
        return np.eye(x)
    H=np.eye(x) - 2*(v@v.T)/(v.T@v)
    return H 

def hcan(v,i):
    e=np.zeros(v.shape)
    if i > v.shape[0]:
        return "i trop grand"
    e[i]=1
    Hi = householder(v-np.linalg.norm(v)*e)
    return Hi

def Hcan(v,i) :
    e=np.zeros(np.shape(v))
    e[i,0]=1
    return householder(v-np.linalg.norm(v)*e)

def QR(A):
    Hx=hcan(A[:,A.shape[1]-1],A.shape[1])
    Hxx = hcan(A[:,0],1)
    for i in range(1,A.shape[1]):
        Hx=Hx@hcan(A[:,i],A.shape[1]-i)
        Hxx=Hxx@hcan(A[:,i],i+1)
    R=Hx@A
    return Hxx, R

def QRhouseholder(A) :
    m,n=A.shape
    Qf=Hcan(A[:,0][np.newaxis].T,0)
    A1=Hcan(A[:,0][np.newaxis].T,0)@A
    for i in range(1,n) :
        Ai=A1[i:,i:]
        Q=np.block([[np.eye(i), np.zeros((i,m-i))],[np.zeros((m-i,i)),Hcan(Ai[:,0][np.newaxis].T,0)]])
        Qf=Qf@Q
        A1=Q@A1
    return A1,Qf #R, Q

def iterationQR(A) :
    R,Q=QRhouseholder(A)
    for i in range(0,5) :
        H=R@Q
        R,Q=QRhouseholder(H)
    return H

def resoltriangsup(A,b):
    x,y =np.shape(A)
    sol = np.zeros(x)
    sol[-1]=b[-1,0]/A[-1,-1]
    for i in range(x-2,-1,-1):
        sol[i]=b[i,0]
        for k in range(i+1,x):
            sol[i]= sol[i]-A[i,k]*sol[k]
        sol[i]=sol[i]/A[i,i]
        
    return sol.reshape((x,1))
    
def resoltrianginf(A,b):
    x,y = np.shape(A)
    sol = np.zeros(x)
    sol[0]=b[0,0]/A[0,0]
    for i in range(1,x):
        sol[i]=b[i,0]
        for k in range(i):
            sol[i]=sol[i]-A[i,k]*sol[k]
        sol[i]=sol[i]/A[i,i]

    return sol.reshape((x,1))

def resolcholesky(A,b):
    n=np.shape(A)[0]
    L=np.zeros((n,n))
    L[0,0]=(A[0,0])**(1/2)
    for j in range(1,n): 
        L[j,0]=A[j,0]/L[0,0] 
    for i in range(1,n-1) :
        L[i,i]=np.sqrt(A[i,i]-np.sum(L[i,:i]**2))
        for j in range(i+1,n) :
            L[j,i]=(A[j,i]-np.sum(L[j,:i]*L[i,:i]))/L[i,i] 
    L[-1,-1]=np.sqrt(A[-1,-1]-np.sum(L[-1,:n-1]**2))     
    y=resoltrianginf(L, b)
    x=resoltriangsup(L.T, y)
    return x

def f(x,sol):
    w=sol[:-1,0].reshape((784,1))
    b=sol[-1,0]
    return w.T @ x + b

def fglobale(x,SOL):
    taux=np.zeros((10,1))
    for i in range(10):
        taux[i]=f(x,SOL[:,i].reshape((785,1)))
    return taux

