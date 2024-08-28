# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:10:10 2021

@author: lhoes
"""

import numpy as np 
import cv2 
import matplotlib.pyplot as plt
from tqdm import tqdm
import fonction as fct

#%%
train_data=np.loadtxt("mnist_train.csv",delimiter=',')
test_data = np.loadtxt("mnist_test.csv",delimiter=",")

#%% Partie 1111111111111111111111111111111111111111111
#%% Construction A, B

nombredetection=0
valeurs=train_data[:,0]
indiceu=np.where(valeurs==nombredetection)
indicev=np.where(valeurs!=nombredetection)
u=train_data[:,1:][indiceu]
v=train_data[:,1:][indicev]


A=np.block([
    [u, np.ones((len(indiceu[0]),1))],
    [v, np.ones((len(indicev[0]),1))]
                     ])
y=np.block([[np.ones((len(indiceu[0]),1))],
            [-np.ones((len(indicev[0]),1))]])


#%%
""" Question 3"""
A=A.T@A
i=fct.iterationQR(A) #C'est un peu long
c=-np.sort(-np.diag(i))
rang=np.where(c==0)[0][0] #rang=716 rang!=taille matrice, donc on a des 0 donc det= 0 pas inversible

#%%
"""Question 4"""
U,S,VT = np.linalg.svd(A) 
rang = np.where(S<1)[0][0] #rang = 713~~716

#%%
"""Question 5.A"""
eps=1
Ae=A+eps*np.eye(785) 
valpropres = np.linalg.eig(Ae)
#Vérification symétrique definie positive
def verif(Ae):
    return (Ae.any()==Ae.T.any()) and (len(np.where(Ae<0)[0])==0) and (len((np.where(np.diag(Ae)<=0)[0]))==0)

print(verif(Ae))

#%%FCT resCHol
"""Question 5.B"""
def resChol(nombredetection,epsilon=1):
    if (nombredetection not in [i for i in range(10)]):
        return "Erreur, nombre détection pas compris entre [0,9]"
    valeurs=train_data[:,0]
    indiceu=np.where(valeurs==nombredetection)
    indicev=np.where(valeurs!=nombredetection)
    u=train_data[:,1:][indiceu]
    v=train_data[:,1:][indicev]

    A=np.block([[u, np.ones((len(indiceu[0]),1))],
                [v, np.ones((len(indicev[0]),1))]])
    y=A.T @ np.block([[np.ones((len(indiceu[0]),1))],
                [-np.ones((len(indicev[0]),1))]])
    
    Ae = A.T@A + epsilon * np.eye(785)
    sol=fct.resolcholesky(Ae, y)
    labels_test=test_data[:,0]
    N=len(labels_test)
    Nvp=0
    Nvn=0
    Nfp=0
    Nfn=0
    for k in range(0,N) :
        T=fct.f(test_data[k,1:],sol)
        if T>0 and labels_test[k]==nombredetection :
            Nvp=Nvp+1
        if T<=0 and labels_test[k]!=nombredetection :
            Nvn=Nvn+1
        if T<=0 and labels_test[k]==nombredetection :
            Nfn=Nfn+1
        if T>0 and labels_test[k]!=nombredetection :
            Nfp=Nfp+1
            
    Tr = (Nvp+Nvn)/10000
    M = np.array([[Nvp, Nfn],
                  [Nfp, Nvn]])

    return Tr,M

#%%
"""Taux réussite et matrice confusion Question 5.C"""
def tr():
    D=np.zeros((10,1))
    D2 = [0 for i in range(10)]
    for i in tqdm(range(10)):
        D[i],D2[i]=resChol(i,1)
    return D,D2
D,D2=tr()

#%%
"""Question 5.D"""
N=50
n=np.linspace(10**(-10),10**9,N)    
Tr=[resChol(0,n[i])[0] for i in tqdm(range(N))]
z=np.where(np.max(Tr)==Tr)[0][0]

plt.plot(n,Tr)
plt.show()

#%% On calcul le epsilon optimal
"""Question 5.E"""
N=50
y=np.zeros((10,1))
n=np.linspace(10**(-10),10**9,N)  
for t in tqdm(range(10)):
    Tr=[resChol(t,n[i])[0] for i in range(N)]
    z=np.where(np.max(Tr)==Tr)[0][0]
    y[t]=n[z]
#np.save('epsilon.npy', y), on garde en commentaire pour être sûr de ne pas supprimer le epsilon déjà fait
    
#%% On construit SOL avec resChol modifié
def resCholSol(nombredetection,epsilon=1):
    if (nombredetection not in [i for i in range(10)]):
        return "Erreur, nombre détection pas compris entre [0,9]"
    valeurs=train_data[:,0]
    indiceu=np.where(valeurs==nombredetection)
    indicev=np.where(valeurs!=nombredetection)
    u=train_data[:,1:][indiceu]
    v=train_data[:,1:][indicev]

    A=np.block([[u, np.ones((len(indiceu[0]),1))],
                [v, np.ones((len(indicev[0]),1))]])
    y=A.T @ np.block([[np.ones((len(indiceu[0]),1))],
                [-np.ones((len(indicev[0]),1))]])
    
    Ae = A.T@A + epsilon * np.eye(785)
    sol=fct.resolcholesky(Ae, y)
    return sol

v=np.load('epsilon.npy')
sol=np.zeros((785,10))
for i in range(10):
    sol[:,i]=resCholSol(i,v[i]).reshape((785,))
    
#np.save('sol.npy',sol)
    
#%% On calcul le taux de réussite
"""Question 6"""
def fglobale(x,SOL):
    taux=np.zeros((10,1))
    for i in range(10):
        taux[i]=fct.f(x,SOL[:,i].reshape((785,1)))
    return taux

SOL=np.load('sol.npy')

V=0
F=0
for i in tqdm(range(test_data.shape[0])):
    imagei=test_data[i,1:]  
    fg=fglobale(imagei,SOL)
    c=np.where(fg==np.max(fg))[0][0]
    if c==test_data[i,0] :
        V+=1
    else : 
        F+=1
        
Tr = V/10000 
print('\n Taux de réussite Tr = {}'.format(Tr*100))
#Valeur du TP : Tr= 86.03%, on trouve 0.42% de mieux
#%% PARTIE 22222222222222222222222222222222222222222222222222
"""Question 1"""
nombredetection=0
valeurs=train_data[:,0]
indiceu = np.where(valeurs==nombredetection)
u=train_data[:,1:][indiceu]
zeromoyen=np.mean(u,axis=0).reshape((1,784))

img=cv2.resize(np.uint8(zeromoyen.reshape((28,28))),[420,420],interpolation=cv2.INTER_AREA)
cv2.imshow('Zero moyen',img)

#%% Fonction chiffre moyen
"""Question 2"""
def chiffremoy(train_data=train_data):
    cm=np.zeros((10,784))
    valeurs=train_data[:,0]
    for i in range(10):
        indiceu = np.where(valeurs==i)
        u=train_data[:,1:][indiceu]
        chiffremoyen=np.mean(u,axis=0).reshape((1,784))
        cm[i,:]=chiffremoyen
    return cm

#%% affichage des chiffres moyen
CM=chiffremoy(train_data)
i=cv2.resize(np.uint8(CM[0,:].reshape((28,28))),[140,140],interpolation=cv2.INTER_AREA)

for k in range(1,10):
    i=np.concatenate((i,cv2.resize(np.uint8(CM[k,:].reshape((28,28))),[140,140],interpolation=cv2.INTER_AREA)),axis=1)
    
cv2.imshow('chiffres moyens',i)

#%%
"""Question 3"""

def Procuste(A,B):
    A,B=A.reshape((1,784)),B.reshape((1,784)) #On passe en colonne ici, donc on rentre des lignes dans la fct
    l,c=A.shape
    u=np.ones((1,c))
    ag,bg=np.zeros((l,1)),np.zeros((l,1))

    for i in range(c):
        ag+=A[:,i].reshape((l,1))
        bg+=B[:,i].reshape((l,1))
    ag,bg=ag/(c), bg/(c)
    Ag, Bg= A-ag@u, B-bg@u
    
    Cg=Ag@Bg.T
    U,S,VT = np.linalg.svd(Cg)
    X=VT.T@U.T
    lam=(np.sum(S))/((np.linalg.norm(Ag,'fro'))**2) #fro pas nécessaire mais on est pas comme les autres tavu
    t=bg-lam*X@ag
    
    phi = lam*X@A + t@u
    return (lam,X,t, np.linalg.norm(B-phi,'fro')**2)
    
#%%
"""Question 4"""
x= train_data[1998,1:].reshape((1,784))

def comparaison(x, CM):
    veccomp=np.zeros((10,1))
    for i in range (10):
        veccomp[i]=Procuste(CM[i,:],x)[3]
    resultat=np.where(veccomp==np.min(veccomp))[0][0]
    return (veccomp,resultat)

c=comparaison(x,CM)
    
#%%
"""Question 5"""
def taux():
    T=0
    for i in tqdm(range (10000)):
        x= test_data[i,1:]
        if comparaison(x,CM)[1]==test_data[i,0]:
            T+=1 
    return T/10000

t=taux()
print('\n Taux de réussite Tr = {}'.format(t*100)) #C'est long, on trouve 82,08%

#%% PARTIE 333333333333333333333333333333333333333333333333333333333333333333333333333333333333
"""Question 1"""
i=0
x = train_data[i,1:].reshape((1,784))
def ff(x,coeff):
    v=coeff * (1/comparaison(x,CM)[0])
    rep= fglobale(x.reshape((784,1)),SOL).reshape(np.shape(v))
    resultat = v+rep
    return np.where(resultat==np.max(resultat))[0][0]
a=ff(x,90)
    
#%% 
"""Question 2"""
# C'est long, voir le rapport pour les conclusions apportées
T=0
n=5
interval = np.linspace(1,10**9,n)
Tr=np.zeros((n,1))
for i in interval:
    T=0
    for j in tqdm(range(10000)):
        x=test_data[j,1:].reshape((1,784))
        if ff(x,i)==test_data[j,0]:
            T+=1
    Tr[np.where(i==interval)[0][0]]=T/10000
    
plt.plot(interval,Tr)
plt.show()
            
            
        
        



