# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:18:09 2021

@author: lhoes
"""

import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random as rnd
import fonction as fct
from PIL import Image, ImageTk
import tkinter.font as tkfont

#%%
test_data = np.loadtxt("mnist_test.csv",delimiter=",")
SOL=np.load('sol.npy')
#%%
N=0
taille = 720
plt.ioff()


fenetre = tk.Tk()
fenetre.geometry('720x720')
fenetre.minsize(taille,taille)
fenetre.maxsize(taille,taille)
fenetre.title('Convertiseur de coordonnées')

font = tkfont.Font(size=12)
fontresultat=tkfont.Font(size=20)

#moitié du haut
#Affichage chiffre
partie1 = tk.Frame(fenetre, borderwidth=5,relief=tk.SOLID,)
partie1.pack(side=tk.TOP,fill=tk.BOTH,expand=1)

can=tk.Canvas(partie1,height=224,width=224,background='white',borderwidth=10,bg='black')
can.pack(side=tk.LEFT)

chiffre = cv2.resize(np.uint8(test_data[0,1:].reshape((28,28))),[224,224],interpolation=cv2.INTER_AREA)
img =  ImageTk.PhotoImage(master=partie1, image=Image.fromarray(chiffre))
imagecan=can.create_image(0,0,anchor=tk.NW, image=img)

#Affichage graphe
can2=tk.Canvas(partie1,height=500,width=500,background='white',borderwidth=5,bg='black')
can2.pack(side=tk.LEFT)
fig, ax = plt.subplots()
ax.bar(np.arange(0,10),fct.fglobale(test_data[N,1:].reshape((784,1)),SOL).reshape(10))
ax.set_title("Réponse de l'algorithme")
ax.set_xticks(np.arange(0,10))
ax.axhline(0, color='grey', linewidth=0.8)
ax.set_ylim(-1.5,1.5)
plt.savefig('bar.png')
plt.close(fig)
valeurs=cv2.imread('bar.png') 
valeurs=cv2.resize(valeurs,(500, 500))
img2 = ImageTk.PhotoImage(master=fenetre, image=Image.fromarray(valeurs))
imagecan2=can2.create_image(0,0,anchor=tk.NW, image = img2)



#moitié du bas
partie2 = tk.Frame(fenetre, borderwidth=5,relief=tk.SOLID,)
partie2.pack(side=tk.TOP,fill=tk.BOTH,expand=1)

#partie choix
choiximage= tk.Frame(partie2, borderwidth=2,relief=tk.SOLID)
choiximage.pack(side=tk.TOP,fill=tk.BOTH,expand=1)

def alea():
    global N
    N=rnd.randint(0,9999)
    echelle.set(N)
    nombre.set(N)
    update()
    

brandom=tk.Button(choiximage,text='Aléatoire',command=alea, activeforeground='white',activebackground='black',borderwidth=5)
brandom.pack(side=tk.LEFT)

echelle=tk.Scale(choiximage,from_=0,to=9999,orient=tk.HORIZONTAL,borderwidth=3,relief=tk.SOLID)
echelle.pack(side=tk.LEFT,expand=1,fill=tk.X)

nombre=tk.IntVar(fenetre,0)
entree = tk.Entry(choiximage,textvariable=nombre,borderwidth=3,relief=tk.SOLID)
entree.pack(side=tk.LEFT)
entree.pack()

def modifentree(event):
    try:
        nombre.get()
    except:
        return
    if nombre.get()>9999 or nombre.get()<0:
        return
    global N
    if N==nombre.get():
        return
    N=nombre.get()
    echelle.set(N)
    update()
    
fenetre.bind('<Any-KeyRelease>',modifentree)

def plusun():
    global N
    if N==9999:
        return
    N+=1
    echelle.set(N)
    nombre.set(N)
    update()
def moinsun():
    global N
    if N==0:
        return
    N+=-1
    echelle.set(N)
    nombre.set(N)
    update()
    

bup = tk.Button(choiximage,text=' ↑ ',command=plusun,borderwidth=5, activeforeground='white',activebackground='black',font=font)
bup.pack(side=tk.TOP,expand=1,fill=tk.BOTH)
bdown= tk.Button(choiximage,text=' ↓ ',command=moinsun,borderwidth=5, activeforeground='white',activebackground='black',font=font)
bdown.pack(side=tk.TOP,expand=1,fill=tk.BOTH)


#partie résultat
resultat = tk.Frame(partie2, borderwidth=2,relief=tk.SOLID)
resultat.pack(side=tk.TOP,fill=tk.BOTH,expand=1)

res=tk.StringVar()
annonce=tk.Label(resultat, textvariable=res,font=fontresultat)
annonce.pack(fill=tk.BOTH,expand=1)


def update():
    global img
    global img2
    global N
    chiffren = cv2.resize(np.uint8(test_data[echelle.get(),1:].reshape((28,28))),[224,224],interpolation=cv2.INTER_AREA)
    img =  ImageTk.PhotoImage(master=partie1, image=Image.fromarray(chiffren))
    can.itemconfig(imagecan, image = img)
    
    fig, ax = plt.subplots()
    ax.bar(np.arange(0,10),fct.fglobale(test_data[N,1:].reshape((784,1)),SOL).reshape(10))
    ax.set_title("Réponse de l'algorithme")
    ax.set_xticks(np.arange(0,10))
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_ylim(-1.5,1.5)
    plt.savefig('bar.png')
    plt.close(fig)
    valeurs=cv2.imread('bar.png') 
    valeurs=cv2.resize(valeurs,(500, 500))
    img2 = ImageTk.PhotoImage(master=fenetre, image=Image.fromarray(valeurs))
    can2.itemconfig(imagecan2,image=img2)
    result()

def result():
    global N
    fg=fct.fglobale(test_data[N,1:],SOL)
    choix=np.where(fg==np.max(fg))[0][0]
    res.set('L\'algorithme voit un {}'.format(choix))
    if choix == test_data[N,0]:
        annonce.config(bg='green')
    elif choix != test_data[N,0]:
        annonce.config(bg='red')
    
    


result()
fenetre.mainloop()
