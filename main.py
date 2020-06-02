from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from PIL import Image as im
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as r
import xlrd
import math  
 
class image:
    id = 0
    mesure = []
    def __repr__(self):
        return str([self.id, self.mesure])

def fill_imgs(images, mesures):
    for id_img in range(0,1000):
        img = image()
        img.id = int(mesures.iloc[id_img,:][0][0:-4])
        # img.mesure = normalize(np.delete(np.array(mesures.iloc[id_img,:]), 0)) 
        img.mesure = np.delete(np.array(mesures.iloc[id_img,:]), 0) 
        images.append(img)
    #tri de la list selon l'id des images
    images.sort(key = lambda i: i.id) 

def k_nearest(images, img_id, dist_array):
    for i in range(0,1000):
        if i == img_id:
            continue 
        dist = distance(images[img_id].mesure,images[i].mesure)
        dist_array.append([i,dist])
        
    dist_array.sort(key = lambda i: i[1])

def display_imgs(dist_array, k, id_mesure):
    if id_mesure == 0:
        print("JCD nearest images: ", end = '')
    if id_mesure == 1:
        print("CEDD nearest images: ", end = '')
    if id_mesure == 2:
        print("PHOG nearest images: ", end = '')
    if id_mesure == 3:
        print("FCTH nearest images: ", end = '')
    if id_mesure == 4:
        print("FCH nearest images: ", end = '')
    for img in dist_array[0:k]:
        print(img[0],end=" ")
        display_image = im.open('Projet\Wang\\'+str(img[0])+'.jpg')
        display_image.show()
    print()

def main():
    #remplir les vecteurs 
    jcd =[]
    cedd =[]
    phog =[]
    fcth =[]
    fch =[]
    images = [[],[],[],[],[]]   # liste des images
    dist_array = [[],[],[],[],[]] #les des distances 
    nb_mesure = 5
    k = int(input("Entrez K\n"))
    img_id = int(input("Entrez l'id de l'image\n"))
    
    print("Reading excel files ...")
    jcd = pd.read_excel (r'Projet\WangSignatures.xls',sheet_name='WangSignaturesJCD',header=None)
    cedd = pd.read_excel (r'Projet\WangSignatures.xls',sheet_name='WangSignaturesCEDD',header=None)
    phog = pd.read_excel (r'Projet\WangSignatures.xls',sheet_name='WangSignaturesPHOG',header=None)
    fcth = pd.read_excel (r'Projet\WangSignatures.xls',sheet_name='WangSignaturesFCTH',header=None)
    fch = pd.read_excel (r'Projet\WangSignatures.xls',sheet_name='WangSignaturesFuzzyColorHistogr',header=None)
    print("Reading is done!") 
    
    mesures = [jcd, cedd, phog, fcth, fch]
    #remplissages des listes de mesures
    for i in range(0,nb_mesure):
        fill_imgs(images[i], mesures[i])
    
    # actual = [] 
    # predicted = [[],[],[],[],[]] 
    # actual.append(int(img_id / 100))
    # for i in range (0, nb_mesure):
    #     k_nearest(images[i], img_id, dist_array[i])
    #     display_imgs(dist_array[i], k, i)
    # vote(dist_array, k , img_id, predicted, nb_mesure)
    while k < 15:
        actual = [] 
        predicted = [[],[],[],[],[]] 
        for img_id in range (0, 1000):
            actual.append(int(img_id / 100))
            # print("ID Image = ", img_id)
            for i in range (0, nb_mesure):
                k_nearest(images[i], img_id, dist_array[i])
            vote(dist_array, k , img_id, predicted, nb_mesure)
            dist_array = [[],[],[],[],[]]
        
        for i in range(0, nb_mesure):
            if i == 0:
                print("JCD Confusion Matrix")
            elif i == 1:
                print("CEDD Confusion Matrix")
            elif i == 2:
                print("PHOG Confusion Matrix")
            elif i == 3:
                print("FCTH Confusion Matrix")
            elif i == 4:
                print("FCH Confusion Matrix")
            print(confusion_matrix(actual, predicted[i])) 
            print(accuracy_score(actual, predicted[i]))
        k+=3

def vote(dist_array, k, img_id, predicted, nb_mesure):
    for j in range(0, nb_mesure):
        classes = [0] * 10
        for i in dist_array[j][0:k]:
            classes[int(i[0]/100)] = classes[int(i[0]/100)] + 1
        predicted[j].append(np.argmax(classes))
        # print_class(np.argmax(classes), img_id, j)

def print_class(c, img_id, id_mesure):
 
    if id_mesure == 0:
        print("JCD prévoit: ", end = '')
    if id_mesure == 1:
        print("CEDD prévoit: ", end = '')
    if id_mesure == 2:
        print("PHOG prévoit: ", end = '')
    if id_mesure == 3:
        print("FCTH prévoit: ", end = '')
    if id_mesure == 4:
        print("FCH prévoit: ", end = '')

    if c == 0:
        print("Jungle est la classe de l'image",img_id,end=" ")
    elif c == 1:
        print("Plage est la classe de l'image",img_id,end=" ")
    elif c == 2:
        print("Monuments est la classe de l'image",img_id,end=" ")
    elif c == 3:
        print("Bus est la classe de l'image",img_id,end=" ")
    elif c == 4:
        print("Dinosaures est la classe de l'image",img_id,end=" ")
    elif c == 5:
        print("Eléphants est la classe de l'image",img_id,end=" ")
    elif c == 6:
        print("Fleurs est la classe de l'image",img_id,end=" ")
    elif c == 7:
        print("Chevaux est la classe de l'image",img_id,end=" ")
    elif c == 8:
        print("Montagne est la classe de l'image",img_id,end=" ")
    elif c == 9:
        print("Plats est la classe de l'image",img_id,end=" ")
    if int(img_id/100) == c:
        print("=> Vrai!")
    else:
        print("=> Faux")
    

def tanimoto (v1,v2):
    v1v2, v1v1, v2v2 = 0., 0., 0.
    for i in range(len(v1)):
        v1v2 += v1[i] * v2[i]
        v1v1 += v1[i] * v1[i]
        v2v2 += v2[i] * v2[i]
    return v1v2 / (v1v1 + v2v2 - v1v2)

def  euclidian (xi, xj):
    sum = 0
    for i in range(0, len(xi)):
        sum += (xj[i] - xi[i]) * (xj[i] -xi[i])
    return math.sqrt(sum)

def distance(my_vec,other_vec):
        return euclidian (my_vec, other_vec)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

if __name__ == "__main__":
    main()

