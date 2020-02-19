import numpy as np
import pandas as pd
import os
import cv2

def getImagesInFolder(folderPath):
    listOfImages = []
    #retorna uma lista contendo os nomes das entradas no diretório fornecido pelo caminho.    
    listOfFiles = os.listdir(folderPath)
    for eachFile in listOfFiles:
        #Verifica se as imagens estão em 3 formatos específicos
        if((".jpg" in eachFile) or (".png" in eachFile) or (".bmp" in eachFile)):
            #Carrega a imagem da pasta
            eachImage = cv2.imread(folderPath + "/" + eachFile, 0)
            #Redimensiona ela para 320x240
            eachImage = cv2.resize(eachImage, (320, 240), cv2.INTER_AREA)
            eachImage2 = eachImage
            #Transforma a matriz em 1D (vetor)
            eachImage = np.ravel(eachImage)
            #Add esse imagem em formato de vetor na lists
            listOfImages.append(eachImage)
            
    return listOfImages

def loadDataset():
    listImagesP = getImagesInFolder("base/pos")
    listImagesN = getImagesInFolder("base/neg")
    
    #Cria uma lista vazia(preenche com 0) do tamanho de listImagesN
    y1 = np.zeros(len(listImagesN), dtype = int)
    
    #Cria uma lista preenchida com 1 do tamanho de listImagesP
    y2 = np.full(len(listImagesP), 1)

    #Uni y1 e y2 em Y
    y = np.concatenate((y1, y2), axis=0)

    #Lista contendo as duas listas (pos e neg)
    X = listImagesP + listImagesN
    
    return X, y

def splitTrainTestSets(X, y, testSize):
    from sklearn.model_selection import train_test_split
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = testSize)

    return XTrain, XTest, yTrain, yTest

