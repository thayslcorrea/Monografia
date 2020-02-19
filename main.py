import preprocessing as pre
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Algoritmos
import logisticregression as lr
import svm as s
import knn as k
import naivebayes as nb
import decisiontree as dt
import randomforest as rf

def computeVP_VN_FP_FN(pred, test):
    vp = 0.0
    vn = 0.0
    fp = 0.0
    fn = 0.0
    for p, t in zip(pred, test):
        #print(p, t)
        if(p == t):
            if(p == 1):
                vp+=1
            else:
                vn+=1
        else:
            if(p == 1 and t == 0):
                fp+=1
            elif(p == 0 and t == 1):
                fn+=1
    #print('\n',vp, vn, fp, fn)
    return vp, vn, fp, fn

def getAccuracy(confusionMatrix):
    accuracy = (confusionMatrix[0][0] + confusionMatrix[1][1]) / (confusionMatrix[0][0] + confusionMatrix[1][0] + confusionMatrix[0][1] + confusionMatrix[1][1])
    
    return accuracy * 100

def getSens(vp, fn):
    sens = vp / (vp + fn)
    return sens * 100

def getEsp(vn, fp):
    esp = vn / (vn + fp)
    return esp * 100

print('------------ 600x166 ----------------\n imagem: 352x264')    


dtArray = []
dtArray1 = []
dtArray2 = []
for i in range(0, 100):
    cmDT, test, pred = dt.DecisionTree.computeExample()
    vp, vn, fp, fn = computeVP_VN_FP_FN(pred, test)
    dtArray.append(getAccuracy(cmDT))
    dtArray1.append(getSens(vp, fn))
    dtArray2.append(getEsp(vn, fp))
print('\n\n------------ Decision Tree ----------------')
print('Acuracia')
print("Média: %.2f" % np.mean(dtArray))
print("Desvio Padrão: %.2f" % np.std(dtArray))

print('\nSensibilidade')
print("Média: %.2f" % np.mean(dtArray1))
print("Desvio Padrão: %.2f" % np.std(dtArray1))

print('\nEspecificidade')
print("Média: %.2f" % np.mean(dtArray2))
print("Desvio Padrão: %.2f" % np.std(dtArray2))

knnArray = []
knnArray1 = []
knnArray2 = []
for i in range(0, 100):
    cmKNN, test, pred = k.KNN.computeExample()
    vp, vn, fp, fn = computeVP_VN_FP_FN(pred, test)

    knnArray.append(getAccuracy(cmKNN))
    knnArray1.append(getSens(vp, fn))
    knnArray2.append(getEsp(vn, fp))
print('\n\n------------ KNN ----------------')    
print('Acuracia')
print("Média: %.2f" % np.mean(knnArray))
print("Desvio Padrão: %.2f" % np.std(knnArray))

print('\nSensibilidade')
print("Média: %.2f" % np.mean(knnArray1))
print("Desvio Padrão: %.2f" % np.std(knnArray1))

print('\nEspecificidade')
print("Média: %.2f" % np.mean(knnArray2))
print("Desvio Padrão: %.2f" % np.std(knnArray2))




lrArray = []
lrArray1 = []
lrArray2 = []
for i in range(0, 100):
    cmLR, test, pred = lr.LogisticRegression.computeExample()
    vp, vn, fp, fn = computeVP_VN_FP_FN(pred, test)

    lrArray.append(getAccuracy(cmLR))
    lrArray1.append(getSens(vp, fn))
    lrArray2.append(getEsp(vn, fp))
print('\n\n------------ Logistic Regression ----------------')    
print('Acuracia')
print("Média: %.2f" % np.mean(lrArray))
print("Desvio Padrão: %.2f" % np.std(lrArray))

print('\nSensibilidade')
print("Média: %.2f" % np.mean(lrArray1))
print("Desvio Padrão: %.2f" % np.std(lrArray1))

print('\nEspecificidade')
print("Média: %.2f" % np.mean(lrArray2))
print("Desvio Padrão: %.2f" % np.std(lrArray2))



nbArray = []
nbArray1 = []
nbArray2 = []
for i in range(0, 100):
    cmNB, test, pred = nb.NaiveBayes.computeExample()
    vp, vn, fp, fn = computeVP_VN_FP_FN(pred, test)

    nbArray.append(getAccuracy(cmNB))
    nbArray1.append(getSens(vp, fn))
    nbArray2.append(getEsp(vn, fp))
print('\n\n------------ Naive Bayes ----------------')    
print('Acuracia')
print("Média: %.2f" % np.mean(nbArray))
print("Desvio Padrão: %.2f" % np.std(nbArray))

print('\nSensibilidade')
print("Média: %.2f" % np.mean(nbArray1))
print("Desvio Padrão: %.2f" % np.std(nbArray1))

print('\nEspecificidade')
print("Média: %.2f" % np.mean(nbArray2))
print("Desvio Padrão: %.2f" % np.std(nbArray2))


svmArray = []
svmArray1 = []
svmArray2 = []
for i in range(0, 100):
    cmSVM, test, pred = s.SVM.computeExample()
    vp, vn, fp, fn = computeVP_VN_FP_FN(pred, test)

    svmArray.append(getAccuracy(cmSVM))
    svmArray1.append(getSens(vp, fn))
    svmArray2.append(getEsp(vn, fp))
print('\n\n------------ SVM ----------------')    
print('Acuracia')
print("Média: %.2f" % np.mean(svmArray))
print("Desvio Padrão: %.2f" % np.std(svmArray))

print('\nSensibilidade')
print("Média: %.2f" % np.mean(svmArray1))
print("Desvio Padrão: %.2f" % np.std(svmArray1))

print('\nEspecificidade')
print("Média: %.2f" % np.mean(svmArray2))
print("Desvio Padrão: %.2f" % np.std(svmArray2))

rfArray = []
rfArray1 = []
rfArray2 = []
for i in range(0, 100):
    cmRF, test, pred = rf.RandomForest.computeExample()
    vp, vn, fp, fn = computeVP_VN_FP_FN(pred, test)
    rfArray.append(getAccuracy(cmRF))
    rfArray1.append(getSens(vp, fn))
    rfArray2.append(getEsp(vn, fp))
print('\n\n------------ Random Forest ----------------')
print('Acuracia')
print("Média: %.2f" % np.mean(rfArray))
print("Desvio Padrão: %.2f" % np.std(rfArray))

print('\nSensibilidade')
print("Média: %.2f" % np.mean(rfArray1))
print("Desvio Padrão: %.2f" % np.std(rfArray1))

print('\nEspecificidade')
print("Média: %.2f" % np.mean(rfArray2))
print("Desvio Padrão: %.2f" % np.std(rfArray2))

plt.plot(dtArray, 'r-', knnArray, 'g--', lrArray, 'b^', nbArray, '^m', svmArray, 'y', rfArray, 'c')
plt.ylabel("Acurácia")
plt.xlabel("Tentativas")
plt.show()

plt.plot(dtArray1, 'r-', knnArray1, 'g--', lrArray1, 'b^', nbArray1, '^m', svmArray1, 'y', rfArray1, 'c')
plt.ylabel("Sensibilidade")
plt.xlabel("Tentativas")
plt.show()

plt.plot(dtArray2, 'r-', knnArray2, 'g--', lrArray2, 'b^', nbArray2, '^m', svmArray2, 'y', rfArray2, 'c')
plt.ylabel("Especificidade")
plt.xlabel("Tentativas")
plt.show()
