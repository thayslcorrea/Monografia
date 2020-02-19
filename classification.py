import preprocessing as pre

class ClassificationModel:
    def __init__(self):
        pass    

    def predictModel(classifier, X):
        return classifier.predict(X)

    def evaluateModel(yPred, yTest):
        from sklearn.metrics import confusion_matrix
        confusionMatrix = confusion_matrix(yTest, yPred)

        return confusionMatrix, yTest, yPred

    def preprocessData():
        X, y = pre.loadDataset()

        XTrain, XTest, yTrain, yTest = pre.splitTrainTestSets(X, y, 0.2)

        return XTrain, XTest, yTrain, yTest
