from classification import ClassificationModel

class NaiveBayes(ClassificationModel):
    def computeModel(XTrain, yTrain):
        from sklearn.naive_bayes import GaussianNB

        classifier = GaussianNB()
        classifier.fit(XTrain, yTrain)

        return classifier

    def computeExample():
        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData()
        classifier = NaiveBayes.computeModel(XTrain, yTrain)
        yPred = ClassificationModel.predictModel(classifier, XTest)
        return ClassificationModel.evaluateModel(yPred, yTest)
