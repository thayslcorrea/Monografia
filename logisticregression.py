from classification import ClassificationModel

class LogisticRegression(ClassificationModel):
    def computeModel(XTrain, yTrain):
        from sklearn.linear_model import LogisticRegression

        classifier = LogisticRegression(solver='lbfgs', max_iter = 4000)
        classifier.fit(XTrain, yTrain)

        return classifier

    def computeExample():
        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData()
        classifier = LogisticRegression.computeModel(XTrain, yTrain)
        yPred = ClassificationModel.predictModel(classifier, XTest)
        return ClassificationModel.evaluateModel(yPred, yTest)
