from classification import ClassificationModel

class DecisionTree(ClassificationModel):
    def computeModel(XTrain, yTrain):
        from sklearn.tree import DecisionTreeClassifier

        classifier = DecisionTreeClassifier(criterion = 'entropy')
        classifier.fit(XTrain, yTrain)

        return classifier

    def computeExample():
        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData()

        classifier = DecisionTree.computeModel(XTrain, yTrain)
        yPred = ClassificationModel.predictModel(classifier, XTest)
        return ClassificationModel.evaluateModel(yPred, yTest)
