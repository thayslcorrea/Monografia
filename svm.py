from classification import ClassificationModel

class SVM(ClassificationModel):
    def computeModel(XTrain, yTrain):
        from sklearn.svm import SVC

        classifier = SVC(gamma='auto')
        classifier.fit(XTrain, yTrain)

        return classifier

    def computeExample():
        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData()
        classifier = SVM.computeModel(XTrain, yTrain)
        yPred = ClassificationModel.predictModel(classifier, XTest)
        return ClassificationModel.evaluateModel(yPred, yTest)
