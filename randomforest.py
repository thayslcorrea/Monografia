from classification import ClassificationModel

class RandomForest(ClassificationModel):
    def computeModel(XTrain, yTrain):
        from sklearn.ensemble import RandomForestClassifier

        classifier = RandomForestClassifier(n_estimators=20, random_state=0)
        classifier.fit(XTrain, yTrain)

        return classifier

    def computeExample():
        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData()

        classifier = RandomForest.computeModel(XTrain, yTrain)
        yPred = ClassificationModel.predictModel(classifier, XTest)
        #yPred = classifier.predict(XTest)
        return ClassificationModel.evaluateModel(yPred, yTest)



