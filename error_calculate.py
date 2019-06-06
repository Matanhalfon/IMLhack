from sklearn import datasets
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
class ErrorCalculator:
    def __init__(self, classifier):
        self.classifier = classifier
    def fit(self,X,Y):
        self.classifier.fit(X, Y)
    def calculate_error(self, X, y):
        error = 0
        predictions = self.classifier.predict(X)
        for i in range(len(predictions)):
            if predictions[i] != y[i]:
                error+=1
        return error/len(predictions)
def main():
    # iris = datasets.load_iris()
    # X, y = iris.data, iris.target
    # print (X)
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    data = pd.read_csv("trainNotRT.csv")
    y  = data["user"]
    X = data.drop(["user","tweet","emojilists","is RT"],axis = 1)
    model = Sequential()
    model.add(Dense(8, input_dim = len(X)-1 , activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

    model.fit(X, y, epochs = 10, batch_size = 2)
    err = ErrorCalculator(model)

    print (err.calculate_error(X,y))


if __name__ == '__main__':
    main()