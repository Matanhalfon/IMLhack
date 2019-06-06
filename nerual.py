import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from data_prce import runMe

from keras.layers import GaussianNoise
from sklearn.feature_extraction.text import CountVectorizer
import re
from keras import regularizers

pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
pattern2 = re.compile('@[A-Za-z0-9_-]+ ')
pattern3 = re.compile('#[A-Za-z0-9_-]+ ')
pattern4 = re.compile('[0-9]+')


def pre_pro (sentence):
    #  remove sites from tweets, @, #
    sentence = sentence.lower()
    r_site = pattern.sub('', sentence)
    # r_strudel = pattern2.sub('', r_site)
    # r_hesteck = pattern3.sub('', r_strudel)
    # print (r_hesteck)
    return r_site

def return_x_y(path):
    """

    :param path:
    :return:
    """
    data = pd.read_csv(path)
    y  = data["user"]
    #encoding
    encoder = LabelEncoder()
    encoder.fit(y)#
    y = encoder.transform(y)
    y = np_utils.to_categorical(y)
    X = data.drop(["user","tweet"],axis = 1)
    return X,y


def test_prediction(model , tweets):
    predicted = []
    for x in tweets:
        print (model.predict_on_batch(x))
        predicted.append(model.predict_on_batch(x))
    return predicted


def calculate_error(predictions, y):
    error = 0
    for i in range(len(predictions)):
        if predictions[i] != y[i]:
            error+=1
    return error/len(predictions)

def create_neural_network(X,y):
    model = Sequential()
    model.add(Dense(8, input_dim = X.shape[1] , activation = 'relu'))
    model.add(Dense(10, activation = 'selu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(GaussianNoise(0.12))
    # model.add(Dense(32, input_dim=32,
    #                 kernel_regularizer=regularizers.l2(0.01),
    #                 activity_regularizer=regularizers.l1(0.01)))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
    model.fit(X, y, epochs = 10, batch_size = 2)
    return model


def main(test_path):
    cv = CountVectorizer(max_features=25000, stop_words='english')
    data = pd.read_csv("train.csv")
    data.tweet = data.tweet.apply(pre_pro)
    f = cv.fit_transform(data.tweet)
    test_data = pd.read_csv(test_path)
    test_data.tweet = test_data.tweet.apply(pre_pro)
    test_x = test_data.tweet
    t = cv.transform(test_x)
    y = data["user"]

    encoder = LabelEncoder()
    encoder.fit(y)#
    y = encoder.transform(y)
    y = np_utils.to_categorical(y)
    model = create_neural_network(f,y)
    from keras.models import load_model
    model.save('my_model.h5')
    test_y = test_data["user"]
    encoder = LabelEncoder()
    encoder.fit(test_y)#
    test_y = encoder.transform(test_y)
    test_y = np_utils.to_categorical(test_y)
    # predictions =
    # print(model.test_on_batch(t,test_y))
    print(model.evaluate(t,test_y))
    # prediction = model.predict_on_batch(t)
    # prediction = test_prediction(model,t)
    # print(calculate_error(prediction,test_y))


if __name__ == '__main__':
    import sys
    main(sys.argv[1])