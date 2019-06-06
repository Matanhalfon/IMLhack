import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
# from data_prce import runMe
RTS="RT @"


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
    X = data.drop(["user","tweet","emojilists","is RT"],axis = 1)
    return X,y


def is_RT(tweet,word="RT @"):
    """
    :return: true if the sample is retweet
    """
    return word in tweet


def test_prediction(model_RT, model_not_RT , tweets):
    predicted = []
    for x in tweets:
        if is_RT(x):
            predicted.append(model_RT.predict(x))
        else:
            predicted.append(model_not_RT.predict(x))
    return predicted


def calculate_error(predictions, y):
    error = 0
    for i in range(len(predictions)):
        if predictions[i] != y[i]:
            error+=1
    return error/len(predictions)

def create_neural_network(path):
    X,y = return_x_y(path)
    model = Sequential()
    model.add(Dense(8, input_dim = X.shape[1] , activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
    model.fit(X, y, epochs = 10, batch_size = 2)
    return model
def create_logistic_regression(path):
    X,y = return_x_y(path)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, y)
    return clf
def train():
    model_not_RT = create_neural_network("trainNotRT.csv")
    model_RT = create_neural_network("trainRT.csv")
    return model_RT,model_not_RT


def main(test_path):
    model_RT,model_not_RT = train()
    data = pd.read_csv(test_path)
    # Rt , NotRT = runMe(test_path)
    test_X = data["tweet"]
    prediction = test_prediction(model_RT,model_not_RT,test_X)
    test_y = data["user"]
    print(calculate_error(prediction,test_y))


if __name__ == '__main__':
    import sys
    main(sys.argv[1])