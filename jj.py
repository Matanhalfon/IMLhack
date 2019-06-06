import  pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import collections
import re
import pylab as pl
import numpy as np

pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
pattern2 = re.compile('@[A-Za-z0-9_-]+ ')
pattern3 = re.compile('#[A-Za-z0-9_-]+ ')


def pre_pro (sentence):
    #  remove sites from tweets, @, #
    sentence = sentence.lower()
    r_site = pattern.sub('', sentence)
    # r_strudel = pattern2.sub('', r_site)
    # r_hesteck = pattern3.sub('', r_strudel)
    # print (r_hesteck)
    return r_site


def word_extraction(sentence):
    with open("english") as stop_words:
        ignore = list(stop_words)
    words = re.sub("[^\w]", " ", sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text

def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)

    words = sorted(list(set(words)))
    return words

def generate_bow(allsentences):
    vocab = tokenize(allsentences)
    file_object = open("words_file", 'w')
    file_object.write("Word List for Document \n{0} \n".format(vocab))
    for sentence in allsentences:
        words = word_extraction(sentence)
        bag_vector = np.zeros(len(vocab))
        for w in words:
            for i, word in enumerate(vocab):
                if word == w:
                    bag_vector[i] += 1

        file_object.write("{0} \n{1}\n".format(sentence, np.array(bag_vector)))
    file_object.close()


def clean_words(label_bag):
    ignore = []
    with open("english") as stop_words:
        for line in stop_words:
            ignore.append(line.replace("\n", ""))
    for person in label_bag:
        for word in ignore:
            if word in label_bag[person]:
                del label_bag[person][word]
    return label_bag

def clean_words_n(label_bag):

    for person in label_bag:
        f_del = []
        for word in label_bag[person]:
            if int(label_bag[person][word])< 50:
                f_del.append(word)
        for w in f_del:
            del label_bag[person][w]
    return label_bag


def main():
    data = pd.read_csv("rawRT.csv")
    data.tweet = data.tweet.apply(pre_pro)
    # generate_bow(data.tweet)
    # data["tweet"] = word_extraction(data.tweet)
    # vectorizer = CountVectorizer()
    labels = data["user"]
    bagsofwords = [ collections.Counter(re.findall(r'\w+', txt))
                    for txt in data.tweet]

    label_bag = {i:Counter() for i in np.arange(10)}

    # all_words = set(bagsofwords)
    for i in range(labels.shape[0]):
        label_bag[labels[i]]+=bagsofwords[i]
    label_bag = clean_words(label_bag)
    label_bag = clean_words_n(label_bag)

    for per in label_bag:

        d = label_bag[per]
        X = np.arange(len(d))
        pl.bar(X, d.values(), align='center', width=0.5)
        pl.xticks(X, d.keys())
        ymax = max(d.values()) + 1
        pl.ylim(0, ymax)
        pl.xticks(rotation='vertical')
        pl.title(per)
        pl.show()








if __name__ == '__main__':

    main()


# print()
# bag= vectorizer.fit_transform(data.tweet)
# print(bag.toarray())

# for i in np.arange(labels.shape[0]):
#     update_dict(bag, words[i],labels[i])
# def create_set_of_words(words):
#     set_of_words= set()
#     for word in words:
#         set_of_words
# def update_dict(bag,sentence, label):
#     for word in sentence:
#         bag[label][word] +=1


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