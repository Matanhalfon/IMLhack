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

def generate_bow(sentence):
    words = word_extraction(sentence)
    bag_vector = np.zeros(len(vocab))
    for w in words:
        for i, word in enumerate(vocab):
            if word == w:
                bag_vector[i] += 1
    return bag_vector



data = pd.read_csv("rawNotRT.csv")
data.tweet = data.tweet.apply(pre_pro)
vocab = tokenize(data.tweet)
file_object = open("words_file", 'w')
file_object.write("Word List for Document \n{0} \n".format(vocab))
file_object.close()
X = data.tweet.apply(generate_bow)
Y = data["user"]





