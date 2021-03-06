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
pattern4 = re.compile('[0-9]+')


def pre_pro (sentence):
    #  remove sites from tweets, @, #
    sentence = sentence.lower()
    r_site = pattern.sub('', sentence)
    # r_strudel = pattern2.sub('', r_site)
    # r_hesteck = pattern3.sub('', r_strudel)
    r = pattern4.sub('', r_site)
    # print (r_hesteck)
    return r


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

def generate_bow1(word):

    bag_vector = np.zeros(len(data.tweet))
    for sentence in data.tweet:
        words = word_extraction(sentence)
        for i, w in enumerate(words):
            if word == w:
                bag_vector[i] += 1
    return bag_vector








