import  pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import collections, re
import json
import re
pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
pattern2 = re.compile('@*')



def pre_pro (sentence):
    #  remove sites from tweets.

    c = sentence
    after = pattern.sub('', sentence)
    return pattern.sub('', sentence)


def remove_strudel (sentence):
    c2 = sentence
    after2 = pattern2.sub('', sentence)
    return  pattern2.sub('', sentence)



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

def main():
    data = pd.read_csv("train.csv")
    # data.tweet = data.tweet.apply(pre_pro)
    # generate_bow(data.tweet)
    # data["tweet"] = word_extraction(data.tweet)
    vectorizer = CountVectorizer()
    labels = data["user"]
    bagsofwords = [ collections.Counter(re.findall(r'\w+', txt))
                for txt in data.tweet]

    label_bag = {i:Counter() for i in np.arange(10)}

    # all_words = set(bagsofwords)
    for i in range(labels.shape[0]):
        label_bag[labels[i]]+=bagsofwords[i]
    sumbags = sum(bagsofwords, collections.Counter())
    print()
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



if __name__ == '__main__':

    main()