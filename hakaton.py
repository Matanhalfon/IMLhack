import  pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

def main():
    data = pd.read_csv("train.csv")
    vectorizer = CountVectorizer()
    labels = data["user"]
    import collections, re
    bagsofwords = [ collections.Counter(re.findall(r'\w+', txt))
                for txt in data.tweet]

    label_bag = {i:Counter() for i in np.arange(9)}

    # all_words = set(bagsofwords)
    # for i in range(labels.shape[0]):
    #     c= label_bag[labels[i]]
    #     label_bag[labels[i]]+=bagsofwords[i]
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