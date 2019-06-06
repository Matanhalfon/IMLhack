import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import pandas as pd
import os
import re
import emoji
import regex

path1="/cs/usr/matanhalfon/Desktop/MLhachton/data"
trainpath="cs/usr/matanhalfon/PycharmProjects/IMLhack"

def readData(path):
    """
    read connect and shuffle the tweets and then write them to
    train validation and test cvs in the path
    :param path: a directory with tweet files
    :return the test part of the data
    """
    data=[]
    for filename in os.listdir(path):
        if filename.endswith("csv"):
            addpath=path+"/"+filename
            to_add=pd.read_csv(addpath)
            data.append(to_add)
        else:
                continue
    addedData = pd.concat(data,sort=True)
    result=addedData.sample(frac=1).reset_index(drop=True)
    train=result[:12000]
    validation=result[12000:25000]
    test=result[25000:]
    train.to_csv(os.path.join(path,r'train.csv'),index=False)
    validation.to_csv(os.path.join(path,r'validation.csv'),index=False)
    test.to_csv(os.path.join(path,r'test.csv'),index=False)
    return train





def extract_emojis(text):
    decode   = text.decode('utf-8')
    allchars = [str for str in decode]
    return [c for c in allchars if c in emoji.UNICODE_EMOJI]

def getMeanWord(text):
    words=text.split()
    avg=sum(len(word) for word in words)/len(words)
    return avg

def isRT(text):
    if "RT @" in text:
        return 1
    else:
        return 0




def RTsplit(data):
    RT = data.loc[data["is RT"] == True]
    not_RT = data.drop(RT.index,axis = 0)
    return RT,not_RT


def writeRawRT(path):
    data=pd.read_csv("train.csv")
    tweets=data["tweet"]
    data["is RT"]=tweets.apply(isRT)
    RT,notRT=RTsplit(data)
    RT = RT.drop(["is RT"],axis = 1)
    notRT = notRT.drop(["is RT"], axis =1)
    RT.to_csv(r'rawRT.csv',index=False)
    notRT.to_csv(r'rawNotRT.csv',index=False)



def runMe(path):
    # data=readData(path)
    data=pd.read_csv("train.csv")
    tweets=data["tweet"]
    data["word count"]=tweets.str.split().apply(len)
    data["wordl len"]=wordlens=tweets.str.len()
    data["numCap"]=tweets.str.findall(r'[A-Z]').str.len()
    data["numHashtags"]=tweets.str.findall(r'#').str.len()
    data["numOfTaging "]=tweets.str.findall(r'@').str.len()
    data["emojilists"]=tweets.apply(extract_emojis)
    data["mean word"]=tweets.apply(getMeanWord)
    data["num of !"]=tweets.str.findall(r'!').str.len()
    data["num of ?"]=tweets.str.findall(r'\?').str.len()
    data["num of dots"]=tweets.str.findall(r'\.').str.len()
    data["num of commas"]=tweets.str.findall(r'\,').str.len()
    data["is RT"]=tweets.apply(isRT)
    labels=data["user"]

    # data.drop(["user"],axis=1,inplace=True)
    # RT,notRT=RTsplit(data)
    RT ,notRT=RTsplit(data)
    RT.to_csv(r'trainRT.csv',index=False)
    notRT.to_csv(r'trainNotRT.csv',index=False)
    return data


runMe(trainpath)