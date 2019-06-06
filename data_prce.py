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


pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
pattern2 = re.compile('@[A-Za-z0-9_-]+ ')
pattern3 = re.compile('#[A-Za-z0-9_-]+ ')

notRTcommenWords=["wall","crime","border","democrats","china","biden","vp","conan","saw","gameofgames","ellentube"
,"thankssponsor","nwatch","kkwbeauty","kkwfragrance","shop","lip","kingjames","striveforgreatness"
,"homie","bro","monsters","joanne","tony","gaga","cr7","cristiano","portugal","nikefootball",
"hello","hala","kimmel","realdonaldtrump","iamguillermo","arnoldsports","great","thank",
"fantastic"]


RTS="RT @"

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

def iscontains(text,word=RTS):
    if word in text:
        return 1
    else:
        return 0



def pre_pro (sentence):
    #  remove sites from tweets, @, #
    sentence = sentence.lower()
    r_site = pattern.sub('', sentence)
    # r_strudel = pattern2.sub('', r_site)
    # r_hesteck = pattern3.sub('', r_strudel)
    # print (r_hesteck)
    return r_site


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
    data.tweet=tweets.apply(pre_pro)
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
    data["is RT"]=tweets.apply(iscontains)
    labels=data["user"]
    # data.drop(["user"],axis=1,inplace=True)
    # RT,notRT=RTsplit(data)
    for comword in notRTcommenWords:
        newcol=tweets.apply(iscontains,word=comword)
        data[comword]=newcol
    RT ,notRT=RTsplit(data)
    # RT.to_csv(r'trainRT.csv',index=False)
    # notRT.to_csv(r'trainNotRT.csv',index=False)
    return RT ,notRT


runMe(trainpath)