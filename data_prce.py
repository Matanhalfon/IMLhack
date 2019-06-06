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

# ["wall","crime","border","democrats","china","biden","vp","conan","saw","gameofgames","ellentube"
#     ,"thankssponsor","nwatch","kkwbeauty","kkwfragrance","shop","lip","kingjames","striveforgreatness"
#     ,"homie","bro","monsters","joanne","tony","gaga","cr7","cristiano","portugal","nikefootball",
#  "hello","hala","kimmel","realdonaldtrump","iamguillermo","arnoldsports","great","thank",
#  "fantastic"]


# ["whitehouse","dbongino","joebiden","vp","theellenshow","kkwbeauty","kkwfragrance"
#     ,"kimkardashian","kuwtk","shop","kingjames","ljfamfoundation","uninterrupted","ladygaga","gaga","lady","btwfoundation"
#     ,"cr","jimmykimmel","arnold","schwarzenegger"]
pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
pattern2 = re.compile('@[A-Za-z0-9_-]+ ')
pattern3 = re.compile('#[A-Za-z0-9_-]+ ')

notRTcommenWords=['border', 'numbers', 'democrats', 'maga', 'crime', 'total', 'witch', 'hunt',
'military', 'democrat', 'collusion', 'southern', 'fbi', 'report', 'dems', 'economy', 'dollars', 'illegal', 'others',
'foxnews', 'biden', 'jill', 'joe', 'vice', 'ryan', 'dr', 'middle', 'class', 'tax', 'saw', 'conan', 'ellentube',
'nwatch', 'clip', 'thankssponsor', 'audience', 'gameofgames', 'splittinguptogether', 'andylassner',
'thekalenallen', 'official_twitch', 'played', 'kkwbeauty', 'pop', 'pst', 'kkwfragrance', 'sold', 'shop', 'lipstick',
'kkw', '12pm', 'classic', 'palette', 'lip', 'liners', 'lipsticks', 'nude', 'striveforgreatness', 'lil', 'bro', 'homie',
'sir', 'brother', 'lol', 'crazy', 'g', 'tony', 'gaga', 'lady', 'joanne', 'monsters', 'training', 'cristiano',
'portugal', 'hi', 'para', 'mercurial', 'cr7', 'nikefootball', 'madrid', 'el', 'collection', 'hola', 'hala', 'e',
'match', 'photos', 'cr7underwear', 'football', 'yesterday', 'hello', 'en', 'que', 'de', 'oscars', 'meantweets',
'realdonaldtrump', 'iamguillermo', 'kimmel', 'thecousinsal', 'trump', 'real', 'old', 'gerrymandering', 'time',
'get', 'love', 'celebapprentice', 'reform', 'great', 'next', 'friend', 'arnoldsports', 'pumped', 'wait', 'see',
'take', 'thank', 'together', 'first', 'one', 'know', 'amp', 'every', 'day', 'world', 'like', 'fantastic',
'asasafterschool', 'back', 'thanks', 'terminator', 'snapchat', 'future', 'join', 'us', 'miss', 'want',
                  'proud', 'today', 'w', 'support', 'go', 'people', 'watch', 'work', 'let', 'best']

RTcommenWords=['whitehouse', 'dbongino', 'president', 'joebiden', 'vp', 'kkwbeauty', 'available', '12pm', 'pst',
'kkwfragrance', 'west', 'collection', 'kuwtk', 'kim', 'shop', 'kimkardashian', 'kkwmafia',
'lipstick', 'palette', 'lip', 'lebron', 'kingjames', 'uninterrupted', 'ljfamfoundation', 'gaga',
'tony', 'lady', 'ladygaga', 'btwfoundation', 'cristiano', 'jimmykimmel', 'rt', 'arnold',
'schwarzenegger', 'amp', 'thearnoldfans', 'great', 'http', 'reddit', 'gerrymandering', 'get', 'n', 'today',
'sabotagemovie', 'snapchat', 'watch', 'live', 'tank', 'new', 'terminator', 'like']


numricfichers=["word count","wordl len","numCap","numCap","numHashtags","numOfTaging ","mean word",
               "num of !" ,   "num of ?","num of dots","num of commas"]

emojilist=["u'\u2728'", "u'\U0001f352'", "u'\U0001f49c'", "u'\u2640'", "u'\U0001f499'", "u'\U0001f351'"
 "u'\U0001f5e3'","u'\U0001f440'", "u'\u203c'", "u'\u2642'", "u'\U0001f926'", "u'\U0001f937'",
"u'\U0001f680'", "u'\u270a'", "u'\U0001f4af'", "u'\U0001f923'", "u'\U0001f601'", "u'\U0001f451'"
"u'\U0001f62d'", "u'\U0001f389'", "u'\U0001f3b6'", "u'\U0001f60a'", "u'\U0001f31f'",
"u'\U0001f3a4'", "u'\U0001f496'", "u'\U0001f607'""u'\U0001f44d'", "u'\U0001f44c'",
"u'\U0001f51d'", "u'\U0001f609'", "u'\u26bd'"]



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

def iscontains(text,word):
    if word in text:
        return 1
    else:
        return 0



def pre_pro (sentence):
    #  remove sites from tweets, @, #
    # sentence = sentence.lower()
    r_site = pattern.sub('', sentence)
    # r_strudel = pattern2.sub('', r_site)
    # r_hesteck = pattern3.sub('', r_strudel)
    # print (r_hesteck)
    return r_site


def RTsplit(data):
    """
    splits to RT and notRT
    """
    RT = data.loc[data["is RT"] == True]
    not_RT = data.drop(RT.index,axis = 0)
    return RT,not_RT


def writeRawRT(path):
    """
    write only the RT and notRT
    """
    data=pd.read_csv("train.csv")
    tweets=data["tweet"]
    data["is RT"]=tweets.apply(iscontains,word="is RT")
    RT,notRT=RTsplit(data)
    RT = RT.drop(["is RT"],axis = 1)
    notRT = notRT.drop(["is RT"], axis =1)
    RT.to_csv(r'rawRT.csv',index=False)
    notRT.to_csv(r'rawNotRT.csv',index=False)

def addcommenwords(tweets,data,flag):

    if flag==1:
        for comword in notRTcommenWords:
            newcol=tweets.apply(iscontains,word=comword)
            data[comword]=newcol
    else:
        for comword in RTcommenWords:
            newcol=tweets.apply(iscontains,word=comword)
            data[comword]=newcol
    # data=addemojis(data)
    return data


def addemojis(data):
    emojilists=data["emojilists"]
    for emoji in emojilist:
        newcol=emojilists.apply(iscontains,word=emoji)
        data[emoji]=newcol
    return data



def  addnmricfichers(tweets,data):
    """
    add numric fichers about the tweet
    :param tweets:
    :param data:
    :return: a data with numric fichers z-score
    """
    data["word count"]=tweets.str.split().apply(len)
    data["wordl len"]=wordlens=tweets.str.len()
    data["numCap"]=tweets.str.findall(r'[A-Z]').str.len()
    data["numHashtags"]=tweets.str.findall(r'#').str.len()
    data["numOfTaging "]=tweets.str.findall(r'@').str.len()
    # data["emojilists"]=tweets.apply(extract_emojis)
    data["mean word"]=tweets.apply(getMeanWord)
    data["num of !"]=tweets.str.findall(r'!').str.len()
    data["num of ?"]=tweets.str.findall(r'\?').str.len()
    data["num of dots"]=tweets.str.findall(r'\.').str.len()
    data["num of commas"]=tweets.str.findall(r'\,').str.len()
    data["is RT"]=tweets.apply(iscontains,word="RT @")
    cols = list(data.columns)
    for col in numricfichers:
        data[col] = (data[col] - data[col].mean())/data[col].std(ddof=0)
    return data


def runMe(path):
    # data=readData(path)
    data=pd.read_csv(path)
    tweets=data["tweet"]
    data.tweet=tweets.apply(pre_pro)
    data=addnmricfichers(tweets,data)
    labels=data["user"]
    RT ,notRT=RTsplit(data)
    RT=addcommenwords(tweets,RT,0)
    notRT=addcommenwords(tweets,notRT,1)
    # RT.to_csv(r'testRT.csv',index=False)
    # notRT.to_csv(r'testNotRT.csv',index=False)
    return RT ,notRT


runMe("test.csv")