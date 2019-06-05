import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import pandas as pd
import os



path="/cs/usr/matanhalfon/Desktop/MLhachton/data"

def readData(path):
    """
    read connect and shafell the tweets and then write them to
    train validation and test cvs
    :param path: a directory with tweet files
    """
    data=[]
    for filename in os.listdir(path):
        if filename.endswith("csv"):
            addpath=path+"/"+filename
            to_add=pd.read_csv(addpath)
            data.append(to_add)
        else:
                continue
    addedData=pd.concat(data,sort=True)
    result=addedData.sample(frac=1).reset_index(drop=True)
    train=result[:12000]
    validtion=result[12000:25000]
    test=result[25000:]
    train.to_csv(os.path.join(path,r'train.csv'),index=False)
    validtion.to_csv(os.path.join(path,r'validation.csv'),index=False)
    test.to_csv(os.path.join(path,r'test.csv'),index=False)


print(readData(path))