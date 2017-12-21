import copy
import math
import numpy as np
from math import sqrt
import pandas as pd
from random import randint


def max_distance(random_number):
    maxval = 0
    for i in range(10):
        if a[random_number][i] > maxval:
            maxval = a[random_number][i]
            max_index = i
    return max_index


def calculate_distance(a,b):
    return abs(a-b)

def new_distance(a,projection_list):
    d=np.zeros(shape=(10,10),dtype=np.float)
    for i in range(10):
        for j in range(10):
            d[i][j] = math.sqrt(((a[i][j])**2) -  (projection_list[i]-projection_list[j])**2)
    return d



def fastmap(a,oa,ob):
    projection_list = []
    for i in range(10):
        if i not in(oa,ob):

            dai = a[oa][i]
            dbi = a[ob][i]
            dab = a[oa][ob]
            xi = float(((dai*dai) + (dab*dab) - (dbi*dbi))/(2*dab))
            projection_list.append(xi)
        elif i == oa:
            projection_list.append(0.0)
        else:
            projection_list.append(float(dab))
    return projection_list

if __name__ == "__main__":
    count = 0
    dataframe = pd.read_csv('fastmap-data.txt', sep='\t',header=None)
    df = dataframe.values
    max_list = []
    a = np.zeros(shape=(10,10),dtype=np.float)
    outputlist = []
    for item in df:
        i = item[0]
        j = item[1]
        a[i-1][j-1] = item[2]
        a[j-1][i-1] = item[2]
    for k in range(2):
        random_number = randint(0, 9)

        max_list.append(random_number)
        max_index = random_number

        while(True):
            max_index = max_distance(max_index)

            max_list.append(max_index)

            startindex = len(max_list) -1

            if len(max_list) > 2:
                if max_list[startindex] == max_list[startindex-2]:
                    oa = max_list[startindex]
                    ob = max_list[startindex-1]
                    break
                else:
                    startindex = startindex+1

        dab = abs(oa-ob)

        if oa<ob:
            projection_list = fastmap(a,oa, ob)
        else:
            projection_list = fastmap(a,ob, oa)
        print projection_list
        outputlist.append(projection_list)
        count = count+1
        if count<2:
            a = new_distance(a,projection_list)

    print outputlist