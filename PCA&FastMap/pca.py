import itertools
import pandas as pd
import numpy as np
import math
import copy
import random

if __name__ == "__main__":
    df = pd.read_csv('pca-data.txt', sep = '\t',header=None)
    #df = df.astype(float)
    data_array = df.as_matrix()
    #data_array = data_array.astype(int)

    count = len(data_array)
    print count
    mean_list = []

    for i in range(0,3):
        sum = 0
        for j in range(0,count):
            sum+=data_array[j][i]
        mean_list.append(sum/count)

    for i in range(0,3):
        for j in range(0,count):
            data_array[j][i]= data_array[j][i]-mean_list[i]

    cov = np.zeros(shape=(3,3))
    cov = (np.matrix(data_array).T* np.matrix(data_array))/count
    print mean_list
    #print np.shape(cov)
    #print data_array
    #print cov

    eigenvalues,eigenvector = np.linalg.eig(cov)
    print eigenvalues
    print "Eigen Vector"
    print eigenvector

    eigen_dict = {eigenvalues[i]:eigenvector[i] for i in range(0,3)}
    print eigen_dict
    sort_eigen = sorted(eigen_dict.items(),reverse = True)

    print "Sorted eigen"
    print sort_eigen

    red_list = []
    tries = 0
    for key,value in eigen_dict.items():
        if(tries == 2):
            break
        tries = tries + 1
        red_list.append(value)

    print  red_list

    trunc_matrix = np.array(red_list)
    trunc_matrix = trunc_matrix[:,0,:]
    print trunc_matrix
    print "trunc"
    print np.shape(trunc_matrix)

    final_matrix =  np.dot(trunc_matrix,(data_array).T)
    print final_matrix
    print np.shape(final_matrix)
    #np.savetxt("test.txt",final_matrix)

    # f = open("test.txt", "w")
    # f.write(final_matrix)
    # f.close()
    # k_eigen = sort_eigen[:,:2]
    # print "K eigen"
    # print k_eigen

