'''
Load data for training RF
'''
import math
import random
import numpy as np

import Generate_motif as gm


def tokenisation(filename):
    data = []
    with open(filename, 'r') as f:
        file = f.readlines()
        j = 1
        for h in file:
            line = h.strip().split(',')
            x_l = [math.log(float(line[0]), 10)]  # take log of the fluorescence value

            # convert string of sequences into numbers
            k = 1
            for a in line[1]:
                if a == 'A':
                    x_l.append(1)
                elif a == 'G':
                    x_l.append(2)
                elif a == 'T':
                    x_l.append(3)
                elif a == 'C':
                    x_l.append(4)
                elif a == 'B':
                    x_l.append(0)
                else:
                    raise ValueError('The '+str(k)+'th letter '+str(j)+'th sequence'
                                     + ' consists a letter other than A G T C')
                k += 1
            
            x_l = gm.count_motif(line[1], x_l)
           
            x_l = np.array(x_l)
            data.append(x_l)
            j += 1
    return data


def shuffle_data(data):
    train_feat = []
    train_id = []

    # randomise the data order
    random.shuffle(data)

    # split data array into x values and y values for training and testing
    for t in data:
        # log(fluorescence)
        train_feat.append(t[1:])  # will be splitted into x_train and x_test in train_test_split

        # sequence in numbers
        train_id.append(t[0])  # will be splitted into y_train and y_test in train_test_split

    train_feat = np.array(train_feat)
    train_id = np.array(train_id)

    return train_feat, train_id
