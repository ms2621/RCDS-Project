import numpy as np
import math
from sklearn.metrics import r2_score
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy import optimize


def load_data(filename):
    train_feat = []
    train_id = []
    data = []
    with open(filename, 'r') as f:
        file = f.readlines()
        for h in file:
            line = h.strip().split(',')
            x_l = [math.log(float(line[0]), 10)]  # take log of the fluorescence value

            # convert string of sequences into numbers
            for a in line[1]:
                if a == 'A':
                    x_l.append(1)
                if a == 'G':
                    x_l.append(2)
                if a == 'T':
                    x_l.append(3)
                if a == 'C':
                    x_l.append(4)
                if a == 'B':
                    x_l.append(0)
            x_l = np.array(x_l)
            data.append(x_l)
    
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


# linear fit
def f_1(x, A, B):
    return A * x + B


# train n times
n = 20
for i in range(n):
    train_feat, train_id = load_data('Data_model_construction.csv')

    normalized_test_data = (train_feat - np.mean(train_feat) / np.std(train_feat))
    X_train, X_test, y_train, y_test = train_test_split(normalized_test_data, train_id,
                                                        test_size=0.1, random_state=0)

    regr = RandomForestRegressor(n_estimators=70)

    regr.fit(X_train, y_train)
    pred = regr.predict(X_test)
    pred2 = regr.predict(X_train)
    score = r2_score(y_test, pred)
    plt.rc('font', family='Times New Roman')

    plt.figure()
    A1, B1 = optimize.curve_fit(f_1, y_test, pred)[0]
    x1 = np.arange(min(y_train), max(y_train), 0.01)
    y1 = A1 * x1 + B1
    plt.plot(x1, y1, c="green")
    plt.title('R2=' + str(round(score, 2)))
    plt.scatter(y_train[:1000], pred2[:1000], s=5, c="pink", marker='o', label="Train")
    plt.scatter(y_test[:200], pred[:200], s=5, c="b", marker='x', label="Test")
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.xlabel('Origin')
    plt.ylabel('Predict')
    plt.legend()

    plt.savefig('Regression_plot/Regression' + str(i+1) + '.png')
