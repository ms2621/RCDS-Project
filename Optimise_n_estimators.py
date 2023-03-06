import numpy as np
import math
import random
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


def load_data(filename):
    train_feat = []
    train_id = []
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
                                     + ' consists a letter other than A G T C B')
                k += 1
            x_l = np.array(x_l)
            data.append(x_l)
            j += 1
    
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


foldername = 'Data_YuDengLab'
datafile = 'Data_model_construction_YuDengLab'
# foldername = 'Data_EVMP'
# datafile = 'Data_model_testing_EVMP'


score_mean = []
trial_values = np.arange(120, 150, 2)
n = 5  # train n times for each n_estimators value

plt.figure(f'Training with {len(trial_values)} n_estimators values')

for j in range(len(trial_values)):
    print(f'>>>>> [Trial {j+1}/{len(trial_values)}] Training with n_estimators = {trial_values[j]} ...')

    score_one_para = []
    for i in range(n):
        train_feat, train_id = load_data(''+str(foldername)+'/'+str(datafile)+'.csv')

        normalized_test_data = train_feat - np.mean(train_feat) / np.std(train_feat)
        X_train, X_test, y_train, y_test = train_test_split(normalized_test_data, train_id,
                                                            test_size=0.1, random_state=0)

        regr = RandomForestRegressor(n_estimators=trial_values[j])

        # cross validation 3 times
        cross_val = cross_val_score(regr, normalized_test_data, train_id, cv=3).mean()

        score_one_para.append(cross_val)
    plt.plot(np.zeros(n)+trial_values[j], score_one_para, '.', ms=5, color='C0', alpha=0.5)
    score_mean.append(np.mean(score_one_para))
print('\n----------End of trials----------')

max_score = np.where(score_mean==np.max(score_mean))[0][0]
print(f"\nThe best n_estimators = {trial_values[max_score]} with R2 = {score_mean[max_score]:.2f}")  
plt.plot(trial_values, score_mean, 'o-', color='C2', label='Mean')
plt.plot(trial_values[max_score], score_mean[max_score], 'x', ms=8, mew=3, color='C1',
         label=f'Best n_estimators')
plt.xlabel('n_estimators')
plt.ylabel('R2 score')
plt.legend()

plt.savefig(f'{foldername}/Optimisation_n_estimators/Value_range_{trial_values[0]}_{trial_values[-1]}_'
            + f'total_{len(trial_values)}_values.png')
