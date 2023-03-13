import numpy as np
import math
from sklearn.metrics import r2_score
import random
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy import optimize


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


            # obtaining the frequency that TATA sequence exits
            start_index = 0
            count = 0
            string = 'TATAAA'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)


            # obtaining the frequency that BRE sequence exits
            start_index = 0
            count = 0
            string = 'GGGCGCC'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'GCGCGCC'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'GGACGCC'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'GCACGCC'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'CCACGCC'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'CCGCGCC'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'CGACGCC'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'CGGCGCC'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)


            # obtaining the frequency that DPE sequence exits
            start_index = 0
            count = 0
            string = 'AGACG'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'AGTTG'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'AGACA'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'AGTTA'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'AGACC'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'AGTTC'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'GGACG'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'GGTTG'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'GGACA'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'GGTTA'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'GGACC'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)

            start_index = 0
            count = 0
            string = 'GGTTC'  # modify the motif here
            str_len = len(string)  # length of the motif
            while line[1].find(string, start_index) != -1:
                count += 1
                start_index = line[1].find(string, start_index) + str_len
            x_l.append(count)
           
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

n = 15  # train n times
score_list = []

cro_val_indicator = False  # whether carrying out cross validation or not
cross_val_num = 5  # number of cross validation

train_feat, train_id = load_data(''+str(foldername)+'/'+str(datafile)+'.csv')
normalized_test_data = train_feat - (np.mean(np.concatenate(train_feat))
                                        / np.std(np.concatenate(train_feat)))

for i in range(n):
    print(f'>>>>> Training and testing trial {i+1}/{n} ...')

    X_train, X_test, y_train, y_test = train_test_split(normalized_test_data, train_id, shuffle=True,
                                                        test_size=0.1, random_state=0)

    regr = RandomForestRegressor(n_estimators=70)

    if cro_val_indicator == True:
        cross_val = cross_val_score(regr, normalized_test_data, train_id, cv=cross_val_num).mean()
        print(f'      Mean score after {cross_val_num} cross validations: {cross_val:.4f}')

    regr.fit(X_train, y_train)
    pred = regr.predict(X_test)
    pred2 = regr.predict(X_train)
    score = r2_score(y_test, pred)

    score_list.append(score)

    plt.rc('font', family='Times New Roman')

    # regression plot
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

    # plt.savefig(''+str(foldername)+'/Regression_plot_single_trial/Regression_'
    #             + ''+str(foldername)+'_'+str(i+1)+'.png')
print('\n----------End of trials----------')

trial_num = np.arange(1, n+1, 1)
score_mean = np.mean(score_list)
score_std = np.std(score_list)
if score_std < 0.01:
    score_std_label = 'Std < 0.01'
else:
    score_std_label = f'Std = {score_std:.2f}'

plt.figure('R2 distribution')
plt.plot(trial_num, np.zeros(len(trial_num))+score_mean, '-', lw=5, color='orange',
         label=f'Mean = {score_mean:.2f}')
plt.fill_between(trial_num, score_mean-score_std, score_mean+score_std, color='grey',
                 alpha=0.3, linewidth=0, label=score_std_label)
plt.plot(trial_num, score_list, 'o', ms=5, color='green', label='R2 score')
plt.xticks(trial_num[int(n/5)-1::int(n/5)])
plt.xlabel('Trial number')
plt.ylabel('R2 score')
plt.legend()

plt.savefig(''+str(foldername)+'/R2_distribution_of_'+str(n)+'_trials_'
            + ''+str(foldername)+'.png')