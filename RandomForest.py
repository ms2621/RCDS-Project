import math
import random
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor

import Load_data as ld

        
# linear fit
def f_1(x, A, B):
    return A * x + B


# data directory
foldername = 'Data_YuDengLab'
datafile = 'Data_model_construction_YuDengLab'

n = 100  # train n times
score_list = []

cro_val_indicator = False  # whether carrying out cross validation or not
cross_val_num = 5  # number of cross validation

plot_individual = False  # whether plotting regression plot for each trial or not

train_feat, train_id = ld.load_data(''+str(foldername)+'/'+str(datafile)+'.csv')
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

    # regression plot
    if plot_individual == True:
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

        plt.savefig(''+str(foldername)+'/Regression_plot_single_trial/Regression_'
                    + ''+str(foldername)+'_'+str(i+1)+'.png')

print('\n----------End of trials----------')


# plot all trials
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
            + ''+str(foldername)+'_with_all_4_motifs.png')