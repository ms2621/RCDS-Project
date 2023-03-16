import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor

import Load_data as ld


# linear fit
def f_1(x, A, B):
    return A * x + B


foldername = 'Data_YuDengLab'
datafile = 'Data_model_construction_YuDengLab'

score_mean = []

# range of n_estimators values for testing
trial_values = np.arange(10, 300, 8)

n = 5  # train n times for each n_estimators value
cro_val_indicator = False  # whether carrying out cross validation or not
cross_val_num = 5  # number of cross validation

plt.figure(f'Training with {len(trial_values)} n_estimators values')

data = ld.tokenisation(''+str(foldername)+'/'+str(datafile)+'.csv')

for j in range(len(trial_values)):
    print(f'>>>>> [Trial {j+1}/{len(trial_values)}] Training with n_estimators = {trial_values[j]} ...')

    score_one_para = []
    for i in range(n):
        train_feat, train_id = ld.shuffle_data(data)
        normalized_test_data = train_feat - (np.mean(np.concatenate(train_feat))
                                             / np.std(np.concatenate(train_feat)))

        X_train, X_test, y_train, y_test = train_test_split(normalized_test_data, train_id,
                                                            test_size=0.1, random_state=0)

        regr = RandomForestRegressor(n_estimators=trial_values[j])

        if cro_val_indicator == True:
            cross_val = cross_val_score(regr, normalized_test_data, train_id, cv=cross_val_num).mean()
            print(f'      Mean score after {cross_val_num} cross validations: {cross_val:.4f}')

        regr.fit(X_train, y_train)
        pred = regr.predict(X_test)
        pred2 = regr.predict(X_train)
        score = r2_score(y_test, pred)

        score_one_para.append(score)

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
