# Prediction of Promoter Strength with Machine Learning (Random Forest Model)

## Quick Intro
We are Interdisciplinary Computing Project Group 4 from the Imperial College I-Explore course. The aim of our project is to use a Random Forest (RF) model to predict promoter strength in DNA. This is an interdisciplinary project that bridges the fields of biochemistry and computing. Our team consists of four members who are second-year biochemists and physicists at Imperial College London.

## File Structure
The `Data_YuDengLab/` folder contains the sample data from **YuDengLab**, and the `Data_EVMP/` folder contains the sample data from **EVMP**  as referenced below.

The `No_cross_validation/` folder in the `Data_YuDengLab/` folder contains the sample plots showing R2 distribution after certain number of trials. The R2 values shown on the plots are obtained *without* performing cross validation when training the RF model. On the other hand, the R2 values of plots in the `With_cross_validation/` folder are obtained *with* performing cross validation when training the RF model.

## Training
During training, the output log is like the following.
```
>>>>> Training and testing trial 1/10 ...
      Mean score after 5 cross validations: 0.6457
>>>>> Training and testing trial 2/10 ...
      Mean score after 5 cross validations: 0.6413
>>>>> Training and testing trial 3/10 ...
      Mean score after 5 cross validations: 0.6387
.....
>>>>> Training and testing trial 10/10 ...
      Mean score after 5 cross validations: 0.6414

----------End of trials----------
```
The regression plots are in the `Regression_plot/` subfolder of the relavent data folder (e.g. `Data_YuDengLab/` folder). An example regression plot using `Data_model_construction_YuDengLab.csv` data is shown below.

![Regression plot](/Data_YuDengLab/Regression_plot/Regression_Data_YuDengLab_2.png "Example regression plot trained with *Data_model_construction_YuDengLab.csv* data.")

The example plot of R2 score distribution for all trials will be stored directly in the relavent data folder (e.g. `Data_YuDengLab/` folder). An example R2 score distribution plot using `Data_model_construction_YuDengLab.csv` data is shown below with 100 trials. No cross validation was performed.

![R2 distribution plot](/Data_YuDengLab/No_cross_validation/R2_distribution_of_100_trials_Data_YuDengLab.png "Example R2 distribution plot trained 100 times with *Data_model_construction_YuDengLab.csv* data.")

## Reference
#### YuDengLab
https://github.com/YuDengLAB/Predictive-the-correlation-between-promoter-base-and-intensity-through-models-comparing

#### EVMP
https://github.com/Tiny-Snow/EVMP