# Prediction of Promoter Strength with Machine Learning (Random Forest Model)


## Quick Intro
We are Interdisciplinary Computing Project Group 4 from the Imperial College I-Explore course. Our aim is to use a Random Forest (RF) model to predict promoter strength in DNA. Promoters are a crucial sequence feature found in eukaryotic genomes and are involved in regulating the expression of genes located downstream of the promoter sequence. This regulation is accomplished through the binding of proteins called transcription factors, which bind at distinct, conserved sequence motifs that can be identified using predictive software. The prediction of promoters and their regulatory strength is essential in biotechnology, particularly for producing novel synthetic promoters. Our project is an interdisciplinary effort that brings together the fields of biochemistry and computing. Our team consists of four members who are second-year biochemistry and physics students at Imperial College London.


## File Structure
The `Data_YuDengLab/` folder contains the sample data from **YuDengLab** as referenced below.

Inside `Data_YuDengLab/` folder:

* The data file of `Data_model_construction_YuDengLab.csv`

* The `Regression_plot_single_trial/` folder contains plots that show the outcome of the RF model for both the training and testing data. It also fits the testing data distribution with a straight line and displays the R2 value. See below for more details.

* The `R2_distribution_plot/` folder contains plots that demonstrate how R2 varies when training the model multiple times. See below for more details.

* The `Optimisation_n_estimators/` folder contains plots generated during the optimization of n_estimators. See below for more details.


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

The regression plots are in the `Regression_plot_single_trial/` subfolder of the relavent data folder (e.g. `Data_YuDengLab/` folder). An example regression plot using `Data_model_construction_YuDengLab.csv` data is shown below.

![Regression plot](/Data_YuDengLab/Regression_plot_single_trial/Regression_Data_YuDengLab_2.png "Example regression plot trained with *Data_model_construction_YuDengLab.csv* data.")

The example plot of R2 score distribution for all trials will be stored directly in the relavent data folder (e.g. `Data_YuDengLab/` folder). An example R2 score distribution plot using `Data_model_construction_YuDengLab.csv` data is shown below with 50 trials. This example plot is stored in `R2_distribution_plot/` folder to for the sake of tidiness.

![R2 distribution plot](/Data_YuDengLab/R2_distribution_plot/R2_distribution_of_50_trials_Data_YuDengLab.png "Example R2 distribution plot trained 100 times with *Data_model_construction_YuDengLab.csv* data.")


## Optimising *n_estimators*
Taking *n_estimators* around 100 would not affect the R2 score too much. Yet, the code for running the RF model between a range of *n_estimators* values is still provided in `Optimise_n_estimators.py`. The example code below varies the value of *n_estimators* from 120 to 148 with 2 as the increment.

```
trial_values = np.arange(120, 150, 2)
```

The output of `Optimise_n_estimators.py` gives a plot of R2 against *n_estimators* values. The example output plot below shows how R2 varies with *n_estimators* ranged from 10 to 298 before the number of counts for motifs are implemented for training.

![Optimisation of n_estimators plot](/Data_YuDengLab/Optimisation_n_estimators/Without_motif_count/Value_range_10_298_total_37_values.png "Example Optimisation of n_estimators plot by running 15 values from *n_estimators* = 10 to 298.")

Another example output plot below shows how R2 varies with *n_estimators* ranged from 10 to 295 after the number of counts for motifs are implemented for training.

![Optimisation of n_estimators plot](/Data_YuDengLab/Optimisation_n_estimators/With_motif_count/Value_range_10_295_total_58_values.png "Example Optimisation of n_estimators plot by running 58 values from *n_estimators* = 10 to 295.")


## Note on Collaboration
Members Seoho Shin and James Pullen looked at non-coding aspects such as data acquisition and results presentation. Member Christine Zhao has been in charge of updating the GitHub repository whilst working on the coding aspect of the project alongside member Miguel Sanchez.


## Reference
#### YuDengLab
https://github.com/YuDengLAB/Predictive-the-correlation-between-promoter-base-and-intensity-through-models-comparing

#### EVMP
https://github.com/Tiny-Snow/EVMP


## Data license
The data used for training was from YuDengLab, and the data license is free without registration.
