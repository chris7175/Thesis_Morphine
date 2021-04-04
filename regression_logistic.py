from __future__ import division
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from datetime import datetime as dt
from sklearn.metrics import accuracy_score
import arviz as az
import pymc3 as pm
from scipy.special import expit



def split_patient(patient_ids, df):
  '''
  This function returns splitting using ids
  '''
  X_train, X_test, y_train, y_test = train_test_split(patient_ids, patient_ids, test_size=0.25)
  df_train = df[df.PERSON_CD.isin(X_train)]
  df_test = df[df.PERSON_CD.isin(X_test)]
  return df_train, df_test


def split_naive(df):
  X_train, X_test, y_train, y_test = train_test_split(df, df, test_size=0.25)
  return X_train, X_test, y_train, y_test


def get_label(df):
    '''
    This functions return splitting labels
    '''
    result = []
    for index, rows in df.iterrows():
        if rows['DIFF_HRZ'] < 0 :
            result.append(1)
        else:
            result.append(0)
    return result




if __name__ == '__main__':

        #Loading Datasets
    df = pd.read_csv('/Users/lin/Desktop/thesis_code/data.csv')

    #Splitting DataSet using ids
    X_train_id, X_test_id = split_patient(df.PERSON_CD.unique(), df = df)
    y_train_id, y_test_id = get_label(X_train_id), get_label(X_test_id)

    #Splitting DataSet naively
    X_train, X_test, _, _ = split_naive(df)
    y_train, y_test = get_label(X_train), get_label(X_test)




    #Params
    numeric_columns = ['AGE', 'DOSE_MG_PER_KG', 'DOPAMINE_RATE_MEAN', 
                    'HYDROMORPHONE_RATE_MEAN', 'MEDIAN_HR_5_Z_y']
    categorical_columns = ['opioid_inf',
                        'sed_inf',
                        'opioid_bolus',
                        'sed_bolus',
                        'vasodi_inf',
                        'vasodi_bolus',
                        'ino_inf',
                        'ino_bolus']

    columns = numeric_columns + categorical_columns


    # Here we define our varying-intercept model using
    # the actual Y training data
    with pm.Model() as vi_model:

        betas = []
        for i in range(len(columns)):
            betas.append(pm.Normal("beta_"+str(i), mu=0, sigma=50))

        LOGIT_P = 0
        for i in range(len(betas)):
            LOGIT_P += betas[i] * X_train[columns[i]]

        obs = pm.Bernoulli(
            "obs",
            logit_p=LOGIT_P,
            observed=y_train
        )

        # Next we set our sampling parameters and draw our posterior samples
        nsamps = 10000
        tune = 2000
        target_accept = 0.90

        # draw posterior samples from our vi_sim_model 
        with vi_model:
            vi_trace = pm.sample(
                draws=nsamps,
                tune=tune,
                target_accept=target_accept,
                # return_inferencedata=False, # silences 'FutureRelease' warning
            ) 
    



    
    from IPython import embed
    embed()
