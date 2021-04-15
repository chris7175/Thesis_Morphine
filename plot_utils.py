from __future__ import division
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from pygam import LinearGAM, s, f
from sklearn.linear_model import LinearRegression

df = pd.read_csv('df_age.csv')

numeric_columns = ['AGE', 
                   'DOSE_MG_PER_KG', 
                   'DOPAMINE_RATE_MEAN', 
                   'HYDROMORPHONE_RATE_MEAN', 
                   'MEDIAN_HR_5_Z_y',
                   'CUM_ME',
                   'dosage_per_kg_cisatracurium_total']


def split_naive(df):
  X_train, X_test, y_train, y_test = train_test_split(df, df, test_size=0.25)
  return X_train, X_test, y_train, y_test


X_train, X_test, _, _ = split_naive(df)
y_train, y_test = X_train['DIFF_HRZ'], X_test['DIFF_HRZ']


def get_best_transform(var, response = 'DIFF_HRZ',df = df ):
    df = df[df[var].between(df[var].quantile(0.1), df[var].quantile(.9))]  
    log_var  = np.log(df[var] - min(df[var]) + 0.0001).values
    log_response = np.log(df[response] - min(df[response]) + 0.0001).values
    length  = len(log_var)
    log_var = log_var.reshape(length, 1)
    log_response = log_response.reshape(length,1)
    raw_reponse = df[response].values.reshape(length,1)
    raw_var = df[var].values.reshape(length,1)
    result = []
    dic = {}
    #log log
    result.append(LinearReg(log_var, log_response))
    result.append(LinearReg(log_var, raw_reponse))
    result.append(LinearReg(raw_var, log_response))
    result.append(LinearReg(raw_var, raw_reponse ))
    print(result)

    return result


def iter_columns(columns,df):
    dic ={}
    for i in columns:
        max_score = np.argmax(get_best_transform(i, 'DIFF_HRZ', df))
        if max_score == 0 :
            dic[i] = 'log-log'
        elif max_score == 1:
            dic[i] = 'log-var'
        elif max_score == 2:
            dic[i] = 'log-response'
        else:
            dic[i]="no transform"
    return dic
    


def plot_scatter_log(var, response,df = df, index = 0 ):
    plt.ion() 
    fig, axs = plt.subplots(1, 3, figsize=(25, 10))
    log_var  = np.log(df[var] - min(df[var]) + 0.0001)
    log_response = np.log(df[response] - min(df[response]) + 0.0001)

    #log-log
    axs[0].scatter(log_var, log_response)
    axs[0].set_xlabel('log ' + var)
    axs[0].set_ylabel('log ' + response)
    axs[0].set_ylim(0.5,3)
    axs[0].set_xlim(0,3)

    #var_log
    axs[0].set_title('log-log plot')
    axs[1].scatter(log_var, df[response])
    axs[1].set_title('semilog plot')
    axs[1].set_xlabel('log ' + var)
    axs[1].set_ylabel(response)

    axs[1].set_xlim(0,3)
    # axs[1].set_ylim(-7,7)

    axs[2].scatter(df[var], log_response)
    axs[2].set_title('semilog plot')
    axs[2].set_xlabel(var)
    axs[2].set_ylabel('log ' +  response)

    # axs[2].set_xlim(-8,8)
    # axs[2].set_ylim(1,3)
    axs[2].set_ylim(-1,3)
    axs[2].set_ylim(0.5,3)

    plt.show()

def LinearReg(X,y):
    reg = LinearRegression().fit(X, y)
    return reg.score(X, y)
        

from IPython import embed
embed()
