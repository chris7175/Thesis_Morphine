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

df = pd.read_csv('/Users/lin/Desktop/thesis_code/df_age.csv')


def plot_scatter_log(var, response,df = df, index = 0 ):
    plt.ion() 
    fig, axs = plt.subplots(1, 3, figsize=(25, 10))
    log_var  = np.log(df[var] - min(df[var]) + 0.0001)
    log_response = np.log(df[response] - min(df[response]) + 0.0001)
    axs[0].scatter(log_var, log_response)
    axs[0].set_xlabel('log ' + var)
    axs[0].set_ylabel('log ' + response)

    axs[0].set_ylim(-1,3)
    # axs[0].set_xlim(-2,3)


    axs[0].set_title('log-log plot')

    axs[1].scatter(log_var, df[response])
    axs[1].set_title('semilog plot')
    axs[1].set_xlabel('log ' + var)
    axs[1].set_ylabel(response)

    # axs[1].set_xlim(-1,3)
    # axs[1].set_ylim(-7,7)

    axs[2].scatter(df[var], log_response)
    axs[2].set_title('semilog plot')
    axs[2].set_xlabel(var)
    axs[2].set_ylabel('log ' +  response)

    # axs[2].set_xlim(-8,8)
    # axs[2].set_ylim(1,3)
    axs[2].set_ylim(-1,3)
    


    plt.savefig(var+'.png')




    plt.show()

        

from IPython import embed
embed()
