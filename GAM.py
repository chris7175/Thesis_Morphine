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

def get_pred_label(pred):
    '''
    This functions return splitting labels
    '''
    result = []
    for i in pred:
        if i < 0 :
            result.append(1)
        else:
            result.append(0)
    return result


#Loading Datasets
df = pd.read_csv('/Users/lin/Desktop/thesis_code/df_age.csv')

# #Splitting DataSet using ids
# X_train_id, X_test_id = split_patient(df.PERSON_CD.unique(), df = df)
# y_train_id, y_test_id = get_label(X_train_id), get_label(X_test_id)



#Splitting DataSet naively
X_train, X_test, _, _ = split_naive(df)
y_train, y_test = X_train['DIFF_HRZ'], X_test['DIFF_HRZ']


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
columns  = categorical_columns + numeric_columns
formula = f(0)
for i,name in enumerate(categorical_columns[1:]):
    formula += f(i)
for j,name in enumerate(numeric_columns):
    formula += s(j+len(categorical_columns))
print(formula)

listings_gam = LinearGAM(formula).fit(X_train[columns], y_train)
# print("GAM Model Test R^2 {:0.3f} (Train was {:0.3f})".format(accuracy_score(y_test, listings_gam.predict(X_test[columns])),
#                                                               accuracy_score(y_train, listings_gam.predict(X_train[columns]))))


label_true = get_label(X_test)                                                                       

preds = listings_gam.predict(X_test[columns])                                                        
pred_label = get_pred_label(preds)    
print("The accuracy score for GAM model is {:0.3f}".format(accuracy_score(label_true, pred_label)))
from IPython import embed
embed()
