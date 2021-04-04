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


#Loading Datasets
df = pd.read_csv('/Users/lin/Desktop/thesis_code/df_age.csv')

# #Splitting DataSet using ids
# X_train_id, X_test_id = split_patient(df.PERSON_CD.unique(), df = df)
# y_train_id, y_test_id = get_label(X_train_id), get_label(X_test_id)



#Splitting DataSet naively
X_train, X_test, _, _ = split_naive(df)
y_train, y_test = X_train['DIFF_HRZ'], X_test['DIFF_HRZ']



##Linear mixed effect
def Linearmixed(model, df_train, df_test):
  md = smf.mixedlm(model, data = df_train, groups=df_train["AgeGroup"])
  mdf = md.fit()
  print(mdf.summary())
  test_result = mdf.predict(df_test)
  result_pred = []
  for i in test_result:
    if i < 0 :
      result_pred.append(1)
    else:
      result_pred.append(0)

  return result_pred


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
                       'ino_bolus',
                       'AgeGroup']

categorical_columns_linear = ['opioid_inf',
                       'sed_inf',
                       'opioid_bolus',
                       'sed_bolus',
                       'vasodi_inf',
                       'vasodi_bolus',
                       'ino_inf',
                       'ino_bolus',
                       'AgeGroup']
columns  = categorical_columns + numeric_columns

column_model = " + ".join(columns)
model = "DIFF_HRZ ~ " + column_model 

prediction_Linear_mixed = Linearmixed(model, X_train, X_test)



##Random Forest
clf = RandomForestClassifier(max_depth=100,
                             bootstrap=True,
                             max_features = 3,
                             min_samples_leaf = 5,
                             min_samples_split = 12,
                             n_estimators = 300)
clf.fit(X_train[columns],get_label(X_train))
random_Forest_result = clf.predict(X_test[columns])


#GAM
formula = f(0)
for i,name in enumerate(categorical_columns[1:]):
    formula += f(i)
for j,name in enumerate(numeric_columns):
    formula += s(j+len(categorical_columns))
print(formula)

listings_gam = LinearGAM(formula).fit(X_train[columns], y_train)

label_true = get_label(X_test)                                                                       
preds = listings_gam.predict(X_test[columns])                                                        
pred_label = get_pred_label(preds)  
pred_label_mixedLinear = get_pred_label(prediction_Linear_mixed)  

print("The accuracy score for GAM model is {:0.3f}".format(accuracy_score(label_true, pred_label)))
print("The accuracy score for Random Forest is: {:0.3f}".format(accuracy_score(label_true,random_Forest_result)))     
print("The accuracy score for linear Mixed effect model is: {:0.3f}".format(accuracy_score(label_true,pred_label_mixedLinear)))     

final_pred=[]
for i in range (len(label_true)):
  if pred_label[i] + random_Forest_result[i] + pred_label_mixedLinear[i] >= 2:
    final_pred.append(1)
  else:
    final_pred.append(0)

print("The accuracy score for final is {:0.3f}".format(accuracy_score(label_true, final_pred)))


df_result = pd.DataFrame(data = {'PERSON_CD':X_test['PERSON_CD'],'Time':X_test['DT_UTC'], 'RandomForest':random_Forest_result, 'LinearMixed':prediction_Linear_mixed, 'y_true':y_test})    
df_result.to_csv('model_result.csv', index = False)

from IPython import embed
embed()
