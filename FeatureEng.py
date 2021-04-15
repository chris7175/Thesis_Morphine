from __future__ import division
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_state(val, threshold):
  if val == 0 :
    return 1
  for i in range(0, len(threshold) +1):
    if i == 0 and val < threshold[0]:
      return 0
    elif i == len(threshold) and val > threshold[-1]:
      return len(threshold)
    else:
      if threshold[i - 1] <= val <= threshold[i]:
        return i



def Add_features(df, id):
    df_patient = df[df['PERSON_CD'] == id].reset_index()
    df_patient = df_patient.sort_values(by='DT_UTC').reset_index()
    df_patient['label'] = [get_state(row['DIFF_HRZ'], [0]) for _, row in df_patient.iterrows()]
    cum_dose = df_patient['DOSE_MG_PER_KG'][0]
    result = [cum_dose]
    count_dose = np.zeros(len(df_patient))
    COUNT_DOSE = 0  
    effective_rate = [1]
    avg_past = [df_patient['DIFF_HRZ'][0]]
    for i in range(1, len(df_patient)):
        result.append(result[-1] + df_patient['DOSE_MG_PER_KG'][i])
        if df_patient['DOSE_MG_PER_KG'][i] != 0:
          COUNT_DOSE += 1
          count_dose[i] = COUNT_DOSE
        
        effective_rate.append(np.sum(df_patient.loc[:i - 1,:]['label'])/(i))
        avg_past.append(np.mean(df_patient.loc[:i - 1,:]['DIFF_HRZ']))
    


    #cumlative dosages
    df_patient['dose_count'] = count_dose  
    df_patient['cumulative_dosage'] = result
    df_patient['prev_effective_rate'] = effective_rate
    df_patient['avg_pasthrz'] = avg_past

    #date since
    date_since = (pd.to_datetime(df_patient['DT_UTC']) - pd.to_datetime(df_patient['DT_UTC']).min()).dt.days.astype(float)
    df_patient['day_since'] = date_since
    return df_patient

def get_df():
    df_new = pd.DataFrame()
    for i in df['PERSON_CD'].unique():
        print(i)
        patient_df = Add_features(df, i)
        df_new = pd.concat([df_new, patient_df])
    return df_new




df = pd.read_csv('df_age.csv')

from IPython import embed
embed()
