from __future__ import division

import random
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy import stats as s
from collections import defaultdict
from sklearn.model_selection import train_test_split
def split_patient(patient_ids, df):
  X_train, X_test, y_train, y_test = train_test_split(patient_ids, patient_ids, test_size=0.25)
  df_train = df[df.PERSON_CD.isin(X_train)]
  df_test = df[df.PERSON_CD.isin(X_test)]
  return df_train, df_test



def get_sum(prediction):
  c = 0
  for i in prediction:
    c += int(i)
  return c

def prediction(df, transition_matrix, N,M,rand =  False):
  patient_ids = df['PERSON_CD'].unique()
  pred_prob=[]
  for i in patient_ids:
    pred_result = ""
    df_patient = df[df.PERSON_CD == i].reset_index()
    if len(df_patient) > N + M:
      #add initial state
      state_initial = [get_state(row['DIFF_HRZ'], [0]) for index, row in df_patient.loc[0:N - 1,:].iterrows()]
      state_initial = "".join(map(str, state_initial))
      pred_result+= state_initial
      for index in range(N, len(df_patient), M):
          #new predictor
          if index + M <= len(df_patient):
              predictor = pred_result[-N:]

              # get the choices and prob for each choices
              choices = list(transition_matrix[predictor].keys())
              prob = list(transition_matrix[predictor].values())

              #check random
              if rand:
                prediction = random.choices(choices, weights = [1/len(prob)]*len(prob), k = 1)[0]
              else:
                prediction = random.choices(choices, weights = prob, k = 1)[0]
              true_val = sum([get_state(row['DIFF_HRZ'], [0]) for index, row in df_patient.loc[index : index + M - 1,:].iterrows()])
              pred_sum = get_sum(prediction)
              if true_val == 0:
                if pred_sum == 0:
                  prob = 1
                else:
                  prob = (M - pred_sum)/ M
              else:
                prob = pred_sum/true_val
              pred_prob.append(prob)
              #adding pred to result of patients
              pred_result += prediction

  return pred_prob

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


def get_transition_matrix_M_N(N, M, df):
    '''
    when the states are consecutive dosages
    '''
    #def transition matrix
    transition_dic ={}
    transition_dic = {}

    patient_ids = df.PERSON_CD.unique()
    # patient_ids = [8096]
    for patient in patient_ids:
      # print("Patient id: {}".format(patient))
      #for each patient
      df_patient = df[df.PERSON_CD == patient].reset_index()
      for index, data in df_patient.iterrows():
        
        if index >= 0 and index + N + M <= len(df_patient) :
            prev = df_patient.loc[index:index + N - 1, :]
            cur = df_patient.loc[index + N: index + N + M - 1, :]

            prev_states = [get_state(row['DIFF_HRZ'], [0]) for _, row in prev.iterrows()]
            cur_states = [get_state(row['DIFF_HRZ'], [0]) for _, row in cur.iterrows()]

            key_prev = "".join(map(str, prev_states))
          
            key_cur = "".join(map(str, cur_states))
            if key_prev in transition_dic:
                if key_cur in transition_dic[key_prev]:
                    transition_dic[key_prev][key_cur] += 1
                else:
                    transition_dic[key_prev][key_cur] = 1
            else:
                transition_dic[key_prev] ={}
                transition_dic[key_prev][key_cur] = 1

        

    result = {}
    for i in transition_dic:
      inner_dic = transition_dic[i]
      dic_sum = sum(inner_dic.values(), 0)
      result[i] ={}
      for j in inner_dic:
        result[i][j] =  inner_dic[j]/dic_sum
    
    return result

def get_state_sum(vals):
  count = 0 
  for i in vals:
    if i < 0 :
      count += 1
  return count

if __name__ == '__main__':

    df = pd.read_csv('df_age.csv')


    pred_result = []
    fake_result = []
    N = 3
    M = 2
    for i in range(20):
        print("iter: ",i)
        df_train, df_test = split_patient(df.PERSON_CD.unique(), df = df)
        transition_matrix_train = get_transition_matrix_M_N(N,M,df_train)
        pred = prediction(df_test, transition_matrix_train,N,M)
        pred_fake = prediction(df_test, transition_matrix_train,N,M, rand=True)
        pred_result.append(sum(pred)/len(pred))
        fake_result.append(sum(pred_fake)/len(pred_fake))
    
    






    from IPython import embed
    embed()
