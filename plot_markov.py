from __future__ import division
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from itertools import permutations   
from sklearn.model_selection import train_test_split

pred = np.loadtxt('pred.txt') 
fake = np.loadtxt('fake.txt')
pred = pred.reshape(56,-1)[:,-25:]
fake = fake.reshape(56,-1)[:,-25:]

NMs = list(permutations([1, 2, 3, 4, 5, 6, 7, 8], 2))

def get_sum(prediction):
  c = 0
  for i in prediction:
    c += int(i)
  return c

def plot_perm(index, pred = pred,fake = fake, NMs = NMs):
    plt.ion()
    pred = pred[index]
    mean_pred = np.mean(pred)
    fake = fake[index]
    mean_fake = np.mean(fake)
    plt.hist(pred, bins = 18, label = "Markov trial")
    plt.hist(fake, bins = 18, label = "random trial")
    plt.axvline(x = mean_pred, linestyle = "dashed", color = "black")
    plt.text(mean_pred + 0.0005, 4, r'$\mu$ = '+str(mean_pred)[:5]) 
    plt.axvline(x = mean_fake, linestyle = "dashed", color = "black")
    plt.text(mean_fake + 0.0005, 3, r'$\mu$ = ' + str(mean_fake)[:5]) 
    plt.xlabel("accuracy")
    plt.ylabel("frequency")
    plt.title("N" +str(NMs[index][0]) + "M"+str(NMs[index][1]) + " Markov v.s. Random trial accuracy distribution" )
    plt.legend(loc ="lower right")
    plt.savefig('result_new/'+"N" +str(NMs[index][0]) + "M"+str(NMs[index][1])+".png")
    plt.clf()


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

def split_patient(patient_ids, df):
  X_train, X_test, y_train, y_test = train_test_split(patient_ids, patient_ids, test_size=0.3)
  df_train = df[df.PERSON_CD.isin(X_train)]
  df_test = df[df.PERSON_CD.isin(X_test)]
  return df_train, df_test

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
            prev = df_patient.loc[index:index + N - 1, :]['label']
            cur = df_patient.loc[index + N: index + N + M - 1, :]['label']

            # prev_states = [get_state(row['DIFF_HRZ'], [0]) for _, row in prev.iterrows()]
            # cur_states = [get_state(row['DIFF_HRZ'], [0]) for _, row in cur.iterrows()]

            key_prev = "".join(map(str, prev))
            key_cur = "".join(map(str, cur))
            try:
                try:
                    transition_dic[key_prev][key_cur] += 1
                except Exception as e:
                    transition_dic[key_prev][key_cur] = 1
            except Exception as e:
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

X = ['N' + str(i[0]) + 'M' + str(i[1]) for i in NMs]  

    

def get_dic(tran,state_num = 7, title =  None):
    dic={} 
   
    dic['#effective'] = np.arange(0,state_num +1,1)
    states = [math.comb(state_num,i) for i in range (0,state_num + 1)]  
    dic['C_prob'] = states/np.sum(states)
    
    for index in tran:
        sums = [get_sum(i) for i in tran[index].keys()]
        set_sums = set(sums)
        dic_random={}
        for i in set_sums:
            dic_random[i] = sums.count(i)
        dic_random = dict(sorted(dic_random.items()))
        for i in dic_random:
            dic_random[i] = dic_random[i]/len(sums)
        if np.sum(np.array(dic['C_prob']) - np.array(list(dic_random.values()))) > 0.001:
             dic['initR_' +index] = list(dic_random.values())
        dic_prob = {}
        for i in tran[index].keys():
            sum_key = get_sum(i)
        
            if sum_key  not in dic_prob:
                dic_prob[sum_key] = tran[index][i]
            else:
                dic_prob[sum_key] += tran[index][i]
        sort_dic = dict(sorted(dic_prob.items()))
        dic['init_' + index] = list(sort_dic.values())
       

    print()
    print("==================================================")
    print(title + ' transition Ms for different initial_states')
    print()
    df = pd.DataFrame(data = dic)
    print(df)
    print("==================================================")
    


from IPython import embed
embed()
