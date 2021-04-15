from __future__ import division
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from itertools import permutations  
from sklearn.metrics import accuracy_score




def get_sum(prediction):
  c = 0
  for i in prediction:
    c += int(i)
  return c

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

        df = pd.DataFrame(data = dic)
        df.to_csv("trans/"+title+".csv")
       

    print()
    print("==================================================")
    print(title + ' transition Ms for different initial_states')
    print()
    print(df)
    print("==================================================")
    


from IPython import embed
embed()
