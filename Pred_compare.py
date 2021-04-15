from __future__ import division

import random
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pandas as pd
from itertools import permutations   
import json
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression



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


test_ids = pd.read_csv('ids_new.csv')
df = pd.read_csv('df_Age.csv')
df['label'] = [get_state(row['DIFF_HRZ'], [0]) for _, row in df.iterrows()]
columns = ['MEDIAN_HR_5_Z_y' ,'dosage_per_kg_cisatracurium_total' ,
    'AGE' , 'dosage_per_kg_vecuronium_total' , 'MORPHINE_RATE_MEAN' ,
    'avgpast_diffhrz' , 'PROCAINAMIDE_RATE_MEAN' , 'dosage_per_kg_milrinone_total' ,
    'opioid_bolus']

from IPython import embed
embed()
