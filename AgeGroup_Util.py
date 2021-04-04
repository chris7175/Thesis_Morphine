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


df = pd.read_csv('/Users/lin/Desktop/thesis_code/data.csv')

#bin the age 
bins = [0,5,10,30,50,70]
# labels=['AgeGroup {} to {}'.format(bins[x-1],bins[x]) for x in range(1, len(bins))]
binned = pd.cut(df['AGE'], bins=bins, labels=[0,1,2,3,4])
df['AgeGroup'] = binned
 

from IPython import embed
embed()
