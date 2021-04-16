setwd('/Users/lin/Desktop/Thesis_Morphine')
library(tidyverse)
library(lme4)
library(glmmLasso)
library(gbm)
library(mgcv)
library(caret)
library(e1071)
library(pROC)

#read csv
df <- read.csv('df_new_param.csv')
test_ids <-read.csv('ids_new.csv')


model_imputed <- df%>% 
  mutate(SBS_pre_clean = ifelse(is.na(SBS_pre_clean), -2, SBS_pre_clean),
         FLACC_pre_clean = ifelse(is.na(FLACC_pre_clean), 0, FLACC_pre_clean),
         PERSON_CD = as.factor(PERSON_CD),
         bin_out = factor(DIFF_HRZ > 0)) %>% 
  select_if(~mean(is.na(.x)) == 0) %>% 
  select(!DT_UTC & !diffHRz_cat)

df_train <- df %>% filter(day_since <= 5)
df_test <- df %>% filter(day_since > 5)  

numeric_columns <- c("MEDIAN_HR_5_Z_y", 
                     "avg_pasthrz",
                     "DOSE_MG_PER_KG", 
                     "prev_effective_rate",
                     "dose_count", 
                     "day_since", 
                     "DOPAMINE_RATE_MEAN", 
                     "EPINEPHRINE_RATE_MEAN", 
                     "CALCIUMGLUCONATE_RATE_MEAN",  
                     "DEXMEDETOMIDINE_RATE_MEAN", 
                     "AGE")
cate_columns <- c("ino_inf",
                  "ino_bolus" ,
                  "sed_inf" ,
                  "sed_bolus" ,
                  "anarrhy_bolus",
                  "anarrhy_inf",
                  "opioid_inf",
                  "opioid_bolus",
                  "vasodi_inf",
                  "vasodi_bolus")

formula_mixed_effect <- paste0("bin_out ~ " ,paste(numeric_columns, collapse = " + ") , " + ", paste(cate_columns, collapse = " + ") , " + ", "(1|PERSON_CD)")


m <- glmer(formula = formula_mixed_effect, data = df_train, family = binomial, control = glmerControl(optimizer = "bobyqa"),
           nAGQ = 10)


