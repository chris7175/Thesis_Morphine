setwd('/Users/lin/Desktop/Thesis_Morphine')
library(tidyverse)
library(lme4)
library(glmmLasso)
install.packages('glmmLasso')

install.packages("mgcv")
library(mgcv)

#read 
df <- read.csv('df_new_param.csv')
test_ids <-read.csv('ids_new.csv')


model_imputed <- df%>% 
  mutate(SBS_pre_clean = ifelse(is.na(SBS_pre_clean), -2, SBS_pre_clean),
         FLACC_pre_clean = ifelse(is.na(FLACC_pre_clean), 0, FLACC_pre_clean),
         PERSON_CD = as.factor(PERSON_CD),
         bin_out = factor(DIFF_HRZ > 0)) %>% 
  select_if(~mean(is.na(.x)) == 0) %>% 
  select(!DT_UTC & !diffHRz_cat)


#test
col = 'X0'
df_train <- model_imputed %>% filter(!(PERSON_CD %in% test_ids[[col]]))
df_test  <- model_imputed %>% filter(PERSON_CD %in% test_ids[[col]])

#PLOTTING
ggplot(data = df_train, aes(x = avg_pasthrz, y = DIFF_HRZ)) +
  geom_point(alpha = 0.1)

#base line
form_base <- formula("bin_out ~ s(MEDIAN_HR_5_Z_y) + s(avg_pasthrz)")
morphine_base <- gam(formula = form_base, family = 'binomial', data = df_train)
summary(morphine_base)

form_2 <- formula("bin_out ~ s(MEDIAN_HR_5_Z_y) + s(avg_pasthrz) + s(DOSE_MG_PER_KG)")
morphine.plus.dose <- gam(formula = form_2, family = 'binomial', data = df_train)
summary(morphine.plus.dose)

#No Age Group
#form_3 <- formula("bin_out ~ s(MEDIAN_HR_5_Z_y) + s(avg_pasthrz) + s(DOSE_MG_PER_KG) + AgeGroup")
#morphine.plus.dose.Age <- gam(formula = form_3, family = 'binomial', data = df_train)
#summary(morphine.plus.dose.Age)
#anova(morphine_base, morphine.plus.dose, test="Chisq")

#prev_effect_rate
form_3 <- formula("bin_out ~  s(MEDIAN_HR_5_Z_y) +
                              s(avg_pasthrz) + 
                              s(DOSE_MG_PER_KG) + 
                              s(prev_effective_rate, k = 4)")
morphine.plus.dose.eff <- gam(formula = form_3, family = 'binomial', data = df_train)
summary(morphine.plus.dose.eff)
anova(morphine.plus.dose, morphine.plus.dose.eff, test="Chisq")

#prev_effect_rate
form_4 <- formula("bin_out ~  s(MEDIAN_HR_5_Z_y) +
                              s(avg_pasthrz) + 
                              s(DOSE_MG_PER_KG) + 
                              s(prev_effective_rate, k = 4) + 
                              s(dose_count)")
morphine.plus.dose.count <- gam(formula = form_4, family = 'binomial', data = df_train)
summary(morphine.plus.dose.count)
anova(morphine.plus.dose.eff, morphine.plus.dose.count, test="Chisq")


form_5 <- formula("bin_out ~  s(MEDIAN_HR_5_Z_y) +
                              s(avg_pasthrz) + 
                              s(DOSE_MG_PER_KG) + 
                              s(prev_effective_rate, k = 4) + 
                              s(dose_count) +
                              s(day_since)")
morphine.plus.day_since <- gam(formula = form_5, family = 'binomial', data = df_train)
summary(morphine.plus.day_since)
anova(morphine.plus.dose.count, morphine.plus.day_since, test="Chisq")

#cumulative_dosage -- 弄need
#form_6 <- formula("bin_out ~  s(MEDIAN_HR_5_Z_y) +
                              #s(avg_pasthrz) + 
                              #s(DOSE_MG_PER_KG) + 
                              #s(prev_effective_rate, k = 4) + 
                              #s(dose_count) +
                              #s(day_since)+
                              #s(cumulative_dosage)")

form_6 <- formula("bin_out ~  s(MEDIAN_HR_5_Z_y) +
                              s(avg_pasthrz) + 
                              s(DOSE_MG_PER_KG) + 
                              s(prev_effective_rate, k = 4) + 
                              s(dose_count) +
                              s(day_since) + 
                              s(AGE)")
morphine.plus.age <- gam(formula = form_6, family = 'binomial', data = df_train)
summary(morphine.plus.age)
anova(morphine.plus.day_since, morphine.plus.age, test="Chisq")


form_7 <- formula("bin_out ~  s(MEDIAN_HR_5_Z_y) +
                              s(avg_pasthrz) + 
                              s(DOSE_MG_PER_KG) + 
                              s(prev_effective_rate, k = 4) + 
                              s(dose_count) +
                              s(day_since) + 
                              s(AGE) +
                              s(DOPAMINE_RATE_MEAN)")

morphine.plus.DOPAMINE_RATE_MEAN <- gam(formula = form_7, family = 'binomial', data = df_train)
summary(morphine.plus.DOPAMINE_RATE_MEAN)
anova(morphine.plus.day_since, morphine.plus.age, test="Chisq")

form_8 <- formula("bin_out ~  s(MEDIAN_HR_5_Z_y) +
                              s(avg_pasthrz) + 
                              s(DOSE_MG_PER_KG) + 
                              s(prev_effective_rate, k = 4) + 
                              s(dose_count) +
                              s(day_since) + 
                              s(AGE) +
                              s(DOPAMINE_RATE_MEAN) +
                              s(EPINEPHRINE_RATE_MEAN)")

morphine.plus.EPINEPHRINE_RATE_MEAN <- gam(formula = form_8, family = 'binomial', data = df_train)
summary(morphine.plus.EPINEPHRINE_RATE_MEAN)
anova(morphine.plus.DOPAMINE_RATE_MEAN, morphine.plus.EPINEPHRINE_RATE_MEAN, test="Chisq")

#form_9 <- formula("bin_out ~  s(MEDIAN_HR_5_Z_y) +
                             # s(avg_pasthrz) + 
                              #s(DOSE_MG_PER_KG) + 
                              #s(prev_effective_rate, k = 4) + 
                              #s(dose_count) +
                              #s(day_since) + 
                              #s(AGE) +
                              #s(DOPAMINE_RATE_MEAN) +
                              #s(EPINEPHRINE_RATE_MEAN) + 
                              #")

#morphine.plus.NOREPINEPHRINE_RATE_MEAN <- gam(formula = form_9, family = 'binomial', data = df_train)
#summary(morphine.plus.NOREPINEPHRINE_RATE_MEAN)




form_10 <- formula("bin_out ~  s(MEDIAN_HR_5_Z_y) +
                              s(avg_pasthrz) + 
                              s(DOSE_MG_PER_KG) + 
                              s(prev_effective_rate, k = 4) + 
                              s(dose_count) +
                              s(day_since) + 
                              s(AGE) +
                              s(DOPAMINE_RATE_MEAN) +
                              s(EPINEPHRINE_RATE_MEAN) +
                              s(CALCIUMGLUCONATE_RATE_MEAN, k = 4)")

morphine.plus.CALCIUMGLUCONATE_RATE_MEAN <- gam(formula = form_10, family = 'binomial', data = df_train)
summary(morphine.plus.CALCIUMGLUCONATE_RATE_MEAN)
anova(morphine.plus.EPINEPHRINE_RATE_MEAN, morphine.plus.CALCIUMGLUCONATE_RATE_MEAN, test="Chisq")


form_11 <- formula("bin_out ~  s(MEDIAN_HR_5_Z_y) +
                              s(avg_pasthrz) + 
                              s(DOSE_MG_PER_KG) + 
                              s(prev_effective_rate, k = 4) + 
                              s(dose_count) +
                              s(day_since) + 
                              s(AGE) +
                              s(DOPAMINE_RATE_MEAN) +
                              s(EPINEPHRINE_RATE_MEAN) +
                              s(CALCIUMGLUCONATE_RATE_MEAN, k = 4) + 
                              s(CISATRACURIUM_RATE_MEAN)")

morphine.plus.CALCIUMGLUCONATE_RATE_MEAN <- gam(formula = form_10, family = 'binomial', data = df_train)
summary(morphine.plus.CALCIUMGLUCONATE_RATE_MEAN)
anova(morphine.plus.EPINEPHRINE_RATE_MEAN, morphine.plus.CALCIUMGLUCONATE_RATE_MEAN, test="Chisq")

png(file="test.png",
    width=11, height=8.5, units="in", res=300)
par(pty='s',mfrow=c(2,2))
plot(morphine.gam, resid=T, rug=F, pch=19, col="blue")
dev.off()


