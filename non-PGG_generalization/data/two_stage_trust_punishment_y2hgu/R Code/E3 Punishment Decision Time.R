#"Deliberating Cost and Impact: Trustworthiness Signals in Punishment and Helping"
#Code for Exp. 3
#Study 1: Decision Time (personal cost); Punishing Context

#loading packages
library(aod)
library(dplyr)
library(emmeans)
library(ggplot2)
library(ggsignif)
library(gvlma)
library(lmtest)
library(olsrr)
library(pscl) 
library(sandwich)
library(tidyr)


#Loading Exp. 3 Player A data (punishment, decision time)
TimeA<- read.csv("puntimeE3a.csv", header=T, sep=";")

#median punishing decision time
summary(TimeA$decisionT) # median 8.44

#Loading Exp. 3 Player B data
TimeB<- read.csv("puntimeE3b.csv", header=T, sep=";")

#recoding levels into 1 = decision process observable & 0 = decision process hidden
TimeA$Condition <- ifelse(TimeA$Condition == 1, 1, 0)
TimeA$Condition<- as.factor(TimeA$Condition)
TimeB$Condition <- ifelse(TimeB$Condition == 1, 1, 0)
TimeB$Condition<- as.factor(TimeB$Condition)

table(TimeA$Condition) #min 653 per condition
#653 (A, obs), 653 (A, hid)
table(TimeB$Condition) #min 653 per condition
#653 (B, obs), 653 (B, hid)

## Descriptives Player A
median(TimeA$Duration) # 374
summary(TimeA$age, na.rm = T) # 18-78, M = 37.44
sd(TimeA$age, na.rm = T) # 12.11
TimeA$gender <- as.factor(TimeA$gender)
summary(TimeA$gender, na.rm = T) # 473 male, 814 female, 19 other

## Descriptives Player B
median(TimeB$Duration) # 394
summary(TimeB$age, na.rm = T) # 18-91, M = 41.96
sd(TimeB$age, na.rm = T) # 13.20
TimeB$gender <- as.factor(TimeB$gender)
summary(TimeB$gender, na.rm = T) # 517 male, 784 female, 3 other, 2 prefer not to say
TimeB$ID<- as.factor(TimeB$ID)

#Player A comprehension score (1 = correct, 0 = incorrect), up to 8
TimeA$comp <- rowSums(TimeA[, c("G1comp1", "G1comp2", "G1comp3", "G1comp4",
                                    "G2comp1", "G2comp2", "G2comp3",
                                    "A2comp4OB", "A2comp4HID")], na.rm = TRUE)
summary(TimeA$comp) # median: 7/8
table(TimeA$comp)

#Player B comprehension score (1 = correct, 0 = incorrect), up to 7
TimeB$comp <- rowSums(TimeB[, c("G1Bcomp1", "G1Bcomp2", "G1Bcomp3", "G1Bcomp4",
                                    "G2Bcomp1", "G2Bcomp2", "G2Bcomp3")], na.rm = TRUE)
summary(TimeB$comp) # median: 5/7
table(TimeB$comp)

#keep only those who got none or one wrong
#TimeA <- subset(TimeA, comp >= 7) 

#keep only those who got none or one wrong
#TimeB <- subset(TimeB, comp >= 6) 


####Is uncalculated punishment used as a signal of trustworthiness? ####
#Prediction: Players A are more likely to act uncalculatingly (decide fast) in
#the process observable condition (when it can confer reputation benefits) 
#than in process hidden condition (when it cannot)

#H1.2b
#TimeA$decisionTlog<- log(TimeA$decisionT)
m1.2b<- lm(decisionT ~ Condition, data = TimeA) # (decision time is Page Submit vs e.g. last click)
summary(m1.2b)
confint(m1.2b, level=0.95) # CIs for model parameters

plot(m1.2b$residuals, pch = 16, col = "red")
ols_plot_resid_qq(m1.2b) 
ols_plot_resid_hist(m1.2b) 
plot(cooks.distance(m1.2b), pch = 16, col = "blue") 

#as its highly skewed natural log transform decisionT
m1.2b_log <- lm(log(decisionT) ~ Condition, data = TimeA) 
summary(m1.2b_log) # exponentiate the coefficient, subtract one from this number, and *100 - gives the % increase in the response for every one-unit increase in the IV
confint(m1.2b_log, level=0.95)
emm_res<- emmeans(m1.2b_log, "Condition")

# Convert emmeans results to a data frame
emm_df <- as.data.frame(emm_res)

# Reorder levels and assign labels for Condition factor
emm_df$Condition <- factor(emm_df$Condition, levels = c("1", "0"), labels = c("Observable", "Hidden"))

# Create a ggplot object with bar plots and error bars
ggplot_object3 <- ggplot(emm_df, aes(x = Condition, y = emmean, fill = Condition)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), position = position_dodge(width = 0.9), width = 0.25) +
  labs(title = "Exp. 3: Pun Cost Decision Time",
       y = "Log-transformed Decision Time",
       x = "Condition") +
  theme_minimal()

ggplot_object3<- ggplot_object3 + geom_signif(comparisons = list(c("Observable", "Hidden")), map_signif_level = TRUE, annotations = c("**"))


####Is uncalculated punishment perceived as a signal of trustworthiness?####
#Prediction: Player Bs send more to punishers who decide fast than to punishers who decide slowly

#Analysis is restricted to observable condition because only here could Player Bs 
#condition their trust on A's decision time; as Players B made decisions based 
#on a median split of Player A decision speed, analyses will reflect this

#select only observable condition B responses 
TimeB2 <- subset(TimeB, Condition == 1)
TimeB2<- subset(TimeB2, select = c(ID, punFast, punSlow, noFast, noSlow))
head(TimeB2)

TimeB2<- TimeB2 %>%
  pivot_longer(-ID) %>%
  mutate(punished = as.numeric(grepl("pun", name)),
         speed   = as.numeric(grepl("Fast",  name))) %>%
  select(ID, punished, speed, value)
#punished: yes - 1, no - 0
#speed: fast (uncalculating) - 1, slow (calculating) - 0 

#rename value to sent
names(TimeB2)[4]<-paste("sent")

#converting from pence sent to % sent
TimeB2$sent<- ifelse(TimeB2$sent >= 1, TimeB2$sent*100/10, TimeB2$sent)

#further subset including punishers only
punishersB<- subset(TimeB2, TimeB2$punished == 1)

#another subset including non-punishers only
nonPunishersB<- subset(TimeB2, TimeB2$punished == 0)


#Prediction: Players B will send more to fast punishers than to slow punishers 
#H2.2b
m2.2b <- lm(sent ~ speed, data = punishersB) #H2b
summary(m2.2b) 
emmeans(m2.2b, "speed")

tapply(punishersB$sent, punishersB$speed, mean)
tapply(punishersB$sent, punishersB$speed, sd)

m2.2bcoeffs_cl <- coeftest(m2.2b, vcov = vcovCL, cluster = ~ID) # clustering on ID
coi_indices <- which(!startsWith(row.names(m2.2bcoeffs_cl), 'ID'))
m2.2bcoeffs_cl[coi_indices,]
m2.2bCIs <- coefci(m2.2b, parm = coi_indices, vcov = vcovCL, cluster = ~ID) #coefci to calculate CIs

confint(m2.2b, level=0.95) # CIs for model parameters without clustering


###Is the predicted positive effect of uncalculated behaviour on trust specific to uncalculated punishment?###
#Prediction:deciding slow will be perceived more negatively when Player A punishes vs doesn't punish
#H8.2b
m8.2b<- lm(sent ~ punished*speed, data = TimeB2) 
summary(m8.2b) 

emmip(m8.2b, punished ~ speed)
m8.2b.emm<- emmeans(m8.2b, ~ punished * speed)
contrast(m8.2b.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

library(interactions) 
summary(plot8.2b <- lm(sent ~ punished * speed, data = TimeB2))
interact_plot(plot8.2b, pred = punished, modx = speed)

m8.2bcoeffs_cl <- coeftest(m8.2b, vcov = vcovCL, cluster = ~ID) # clustering on PID
coi_indices <- which(!startsWith(row.names(m8.2bcoeffs_cl), 'ID'))
m8.2bcoeffs_cl[coi_indices,]
m8.2bCIs <- coefci(m8.2b, parm = coi_indices, vcov = vcovCL, cluster = ~ID) 

confint(m8.2b, level=0.95) # CIs without clustering


#Prediction: Player Bs will send more to fast non-punishes than slow non-punishers 
#H7.2b
m7.2b<- lm(sent ~ speed, data = nonPunishersB) 
summary(m7.2b) 
emmeans(m7.2b, "speed")

tapply(nonPunishersB$sent, nonPunishersB$speed, mean)
tapply(nonPunishersB$sent, nonPunishersB$speed, sd)

m7.2bcoeffs_cl <- coeftest(m7.2b, vcov = vcovCL, cluster = ~ID) # clustering on PID
coi_indices <- which(!startsWith(row.names(m7.2bcoeffs_cl), 'ID'))
m7.2bcoeffs_cl[coi_indices,]
m7.2bCIs <- coefci(m7.2b, parm = coi_indices, vcov = vcovCL, cluster = ~ID) 

confint(m7.2b, level=0.95) # model parameter CIs without clustering


####Is uncalculated TPP actually a signal of trustworthiness?###
#Prediction: uncalculated punishers are more trustworthy than calculated punishers
#Both conditions are included as we have data on Player A decision processes in both

#creating the comprehension time control (sum of time taken on comprehension questions)
TimeA$Timer3<- coalesce(TimeA$Timer3obs_Page.Submit, TimeA$Timer3hid_Page.Submit)
TimeA$comprT <- TimeA$Timer1_Page.Submit + TimeA$Timer2_Page.Submit + TimeA$Timer3

#subset including punishers only
punishersA<- subset(TimeA, TimeA$punish == 1)

#another subset including non-punishers only
nonPunishersA<- subset(TimeA, TimeA$punish == 0)

#Prediction: fast punishers will return more than slow punishers
#H14.2b
m14.2b<- lm(return ~ log(decisionT) + log(comprT), data = punishersA) 
summary(m14.2b) 
# divide the coefficient by 100. This tells us that a 1% increase in the IV increases the DV by (coefficient/100) units. 
confint(m14.2b, level=0.95) # CIs for model parameters

# Scatter plot with regression line
E3return<- ggplot(punishersA, aes(x = log(decisionT), y = return)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "Log-transformed Decision Time", y = "Percentage endowment returned") +
  theme_classic()

# Scatter plot with regression line
E3returnNon<- ggplot(nonPunishersA, aes(x = log(decisionT), y = return)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "Log-transformed Decision Time", y = "Percentage endowment returned") +
  theme_classic()

##Is the predicted positive effect of uncalculated behaviour on trustworthiness specific to uncalculated punishment?###
#Prediction: decision time is a stronger predictor of untrustworthiness when A punished vs did not
#H9.2b
m9.2b<- lm(return ~ punishing*log(decisionT) + log(comprT), data = TimeA) 
summary(m9.2b) 

#shows diff sent between punishing with decisionT constant
m9.2b.emm<- emmeans(m9.2b, ~ log(decisionT)*punishing)

TimeA$punishing<- as.factor(TimeA$punishing)

qplot(x = log(decisionT), y = return, data = TimeA, color = punishing) +
  geom_smooth(method = "lm") 


#Prediction: fast non-punishers will return less than slow non-punishers
#H15.2b
m15.2b<- lm(return ~ log(decisionT) + log(comprT), data = nonPunishersA) 
summary(m15.2b) 


#############################
##### Bayesian Analyses #####
#############################


library(rstan)
library(rstanarm)
library(brms)
library(bridgesampling)
library(BayesFactor)

#if you are having issues, try the below
#remove.packages(c("StanHeaders", "rstan"))
#install.packages("StanHeaders", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
#install.packages("rstan", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))


TimeA$decisionT_log<- log(TimeA$decisionT)
TimeA$comprT_log<- log(TimeA$comprT)
punishersA$decisionT_log<- log(punishersA$decisionT)
punishersA$comprT_log<- log(punishersA$comprT)
nonPunishersA$decisionT_log<- log(nonPunishersA$decisionT)
nonPunishersA$comprT_log<- log(nonPunishersA$comprT)

# Return decisions using BayesFactor package

#lmBF provides an interface for computing Bayes factors for specific linear 
#models against the intercept-only null; other tests may be obtained by 
#computing two models and dividing their Bayes factors

#"Currently, the function [lmBF] does not allow for general linear models, containing both continuous and categorical predcitors, but this support will be added in the future."
#H9.2b (return ~ punishing * decisionT_log + comprT_log, data = TimeA) will therefore be analysed using brms


#H1.2b
H1.2b_bf<- lmBF(decisionT_log ~ Condition, data = TimeA, rscalefixed = 0.5, posterior = FALSE)

#H14.2b
H14.2b_bf<- lmBF(return ~ decisionT_log + comprT_log, data = punishersA, rscalefixed = 0.5, posterior = FALSE)

#H15.2b
H15.2b_bf<- lmBF(return ~ decisionT_log + comprT_log, data = nonPunishersA, rscalefixed = 0.5, posterior = FALSE)


# Models using the brms package
#brms: default 4 chains, and warmup default iter/2


## H2.2b ##
get_prior(sent ~ speed + (1|ID), data = punishersB)

priorH2.2b <- c(set_prior("student_t(4, 0, 8.9507)", class = "b", coef = "speed"),
                set_prior("student_t(4, 0, 10)", class = "Intercept"),
                set_prior("student_t(4, 0, 10)", class = "sigma"),
                set_prior("student_t(4, 0, 10)", class = "sd"),
                set_prior("student_t(4, 0, 10)", class = "sd", group = "ID"),
                set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "ID"))

priorH2.2b_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
                  set_prior("student_t(4, 0, 10)", class = "sigma"),
                  set_prior("student_t(4, 0, 10)", class = "sd"),
                  set_prior("student_t(4, 0, 10)", class = "sd", group = "ID"),
                  set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "ID"))
c(get_prior(sent ~ speed + (1|ID), data = punishersB), priorH2.2b, replace = TRUE)

#sometimes setting a larger max_treedepth may improve the mixing of the MCMC chain and reduce autocorrelation between samples, which can increase the ESS and improve the reliability of the posterior inference.
full_H2.2b = brm(sent ~ speed + (1|ID), data = punishersB, prior = priorH2.2b, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H2.2b = brm(sent ~ 1 + (1|ID), data = punishersB, prior = priorH2.2b_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000)

#compute + store Bayes factor with bridgesampling
H2.2b_Full <- bridge_sampler(full_H2.2b) 
H2.2b_Null <- bridge_sampler(null_H2.2b) 
BF10_H2.2b<- bayes_factor(H2.2b_Full, H2.2b_Null)$bf



## H7.2b ##
get_prior(sent ~ speed + (1|ID), data = nonPunishersB)

priorH7.2b <- c(set_prior("student_t(4, 0, 4.457)", class = "b", coef = "speed"),
                set_prior("student_t(4, 0, 10)", class = "Intercept"),
                set_prior("student_t(4, 0, 10)", class = "sigma"),
                set_prior("student_t(4, 0, 10)", class = "sd"),
                set_prior("student_t(4, 0, 10)", class = "sd", group = "ID"),
                set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "ID"))

priorH7.2b_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
                  set_prior("student_t(4, 0, 10)", class = "sigma"),
                  set_prior("student_t(4, 0, 10)", class = "sd"),
                  set_prior("student_t(4, 0, 10)", class = "sd", group = "ID"),
                  set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "ID"))
c(get_prior(sent ~ speed + (1|ID), data = nonPunishersB), priorH7.2b, replace = TRUE)

full_H7.2b = brm(sent ~ speed + (1|ID), data = nonPunishersB, prior = priorH7.2b, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H7.2b = brm(sent ~ 1 + (1|ID), data = nonPunishersB, prior = priorH7.2b_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 

#compute + store Bayes factor with bridgesampling
H7.2b_Full <- bridge_sampler(full_H7.2b)
H7.2b_Null <- bridge_sampler(null_H7.2b)
BF10_H7.2b<- bayes_factor(H7.2b_Full, H7.2b_Null)$bf



## H8.2b ##
get_prior(sent ~ punished*speed + (1|ID), data = TimeB2)

priorH8.2b <- c(set_prior("student_t(4, 0, 10)", class = "b", coef = "speed"),
                set_prior("student_t(4, 0, 10)", class = "b", coef = "punished"),
                set_prior("student_t(4, 0, 13.41)", class = "b", coef = "punished:speed"),
                set_prior("student_t(4, 0, 10)", class = "Intercept"),
                set_prior("student_t(4, 0, 10)", class = "sigma"),
                set_prior("student_t(4, 0, 10)", class = "sd"),
                set_prior("student_t(4, 0, 10)", class = "sd", group = "ID"),
                set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "ID"))

priorH8.2b_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
                  set_prior("student_t(4, 0, 10)", class = "b", coef = "speed"),
                  set_prior("student_t(4, 0, 10)", class = "b", coef = "punished"),
                  set_prior("student_t(4, 0, 10)", class = "sigma"),
                  set_prior("student_t(4, 0, 10)", class = "sd"),
                  set_prior("student_t(4, 0, 10)", class = "sd", group = "ID"),
                  set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "ID"))
c(get_prior(sent ~ punished*speed + (1|ID), data = TimeB2), priorH8.2b, replace = TRUE)

full_H8.2b = brm(sent ~ punished*speed + (1|ID), data = TimeB2, prior = priorH8.2b, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H8.2b = brm(sent ~ punished + speed + (1|ID), data = TimeB2, prior = priorH8.2b_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) #

#compute + store Bayes factor with bridgesampling
H8.2b_Full <- bridge_sampler(full_H8.2b) 
H8.2b_Null <- bridge_sampler(null_H8.2b) 
BF10_H8.2b<- bayes_factor(H8.2b_Full, H8.2b_Null)$bf


## H9.2b ##
get_prior(return ~ punishing*decisionT_log + comprT_log, data = TimeA)

priorH9.2b <- c(set_prior("student_t(4, 0, 10)", class = "b", coef = "comprT_log"),
                set_prior("student_t(4, 0, 10)", class = "b", coef = "decisionT_log"),
                set_prior("student_t(4, 0, 10)", class = "b", coef = "punishing1"),
                set_prior("student_t(4, 0, 6.995)", class = "b", coef = "punishing1:decisionT_log"),
                set_prior("student_t(4, 0, 10)", class = "Intercept"),
                set_prior("student_t(4, 0, 10)", class = "sigma"))

priorH9.2b_0 <- c(set_prior("student_t(4, 0, 10)", class = "b", coef = "comprT_log"),
                                set_prior("student_t(4, 0, 10)", class = "b", coef = "decisionT_log"),
                                set_prior("student_t(4, 0, 10)", class = "b", coef = "punishing1"),
                                set_prior("student_t(4, 0, 10)", class = "Intercept"),
                                set_prior("student_t(4, 0, 10)", class = "sigma"))
c(get_prior(return ~ punishing*decisionT_log + comprT_log, data = TimeA), priorH9.2b, replace = TRUE)

full_H9.2b = brm(return ~ punishing*decisionT_log + comprT_log, data = TimeA, prior = priorH9.2b, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H9.2b = brm(return ~ punishing + decisionT_log + comprT_log, data = TimeA, prior = priorH9.2b_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 

#compute + store Bayes factor with bridgesampling
H9.2b_Full <- bridge_sampler(full_H9.2b) 
H9.2b_Null <- bridge_sampler(null_H9.2b) 
BF10_H9.2b<- bayes_factor(H9.2b_Full, H9.2b_Null)$bf

