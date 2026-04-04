#"Deliberating Cost and Impact: Trustworthiness Signals in Punishment and Helping"
#Code for Exp. 2
#Study 1: Personal Cost Checking; Punishing Context

#loading packages
library(aod)
library(dplyr)
library(emmeans)
library(ggplot2)
library(ggsignif)
library(gvlma)
library(interactions)
library(lmtest)
library(pscl)
library(pwr)
library(sandwich)
library(tidyr)


#Loading Exp. 2 data (punishment, cost checking)
E2<- read.csv("puncostcheckE2.csv", header=T, sep=";")

table(E2$Condition) #min 653 per condition
#656 (A, obs), 653 (A, hid), 653 (B obs), 653 (B hid)

## Descriptives
median(E2$Duration) # 405
summary(E2$age, na.rm = T) # 18-79, M = 39.16
sd(E2$age, na.rm = T) # 12.36
E2$gender <- as.factor(E2$gender)
summary(E2$gender, na.rm = T) # 1071 male, 1519 female, 18 other, 7 prefer not to say
E2$PID<- as.factor(E2$PID)

#creating subsets for Players A and Players B
#condition: 3 = Player A observable, 4 = Player A hidden, 5 = Player B observable, 6 = Player B hidden
PlayerA <- subset(E2, Condition %in% c(3, 4))
PlayerB <- subset(E2, Condition %in% c(5, 6))

PlayerA <- PlayerA[, c(1:16, 30:32)]
PlayerB<- PlayerB[, c(1:2, 17:32)]

#Player A comprehension score (1 = correct, 0 = incorrect), up to 8
PlayerA$comp <- rowSums(PlayerA[, c("A1comp1", "A1comp2", "A1comp3", "A1comp4",
                                    "A2comp1", "A2comp2", "A2comp3",
                                    "A2comp4OB", "A2comp4HID")], na.rm = TRUE)
summary(PlayerA$comp) # median: 6/8

#Player B comprehension score (1 = correct, 0 = incorrect), up to 7
PlayerB$comp <- rowSums(PlayerB[, c("G1Bcomp1", "G1Bcomp2", "G1Bcomp3", "G1Bcomp4",
                                    "G2Bcomp1", "G2Bcomp2", "G2Bcomp3")], na.rm = TRUE)
summary(PlayerB$comp) # median: 5/7

#remove those who got 1 or less out of 8 wrong
#PlayerA <- subset(PlayerA, comp >= 7) 

#remove those who got 1 or less out of 7 wrong
#PlayerB <- subset(PlayerB, comp >= 6) # 

#recoding levels into 1 = decision process observable & 0 = decision process hidden
PlayerA$conditionA <- ifelse(PlayerA$Condition == 3, 1, 0)
PlayerA$conditionA<- as.factor(PlayerA$conditionA)
PlayerB$conditionB <- ifelse(PlayerB$Condition == 5, 1, 0)
PlayerB$conditionB<- as.factor(PlayerB$conditionB)

#combining info from two variables; coalesce returns the first non-NA value from the specified columns
PlayerA$checking<- coalesce(PlayerA$checkObs, PlayerA$checkHid) # 1 = checked the cost, 0 = did not check the cost
PlayerA$punishing<- coalesce(PlayerA$calcPun, PlayerA$uncalcPun) # 1 = punished, 0 = did not punish


####Is uncalculated punishment used as a signal of trustworthiness? ####
#Prediction: Player As are more likely to act uncalculatingly (to not check) in 
#the process observable condition (when it can confer reputational benefits) 
#than in process hidden condition (when it cannot)

#H1.2a
m1.2a <- glm(checking ~ conditionA, data = PlayerA, family = "binomial") 
summary(m1.2a) #checked (1), did not check (0), observable(1), hidden(0)

confint(m1.2a) # CIs for coefficient estimates (for logistic models CIs are based on the profiled log-likelihood function)
exp(cbind(OR = coef(m1.2a), confint(m1.2a))) #get odds ratios and their CIs 

# % of players who checked the personal cost in the hidden vs observable condition
HiddenA<- subset(PlayerA, conditionA == 0)
ObservableA<- subset(PlayerA, conditionA == 1)
checkPercObs<- sum(ObservableA$checkObs)/nrow(ObservableA)*100 #54% checked in the observable condition
checkPercHid<- sum(HiddenA$checkHid)/nrow(HiddenA)*100 #67% checked in the hidden condition 

#Figure
# Extract predicted probabilities and confidence intervals
pred_results <- predict(m1.2a, type = "response", se.fit = TRUE)

# Calculate percentages for each condition
checkPercObs <- 100 * pred_results$fit[1]  # Percentage for observable condition
checkPercHid <- 100 * pred_results$fit[2]  # Percentage for hidden condition

# Calculate confidence intervals
CI_Low <- 100 * (pred_results$fit - qnorm(0.975) * pred_results$se.fit)
CI_High <- 100 * (pred_results$fit + qnorm(0.975) * pred_results$se.fit)

# Create a data frame for plotting
plot_data <- data.frame(
  Condition = factor(c("Observable", "Hidden"), levels = c("Observable", "Hidden")),
  Percentage = c(checkPercObs, checkPercHid),
  CI_low = c(CI_Low[1], CI_Low[2]),
  CI_high = c(CI_High[1], CI_High[2])
)

# Create a bar plot with error bars using ggplot2
ggplot_object2<- ggplot(plot_data, aes(x = Condition, y = Percentage, fill = Condition)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  geom_errorbar(aes(ymin = CI_low, ymax = CI_high), position = position_dodge(width = 0.9), width = 0.25) +
  labs(title = "Exp. 2: Pun Cost Checking",
       y = "Percentage who check",
       x = "Condition") +
  theme_minimal()

ggplot_object2<- ggplot_object2 + geom_signif(comparisons = list(c("Observable", "Hidden")), map_signif_level = TRUE, annotations = c("***"))

####Is uncalculated punishment perceived as a signal of trustworthiness?####
#Analysis is restricted to the observable condition because Player Bs could 
#only condition their trust on Player A cost checking decisions in this condition

#select only observable condition B responses
PlayerB2 <- subset(PlayerB, conditionB == 1)
PlayerB2<- subset(PlayerB2, select = c(PID, punUncalc, punCalc, noUncalc, noCalc))
head(PlayerB2)

PlayerB2<- PlayerB2 %>%
  pivot_longer(-PID) %>%
  mutate(punishing = as.numeric(grepl("pun", name)),
         checking   = as.numeric(grepl("Calc",  name))) %>%
  select(PID, punishing, checking, value)
#punishing: 1 yes, 0 no
#checking: 1 yes (calculating), 0 no (uncalting)

#rename value to sent
names(PlayerB2)[4]<-paste("sent")

#converting from pence sent to % sent
PlayerB2$sent<- ifelse(PlayerB2$sent >= 1, PlayerB2$sent*100/10, PlayerB2$sent)

#further subset including punishers only
punishersB<- subset(PlayerB2, PlayerB2$punishing == 1)

#another subset including non-punishers only
nonPunishersB<- subset(PlayerB2, PlayerB2$punishing == 0)

#Prediction: Observers will send more to punishers who don't check the cost than punishers who do
#H2.2a
m2.2a <- lm(sent ~ checking, data = punishersB) 
summary(m2.2a) 
emmeans(m2.2a, "checking")

tapply(punishersB$sent, punishersB$checking, mean)
tapply(punishersB$sent, punishersB$checking, sd)

m2.2acoeffs_cl <- coeftest(m2.2a, vcov = vcovCL, cluster = ~PID) # clustering on ID
coi_indices <- which(!startsWith(row.names(m2.2acoeffs_cl), 'PID'))
m2.2acoeffs_cl[coi_indices,]
m2.2aCIs <- coefci(m2.2a, parm = coi_indices, vcov = vcovCL, cluster = ~PID) #coefci to calculate CIs


###Is the predicted positive effect of uncalculated behaviour on trust specific to uncalculated punishment?###
#Prediction: Cost checking will be perceived more negatively when A punishes vs does not punish
#H8.2a
m8.2a<- lm(sent ~ punishing*checking, data = PlayerB2)
summary(m8.2a) 

emmip(m8.2a, punishing ~ checking)
m8.2a.emm<- emmeans(m8.2a, ~ punishing * checking)
contrast(m8.2a.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

summary(plot8.2a <- lm(sent ~ punishing * checking, data = PlayerB2))
interact_plot(plot8.2a, pred = punishing, modx = checking)
#interact_plot(plot8.2a, pred = checking, modx = punishing)

m8.2acoeffs_cl <- coeftest(m8.2a, vcov = vcovCL, cluster = ~PID) # clustering on participant ID
coi_indices <- which(!startsWith(row.names(m8.2acoeffs_cl), 'PID'))
m8.2acoeffs_cl[coi_indices,]
m8.2aCIs <- coefci(m8.2a, parm = coi_indices, vcov = vcovCL, cluster = ~PID) 


#Prediction: Player Bs send less to non-punishers who checked the cost than those who did not
#H7.2a
m7.2a<- lm(sent ~ checking, data = nonPunishersB) 
summary(m7.2a) 
emmeans(m7.2a, "checking")

tapply(nonPunishersB$sent, nonPunishersB$checking, mean)
tapply(nonPunishersB$sent, nonPunishersB$checking, sd)

m7.2acoeffs_cl <- coeftest(m7.2a, vcov = vcovCL, cluster = ~PID) # clustering on PID
coi_indices <- which(!startsWith(row.names(m7.2acoeffs_cl), 'PID'))
m7.2acoeffs_cl[coi_indices,]
m7.2aCIs <- coefci(m7.2a, parm = coi_indices, vcov = vcovCL, cluster = ~PID) 


####Is uncalculated TPP actually a signal of trustworthiness?####
#Prediction: uncalculated punishers are more trustworthy than calculated punishers
#both conditions are included as we have data on Player A decision processes in both

#subset including punishers only
punishersA<- subset(PlayerA, PlayerA$punishing == 1)

#another subset including non-punishers only
nonPunishersA<- subset(PlayerA, PlayerA$punishing == 0)

#Prediction: punishers who do not check the cost will return more than punishers who do
#H14.2a
m14.2a<- lm(return ~ checking, data = punishersA) 
summary(m14.2a) 
emmeans(m14.2a, "checking")

tapply(punishersA$return, punishersA$checking, mean)
tapply(punishersA$return, punishersA$checking, sd)


###Is the predicted positive effect of uncalculated behaviour on trustworthiness specific to uncalculated punishment?###
#Prediction: cost checking will affect untrustworthiness more when Player A punishes vs does not punish
#H9.2a
m9.2a<- lm(return ~ punishing*checking, data = PlayerA) 
summary(m9.2a) 

emmip(m9.2a, punishing ~ checking)
m9.2a.emm<- emmeans(m9.2a, ~ punishing * checking)
contrast(m9.2a.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

summary(plot9.2a <- lm(return ~ punishing * checking, data = PlayerA))
interact_plot(plot9.2a, pred = punishing, modx = checking)
#interact_plot(plot9.2a, pred = checking, modx = punishing)


#Prediction: non-punishers who checked the cost will return less than non-punishers who didn't
#H15.2a
m15.2a<- lm(return ~ checking, data = nonPunishersA) 
summary(m15.2a) 
emmeans(m15.2a, "checking")

tapply(nonPunishersA$return, nonPunishersA$checking, mean)
tapply(nonPunishersA$return, nonPunishersA$checking, sd)


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


# Return decisions using BayesFactor package

#lmBF provides an interface for computing Bayes factors for specific linear 
#models against the intercept-only null; other tests may be obtained by 
#computing two models and dividing their Bayes factors

#H14.2a
### Bayes factor of full model against null
H14.2a_bf<- lmBF(return ~ checking, data = punishersA, rscalefixed = 0.5, posterior = FALSE)

#H15.2a
### Bayes factor of full model against null
H15.2a_bf<- lmBF(return ~ checking, data = nonPunishersA, rscalefixed = 0.5, posterior = FALSE)

#H9.2a
H9.2a_bf <- lmBF(return ~ punishing*checking, data = PlayerA, rscalefixed = 0.5, posterior = FALSE)
H9.2a_bf0 <- lmBF(return ~ punishing + checking, data = PlayerA, rscalefixed = 0.5, posterior = FALSE)
#compare full model to main-effects model
H9.2a_bf/H9.2a_bf0



# Sending decisions using the brms package
#brms: default 4 chains, and warmup default iter/2

## H2.2a ##
get_prior(sent ~ checking + (1|PID), data = punishersB)

priorH2.2a <- c(set_prior("student_t(4, 0, 0.765)", class = "b", coef = "checking"),
               set_prior("student_t(4, 0, 10)", class = "Intercept"),
               set_prior("student_t(4, 0, 10)", class = "sigma"),
               set_prior("student_t(4, 0, 10)", class = "sd"),
               set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
               set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))

priorH2.2a_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
                 set_prior("student_t(4, 0, 10)", class = "sigma"),
                 set_prior("student_t(4, 0, 10)", class = "sd"),
                 set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
                 set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))
c(get_prior(sent ~ checking + (1|PID), data = punishersB), priorH2.2a, replace = TRUE)

#sometimes setting a larger max_treedepth may improve the mixing of the MCMC chain and reduce autocorrelation between samples, which can increase the ESS and improve the reliability of the posterior inference.
full_H2.2a = brm(sent ~ checking + (1|PID), data = punishersB, prior = priorH2.2a, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H2.2a = brm(sent ~ 1 + (1|PID), data = punishersB, prior = priorH2.2a_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 

#compute + store Bayes factor with bridgesampling
H2.2a_Full <- bridge_sampler(full_H2.2a) 
H2.2a_Null <- bridge_sampler(null_H2.2a)
BF10_H2.2a<- bayes_factor(H2.2a_Full, H2.2a_Null)$bf



## H7.2a ##
get_prior(sent ~ checking + (1|PID), data = nonPunishersB)

priorH7.2a <- c(set_prior("student_t(4, 0, 1.87)", class = "b", coef = "checking"),
               set_prior("student_t(4, 0, 10)", class = "Intercept"),
               set_prior("student_t(4, 0, 10)", class = "sigma"),
               set_prior("student_t(4, 0, 10)", class = "sd"),
               set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
               set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))

priorH7.2a_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
                 set_prior("student_t(4, 0, 10)", class = "sigma"),
                 set_prior("student_t(4, 0, 10)", class = "sd"),
                 set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
                 set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))
c(get_prior(sent ~ checking + (1|PID), data = nonPunishersB), priorH7.2a, replace = TRUE)

full_H7.2a = brm(sent ~ checking + (1|PID), data = nonPunishersB, prior = priorH7.2a, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H7.2a = brm(sent ~ 1 + (1|PID), data = nonPunishersB, prior = priorH7.2a_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 

#compute + store Bayes factor with bridgesampling
H7.2a_Full <- bridge_sampler(full_H7.2a) 
H7.2a_Null <- bridge_sampler(null_H7.2a) 
BF10_H7.2a<- bayes_factor(H7.2a_Full, H7.2a_Null)$bf




## H8.2a ##
get_prior(sent ~ punishing*checking + (1|PID), data = PlayerB2)

priorH8.2a <- c(set_prior("student_t(4, 0, 10)", class = "b", coef = "checking"),
               set_prior("student_t(4, 0, 10)", class = "b", coef = "punishing"),
               set_prior("student_t(4, 0, 7.03)", class = "b", coef = "punishing:checking"),
               set_prior("student_t(4, 0, 10)", class = "Intercept"),
               set_prior("student_t(4, 0, 10)", class = "sigma"),
               set_prior("student_t(4, 0, 10)", class = "sd"),
               set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
               set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))

priorH8.2a_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
                 set_prior("student_t(4, 0, 10)", class = "b", coef = "checking"),
                 set_prior("student_t(4, 0, 10)", class = "b", coef = "punishing"),
                 set_prior("student_t(4, 0, 10)", class = "sigma"),
                 set_prior("student_t(4, 0, 10)", class = "sd"),
                 set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
                 set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))
c(get_prior(sent ~ punishing*checking + (1|PID), data = PlayerB2), priorH8.2a, replace = TRUE)

full_H8.2a = brm(sent ~ punishing*checking + (1|PID), data = PlayerB2, prior = priorH8.2a, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H8.2a = brm(sent ~ punishing + checking + (1|PID), data = PlayerB2, prior = priorH8.2a_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 

#compute + store Bayes factor with bridgesampling
H8.2a_Full <- bridge_sampler(full_H8.2a) 
H8.2a_Null <- bridge_sampler(null_H8.2a)
BF10_H8.2a<- bayes_factor(H8.2a_Full, H8.2a_Null)$bf



## logistic regression

#H1.2a
get_prior(checking ~ conditionA, data = PlayerA)

#bernoulli can't use sigma 
priorH1.2a <- c(set_prior("student_t(4, 0, 0.446)", class = "b", coef = "conditionA1"),
               set_prior("student_t(4, 0, 10)", class = "Intercept"))                

priorH1.2a_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"))

c(get_prior(checking ~ conditionA, data = PlayerA), priorH1.2a, replace = TRUE)

full_H1.2a <- brm(checking ~ conditionA, data = PlayerA, family = bernoulli(), prior = priorH1.2a, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15, stepsize = 0.01), chains = 4, iter = 20000, warmup = 10000)
null_H1.2a <- brm(checking ~ 1, data = PlayerA, family = bernoulli(), prior = priorH1.2a_0, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15, stepsize = 0.01), chains = 4, iter = 20000, warmup = 10000)

#compute + store Bayes factor with bridgesampling
H1.2aFull <- bridge_sampler(full_H1.2a) 
H1.2aNull <- bridge_sampler(null_H1.2a) 
BF10_H1.2a <- bayes_factor(H1.2aFull, H1.2aNull)$bf 


