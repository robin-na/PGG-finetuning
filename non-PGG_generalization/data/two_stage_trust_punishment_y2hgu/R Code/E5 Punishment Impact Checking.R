#"Deliberating Cost and Impact: Trustworthiness Signals in Punishment and Helping"
#Code for Exp. 5
#Study 2: Target Impact Checking; Punishing Context

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


#Loading Exp. 5 data (punishment, impact checking)
E5<- read.csv("punimpactcheckE5.csv", header=T, sep=";")

table(E5$Condition) #min 653 per condition
# 653 (A, obs), 653 (A, hid), 653 (B obs), 653 (B hid)

## Descriptives
median(E5$Duration) # 384
summary(E5$age, na.rm = T) # 18-78, M = 38.43
sd(E5$age, na.rm = T) # 12.45
E5$gender <- as.factor(E5$gender)
summary(E5$gender, na.rm = T) # 1002 male, 1581 female, 20 other, 9 prefer not to say
E5$PID<- as.factor(E5$PID)

#creating subsets for Players A and Players B
#condition: 3 = Player A observable, 4 = Player A hidden, 5 = Player B observable, 6 = Player B hidden
PlayerA <- subset(E5, Condition %in% c(3, 4))
PlayerB <- subset(E5, Condition %in% c(5, 6))

PlayerA <- PlayerA[, c(1:16, 30:32)] 
PlayerB<- PlayerB[, c(1:2, 17:32)] 

#Player A comprehension score (1 = correct, 0 = incorrect), up to 8
PlayerA$comp <- rowSums(PlayerA[, c("A1comp1", "A1comp2", "A1comp3", "A1comp4",
                                    "A2comp1", "A2comp2", "A2comp3",
                                    "A2comp4OB", "A2comp4HID")], na.rm = TRUE)
summary(PlayerA$comp) # median: 6/8
table(PlayerA$comp)

#Player B comprehension score (1 = correct, 0 = incorrect), up to 7
PlayerB$comp <- rowSums(PlayerB[, c("G1Bcomp1", "G1Bcomp2", "G1Bcomp3", "G1Bcomp4",
                                    "G2Bcomp1", "G2Bcomp2", "G2Bcomp3")], na.rm = TRUE)
summary(PlayerB$comp) # median: 5/7
table(PlayerB$comp)

#include only those who got none or one wrong
#PlayerA <- subset(PlayerA, comp >= 7) 

#include only those who got none or one wrong
#PlayerB <- subset(PlayerB, comp >= 6) 

#recoding levels into 1 = decision process observable & 0 = decision process hidden
PlayerA$conditionA <- ifelse(PlayerA$Condition == 3, 1, 0)
PlayerA$conditionA<- as.factor(PlayerA$conditionA)
PlayerB$conditionB <- ifelse(PlayerB$Condition == 5, 1, 0)
PlayerB$conditionB<- as.factor(PlayerB$conditionB)

#combining info from two variables; coalesce returns the first non-NA value from the specified columns
PlayerA$checking<- coalesce(PlayerA$checkObs, PlayerA$checkHid) # 1 = checked the impact, 0 = did not check the impact
PlayerA$punishing<- coalesce(PlayerA$calcPun, PlayerA$uncalcPun) # 1 = punished, 0 = did not punish



####Is uncalculated punishment used as a signal of trustworthiness? ####
#Prediction: Player As are more likely to act calculatingly (to check) in 
#the process observable condition (when it can confer reputation benefits) 
#than in process hidden condition (when it cannot)

#H5.2
m5.2 <- glm(checking ~ conditionA, data = PlayerA, family = "binomial") 
summary(m5.2) #checked (1), did not check (0), observable(1), hidden(0)

confint(m5.2) # CIs for coefficient estimates (for logistic models CIs are based on the profiled log-likelihood function)
exp(cbind(OR = coef(m5.2), confint(m5.2))) #get odds ratios and their CIs 

# % of players who checked the personal cost in the hidden vs observable condition
HiddenA<- subset(PlayerA, conditionA == 0)
ObservableA<- subset(PlayerA, conditionA == 1)
checkPercObs<- sum(ObservableA$checkObs)/nrow(ObservableA)*100 # 69% checked in the observable condition
checkPercHid<- sum(HiddenA$checkHid)/nrow(HiddenA)*100 # 72% checked in the hidden condition 

#Figure
#(note: this is for the version where only those with excellent comprehension are included)
#(if trying the same fig for all participants, swap [1] & [2] refs)
# Extract predicted probabilities and confidence intervals
pred_results <- predict(m5.2, type = "response", se.fit = TRUE)

# Calculate percentages for each condition
checkPercObs <- 100 * pred_results$fit[2]  # Percentage for observable condition
checkPercHid <- 100 * pred_results$fit[1]  # Percentage for hidden condition

# Calculate confidence intervals
CI_Low <- 100 * (pred_results$fit - qnorm(0.975) * pred_results$se.fit)
CI_High <- 100 * (pred_results$fit + qnorm(0.975) * pred_results$se.fit)

# Create a data frame for plotting
plot_data <- data.frame(
  Condition = factor(c("Observable", "Hidden"), levels = c("Observable", "Hidden")),
  Percentage = c(checkPercObs, checkPercHid),
  CI_low = c(CI_Low[2], CI_Low[1]),
  CI_high = c(CI_High[2], CI_High[1])
)

# Create a bar plot with error bars using ggplot2
ggplot_object5<- ggplot(plot_data, aes(x = Condition, y = Percentage, fill = Condition)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  geom_errorbar(aes(ymin = CI_low, ymax = CI_high), position = position_dodge(width = 0.9), width = 0.25) +
  labs(title = "Exp. 5: Pun Impact Checking",
       y = "Percentage who check",
       x = "Condition") +
  theme_minimal()

ggplot_object5<- ggplot_object5 + geom_signif(comparisons = list(c("Observable", "Hidden")), map_signif_level = TRUE, annotations = c("*"))


####Is calculated punishment perceived as a signal of trustworthiness?####
#Analysis is restricted to the observable condition because Players B could 
#only condition their trust on Player A checking decisions in this condition

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


#Prediction: Observers will send more to punishers who check the impact than to punishers who do not
#H6.2
m6.2 <- lm(sent ~ checking, data = punishersB) 
summary(m6.2) 
emmeans(m6.2, "checking")

tapply(punishersB$sent, punishersB$checking, mean)
tapply(punishersB$sent, punishersB$checking, sd)

m6.2coeffs_cl <- coeftest(m6.2, vcov = vcovCL, cluster = ~PID) # clustering on ID
coi_indices <- which(!startsWith(row.names(m6.2coeffs_cl), 'PID'))
m6.2coeffs_cl[coi_indices,]
m6.2CIs <- coefci(m6.2, parm = coi_indices, vcov = vcovCL, cluster = ~PID) #coefci to calculate CIs


###Is the predicted positive effect of calculated behaviour on trust specific to calculated punishment?###
#Prediction: Impact checking will be perceived more positively when A punishes vs does not punish
#H12.2
m12.2<- lm(sent ~ punishing*checking, data = PlayerB2)
summary(m12.2) 

emmip(m12.2, punishing ~ checking)
m12.2.emm<- emmeans(m12.2, ~ punishing * checking)
contrast(m12.2.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

summary(plot12.2 <- lm(sent ~ punishing * checking, data = PlayerB2))
interact_plot(plot12.2, pred = punishing, modx = checking)
#interact_plot(plot12.2, pred = checking, modx = punishing)

m12.2coeffs_cl <- coeftest(m12.2, vcov = vcovCL, cluster = ~PID) # clustering on participant ID
coi_indices <- which(!startsWith(row.names(m12.2coeffs_cl), 'PID'))
m12.2coeffs_cl[coi_indices,]
m12.2CIs <- coefci(m12.2, parm = coi_indices, vcov = vcovCL, cluster = ~PID) 


#Prediction: Observers will send more to non-punishers who checked the impact than to those who did not
#11.2
m11.2<- lm(sent ~ checking, data = nonPunishersB) 
summary(m11.2) 
emmeans(m11.2, "checking")

tapply(nonPunishersB$sent, nonPunishersB$checking, mean)
tapply(nonPunishersB$sent, nonPunishersB$checking, sd)


m11.2coeffs_cl <- coeftest(m11.2, vcov = vcovCL, cluster = ~PID) # clustering on PID
coi_indices <- which(!startsWith(row.names(m11.2coeffs_cl), 'PID'))
m11.2coeffs_cl[coi_indices,]
m11.2CIs <- coefci(m11.2, parm = coi_indices, vcov = vcovCL, cluster = ~PID) 


####Is calculated TPP actually a signal of trustworthiness?####
#Prediction: calculated punishers are more trustworthy than uncalculated punishers
#Both conditions are included as we have data on Player A decision processes in both

#subset including punishers only
punishersA<- subset(PlayerA, PlayerA$punishing == 1)

#another subset including non-punishers only
nonPunishersA<- subset(PlayerA, PlayerA$punishing == 0)

#Prediction: Punishers who checked the impact will return more than those who did not 
#H19.2
m19.2<- lm(return ~ checking, data = punishersA) 
summary(m19.2) 
confint(m19.2)
emmeans(m19.2, "checking")

tapply(punishersA$return, punishersA$checking, mean)
tapply(punishersA$return, punishersA$checking, sd)


###Is the predicted positive effect of calculated behaviour on trustworthiness specific to calculated punishment?###
#Prediction: impact checking will affect trustworthiness more when Player A punishes vs does not punish
#H13.2
m13.2<- lm(return ~ punishing*checking, data = PlayerA) 
summary(m13.2)
confint(m13.2)

emmip(m13.2, punishing ~ checking)
m13.2.emm<- emmeans(m13.2, ~ punishing * checking)
contrast(m13.2.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

summary(plot13.2 <- lm(return ~ punishing * checking, data = PlayerA))
interact_plot(plot13.2, pred = punishing, modx = checking)
#interact_plot(plot13.2, pred = checking, modx = punishing)


#Prediction: non-punishers who checked the impact will return more than non-punishers who did not
#H20.2
m20.2<- lm(return ~ checking, data = nonPunishersA) 
summary(m20.2) 
confint(m20.2)
emmeans(m20.2, "checking")

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

#H19.2
### Bayes factor of full model against null
H19.2_bf<- lmBF(return ~ checking, data = punishersA, rscalefixed = 0.5, posterior = FALSE)

#H20.2
### Bayes factor of full model against null
H20.2_bf<- lmBF(return ~ checking, data = nonPunishersA, rscalefixed = 0.5, posterior = FALSE)

#H13.2
H13.2_bf <- lmBF(return ~ punishing*checking, data = PlayerA, rscalefixed = 0.5, posterior = FALSE)
H13.2_bf0 <- lmBF(return ~ punishing + checking, data = PlayerA, rscalefixed = 0.5, posterior = FALSE)
#compare full model to main-effects model
H13.2_bf/H13.2_bf0




# Sending decisions using the brms package
#brms: default 4 chains, and warmup default iter/2

## H6.2 ##
get_prior(sent ~ checking + (1|PID), data = punishersB)

priorH6.2 <- c(set_prior("student_t(4, 0, 0.765)", class = "b", coef = "checking"),
               set_prior("student_t(4, 0, 10)", class = "Intercept"),
               set_prior("student_t(4, 0, 10)", class = "sigma"),
               set_prior("student_t(4, 0, 10)", class = "sd"),
               set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
               set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))

priorH6.2_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
                 set_prior("student_t(4, 0, 10)", class = "sigma"),
                 set_prior("student_t(4, 0, 10)", class = "sd"),
                 set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
                 set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))
c(get_prior(sent ~ checking + (1|PID), data = punishersB), priorH6.2, replace = TRUE)

#sometimes setting a larger max_treedepth may improve the mixing of the MCMC chain and reduce autocorrelation between samples, which can increase the ESS and improve the reliability of the posterior inference.
full_H6.2 = brm(sent ~ checking + (1|PID), data = punishersB, prior = priorH6.2, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H6.2 = brm(sent ~ 1 + (1|PID), data = punishersB, prior = priorH6.2_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 

#compute + store Bayes factor with bridgesampling
H6.2_Full <- bridge_sampler(full_H6.2) 
H6.2_Null <- bridge_sampler(null_H6.2) 
BF10_H6.2<- bayes_factor(H6.2_Full, H6.2_Null)$bf



## H11.2 ##
get_prior(sent ~ checking + (1|PID), data = nonPunishersB)

priorH11.2 <- c(set_prior("student_t(4, 0, 1.87)", class = "b", coef = "checking"),
                set_prior("student_t(4, 0, 10)", class = "Intercept"),
                set_prior("student_t(4, 0, 10)", class = "sigma"),
                set_prior("student_t(4, 0, 10)", class = "sd"),
                set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
                set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))

priorH11.2_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
                  set_prior("student_t(4, 0, 10)", class = "sigma"),
                  set_prior("student_t(4, 0, 10)", class = "sd"),
                  set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
                  set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))
c(get_prior(sent ~ checking + (1|PID), data = nonPunishersB), priorH11.2, replace = TRUE)

full_H11.2 = brm(sent ~ checking + (1|PID), data = nonPunishersB, prior = priorH11.2, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H11.2 = brm(sent ~ 1 + (1|PID), data = nonPunishersB, prior = priorH11.2_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000)

#compute + store Bayes factor with bridgesampling
H11.2_Full <- bridge_sampler(full_H11.2) 
H11.2_Null <- bridge_sampler(null_H11.2) 
BF10_H11.2<- bayes_factor(H11.2_Full, H11.2_Null)$bf


## H12.2 ##
get_prior(sent ~ punishing*checking + (1|PID), data = PlayerB2)

priorH12.2 <- c(set_prior("student_t(4, 0, 10)", class = "b", coef = "checking"),
                set_prior("student_t(4, 0, 10)", class = "b", coef = "punishing"),
                set_prior("student_t(4, 0, 7.03)", class = "b", coef = "punishing:checking"),
                set_prior("student_t(4, 0, 10)", class = "Intercept"),
                set_prior("student_t(4, 0, 10)", class = "sigma"),
                set_prior("student_t(4, 0, 10)", class = "sd"),
                set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
                set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))

priorH12.2_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
                  set_prior("student_t(4, 0, 10)", class = "b", coef = "checking"),
                  set_prior("student_t(4, 0, 10)", class = "b", coef = "punishing"),
                  set_prior("student_t(4, 0, 10)", class = "sigma"),
                  set_prior("student_t(4, 0, 10)", class = "sd"),
                  set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
                  set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))
c(get_prior(sent ~ punishing*checking + (1|PID), data = PlayerB2), priorH12.2, replace = TRUE)

full_H12.2 = brm(sent ~ punishing*checking + (1|PID), data = PlayerB2, prior = priorH12.2, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H12.2 = brm(sent ~ punishing + checking + (1|PID), data = PlayerB2, prior = priorH12.2_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 

#compute + store Bayes factor with bridgesampling
H12.2_Full <- bridge_sampler(full_H12.2) 
H12.2_Null <- bridge_sampler(null_H12.2) 
BF10_H12.2<- bayes_factor(H12.2_Full, H12.2_Null)$bf



## logistic regression

#H5.2
get_prior(checking ~ conditionA, data = PlayerA)

#bernoulli can't use sigma 
priorH5.2 <- c(set_prior("student_t(4, 0, 0.446)", class = "b", coef = "conditionA1"),
               set_prior("student_t(4, 0, 10)", class = "Intercept"))                

priorH5.2_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"))

c(get_prior(checking ~ conditionA, data = PlayerA), priorH5.2, replace = TRUE)

full_H5.2 <- brm(checking ~ conditionA, data = PlayerA, family = bernoulli(), prior = priorH5.2, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15, stepsize = 0.01), chains = 4, iter = 20000, warmup = 10000)
null_H5.2 <- brm(checking ~ 1, data = PlayerA, family = bernoulli(), prior = priorH5.2_0, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15, stepsize = 0.01), chains = 4, iter = 20000, warmup = 10000)

#compute + store Bayes factor with bridgesampling
H5.2Full <- bridge_sampler(full_H5.2)
H5.2Null <- bridge_sampler(null_H5.2)
BF10_H5.2 <- bayes_factor(H5.2Full, H5.2Null)$bf 
