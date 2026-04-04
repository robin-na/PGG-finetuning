#"Deliberating Cost and Impact: Trustworthiness Signals in Punishment and Helping"
#Code for Exp. 4
#Study 2: Target Impact Checking; Helping Context

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

#Loading Exp. 4 data
E4<- read.csv("helpimpactcheckE4.csv", header=T, sep=";")

table(E4$Condition) #min 653 per condition
#657 (A, obs), 654 (A, hid), 653 (B obs), 653 (B hid)

## Descriptives
median(E4$Duration) # 383s
summary(E4$age, na.rm = T) #18-80, Mean: 37.28
sd(E4$age, na.rm = T) # 11.79
E4$gender <- as.factor(E4$gender)
summary(E4$gender, na.rm = T) # 949 male, 1645 female, 21 other, 2 prefer not to say
E4$PID<- as.factor(E4$PID)

#creating subsets for Players A and Players B
#condition: 3 = Player A observable, 4 = Player A hidden, 5 = Player B observable, 6 = Player B hidden
PlayerA <- subset(E4, Condition %in% c(3, 4))
PlayerB <- subset(E4, Condition %in% c(5, 6))

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
PlayerA$checking<- coalesce(PlayerA$checkObs, PlayerA$checkHid) # 1 = checked the cost, 0 = did not check the cost
PlayerA$helping<- coalesce(PlayerA$calcHelp, PlayerA$uncalcHelp) # 1 = helped, 0 = did not help



####Is uncalculated help used as a signal of trustworthiness? ####
#Prediction: Players A are more likely to act uncalculatingly (to not check) in 
#the process observable condition (when it can confer reputational benefits) 
#than in process hidden condition (when it cannot)
#H5.1
m5.1 <- glm(checking ~ conditionA, data = PlayerA, family = "binomial") 
summary(m5.1) #checked (1), did not check (0), observable(1), hidden(0)

confint(m5.1) # CIs for coefficient estimates (for logistic models CIs are based on the profiled log-likelihood function)
exp(cbind(OR = coef(m5.1), confint(m5.1))) #get odds ratios and their CIs 

# % of players who checked the target impact in the hidden vs observable condition
HiddenA<- subset(PlayerA, conditionA == 0)
ObservableA<- subset(PlayerA, conditionA == 1)
checkPercObs<- sum(ObservableA$checkObs)/nrow(ObservableA)*100 #78% checked in the observable condition
checkPercHid<- sum(HiddenA$checkHid)/nrow(HiddenA)*100 #83% checked in the hidden condition 

#Figure
# Create a data frame for plotting
plot_data <- data.frame(
  Condition = factor(c("Observable", "Hidden"), levels = c("Observable", "Hidden")),
  Percentage = c(checkPercObs, checkPercHid),
  CI_low = c(checkPercObs - qnorm(0.975) * sqrt((checkPercObs * (100 - checkPercObs)) / nrow(ObservableA)),
             checkPercHid - qnorm(0.975) * sqrt((checkPercHid * (100 - checkPercHid)) / nrow(HiddenA))),
  CI_high = c(checkPercObs + qnorm(0.975) * sqrt((checkPercObs * (100 - checkPercObs)) / nrow(ObservableA)),
              checkPercHid + qnorm(0.975) * sqrt((checkPercHid * (100 - checkPercHid)) / nrow(HiddenA)))
)

# Create a bar plot with error bars using ggplot2
ggplot_object4<- ggplot(plot_data, aes(x = Condition, y = Percentage, fill = Condition)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  geom_errorbar(aes(ymin = CI_low, ymax = CI_high), position = position_dodge(width = 0.9), width = 0.25) +
  labs(title = "Exp. 4: Help Impact Checking",
       y = "Percentage who check",
       x = "Condition") +
  theme_minimal()

#ggplot_object4<- ggplot_object4 + geom_signif(comparisons = list(c("Observable", "Hidden")), map_signif_level = TRUE, annotations = c("NS"))

####Is uncalculated help perceived as a signal of trustworthiness?####
#Analysis is restricted to the observable condition because Players B could 
#only condition their trust on Player A looking decisions in this condition

#select only observable condition B responses
PlayerB2 <- subset(PlayerB, conditionB == 1)
PlayerB2<- subset(PlayerB2, select = c(PID, helpUncalc, helpCalc, noUncalc, noCalc))
head(PlayerB2)

PlayerB2<- PlayerB2 %>%
  pivot_longer(-PID) %>%
  mutate(helping = as.numeric(grepl("help", name)),
         checking   = as.numeric(grepl("Calc",  name))) %>%
  select(PID, helping, checking, value)
#helping: 1 yes, 0 no
#checking: 1 yes, 0 no

#rename value to sent
names(PlayerB2)[4]<-paste("sent")

#converting from pence sent to % sent
PlayerB2$sent<- ifelse(PlayerB2$sent >= 1, PlayerB2$sent*100/10, PlayerB2$sent)

#further subset including helpers only
helpersB<- subset(PlayerB2, PlayerB2$helping == 1)

#another subset including non-helpers only
nonHelpersB<- subset(PlayerB2, PlayerB2$helping == 0)


#Prediction: Observers send more to helpers who did not check the impact than to helpers who did
m6.1 <- lm(sent ~ checking, data = helpersB) 
summary(m6.1) 
emmeans(m6.1, "checking")

tapply(helpersB$sent, helpersB$checking, mean)
tapply(helpersB$sent, helpersB$checking, sd)

m6.1coeffs_cl <- coeftest(m6.1, vcov = vcovCL, cluster = ~PID) # clustering on participant ID
coi_indices <- which(!startsWith(row.names(m6.1coeffs_cl), 'PID'))
m6.1coeffs_cl[coi_indices,]
m6.1CIs <- coefci(m6.1, parm = coi_indices, vcov = vcovCL, cluster = ~PID) #coefci to calculate CIs


###Is the predicted positive effect of uncalculated behaviour on trust specific to uncalculated help?###
#Prediction: Impact checking will be perceived more negatively when A helped vs did not help
#H12.1
m12.1<- lm(sent ~ helping*checking, data = PlayerB2) 
summary(m12.1) 

emmip(m12.1, helping ~ checking)
m12.1.emm<- emmeans(m12.1, ~ helping * checking)
contrast(m12.1.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

summary(plot12.1 <- lm(sent ~ helping * checking, data = PlayerB2))
interact_plot(plot12.1, pred = helping, modx = checking)
#interact_plot(plot12.1, pred = checking, modx = helping)

m12.1coeffs_cl <- coeftest(m12.1, vcov = vcovCL, cluster = ~PID) # clustering on ID
coi_indices <- which(!startsWith(row.names(m12.1coeffs_cl), 'PID'))
m12.1coeffs_cl[coi_indices,]
m12.1CIs <- coefci(m12.1, parm = coi_indices, vcov = vcovCL, cluster = ~PID) 


#Prediction: Player Bs send more to non-helpers who impact-check than to non-helpers who don't
#H11.1
m11.1<- lm(sent ~ checking, data = nonHelpersB)
summary(m11.1) 
emmeans(m11.1, "checking")

tapply(nonHelpersB$sent, nonHelpersB$checking, mean)
tapply(nonHelpersB$sent, nonHelpersB$checking, sd)

m11.1coeffs_cl <- coeftest(m11.1, vcov = vcovCL, cluster = ~PID) # clustering on PID
coi_indices <- which(!startsWith(row.names(m11.1coeffs_cl), 'PID'))
m11.1coeffs_cl[coi_indices,]
m11.1CIs <- coefci(m11.1, parm = coi_indices, vcov = vcovCL, cluster = ~PID) 


####Is uncalculated helping actually a signal of trustworthiness?####
#Prediction: uncalculated helpers are more trustworthy than calculated helpers
#Both conditions are included as we have data on Player A decision processes in both

#subset including helpers only
helpersA<- subset(PlayerA, PlayerA$helping == 1)

#another subset including non-helpers only
nonHelpersA<- subset(PlayerA, PlayerA$helping == 0)

#Prediction: helpers who did checked the impact of helping will return less than
#helpers who did not check the impact
#H19.1
m19.1<- lm(return ~ checking, data = helpersA)
summary(m19.1) 
confint(m19.1)
emmeans(m19.1, "checking")

tapply(helpersA$return, helpersA$checking, mean)
tapply(helpersA$return, helpersA$checking, sd)
  

###Is the predicted positive effect of uncalculated behaviour on trustworthiness specific to uncalculated helping?###
#Prediction: impact-checking will affect untrustworthiness more when Player A helps vs does not help
#H13.1
m13.1<- lm(return ~ helping*checking, data = PlayerA) 
summary(m13.1) 
confint(m13.1)

emmip(m13.1, helping ~ checking)
m13.1.emm<- emmeans(m13.1, ~ helping * checking)
contrast(m13.1.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

summary(plot13.1 <- lm(return ~ helping * checking, data = PlayerA))
interact_plot(plot13.1, pred = helping, modx = checking)
#interact_plot(plot13.1, pred = checking, modx = helping)


#Prediction: non-helpers who checked the impact of helping will return more than
#non-helpers who didn't check the impact 
#H20.1
m20.1<- lm(return ~ checking, data = nonHelpersA)
summary(m20.1) 
confint(m20.1)
emmeans(m20.1, "checking")

tapply(nonHelpersA$return, nonHelpersA$checking, mean)
tapply(nonHelpersA$return, nonHelpersA$checking, sd)


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

#H19.1
### Bayes factor of full model against null
H19.1_bf<- lmBF(return ~ checking, data = helpersA, rscalefixed = 0.5, posterior = FALSE)

#H20.1
### Bayes factor of full model against null
H20.1_bf<- lmBF(return ~ checking, data = nonHelpersA, rscalefixed = 0.5, posterior = FALSE)

#H13.1
H13.1_bf <- lmBF(return ~ helping*checking, data = PlayerA, rscalefixed = 0.5, posterior = FALSE)
H13.1_bf0 <- lmBF(return ~ helping + checking, data = PlayerA, rscalefixed = 0.5, posterior = FALSE)
#compare full model to main-effects model
H13.1_bf/H13.1_bf0



# Sending decisions using the brms package
#brms: default 4 chains, and warmup default iter/2

## H6.1 ##
get_prior(sent ~ checking + (1|PID), data = helpersB)

priorH6.1 <- c(set_prior("student_t(4, 0, 0.765)", class = "b", coef = "checking"),
               set_prior("student_t(4, 0, 10)", class = "Intercept"),
               set_prior("student_t(4, 0, 10)", class = "sigma"),
               set_prior("student_t(4, 0, 10)", class = "sd"),
               set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
               set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))

priorH6.1_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
                 set_prior("student_t(4, 0, 10)", class = "sigma"),
                 set_prior("student_t(4, 0, 10)", class = "sd"),
                 set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
                 set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))
c(get_prior(sent ~ checking + (1|PID), data = helpersB), priorH6.1, replace = TRUE)

#sometimes setting a larger max_treedepth may improve the mixing of the MCMC chain and reduce autocorrelation between samples, which can increase the ESS and improve the reliability of the posterior inference.
full_H6.1 = brm(sent ~ checking + (1|PID), data = helpersB, prior = priorH6.1, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H6.1 = brm(sent ~ 1 + (1|PID), data = helpersB, prior = priorH6.1_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 

#compute + store Bayes factor with bridgesampling
H6.1_Full <- bridge_sampler(full_H6.1) 
H6.1_Null <- bridge_sampler(null_H6.1) 
BF10_H6.1<- bayes_factor(H6.1_Full, H6.1_Null)$bf


## H11.1 ##
get_prior(sent ~ checking + (1|PID), data = nonHelpersB)

priorH11.1 <- c(set_prior("student_t(4, 0, 1.87)", class = "b", coef = "checking"),
               set_prior("student_t(4, 0, 10)", class = "Intercept"),
               set_prior("student_t(4, 0, 10)", class = "sigma"),
               set_prior("student_t(4, 0, 10)", class = "sd"),
               set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
               set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))

priorH11.1_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
                 set_prior("student_t(4, 0, 10)", class = "sigma"),
                 set_prior("student_t(4, 0, 10)", class = "sd"),
                 set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
                 set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))
c(get_prior(sent ~ checking + (1|PID), data = nonHelpersB), priorH11.1, replace = TRUE)

full_H11.1 = brm(sent ~ checking + (1|PID), data = nonHelpersB, prior = priorH11.1, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H11.1 = brm(sent ~ 1 + (1|PID), data = nonHelpersB, prior = priorH11.1_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 

#compute + store Bayes factor with bridgesampling
H11.1_Full <- bridge_sampler(full_H11.1) 
H11.1_Null <- bridge_sampler(null_H11.1) 
BF10_H11.1<- bayes_factor(H11.1_Full, H11.1_Null)$bf



## H12.1 ##
get_prior(sent ~ helping*checking + (1|PID), data = PlayerB2)

priorH12.1 <- c(set_prior("student_t(4, 0, 10)", class = "b", coef = "checking"),
               set_prior("student_t(4, 0, 10)", class = "b", coef = "helping"),
               set_prior("student_t(4, 0, 7.03)", class = "b", coef = "helping:checking"),
               set_prior("student_t(4, 0, 10)", class = "Intercept"),
               set_prior("student_t(4, 0, 10)", class = "sigma"),
               set_prior("student_t(4, 0, 10)", class = "sd"),
               set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
               set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))

priorH12.1_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
                 set_prior("student_t(4, 0, 10)", class = "b", coef = "checking"),
                 set_prior("student_t(4, 0, 10)", class = "b", coef = "helping"),
                 set_prior("student_t(4, 0, 10)", class = "sigma"),
                 set_prior("student_t(4, 0, 10)", class = "sd"),
                 set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
                 set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))
c(get_prior(sent ~ helping*checking + (1|PID), data = PlayerB2), priorH12.1, replace = TRUE)

full_H12.1 = brm(sent ~ helping*checking + (1|PID), data = PlayerB2, prior = priorH12.1, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H12.1 = brm(sent ~ helping + checking + (1|PID), data = PlayerB2, prior = priorH12.1_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 

#compute + store Bayes factor with bridgesampling
H12.1_Full <- bridge_sampler(full_H12.1) 
H12.1_Null <- bridge_sampler(null_H12.1) 
BF10_H12.1<- bayes_factor(H12.1_Full, H12.1_Null)$bf



## logistic regression

#H5.1
get_prior(checking ~ conditionA, data = PlayerA)

#bernoulli can't use sigma 
priorH5.1 <- c(set_prior("student_t(4, 0, 0.446)", class = "b", coef = "conditionA1"),
               set_prior("student_t(4, 0, 10)", class = "Intercept"))                

priorH5.1_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"))

c(get_prior(checking ~ conditionA, data = PlayerA), priorH5.1, replace = TRUE)

full_H5.1 <- brm(checking ~ conditionA, data = PlayerA, family = bernoulli(), prior = priorH5.1, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15, stepsize = 0.01), chains = 4, iter = 20000, warmup = 10000)
null_H5.1 <- brm(checking ~ 1, data = PlayerA, family = bernoulli(), prior = priorH5.1_0, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15, stepsize = 0.01), chains = 4, iter = 20000, warmup = 10000)

#compute + store Bayes factor with bridgesampling
H5.1Full <- bridge_sampler(full_H5.1) 
H5.1Null <- bridge_sampler(null_H5.1) 
BF10_H5.1 <- bayes_factor(H5.1Full, H5.1Null)$bf 


