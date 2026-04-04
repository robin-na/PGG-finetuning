#"Deliberating Cost and Impact: Trustworthiness Signals in Punishment and Helping"
#Code for Exp. 1
#Study 1: Personal Cost Checking; Helping Context
#H1.1. H2.1. H7.1 H8.1 H9.1 H14.1 H15.1

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

#Power Analysis
pwr.f2.test(u = 1, f2 = 0.02/(1 - 0.02), sig.level = 0.05, power = 0.95)

#Loading Exp. 1 data (help, cost checking)
E1<- read.csv("helpcostcheckE1.csv", header=T, sep=";")

table(E1$Condition) #min 653 per condition
#654 (A, obs), 657 (A, hid), 653 (B obs), 653 (B hid)

## Descriptives
median(E1$Duration) #410
summary(E1$age, na.rm = T) #18-80, Mean: 39.45
sd(E1$age, na.rm = T) # 12.51
E1$gender <- as.factor(E1$gender)
summary(E1$gender, na.rm = T) # 1207 male, 1381 female, 17 other, 7 prefer not to say
E1$PID<- as.factor(E1$PID)

#creating subsets for Players A and Players B
#condition: 3 = Player A observable, 4 = Player A hidden, 5 = Player B observable, 6 = Player B hidden
PlayerA <- subset(E1, Condition %in% c(3, 4))
PlayerB <- subset(E1, Condition %in% c(5, 6))

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

#remove those who got 1 or more out of 8 wrong
#PlayerA <- subset(PlayerA, comp >= 7)

#remove those who got 1 or more out of 7 wrong
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
#Prediction: Players A are more likely to act uncalculatingly (to not check the cost of helping) 
#in the process observable condition (when it can confer reputation benefits) 
#than in process hidden condition (when it cannot).

#H1.1
m1.1 <- glm(checking ~ conditionA, data = PlayerA, family = "binomial") 
summary(m1.1) #checked (1), did not check (0), observable(1), hidden(0)

confint(m1.1) # CIs for coefficient estimates (for logistic models CIs are based on the profiled log-likelihood function)
exp(cbind(OR = coef(m1.1), confint(m1.1))) #get odds ratios and their CIs 

# % of players who checked the personal cost in the hidden vs observable condition
HiddenA<- subset(PlayerA, conditionA == 0)
ObservableA<- subset(PlayerA, conditionA == 1)
checkPercObs<- sum(ObservableA$checkObs)/nrow(ObservableA)*100 #69% checked in the observable condition
checkPercHid<- sum(HiddenA$checkHid)/nrow(HiddenA)*100 #78% checked in the hidden condition 


#Figure
# Extract predicted probabilities and confidence intervals
pred_results <- predict(m1.1, type = "response", se.fit = TRUE)

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
ggplot_object1<- ggplot(plot_data, aes(x = Condition, y = Percentage, fill = Condition)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  geom_errorbar(aes(ymin = CI_low, ymax = CI_high), position = position_dodge(width = 0.9), width = 0.25) +
  labs(title = "Exp. 1: Help Cost Checking",
       y = "Percentage who check",
       x = "Condition") +
  theme_minimal()

ggplot_object1<- ggplot_object1 + geom_signif(comparisons = list(c("Observable", "Hidden")), map_signif_level = TRUE, annotations = c("***"))


####Is uncalculated help perceived as a signal of trustworthiness?####
#Analysis is restricted to the observable condition because Players B could 
#only condition their trust on Player A checking decisions in this condition

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

#Prediction: Observers send more to helpers who don't check the cost than to helpers who check the cost
#H2.1
m2.1 <- lm(sent ~ checking, data = helpersB) 
summary(m2.1) 
emmeans(m2.1, "checking")

tapply(helpersB$sent, helpersB$checking, mean)
tapply(helpersB$sent, helpersB$checking, sd)

m2.1coeffs_cl <- coeftest(m2.1, vcov = vcovCL, cluster = ~PID) # clustering on participant ID
coi_indices <- which(!startsWith(row.names(m2.1coeffs_cl), 'PID'))
m2.1coeffs_cl[coi_indices,]
m2.1CIs <- coefci(m2.1, parm = coi_indices, vcov = vcovCL, cluster = ~PID) #coefci to calculate CIs


###Is the predicted positive effect of uncalculated behaviour on trust specific to uncalculated help?###
#Prediction: Cost checking will be perceived more negatively when A helped vs did not help
#H8.1
m8.1<- lm(sent ~ helping*checking, data = PlayerB2) 
summary(m8.1) 

emmip(m8.1, helping ~ checking)
m8.1.emm<- emmeans(m8.1, ~ helping * checking)
contrast(m8.1.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

summary(plot8.1 <- lm(sent ~ helping * checking, data = PlayerB2))
interact_plot(plot8.1, pred = helping, modx = checking)
#interact_plot(plot8.1, pred = checking, modx = helping)

m8.1coeffs_cl <- coeftest(m8.1, vcov = vcovCL, cluster = ~PID) # clustering on ID
coi_indices <- which(!startsWith(row.names(m8.1coeffs_cl), 'PID'))
m8.1coeffs_cl[coi_indices,]
m8.1CIs <- coefci(m8.1, parm = coi_indices, vcov = vcovCL, cluster = ~PID) 


#Prediction: Player Bs send more to non-helpers who checked the cost than 
#to non-helpers who did not check the cost
#H7.1
m7.1<- lm(sent ~ checking, data = nonHelpersB)
summary(m7.1) 
emmeans(m7.1, "checking")

tapply(nonHelpersB$sent, nonHelpersB$checking, mean)
tapply(nonHelpersB$sent, nonHelpersB$checking, sd)


m7.1coeffs_cl <- coeftest(m7.1, vcov = vcovCL, cluster = ~PID) # clustering on PID
coi_indices <- which(!startsWith(row.names(m7.1coeffs_cl), 'PID'))
m7.1coeffs_cl[coi_indices,]
m7.1CIs <- coefci(m7.1, parm = coi_indices, vcov = vcovCL, cluster = ~PID) 


####Is uncalculated helping actually a signal of trustworthiness?####
#Prediction: uncalculated helpers are more trustworthy than calculated helpers
#Both conditions are included as we have data on Player A decision processes in both

#subset including helpers only
helpersA<- subset(PlayerA, PlayerA$helping == 1)

#another subset including non-helpers only
nonHelpersA<- subset(PlayerA, PlayerA$helping == 0)

#Prediction: helpers who check the cost will return less than helpers who do not check the cost
#H14.1
m14.1<- lm(return ~ checking, data = helpersA)
summary(m14.1) 
confint(m14.1)
emmeans(m14.1, "checking")

tapply(helpersA$return, helpersA$checking, mean)
tapply(helpersA$return, helpersA$checking, sd)


###Is the positive effect of uncalculated behaviour on trustworthiness specific to uncalculated helping?###
#Prediction: cost checking will affect untrustworthiness more when Player A helped vs did not help
#H9.1
m9.1<- lm(return ~ helping*checking, data = PlayerA) 
summary(m9.1) 

emmip(m9.1, helping ~ checking)
m9.1.emm<- emmeans(m9.1, ~ helping * checking)
contrast(m9.1.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

summary(plot9.1 <- lm(return ~ helping * checking, data = PlayerA))
interact_plot(plot9.1, pred = helping, modx = checking)
#interact_plot(plot9.1, pred = checking, modx = helping)

#Prediction: non-helpers who checked the cost will return more than non-helpers who didn't check the cost
#H15.1
m15.1<- lm(return ~ checking, data = nonHelpersA)
summary(m15.1) 
confint(m15.1)
emmeans(m15.1, "checking")

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

#H14.1
### Bayes factor of full model against null
H14.1_bf<- lmBF(return ~ checking, data = helpersA, rscalefixed = 0.5, posterior = FALSE)

#H15.1
### Bayes factor of full model against null
H15.1_bf<- lmBF(return ~ checking, data = nonHelpersA, rscalefixed = 0.5, posterior = FALSE)

#H9.1
H9.1_bf <- lmBF(return ~ helping*checking, data = PlayerA, rscalefixed = 0.5, posterior = FALSE)
H9.1_bf0 <- lmBF(return ~ helping + checking, data = PlayerA, rscalefixed = 0.5, posterior = FALSE)
#compare full model to main-effects model
H9.1_bf/H9.1_bf0



# Sending decisions using the brms package
#brms: default 4 chains, and warmup default iter/2

## H2.1 ##
get_prior(sent ~ checking + (1|PID), data = helpersB)

priorH2.1 <- c(set_prior("student_t(4, 0, 0.765)", class = "b", coef = "checking"),
              set_prior("student_t(4, 0, 10)", class = "Intercept"),
              set_prior("student_t(4, 0, 10)", class = "sigma"),
              set_prior("student_t(4, 0, 10)", class = "sd"),
              set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
              set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))

priorH2.1_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
               set_prior("student_t(4, 0, 10)", class = "sigma"),
               set_prior("student_t(4, 0, 10)", class = "sd"),
               set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
               set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))
c(get_prior(sent ~ checking + (1|PID), data = helpersB), priorH2.1, replace = TRUE)

#sometimes setting a larger max_treedepth may improve the mixing of the MCMC chain and reduce autocorrelation between samples, which can increase the ESS and improve the reliability of the posterior inference.
full_H2.1 = brm(sent ~ checking + (1|PID), data = helpersB, prior = priorH2.1, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H2.1 = brm(sent ~ 1 + (1|PID), data = helpersB, prior = priorH2.1_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000)  

#compute + store Bayes factor with bridgesampling
H2.1_Full <- bridge_sampler(full_H2.1) 
H2.1_Null <- bridge_sampler(null_H2.1) 
BF10_H2.1<- bayes_factor(H2.1_Full, H2.1_Null)$bf


## H7.1 ##
get_prior(sent ~ checking + (1|PID), data = nonHelpersB)

priorH7.1 <- c(set_prior("student_t(4, 0, 1.87)", class = "b", coef = "checking"),
               set_prior("student_t(4, 0, 10)", class = "Intercept"),
               set_prior("student_t(4, 0, 10)", class = "sigma"),
               set_prior("student_t(4, 0, 10)", class = "sd"),
               set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
               set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))

priorH7.1_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
                 set_prior("student_t(4, 0, 10)", class = "sigma"),
                 set_prior("student_t(4, 0, 10)", class = "sd"),
                 set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
                 set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))
c(get_prior(sent ~ checking + (1|PID), data = nonHelpersB), priorH7.1, replace = TRUE)

full_H7.1 = brm(sent ~ checking + (1|PID), data = nonHelpersB, prior = priorH7.1, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H7.1 = brm(sent ~ 1 + (1|PID), data = nonHelpersB, prior = priorH7.1_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 

#compute + store Bayes factor with bridgesampling
H7.1_Full <- bridge_sampler(full_H7.1) 
H7.1_Null <- bridge_sampler(null_H7.1)
BF10_H7.1<- bayes_factor(H7.1_Full, H7.1_Null)$bf



## H8.1 ##
get_prior(sent ~ helping*checking + (1|PID), data = PlayerB2)

priorH8.1 <- c(set_prior("student_t(4, 0, 10)", class = "b", coef = "checking"),
               set_prior("student_t(4, 0, 10)", class = "b", coef = "helping"),
               set_prior("student_t(4, 0, 7.03)", class = "b", coef = "helping:checking"),
               set_prior("student_t(4, 0, 10)", class = "Intercept"),
               set_prior("student_t(4, 0, 10)", class = "sigma"),
               set_prior("student_t(4, 0, 10)", class = "sd"),
               set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
               set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))

priorH8.1_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
                 set_prior("student_t(4, 0, 10)", class = "b", coef = "checking"),
                 set_prior("student_t(4, 0, 10)", class = "b", coef = "helping"),
                 set_prior("student_t(4, 0, 10)", class = "sigma"),
                 set_prior("student_t(4, 0, 10)", class = "sd"),
                 set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
                 set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))
c(get_prior(sent ~ helping*checking + (1|PID), data = PlayerB2), priorH8.1, replace = TRUE)

full_H8.1 = brm(sent ~ helping*checking + (1|PID), data = PlayerB2, prior = priorH8.1, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H8.1 = brm(sent ~ helping + checking + (1|PID), data = PlayerB2, prior = priorH8.1_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
#compute + store Bayes factor with bridgesampling
H8.1_Full <- bridge_sampler(full_H8.1) 
H8.1_Null <- bridge_sampler(null_H8.1)
BF10_H8.1<- bayes_factor(H8.1_Full, H8.1_Null)$bf



## logistic regression

#H1.1
get_prior(checking ~ conditionA, data = PlayerA)

#bernoulli can't use sigma 
priorH1.1 <- c(set_prior("student_t(4, 0, 0.446)", class = "b", coef = "conditionA1"),
                 set_prior("student_t(4, 0, 10)", class = "Intercept"))                

priorH1.1_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"))

c(get_prior(checking ~ conditionA, data = PlayerA), priorH1.1, replace = TRUE)

full_H1.1 <- brm(checking ~ conditionA, data = PlayerA, family = bernoulli(), prior = priorH1.1, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15, stepsize = 0.01), chains = 4, iter = 20000, warmup = 10000)
null_H1.1 <- brm(checking ~ 1, data = PlayerA, family = bernoulli(), prior = priorH1.1_0, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15, stepsize = 0.01), chains = 4, iter = 20000, warmup = 10000)

#compute + store Bayes factor with bridgesampling
logFull <- bridge_sampler(full_H1.1) 
logNull <- bridge_sampler(null_H1.1) 
BF10_H1.1 <- bayes_factor(logFull, logNull)$bf 


