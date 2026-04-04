#Hypotheses across studies (E1 & E2; E2 & E3)
#H3, H4, H10, H16, H17, H18 (as well as two additional analyses)

#loading packages
library(aod)
library(dplyr)
library(emmeans)
library(ggplot2)
library(gvlma)
library(interactions)
library(lmtest)
library(pscl)
library(pwr)
library(sandwich)
library(tidyr)

###############################
#### Preparing Exp. 1 Data ####
###############################

#loading Exp. 1 data
E1<- read.csv("helpcostcheckE1.csv", header=T, sep=";")

#creating subsets for Players A and Players B
#condition: 3 = Player A observable, 4 = Player A hidden, 5 = Player B observable, 6 = Player B hidden
PlayerAe1 <- subset(E1, Condition %in% c(3, 4))
PlayerBe1 <- subset(E1, Condition %in% c(5, 6))

PlayerAe1 <- PlayerAe1[, c(1:16, 30:32)]
PlayerBe1<- PlayerBe1[, c(1:2, 17:32)] 

#Player A comprehension score (1 = correct, 0 = incorrect), up to 8
PlayerAe1$comp <- rowSums(PlayerAe1[, c("A1comp1", "A1comp2", "A1comp3", "A1comp4",
                                        "A2comp1", "A2comp2", "A2comp3",
                                        "A2comp4OB", "A2comp4HID")], na.rm = TRUE)

#Player B comprehension score (1 = correct, 0 = incorrect), up to 7
PlayerBe1$comp <- rowSums(PlayerBe1[, c("G1Bcomp1", "G1Bcomp2", "G1Bcomp3", "G1Bcomp4",
                                        "G2Bcomp1", "G2Bcomp2", "G2Bcomp3")], na.rm = TRUE)

# including only those who got none or one wrong
#PlayerAe1 <- subset(PlayerAe1, comp >= 7) 

# including only those who got none or one wrong
#PlayerBe1 <- subset(PlayerBe1, comp >= 6) 

#recoding levels into 1 = decision process observable & 0 = decision process hidden
PlayerAe1$conditionA <- ifelse(PlayerAe1$Condition == 3, 1, 0)
PlayerAe1$conditionA<- as.factor(PlayerAe1$conditionA)
PlayerBe1$conditionB <- ifelse(PlayerBe1$Condition == 5, 1, 0)
PlayerBe1$conditionB<- as.factor(PlayerBe1$conditionB)

#combining info from two variables; coalesce returns the first non-NA value from the specified columns
PlayerAe1$calc<- coalesce(PlayerAe1$checkObs, PlayerAe1$checkHid) # 1 = calculated, 0 = uncalculated
PlayerAe1$helping<- coalesce(PlayerAe1$calcHelp, PlayerAe1$uncalcHelp) # 1 = punished, 0 = did not punish

#subset including punishers only
helpersAe1<- subset(PlayerAe1, PlayerAe1$helping == 1)
#another subset including non-punishers only
nonHelpersAe1<- subset(PlayerAe1, PlayerAe1$helping == 0)


#select only observable condition B responses
PlayerB2e1 <- subset(PlayerBe1, conditionB == 1)
PlayerB2e1<- subset(PlayerB2e1, select = c(PID, helpUncalc, helpCalc, noUncalc, noCalc))

PlayerB2e1<- PlayerB2e1 %>%
  pivot_longer(-PID) %>%
  mutate(helping = as.numeric(grepl("help", name)),
         calc   = as.numeric(grepl("Calc",  name))) %>%
  select(PID, helping, calc, value)
#helping: 1 yes, 0 no
#calc: 1 calculated, 0 uncalculated

#rename value to sent
names(PlayerB2e1)[4]<-paste("sent")

#converting from pence sent to % sent
PlayerB2e1$sent<- ifelse(PlayerB2e1$sent >= 1, PlayerB2e1$sent*100/10, PlayerB2e1$sent)

#further subset including punishers only
helpersBe1<- subset(PlayerB2e1, PlayerB2e1$helping == 1)

#another subset including non-punishers only
nonHelpersBe1<- subset(PlayerB2e1, PlayerB2e1$helping == 0)

#adding experiment as a variable
helpersAe1$experiment <- "1" 
helpersBe1$experiment <- "1"
nonHelpersAe1$experiment <- "1" 
nonHelpersBe1$experiment <- "1" 

#as factor
helpersAe1$experiment <- as.factor(helpersAe1$experiment)
helpersBe1$experiment <- as.factor(helpersBe1$experiment)
nonHelpersAe1$experiment <- as.factor(nonHelpersAe1$experiment)
nonHelpersBe1$experiment <- as.factor(nonHelpersBe1$experiment)

#selecting relevant columns
helpersAe1<- helpersAe1 %>% select(PID, experiment, calc, return)
nonHelpersAe1<- nonHelpersAe1 %>% select(PID, experiment, calc, return)

helpersBe1<- helpersBe1 %>% select(PID, experiment, calc, sent)
nonHelpersBe1<- nonHelpersBe1 %>% select(PID, experiment, calc, sent)

#as factor
helpersAe1$calc<- as.factor(helpersAe1$calc)
nonHelpersAe1$calc<- as.factor(nonHelpersAe1$calc)
helpersBe1$calc<- as.factor(helpersBe1$calc)
nonHelpersBe1$calc<- as.factor(nonHelpersBe1$calc)


###########################
#### Preparing E2 Data ####
###########################

#loading Exp. 2 data
E2<- read.csv("puncostcheckE2.csv", header=T, sep=";")

#giving the participants different IDs across studies
#chose E2 as it will be compared both with E1 and E3 data
E2$PID<- E2$PID + 5000

#creating subsets for Players A and Players B
#condition: 3 = Player A observable, 4 = Player A hidden, 5 = Player B observable, 6 = Player B hidden
PlayerAe2 <- subset(E2, Condition %in% c(3, 4))
PlayerBe2 <- subset(E2, Condition %in% c(5, 6))

PlayerAe2 <- PlayerAe2[, c(1:16, 30:32)] 
PlayerBe2<- PlayerBe2[, c(1:2, 17:32)] 

#Player A comprehension score (1 = correct, 0 = incorrect), up to 8
PlayerAe2$comp <- rowSums(PlayerAe2[, c("A1comp1", "A1comp2", "A1comp3", "A1comp4",
                                    "A2comp1", "A2comp2", "A2comp3",
                                    "A2comp4OB", "A2comp4HID")], na.rm = TRUE)

#Player B comprehension score (1 = correct, 0 = incorrect), up to 7
PlayerBe2$comp <- rowSums(PlayerBe2[, c("G1Bcomp1", "G1Bcomp2", "G1Bcomp3", "G1Bcomp4",
                                    "G2Bcomp1", "G2Bcomp2", "G2Bcomp3")], na.rm = TRUE)

# including only those who got none or one wrong
#PlayerAe2 <- subset(PlayerAe2, comp >= 7)

#remove those who got 2 or more out of 7 wrong
#PlayerBe2 <- subset(PlayerBe2, comp >= 6) 

#recoding levels into 1 = decision process observable & 0 = decision process hidden
PlayerAe2$conditionA <- ifelse(PlayerAe2$Condition == 3, 1, 0)
PlayerAe2$conditionA<- as.factor(PlayerAe2$conditionA)
PlayerBe2$conditionB <- ifelse(PlayerBe2$Condition == 5, 1, 0)
PlayerBe2$conditionB<- as.factor(PlayerBe2$conditionB)

#combining info from two variables; coalesce returns the first non-NA value from the specified columns
PlayerAe2$calc<- coalesce(PlayerAe2$checkObs, PlayerAe2$checkHid) # 1 = calculated, 0 = uncalculated
PlayerAe2$punishing<- coalesce(PlayerAe2$calcPun, PlayerAe2$uncalcPun) # 1 = punished, 0 = did not punish

#subset including punishers only
punishersAe2<- subset(PlayerAe2, PlayerAe2$punishing == 1)
#another subset including non-punishers only
nonPunishersAe2<- subset(PlayerAe2, PlayerAe2$punishing == 0)


#select only observable condition B responses
PlayerB2e2 <- subset(PlayerBe2, conditionB == 1)
PlayerB2e2<- subset(PlayerB2e2, select = c(PID, punUncalc, punCalc, noUncalc, noCalc))

PlayerB2e2<- PlayerB2e2 %>%
  pivot_longer(-PID) %>%
  mutate(punishing = as.numeric(grepl("pun", name)),
         calc   = as.numeric(grepl("Calc",  name))) %>%
  select(PID, punishing, calc, value)
#punishing: 1 yes, 0 no
#calc: 1 calculated, 0 uncalculated

#rename value to sent
names(PlayerB2e2)[4]<-paste("sent")

#converting from pence sent to % sent
PlayerB2e2$sent<- ifelse(PlayerB2e2$sent >= 1, PlayerB2e2$sent*100/10, PlayerB2e2$sent)

#further subset including punishers only
punishersBe2<- subset(PlayerB2e2, PlayerB2e2$punishing == 1)

#another subset including non-punishers only
nonPunishersBe2<- subset(PlayerB2e2, PlayerB2e2$punishing == 0)


#adding experiment as a variable (to compare with Exp. 1)
punishersAe2$experiment <- "0" 
punishersBe2$experiment <- "0" 
nonPunishersAe2$experiment <- "0" 
nonPunishersBe2$experiment <- "0" 
#^^ to compare with Exp. 1

#adding experiment as a variable (to compare with Exp. 3)
punishersAe2$experiment <- "1" 
punishersBe2$experiment <- "1" 
nonPunishersAe2$experiment <- "1" 
nonPunishersBe2$experiment <- "1" 
#^^ to compare with Exp. 3

#as factor
punishersAe2$experiment <- as.factor(punishersAe2$experiment)
punishersBe2$experiment <- as.factor(punishersBe2$experiment)
nonPunishersAe2$experiment <- as.factor(nonPunishersAe2$experiment)
nonPunishersBe2$experiment <- as.factor(nonPunishersBe2$experiment)

#selecting relevant columns
punishersAe2<- punishersAe2 %>% select(PID, experiment, calc, return)
nonPunishersAe2<- nonPunishersAe2 %>% select(PID, experiment, calc, return)

punishersBe2<- punishersBe2 %>% select(PID, experiment, calc, sent)
nonPunishersBe2<- nonPunishersBe2 %>% select(PID, experiment, calc, sent)

#as factor
punishersAe2$calc<- as.factor(punishersAe2$calc)
nonPunishersAe2$calc<- as.factor(nonPunishersAe2$calc)
punishersBe2$calc<- as.factor(punishersBe2$calc)
nonPunishersBe2$calc<- as.factor(nonPunishersBe2$calc)


###########################
#### Preparing E3 Data ####
###########################

#Loading Exp. 3 Player A data (punishment, decision time)
TimeA<- read.csv("puntimeE3a.csv", header=T, sep=";")

#Loading Exp. 3 Player B data
TimeB<- read.csv("puntimeE3b.csv", header=T, sep=";")

#renaming participant ID (ID) to PID, to match the other study
TimeA<- rename(TimeA, PID = ID)
TimeB<- rename(TimeB, PID = ID)

#median split on decision speed; 0 = fast/uncalculated, 1 = slow/calculated
median(TimeA$decisionT) #8.441
TimeA$calc <- ifelse(TimeA$decisionT < 8.441, 0, 1) #there are no values that are exactly 8.441

#recoding level: making 1 = decision process observable & 0 = decision process hidden
TimeA$Condition <- ifelse(TimeA$Condition == 1, 1, 0)
TimeA$Condition<- as.factor(TimeA$Condition)
TimeB$Condition <- ifelse(TimeB$Condition == 1, 1, 0)
TimeB$Condition<- as.factor(TimeB$Condition)

#Player A comprehension score (1 = correct, 0 = incorrect), up to 8
TimeA$comp <- rowSums(TimeA[, c("G1comp1", "G1comp2", "G1comp3", "G1comp4",
                                "G2comp1", "G2comp2", "G2comp3",
                                "A2comp4OB", "A2comp4HID")], na.rm = TRUE)

#Player B comprehension score (1 = correct, 0 = incorrect), up to 7
TimeB$comp <- rowSums(TimeB[, c("G1Bcomp1", "G1Bcomp2", "G1Bcomp3", "G1Bcomp4",
                                "G2Bcomp1", "G2Bcomp2", "G2Bcomp3")], na.rm = TRUE)

# including only those who got none or one wrong
#TimeA <- subset(TimeA, comp >= 7) 

# including only those who got none or one wrong
#TimeB <- subset(TimeB, comp >= 6) 

#select only observable condition B responses 
TimeB2 <- subset(TimeB, Condition == 1)
TimeB2<- subset(TimeB2, select = c(PID, punFast, punSlow, noFast, noSlow))
head(TimeB2)

TimeB2<- TimeB2 %>%
  pivot_longer(-PID) %>%
  mutate(punished = as.numeric(grepl("pun", name)),
         speed   = as.numeric(grepl("Fast",  name))) %>%
  select(PID, punished, speed, value)
#punished: yes - 1, no - 0
#speed: fast (uncalculating) - 1, slow (calculating) - 0 

#rename value to sent
names(TimeB2)[4]<-paste("sent")

#converting from pence sent to % sent
TimeB2$sent<- ifelse(TimeB2$sent >= 1, TimeB2$sent*100/10, TimeB2$sent)

#further subset including punishers only
punishersBe3<- subset(TimeB2, TimeB2$punished == 1)

#another subset including non-punishers only
nonPunishersBe3<- subset(TimeB2, TimeB2$punished == 0)

#subset including punishers only
punishersAe3<- subset(TimeA, TimeA$punish == 1)

#another subset including non-punishers only
nonPunishersAe3<- subset(TimeA, TimeA$punish == 0)

#adding experiment as a variable
punishersAe3$experiment <- "0" 
punishersBe3$experiment <- "0" 
nonPunishersAe3$experiment <- "0" 
nonPunishersBe3$experiment <- "0" 

#as factor
punishersAe3$experiment <- as.factor(punishersAe3$experiment)
punishersBe3$experiment <- as.factor(punishersBe3$experiment)
nonPunishersAe3$experiment <- as.factor(nonPunishersAe3$experiment)
nonPunishersBe3$experiment <- as.factor(nonPunishersBe3$experiment)

#recode to match other studies: now fast (prev. 1) is uncalculating (0), 
#and slow (prev. 0) is calculating (1)
punishersBe3<- punishersBe3 %>% mutate(
  calc = recode(speed,
                "1" = "0",
                "0" = "1"
  ))

#again recode so that fast (1) is uncalculated (0) & slow(0) is calculated (1)
nonPunishersBe3<- nonPunishersBe3 %>% mutate(
  calc = recode(speed,
                "1" = "0",
                "0" = "1"
  ))


#selecting relevant columns
punishersAe3<- punishersAe3 %>% select(PID, experiment, calc, return)
nonPunishersAe3<- nonPunishersAe3 %>% select(PID, experiment, calc, return)

punishersBe3<- punishersBe3 %>% select(PID, experiment, calc, sent)
nonPunishersBe3<- nonPunishersBe3 %>% select(PID, experiment, calc, sent)

#as factor
punishersAe3$calc<- as.factor(punishersAe3$calc)
nonPunishersAe3$calc<- as.factor(nonPunishersAe3$calc)
punishersBe3$calc<- as.factor(punishersBe3$calc)
nonPunishersBe3$calc<- as.factor(nonPunishersBe3$calc)

#########################
#### Merging E1 & E2 ####
#########################

ActorsA<- full_join(punishersAe2, helpersAe1)
ActorsB<- full_join(punishersBe2, helpersBe1)
NonActorsA<- full_join(nonPunishersAe2, nonHelpersAe1)  
NonActorsB<- full_join(nonPunishersBe2, nonHelpersBe1)

#########################
#### Merging E2 & E3 ####
#########################

PUNISHERSa<- full_join(punishersAe3, punishersAe2)
PUNISHERSb<- full_join(punishersBe3, punishersBe2)
NONPUNISHERSa<- full_join(nonPunishersAe3, nonPunishersAe2)  
NONPUNISHERSb<- full_join(nonPunishersBe3, nonPunishersBe2)

###################
#### Analyses ####
##################

#####
#H3
#####

#Prediction: Observers will send less to cost-checking punishers than to 
#slow punishers 
m3<- lm(sent ~ experiment*calc, data = PUNISHERSb) #Players B in the observable condition of E2 & E3
summary(m3) #1 = E2 (cost checking), 0 = E3 (decision time); 1 = calculated, 0 = uncalculated

emmip(m3, experiment ~ calc)
m3.emm<- emmeans(m3, ~ experiment * calc)
contrast(m3.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

cat_plot(m3, pred = experiment, modx = calc, geom = "line", point.shape = TRUE, vary.lty = TRUE)
#below is calc & experiment flipped and alternative way of plotting
#cat_plot(mIDK1, pred = calc, modx = experiment, plot.points = TRUE)

m3coeffs_cl <- coeftest(m3, vcov = vcovCL, cluster = ~PID) # clustering on PID idk if necessary
coi_indices <- which(!startsWith(row.names(m3coeffs_cl), 'PID'))
m3coeffs_cl[coi_indices,]
m3CIs <- coefci(m3, parm = coi_indices, vcov = vcovCL, cluster = ~PID) 


#####
#H10
#####

#Prediction: Observers will send less to cost-checking non-punishers than to slow non-punishers 
m10<- lm(sent ~ experiment*calc, data = NONPUNISHERSb) #Players B in the observable condition of E2 & E3
summary(m10) #1 = E2 (cost checking), 0 = E3 (decision time); 1 = calculated, 0 = uncalculated

emmip(m10, experiment ~ calc)
m10.emm<- emmeans(m10, ~ experiment * calc)
contrast(m10.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

cat_plot(m10, pred = experiment, modx = calc, geom = "line", point.shape = TRUE, vary.lty = TRUE)

m10coeffs_cl <- coeftest(m10, vcov = vcovCL, cluster = ~PID) # clustering on ID idk if necessary
coi_indices <- which(!startsWith(row.names(m10coeffs_cl), 'PID'))
m10coeffs_cl[coi_indices,]
m10CIs <- coefci(m10, parm = coi_indices, vcov = vcovCL, cluster = ~PID) 



#####
#H16
#####

#Prediction: Cost-checking punishers will return less than slow punishers 
m16<- lm(return ~ experiment*calc, data = PUNISHERSa) #Punishers in the observable condition of E2 & E3
summary(m16) #1 = E2 (cost checking), 0 = E3 (decision time); 1 = calculated, 0 = uncalculated
confint(m16)

emmip(m16, experiment ~ calc)
m16.emm<- emmeans(m16, ~ experiment * calc)
contrast(m16.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

cat_plot(m16, pred = experiment, modx = calc, geom = "line", point.shape = TRUE, vary.lty = TRUE)



#####
#H17
#####

#Prediction: Cost-checking non-punishers will return less than slow non-punishers 
#H17
m17<- lm(return ~ experiment*calc, data = NONPUNISHERSa) #Non-punishers in the observable condition of E2 & E3
summary(m17) #1 = E2 (cost checking), 0 = E3 (decision time); 1 = calculated, 0 = uncalculated
confint(m17)

emmip(m17, experiment ~ calc)
m17.emm<- emmeans(m17, ~ experiment * calc)
contrast(m17.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

cat_plot(m17, pred = experiment, modx = calc, geom = "line", point.shape = TRUE, vary.lty = TRUE)



####
#H4. sent ~ behaviour * calc (B obs in E1 & E2)	
####

#Prediction: Observers will send less to cost-checking HELPERS than to cost-checking PUNISHERS
m4<- lm(sent ~ experiment*calc, data = ActorsB) #Players B in the observable condition of E1 & E2 (restricted to decisions around helpers/punishers, not non-helpers/punishers)
summary(m4) #1 = helping, 0 = punishing; 1 = calculated, 0 = uncalculated

tapply(ActorsB$sent, list(ActorsB$calc, ActorsB$experiment), mean)
tapply(ActorsB$sent, list(ActorsB$calc, ActorsB$experiment), sd)

emmip(m4, experiment ~ calc)
m4.emm<- emmeans(m4, ~ experiment * calc)
contrast(m4.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

cat_plot(m4, pred = experiment, modx = calc, geom = "line", point.shape = TRUE, vary.lty = TRUE)

m4coeffs_cl <- coeftest(m4, vcov = vcovCL, cluster = ~PID) # clustering on PID idk if necessary
coi_indices <- which(!startsWith(row.names(m4coeffs_cl), 'PID'))
m4coeffs_cl[coi_indices,]
m4CIs <- coefci(m4, parm = coi_indices, vcov = vcovCL, cluster = ~PID) 


#####
#H18. return ~ behaviour * calc  (Helpers/punishers in E1 & E2)
#####

#Prediction: Cost-checking HELPERS will return less than cost-checking PUNISHERS
m18<- lm(return ~ experiment*calc, data = ActorsA) #helpers/punishers in the observable condition of E1 & E2
summary(m18) #1 = helping, 0 = punishing; 1 = calculated, 0 = uncalculated
confint(m18)

emmip(m18, experiment ~ calc)
m18.emm<- emmeans(m18, ~ experiment * calc)
contrast(m18.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

cat_plot(m18, pred = experiment, modx = calc, geom = "line", point.shape = TRUE, vary.lty = TRUE)


##################
#### Additional ###
###################

####
#H4-non / add.1: sent ~ behaviour * calc (B obs in E1 & E2)	
####

#Prediction: Observers will send less to cost-checking HELPERS than to cost-checking PUNISHERS
add1<- lm(sent ~ experiment*calc, data = NonActorsB) #Players B in the observable condition of E1 & E2 (restricted to decisions around helpers/punishers, not non-helpers/punishers)
summary(add1) #1 = helping, 0 = punishing; 1 = calculated, 0 = uncalculated

tapply(ActorsB$sent, list(NonActorsB$calc, NonActorsB$experiment), mean)
tapply(ActorsB$sent, list(NonActorsB$calc, NonActorsB$experiment), sd)

emmip(add1, experiment ~ calc)
add1.emm<- emmeans(add1, ~ experiment * calc)
contrast(add1.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

cat_plot(add1, pred = experiment, modx = calc, geom = "line", point.shape = TRUE, vary.lty = TRUE)

add1coeffs_cl <- coeftest(add1, vcov = vcovCL, cluster = ~PID) # clustering on PID idk if necessary
coi_indices <- which(!startsWith(row.names(add1coeffs_cl), 'PID'))
add1coeffs_cl[coi_indices,]
add1CIs <- coefci(add1, parm = coi_indices, vcov = vcovCL, cluster = ~PID) 


#####
#H18-non / add.2: return ~ behaviour * calc  (Helpers/punishers in E1 & E2)
#####

#Prediction: Cost-checking HELPERS will return less than cost-checking PUNISHERS
add2<- lm(return ~ experiment*calc, data = NonActorsA) #helpers/punishers in the observable condition of E1 & E2
summary(add2) #1 = helping, 0 = punishing; 1 = calculated, 0 = uncalculated
confint(add2)

tapply(NonActorsA$return, list(NonActorsA$calc, NonActorsA$experiment), mean)
tapply(NonActorsA$return, list(NonActorsA$calc, NonActorsA$experiment), sd)


emmip(add2, experiment ~ calc)
add2.emm<- emmeans(add2, ~ experiment * calc)
contrast(add2.emm, "consec", simple = "each", combine = TRUE, adjust = "mvt")

cat_plot(add2, pred = experiment, modx = calc, geom = "line", point.shape = TRUE, vary.lty = TRUE)



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



###
## comparison between E2 & E3 ##
###


## H3 ##
get_prior(sent ~ experiment*calc + (1|PID), data = PUNISHERSb)

priorH3 <- c(set_prior("student_t(4, 0, 10)", class = "b", coef = "calc1"),
                set_prior("student_t(4, 0, 10)", class = "b", coef = "experiment1"),
                set_prior("student_t(4, 0, 6.995)", class = "b", coef = "experiment1:calc1"),
                set_prior("student_t(4, 0, 10)", class = "Intercept"),
                set_prior("student_t(4, 0, 10)", class = "sigma"),
                set_prior("student_t(4, 0, 10)", class = "sd"),
                set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
                set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))

priorH3_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
                  set_prior("student_t(4, 0, 10)", class = "b", coef = "experiment1"),
                  set_prior("student_t(4, 0, 10)", class = "b", coef = "calc1"),
                  set_prior("student_t(4, 0, 10)", class = "sigma"),
                  set_prior("student_t(4, 0, 10)", class = "sd"),
                  set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
                  set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))
c(get_prior(sent ~ experiment*calc + (1|PID), data = PUNISHERSb), priorH3, replace = TRUE)

#sometimes setting a larger max_treedepth may improve the mixing of the MCMC chain and reduce autocorrelation between samples, which can increase the ESS and improve the reliability of the posterior inference.
full_H3 = brm(sent ~ experiment*calc + (1|PID), data = PUNISHERSb, prior = priorH3, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H3 = brm(sent ~ experiment + calc + (1|PID), data = PUNISHERSb, prior = priorH3_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
BF_brms_H3 = bayes_factor(full_H3, null_H3)

#compute + store Bayes factor with bridgesampling
H3_Full <- bridge_sampler(full_H3)
H3_Null <- bridge_sampler(null_H3) 
BF10_3<- bayes_factor(H3_Full, H3_Null)$bf



## H10 ##
get_prior(sent ~ experiment*calc + (1|PID), data = NONPUNISHERSb)

priorH10 <- c(set_prior("student_t(4, 0, 10)", class = "b", coef = "calc1"),
             set_prior("student_t(4, 0, 10)", class = "b", coef = "experiment1"),
             set_prior("student_t(4, 0, 6.995)", class = "b", coef = "experiment1:calc1"),
             set_prior("student_t(4, 0, 10)", class = "Intercept"),
             set_prior("student_t(4, 0, 10)", class = "sigma"),
             set_prior("student_t(4, 0, 10)", class = "sd"),
             set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
             set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))

priorH10_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
               set_prior("student_t(4, 0, 10)", class = "b", coef = "experiment1"),
               set_prior("student_t(4, 0, 10)", class = "b", coef = "calc1"),
               set_prior("student_t(4, 0, 10)", class = "sigma"),
               set_prior("student_t(4, 0, 10)", class = "sd"),
               set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
               set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))
c(get_prior(sent ~ experiment*calc + (1|PID), data = NONPUNISHERSb), priorH10, replace = TRUE)

full_H10 = brm(sent ~ experiment*calc + (1|PID), data = NONPUNISHERSb, prior = priorH10, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H10 = brm(sent ~ experiment + calc + (1|PID), data = NONPUNISHERSb, prior = priorH10_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 

#compute + store Bayes factor with bridgesampling
H10_Full <- bridge_sampler(full_H10) 
H10_Null <- bridge_sampler(null_H10) 
BF10_10<- bayes_factor(H10_Full, H10_Null)$bf


## H16 ##
H16_bf <- lmBF(return ~ experiment*calc, data = PUNISHERSa, posterior = FALSE)
H16_bf0 <- lmBF(return ~ experiment + calc, data = PUNISHERSa, posterior = FALSE)
#compare full model to main-effects model
H16_bf/H16_bf0


## H17 ##
H17_bf <- lmBF(return ~ experiment*calc, data = NONPUNISHERSa, posterior = FALSE)
H17_bf0 <- lmBF(return ~ experiment + calc, data = NONPUNISHERSa, posterior = FALSE)
#compare full model to main-effects model
H17_bf/H17_bf0




###
## comparison between E1 & E2 ##
###

## H4 ##
get_prior(sent ~ experiment*calc + (1|PID), data = ActorsB)

priorH4 <- c(set_prior("student_t(4, 0, 10)", class = "b", coef = "calc1"),
              set_prior("student_t(4, 0, 10)", class = "b", coef = "experiment1"),
              set_prior("student_t(4, 0, 6.995)", class = "b", coef = "experiment1:calc1"),
              set_prior("student_t(4, 0, 10)", class = "Intercept"),
              set_prior("student_t(4, 0, 10)", class = "sigma"),
              set_prior("student_t(4, 0, 10)", class = "sd"),
              set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
              set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))

priorH4_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
                set_prior("student_t(4, 0, 10)", class = "b", coef = "experiment1"),
                set_prior("student_t(4, 0, 10)", class = "b", coef = "calc1"),
                set_prior("student_t(4, 0, 10)", class = "sigma"),
                set_prior("student_t(4, 0, 10)", class = "sd"),
                set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
                set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))
c(get_prior(sent ~ experiment*calc + (1|PID), data = ActorsB), priorH4, replace = TRUE)

full_H4 = brm(sent ~ experiment*calc + (1|PID), data = ActorsB, prior = priorH4, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H4 = brm(sent ~ experiment + calc + (1|PID), data = ActorsB, prior = priorH4_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 

#compute + store Bayes factor with bridgesampling
H4_Full <- bridge_sampler(full_H4) 
H4_Null <- bridge_sampler(null_H4) 
BF10_4<- bayes_factor(H4_Full, H4_Null)$bf



## H18 ##
H18_bf <- lmBF(return ~ experiment*calc, data = ActorsA, posterior = FALSE)
H18_bf0 <- lmBF(return ~ experiment + calc, data = ActorsA, posterior = FALSE)
#compare full model to main-effects model
H18_bf/H18_bf0




####################
#### Additional ###


## H4-non, add.1 ##
get_prior(sent ~ experiment*calc + (1|PID), data = NonActorsB)

priorH4n <- c(set_prior("student_t(4, 0, 10)", class = "b", coef = "calc1"),
             set_prior("student_t(4, 0, 10)", class = "b", coef = "experiment1"),
             set_prior("student_t(4, 0, 6.995)", class = "b", coef = "experiment1:calc1"),
             set_prior("student_t(4, 0, 10)", class = "Intercept"),
             set_prior("student_t(4, 0, 10)", class = "sigma"),
             set_prior("student_t(4, 0, 10)", class = "sd"),
             set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
             set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))

priorH4n_0 <- c(set_prior("student_t(4, 0, 10)", class = "Intercept"),
               set_prior("student_t(4, 0, 10)", class = "b", coef = "experiment1"),
               set_prior("student_t(4, 0, 10)", class = "b", coef = "calc1"),
               set_prior("student_t(4, 0, 10)", class = "sigma"),
               set_prior("student_t(4, 0, 10)", class = "sd"),
               set_prior("student_t(4, 0, 10)", class = "sd", group = "PID"),
               set_prior("student_t(4, 0, 10)", class = "sd", coef = "Intercept", group = "PID"))
c(get_prior(sent ~ experiment*calc + (1|PID), data = ActorsB), priorH4, replace = TRUE)

full_H4n = brm(sent ~ experiment*calc + (1|PID), data = NonActorsB, prior = priorH4, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 
null_H4n = brm(sent ~ experiment + calc + (1|PID), data = NonActorsB, prior = priorH4_0, sample_prior = TRUE, save_pars = save_pars(all = TRUE), control = list(adapt_delta = 0.99, max_treedepth=15), chains = 4, iter = 40000, warmup = 20000) 

#compute + store Bayes factor with bridgesampling
H4_Fulln <- bridge_sampler(full_H4n) 
H4_Nulln <- bridge_sampler(null_H4n) 
BF10_4n<- bayes_factor(H4_Fulln, H4_Nulln)$bf



## H18-non, add.2 ##
add2_bf <- lmBF(return ~ experiment*calc, data = NonActorsA, posterior = FALSE)
add2_bf0 <- lmBF(return ~ experiment + calc, data = NonActorsA, posterior = FALSE)
#compare full model to main-effects model
add2_bf/add2_bf0


