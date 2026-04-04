#Deliberating Cost and Impact: Trustworthiness Signals in Punishment and Helping
#Figures

#loading packages
library(dplyr)
library(ggplot2)
library(ggsignif)
library(tidyr)

################
#### Exp. 1 ####
################

#Loading Exp. 1 data (help, cost checking)
E1<- read.csv("helpcostcheckE1.csv", header=T, sep=";")

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

#Player B comprehension score (1 = correct, 0 = incorrect), up to 7
PlayerB$comp <- rowSums(PlayerB[, c("G1Bcomp1", "G1Bcomp2", "G1Bcomp3", "G1Bcomp4",
                                    "G2Bcomp1", "G2Bcomp2", "G2Bcomp3")], na.rm = TRUE)

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
PlayerA$deliberation<- coalesce(PlayerA$checkObs, PlayerA$checkHid) # 1 = checked the cost, 0 = did not check the cost
PlayerA$helping<- coalesce(PlayerA$calcHelp, PlayerA$uncalcHelp) # 1 = helped, 0 = did not help

#select only observable condition B responses
PlayerB2 <- subset(PlayerB, conditionB == 1)
PlayerB2<- subset(PlayerB2, select = c(PID, helpUncalc, helpCalc, noUncalc, noCalc))
head(PlayerB2)

PlayerB2<- PlayerB2 %>%
  pivot_longer(-PID) %>%
  mutate(helping = as.numeric(grepl("help", name)),
         deliberation   = as.numeric(grepl("Calc",  name))) %>%
  select(PID, helping, deliberation, value)
#helping: 1 yes, 0 no
#deliberation: 1 yes, 0 no

#rename value to sent
names(PlayerB2)[4]<-paste("sent")

#converting from pence sent to % sent
PlayerB2$sent<- ifelse(PlayerB2$sent >= 1, PlayerB2$sent*100/10, PlayerB2$sent)

#further subset including helpers only
Exp1B<- subset(PlayerB2, PlayerB2$helping == 1)
# Adding a new variable "Experiment" with a constant value "Exp. 1"
Exp1B$Experiment <- "1. Help Cost Check"
Exp1B<- Exp1B[c("sent", "deliberation", "Experiment")]

#subset including helpers only
Exp1A<- subset(PlayerA, PlayerA$helping == 1)
# Adding a new variable "Experiment" with a constant value "Exp. 1"
Exp1A$Experiment <- "1. Help Cost Check"
Exp1A<- Exp1A[c("return", "deliberation", "Experiment")]

#further subset including non-helpers only
Exp1Bn<- subset(PlayerB2, PlayerB2$helping == 0)
# Adding a new variable "Experiment" with a constant value "Exp. 1"
Exp1Bn$Experiment <- "1. Help Cost Check"
Exp1Bn<- Exp1Bn[c("sent", "deliberation", "Experiment")]

#subset including non-helpers only
Exp1An<- subset(PlayerA, PlayerA$helping == 0)
# Adding a new variable "Experiment" with a constant value "Exp. 1"
Exp1An$Experiment <- "1. Help Cost Check"
Exp1An<- Exp1An[c("return", "deliberation", "Experiment")]



################
#### Exp. 2 ####
################

#Loading Exp. 2 data (punishment, cost checking)
E2<- read.csv("puncostcheckE2.csv", header=T, sep=";")

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

#Player B comprehension score (1 = correct, 0 = incorrect), up to 7
PlayerB$comp <- rowSums(PlayerB[, c("G1Bcomp1", "G1Bcomp2", "G1Bcomp3", "G1Bcomp4",
                                    "G2Bcomp1", "G2Bcomp2", "G2Bcomp3")], na.rm = TRUE)

#remove those who got 1 or less out of 8 wrong
#PlayerA <- subset(PlayerA, comp >= 7) 

#remove those who got 1 or less out of 7 wrong
#PlayerB <- subset(PlayerB, comp >= 6) 

#recoding levels into 1 = decision process observable & 0 = decision process hidden
PlayerA$conditionA <- ifelse(PlayerA$Condition == 3, 1, 0)
PlayerA$conditionA<- as.factor(PlayerA$conditionA)
PlayerB$conditionB <- ifelse(PlayerB$Condition == 5, 1, 0)
PlayerB$conditionB<- as.factor(PlayerB$conditionB)

#combining info from two variables; coalesce returns the first non-NA value from the specified columns
PlayerA$deliberation<- coalesce(PlayerA$checkObs, PlayerA$checkHid) # 1 = checked the cost, 0 = did not check the cost
PlayerA$punishing<- coalesce(PlayerA$calcPun, PlayerA$uncalcPun) # 1 = punished, 0 = did not punish

#select only observable condition B responses
PlayerB2 <- subset(PlayerB, conditionB == 1)
PlayerB2<- subset(PlayerB2, select = c(PID, punUncalc, punCalc, noUncalc, noCalc))
head(PlayerB2)

PlayerB2<- PlayerB2 %>%
  pivot_longer(-PID) %>%
  mutate(punishing = as.numeric(grepl("pun", name)),
         deliberation   = as.numeric(grepl("Calc",  name))) %>%
  select(PID, punishing, deliberation, value)
#punishing: 1 yes, 0 no
#deliberation: 1 yes (calculating), 0 no (uncalculating)

#rename value to sent
names(PlayerB2)[4]<-paste("sent")

#converting from pence sent to % sent
PlayerB2$sent<- ifelse(PlayerB2$sent >= 1, PlayerB2$sent*100/10, PlayerB2$sent)

#further subset including punishers only
Exp2B<- subset(PlayerB2, PlayerB2$punishing == 1)
# Adding a new variable "Experiment" with a constant value "Exp. 2"
Exp2B$Experiment <- "2. Pun Cost Check"
Exp2B<- Exp2B[c("sent", "deliberation", "Experiment")]

#subset including punishers only
Exp2A<- subset(PlayerA, PlayerA$punishing == 1)
# Adding a new variable "Experiment" with a constant value "Exp. 2"
Exp2A$Experiment <- "2. Pun Cost Check"
Exp2A<- Exp2A[c("return", "deliberation", "Experiment")]

#further subset including non-punishers only
Exp2Bn<- subset(PlayerB2, PlayerB2$punishing == 0)
# Adding a new variable "Experiment" with a constant value "Exp. 2"
Exp2Bn$Experiment <- "2. Pun Cost Check"
Exp2Bn<- Exp2Bn[c("sent", "deliberation", "Experiment")]

#subset including non-punishers only
Exp2An<- subset(PlayerA, PlayerA$punishing == 0)
# Adding a new variable "Experiment" with a constant value "Exp. 2"
Exp2An$Experiment <- "2. Pun Cost Check"
Exp2An<- Exp2An[c("return", "deliberation", "Experiment")]


################
#### Exp. 3 ####
################

#Loading Exp. 3 data (punishment, decision time)
TimeA<- read.csv("puntimeE3a.csv", header=T, sep=";")
TimeB<- read.csv("puntimeE3b.csv", header=T, sep=";")

#recoding levels into 1 = decision process observable & 0 = decision process hidden
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

#keep only those who got none or one wrong
#TimeA <- subset(TimeA, comp >= 7) 

#keep only those who got none or one wrong
#TimeB <- subset(TimeB, comp >= 6) 

#select only observable condition B responses 
TimeB2 <- subset(TimeB, Condition == 1)
TimeB2<- subset(TimeB2, select = c(ID, punFast, punSlow, noFast, noSlow))
head(TimeB2)

TimeB2<- TimeB2 %>%
  pivot_longer(-ID) %>%
  mutate(punished = as.numeric(grepl("pun", name)),
         deliberation   = as.numeric(grepl("Fast",  name))) %>%
  select(ID, punished, deliberation, value)
#punished: yes - 1, no - 0
#deliberation: fast (uncalculating) - 1, slow (calculating) - 0 
#recode so 1 = calculating & 0 = uncalculating
TimeB2 <- TimeB2 %>%
  mutate(deliberation = recode(deliberation, `0` = 1, `1` = 0))

#rename value to sent
names(TimeB2)[4]<-paste("sent")

#converting from pence sent to % sent
TimeB2$sent<- ifelse(TimeB2$sent >= 1, TimeB2$sent*100/10, TimeB2$sent)

#further subset including punishers only
Exp3B<- subset(TimeB2, TimeB2$punished == 1)
# Adding a new variable "Experiment" with a constant value "Exp. 3"
Exp3B$Experiment <- "3. Pun Decision Time"
Exp3B<- Exp3B[c("sent", "deliberation", "Experiment")]

#further subset including non-punishers only
Exp3Bn<- subset(TimeB2, TimeB2$punished == 0)
# Adding a new variable "Experiment" with a constant value "Exp. 3"
Exp3Bn$Experiment <- "3. Pun Decision Time"
Exp3Bn<- Exp3Bn[c("sent", "deliberation", "Experiment")]

################
#### Exp. 4 ####
################

#Loading Exp. 4 data
E4<- read.csv("helpimpactcheckE4.csv", header=T, sep=";")

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

#Player B comprehension score (1 = correct, 0 = incorrect), up to 7
PlayerB$comp <- rowSums(PlayerB[, c("G1Bcomp1", "G1Bcomp2", "G1Bcomp3", "G1Bcomp4",
                                    "G2Bcomp1", "G2Bcomp2", "G2Bcomp3")], na.rm = TRUE)
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
PlayerA$deliberation<- coalesce(PlayerA$checkObs, PlayerA$checkHid) # 1 = checked the cost, 0 = did not check the cost
PlayerA$helping<- coalesce(PlayerA$calcHelp, PlayerA$uncalcHelp) # 1 = helped, 0 = did not help

#select only observable condition B responses
PlayerB2 <- subset(PlayerB, conditionB == 1)
PlayerB2<- subset(PlayerB2, select = c(PID, helpUncalc, helpCalc, noUncalc, noCalc))
head(PlayerB2)

PlayerB2<- PlayerB2 %>%
  pivot_longer(-PID) %>%
  mutate(helping = as.numeric(grepl("help", name)),
         deliberation   = as.numeric(grepl("Calc",  name))) %>%
  select(PID, helping, deliberation, value)
#helping: 1 yes, 0 no
#deliberation: 1 yes, 0 no

#rename value to sent
names(PlayerB2)[4]<-paste("sent")

#converting from pence sent to % sent
PlayerB2$sent<- ifelse(PlayerB2$sent >= 1, PlayerB2$sent*100/10, PlayerB2$sent)

#further subset including helpers only
Exp4B<- subset(PlayerB2, PlayerB2$helping == 1)
# Adding a new variable "Experiment" with a constant value "Exp. 4"
Exp4B$Experiment <- "4. Help Impact Check"
Exp4B<- Exp4B[c("sent", "deliberation", "Experiment")]

#subset including helpers only
Exp4A<- subset(PlayerA, PlayerA$helping == 1)
# Adding a new variable "Experiment" with a constant value "Exp. 4"
Exp4A$Experiment <- "4. Help Impact Check"
Exp4A<- Exp4A[c("return", "deliberation", "Experiment")]

#further subset including non-helpers only
Exp4Bn<- subset(PlayerB2, PlayerB2$helping == 0)
# Adding a new variable "Experiment" with a constant value "Exp. 4"
Exp4Bn$Experiment <- "4. Help Impact Check"
Exp4Bn<- Exp4Bn[c("sent", "deliberation", "Experiment")]

#subset including non-helpers only
Exp4An<- subset(PlayerA, PlayerA$helping == 0)
# Adding a new variable "Experiment" with a constant value "Exp. 4"
Exp4An$Experiment <- "4. Help Impact Check"
Exp4An<- Exp4An[c("return", "deliberation", "Experiment")]


################
#### Exp. 5 ####
################

#Loading Exp. 5 data (punishment, impact checking)
E5<- read.csv("punimpactcheckE5.csv", header=T, sep=";")

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
#Player B comprehension score (1 = correct, 0 = incorrect), up to 7
PlayerB$comp <- rowSums(PlayerB[, c("G1Bcomp1", "G1Bcomp2", "G1Bcomp3", "G1Bcomp4",
                                    "G2Bcomp1", "G2Bcomp2", "G2Bcomp3")], na.rm = TRUE)
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
PlayerA$deliberation<- coalesce(PlayerA$checkObs, PlayerA$checkHid) # 1 = checked the impact, 0 = did not check the impact
PlayerA$punishing<- coalesce(PlayerA$calcPun, PlayerA$uncalcPun) # 1 = punished, 0 = did not punish

#select only observable condition B responses
PlayerB2 <- subset(PlayerB, conditionB == 1)
PlayerB2<- subset(PlayerB2, select = c(PID, punUncalc, punCalc, noUncalc, noCalc))
head(PlayerB2)

PlayerB2<- PlayerB2 %>%
  pivot_longer(-PID) %>%
  mutate(punishing = as.numeric(grepl("pun", name)),
         deliberation   = as.numeric(grepl("Calc",  name))) %>%
  select(PID, punishing, deliberation, value)
#punishing: 1 yes, 0 no
#deliberation: 1 yes (calculating), 0 no (uncalting)

#rename value to sent
names(PlayerB2)[4]<-paste("sent")

#converting from pence sent to % sent
PlayerB2$sent<- ifelse(PlayerB2$sent >= 1, PlayerB2$sent*100/10, PlayerB2$sent)

#further subset including punishers only
Exp5B<- subset(PlayerB2, PlayerB2$punishing == 1)
# Adding a new variable "Experiment" with a constant value "Exp. 5"
Exp5B$Experiment <- "5. Pun Impact Check"
Exp5B<- Exp5B[c("sent", "deliberation", "Experiment")]

#subset including punishers only
Exp5A<- subset(PlayerA, PlayerA$punishing == 1)
# Adding a new variable "Experiment" with a constant value "Exp. 5"
Exp5A$Experiment <- "5. Pun Impact Check"
Exp5A<- Exp5A[c("return", "deliberation", "Experiment")]

#further subset including non-punishers only
Exp5Bn<- subset(PlayerB2, PlayerB2$punishing == 0)
# Adding a new variable "Experiment" with a constant value "Exp. 5"
Exp5Bn$Experiment <- "5. Pun Impact Check"
Exp5Bn<- Exp5Bn[c("sent", "deliberation", "Experiment")]

#subset including non-punishers only
Exp5An<- subset(PlayerA, PlayerA$punishing == 0)
# Adding a new variable "Experiment" with a constant value "Exp. 5"
Exp5An$Experiment <- "5. Pun Impact Check"
Exp5An<- Exp5An[c("return", "deliberation", "Experiment")]

############################
##### MERGING DATA SETS ####
############################

### Action

# Combine/merge the datasets for sending decisions
sending <- rbind(Exp1B, Exp2B, Exp3B, Exp4B, Exp5B)
sending$Experiment<- as.factor(sending$Experiment)
sending$Deliberation<- as.factor(sending$deliberation)

# Combine/merge the datasets for returning decisions
returning <- rbind(Exp1A, Exp2A, Exp4A, Exp5A)
returning$Experiment<- as.factor(returning$Experiment)
returning$Deliberation<- as.factor(returning$deliberation)

### Inaction

# Combine/merge the datasets for sending decisions
sendingNon <- rbind(Exp1Bn, Exp2Bn, Exp3Bn, Exp4Bn, Exp5Bn)
sendingNon$Experiment<- as.factor(sendingNon$Experiment)
sendingNon$Deliberation<- as.factor(sendingNon$deliberation)

# Combine/merge the datasets for returning decisions
returningNon <- rbind(Exp1An, Exp2An, Exp4An, Exp5An)
returningNon$Experiment<- as.factor(returningNon$Experiment)
returningNon$Deliberation<- as.factor(returningNon$deliberation)


#################
#### Figures ####
#################

#sending and returning decisions for helpers and punishers

#default trim = TRUE (tails of the violins are trimmed). FALSE doesnt trim the tails

##
# Sending #
##

s<- ggplot(sending, aes(x = Experiment, y = sent, fill = Deliberation)) +
  geom_violin(trim = TRUE) +
  labs(x = "Experiment", y = "Percentage endowment entrusted") +
  theme_classic() +
  scale_fill_manual(
    values = c("0" = "lightgrey", "1" = "darksalmon"),
    labels = c("0" = "uncalculating", "1" = "calculating")
  )

# Define a custom summary function
data_summary <- function(x) {
  m <- mean(x)
  ymin <- m - sd(x)
  ymax <- m + sd(x)
  return(c(y = m, ymin = ymin, ymax = ymax))
}

# Adding mean and standard deviation
s<- s + stat_summary(
  fun.data = data_summary,
  geom = "point",
  position = position_dodge(width = 0.9),
)

s<- s + stat_summary(
  fun.data = data_summary,
  geom = "errorbar",
  position = position_dodge(width = 0.9),
  width = 0.2
)

# adding significance indicator
s<- s + geom_signif(y_position = c(102), xmin = c(0.8), xmax = c(1.2),
            annotation=c("*"), tip_length=0)

##
# Returning #
##

r<- ggplot(returning, aes(x = Experiment, y = return, fill = Deliberation)) +
  geom_violin() +
  labs(x = "Experiment", y = "Percentage endowment returned") +
  theme_classic() +
  scale_fill_manual(
    values = c("0" = "lightgrey", "1" = "darksalmon"),
    labels = c("0" = "uncalculating", "1" = "calculating")
  )

# Define a custom summary function
data_summary <- function(x) {
  m <- mean(x)
  ymin <- m - sd(x)
  ymax <- m + sd(x)
  return(c(y = m, ymin = ymin, ymax = ymax))
}

# Adding mean and standard deviation
r<- r + stat_summary(
  fun.data = data_summary,
  geom = "point",
  position = position_dodge(width = 0.9),
)

r + stat_summary(
  fun.data = data_summary,
  geom = "errorbar",
  position = position_dodge(width = 0.9),
  width = 0.2
)

# adding significance indicator
r<- r + geom_signif(y_position = c(102), xmin = c(0.8), xmax = c(1.2),
                annotation=c("***"), tip_length=0)


## run "E3 Punishment Decision Time" code as well

library(patchwork)
r / E3return + plot_annotation(tag_levels = 'A')





#####
# Non-Action
#####

##
# Sending #
##

sn<- ggplot(sendingNon, aes(x = Experiment, y = sent, fill = Deliberation)) +
  geom_violin(trim = TRUE) +
  labs(x = "Experiment", y = "Percentage endowment entrusted") +
  theme_classic() +
  scale_fill_manual(
    values = c("0" = "lightgrey", "1" = "darksalmon"),
    labels = c("0" = "uncalculating", "1" = "calculating")
  )

# Define a custom summary function
data_summary <- function(x) {
  m <- mean(x)
  ymin <- m - sd(x)
  ymax <- m + sd(x)
  return(c(y = m, ymin = ymin, ymax = ymax))
}

# Adding mean and standard deviation
sn<- sn + stat_summary(
  fun.data = data_summary,
  geom = "point",
  position = position_dodge(width = 0.9),
)

sn<- sn + stat_summary(
  fun.data = data_summary,
  geom = "errorbar",
  position = position_dodge(width = 0.9),
  width = 0.2
)



##
# Returning #
##

rn<- ggplot(returningNon, aes(x = Experiment, y = return, fill = Deliberation)) +
  geom_violin() +
  labs(x = "Experiment", y = "Percentage endowment returned") +
  theme_classic() +
  scale_fill_manual(
    values = c("0" = "lightgrey", "1" = "darksalmon"),
    labels = c("0" = "uncalculating", "1" = "calculating")
  )

# Define a custom summary function
data_summary <- function(x) {
  m <- mean(x)
  ymin <- m - sd(x)
  ymax <- m + sd(x)
  return(c(y = m, ymin = ymin, ymax = ymax))
}

# Adding mean and standard deviation
rn<- rn + stat_summary(
  fun.data = data_summary,
  geom = "point",
  position = position_dodge(width = 0.9),
)

rn<- rn + stat_summary(
  fun.data = data_summary,
  geom = "errorbar",
  position = position_dodge(width = 0.9),
  width = 0.2
)

rn<- rn + geom_signif(y_position = c(102), xmin = c(1.8), xmax = c(2.2),
                    annotation=c("**"), tip_length=0)


## run "E3 Punishment Decision Time" code as well

library(patchwork)
rn / E3returnNon + plot_annotation(tag_levels = 'A')


###########
## Q1/Q5 ##
###########


# this section assumes the 5 experiment code files have been run

library(ggpubr)
ggarrange(ggplot_object1, ggplot_object2, ggplot_object3, ggplot_object4, ggplot_object5, common.legend = TRUE, legend = "bottom")


library(patchwork)
# Combine the plots
combined_plots <- ggplot_object1 + ggplot_object2 + ggplot_object3 + ggplot_object4 + ggplot_object5










