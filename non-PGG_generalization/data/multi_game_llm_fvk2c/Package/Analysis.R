################################################################################
# Data Analysis
# Adverse Reactions to the Use of Large Language Models in Social Interactions
# F. Dvorak, R. Stumpf, S. Fehrler & U. Fischbacher
# PNAS Nexus
################################################################################
library(tidyverse)
library(readxl)
library(ds4psy)
library(kableExtra)
library(unikn)
library(DescTools)
library(hunspell)
library(openxlsx)
library(ggsignif)
library(rstatix)
library(jtools)
library(stargazer)
library(gtools)
library(ggpubr)
library(gridExtra)
library(cowplot)
library(glmnet)
library(ggplot2)
rm(list = ls())                          # clear workspace
setwd(dirname(sys.frame(1)$ofile))       # set working directory
set.seed(1)                              # random seed (1)
BS <- 10000                              # number of bootstrap samples 

###############################################################################
## Load main data and exclude subjects
###############################################################################
df <- read_csv("./data/MainData.csv", show_col_types = FALSE)

df$DurationInMin <- as.numeric(df$DurationInSec)/60
df$DurationInSec_log <- log(as.numeric(df$DurationInSec))

cutoff_time_upper <- mean(df$DurationInSec_log) + 3*sd(df$DurationInSec_log)
cutoff_time_lower <- mean(df$DurationInSec_log) - 3*sd(df$DurationInSec_log)

N_excluded_time <- df %>% group_by(SubjectID) %>% filter(row_number() == 1)
N_excluded_time <- N_excluded_time %>% filter(DurationInSec_log > cutoff_time_upper|DurationInSec_log < cutoff_time_lower) %>% ungroup() %>% summarise(N = n())
N_excluded_time

df <- df %>% filter(DurationInSec_log <= cutoff_time_upper & DurationInSec_log >= cutoff_time_lower)

df %>% group_by(Treatment) %>% summarise(N = length(unique(SubjectID))) %>% kbl() %>% kable_styling(full_width = F)
df %>% group_by(PersonalizedTreatment) %>% summarise(N = length(unique(SubjectID))) %>% kbl() %>% kable_classic_2(full_width = F)
df %>% group_by(TreatmentCode) %>% summarise(N = length(unique(SubjectID))) %>% kbl() %>% kable_minimal(full_width = F)

df$ExcludeSubject <- 0

sum(grepl("Yes|yes", df$Exclusion_Disturb))
df$SubjectID[grepl("Yes|yes", df$Exclusion_Disturb)]

sum(grepl("Yes|yes", df$ExclusionUsageGPT))
df$SubjectID[grepl("Yes|yes", df$ExclusionUsageGPT)]

df$ExcludeSubject[grepl("Yes|yes", df$Exclusion_Disturb)] <- 1
df$ExcludeSubject[grepl("Yes|yes", df$ExclusionUsageGPT)] <- 1

N_excluded_other <- df %>% group_by(SubjectID) %>% filter(row_number() == 1)
N_excluded_other <- N_excluded_other %>% filter(ExcludeSubject == 1) %>% ungroup() %>% summarise(N = n())
N_excluded_other

df <- df %>% filter(ExcludeSubject == 0)
df_short <- df %>% group_by(SubjectID) %>% filter(row_number() == 1)

# AI flavours
df$Q1 = as.numeric(df$IntuitionThoughtfulness) - 1
df$Q2 = as.numeric(df$IntroversionExtraverison) - 1
df$Q3 = as.numeric(df$FairnessEfficiency) - 1
df$Q4 = as.numeric(df$ChaosBoredom) - 1
df$Q5 = as.numeric(df$SelfishnessAltruism) - 1
df$Q6 = as.numeric(df$NoveltyReliability) - 1
df$Q7 = as.numeric(df$TruthHarmony) - 1
df$AI_flavour <- 64*df$Q1 + 32*df$Q2 + 16*df$Q3 + 8*df$Q4 + 4*df$Q5 + 2*df$Q6 + df$Q7 + 1

###############################################################################
## Load turing data and exclude subjects
###############################################################################
df_t <- read_csv("./data/TuringData.csv", show_col_types = FALSE)

df_t$DurationInSec_log <- log(as.numeric(df_t$DurationInSec))

cutoff_time_upper <- mean(df_t$DurationInSec_log) + 3*sd(df_t$DurationInSec_log)
cutoff_time_lower <- mean(df_t$DurationInSec_log) - 3*sd(df_t$DurationInSec_log)

N_excluded_time_turing <- df_t %>% group_by(SubjectID) %>% filter(row_number() == 1) 
N_excluded_time_turing <- N_excluded_time_turing %>% filter(DurationInSec_log > cutoff_time_upper|DurationInSec_log < cutoff_time_lower) %>% ungroup() %>% summarise(N = n())
N_excluded_time_turing

df_t <- df_t %>% filter(DurationInSec_log <= cutoff_time_upper & DurationInSec_log >= cutoff_time_lower)

df_t$UsageChatGPT[is.na(df_t$UsageChatGPT)] <- ""
df_t <- df_t %>% select(-RecordedDate)

df_t$ExcludeSubject <- 0

sum(grepl("Yes|yes", df_t$Exclusion_Disturb))
sum(grepl("Yes|yes", df_t$ExclusionUsageGPT))

df_t <- df_t %>% select(-ExcludeSubject)

df_t$Treatment <- ""
df_t$Treatment[grepl("TR", df_t$TreatmentCode)] <- "TransparentRandom"
df_t$Treatment[grepl("TD", df_t$TreatmentCode)] <- "TransparentDelegation"
df_t$Treatment[grepl("O", df_t$TreatmentCode)]  <- "OpaqueDelegation"

df_t <- df_t %>% group_by(SubjectID) %>% mutate(MeanRatingPerSubject = mean(RatingIsAI))
df_t <- df_t %>% group_by(SubjectID) %>% mutate(MeanRatingPerSubject_FirstDecision = mean(ifelse(grepl("1", Situation), RatingIsAI, NA), na.rm = TRUE))
df_t <- df_t %>% group_by(SubjectID) %>% mutate(MeanRatingPerSubject_SecondDecision = mean(ifelse(grepl("2", Situation), RatingIsAI, NA), na.rm = TRUE))

df_t$Situation[df_t$Situation == "TGReceiver"] <- "TG_R"
df_t$Situation[df_t$Situation == "TGSender"] <- "TG_S"
df_t$Situation[df_t$Situation == "UGProposer"] <- "UG_P"
df_t$Situation[df_t$Situation == "UGResponder"] <- "UG_R"

df_t$Situation[df_t$Situation == "TGReceiver1"] <- "TG_R1"
df_t$Situation[df_t$Situation == "TGSender1"] <- "TG_S1"
df_t$Situation[df_t$Situation == "UGProposer1"] <- "UG_P1"
df_t$Situation[df_t$Situation == "UGResponder1"] <- "UG_R1"

df_t$Situation[df_t$Situation == "TGReceiver2"] <- "TG_R2"
df_t$Situation[df_t$Situation == "TGSender2"] <- "TG_S2"
df_t$Situation[df_t$Situation == "UGProposer2"] <- "UG_P2"
df_t$Situation[df_t$Situation == "UGResponder2"] <- "UG_R2"

df_t$Decision[df_t$Treatment != "OpaqueDelegation"] <- "Justification"
df_t$Decision[df_t$Treatment == "OpaqueDelegation" & grepl("1", df_t$Situation)] <- "Behavior"
df_t$Decision[df_t$Treatment == "OpaqueDelegation" & grepl("2", df_t$Situation)] <- "Justification"

## Adjusting outcome variable for insufficient randomization
df_t$HumanAISameDecision <- ifelse(df_t$Human_DEC == df_t$AI_DEC,1,0)
df_t$HumanAISameDecision <- as.factor(df_t$HumanAISameDecision)
df_t$AIStatementLeft[grepl("UG_R|TG_R|SH", df_t$Situation)] <- 1
df_t$AIStatementLeft[grepl("UG_P|TG_S|PD|C", df_t$Situation)] <- 0

df_t$RatingIsAI_adjusted <- df_t$RatingIsAI
df_t$RatingIsAI_adjusted[df_t$Decision == "Behavior" & df_t$HumanAISameDecision == 1] <- 0.5

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

df_t$Game <- df_t$Situation
df_t$Game[grepl("UG",df_t$Game)] <- "UG"
df_t$Game[grepl("TG",df_t$Game)] <- "TG"

df_t$Situation[df_t$Treatment == "OpaqueDelegation"] <- gsub('.{1}$', "",df_t$Situation[df_t$Treatment == "OpaqueDelegation"] )

################################################################################
# Load ChatGPT data
################################################################################
gpt_data <- read.csv("./data/AIData.csv")

 # Type unpersonalized or personalized
Type <- factor(gpt_data$Type)

# Coordination
gpt_data$C <- factor(gsub("\\.[0-9]*$","", gpt_data$DEC.C), 
            levels= c('Mercury', 'Venus', 'Earth', 'Mars', 'Saturn'))
gpt_data$C2 <- as.numeric(gpt_data$C)-3
gpt_data$Earth <- as.numeric(gpt_data$C == "Earth")

# Trust
gpt_data$Trust <- gsub("\\.[0-9]*$","", gpt_data$DEC.TG.trustor)
gpt_data$Trust[gpt_data$Trust == "yes" | gpt_data$Trust == "Yes"] <- 1
gpt_data$Trust[gpt_data$Trust == "no" | gpt_data$Trust == "No"] <- 0
gpt_data$TG.sender <- as.numeric(gpt_data$Trust)

# Trustworthiness
gpt_data$TG.receiver <- gpt_data$DEC.TG.trustee

# PD 
gpt_data$PD <- gpt_data$DEC.PD
gpt_data$PD[gpt_data$PD == "A"] <- 1
gpt_data$PD[gpt_data$PD == "B"] <- 0
gpt_data$PD <- as.numeric(gpt_data$PD)

# SH 
gpt_data$SH <- gpt_data$DEC.SH
gpt_data$SH[gpt_data$SH == "A"] <- 1
gpt_data$SH[gpt_data$SH == "B"] <- 0
gpt_data$SH <- as.numeric(gpt_data$SH)

# Fairness (UG)
gpt_data$UG.proposer <- gpt_data$DEC.UG.proposer
gpt_data$UG.responder <- gpt_data$DEC.UG.responder

decisions_list <- list(C=gpt_data$Earth, Sender=gpt_data$TG.sender, 
                       Receiver=gpt_data$TG.receiver, Proposer=gpt_data$UG.proposer,
                       Responder=gpt_data$UG.responder, PD=gpt_data$PD, SH=gpt_data$SH)

decision_cols <- names(decisions_list)

################################################################################
# Descriptive statistics
################################################################################
N_total <- length(unique(df$SubjectID))
df$Age <- as.numeric(df$Age)
Mean_age <- round(mean(df$Age), 1)
Mean_man <- round(mean(df$Gender == "Man")*100, 1)
Mean_woman <- round(mean(df$Gender == "Woman")*100, 1)
Mean_nonbinary <- round(mean(df$Gender == "Non-binary")*100, 1)
Mean_college <- round(mean(df$Education %in% c("College or university","Post-graduate degree"))*100, 1)
Med_duration <- round(median(df$DurationInMin), 1)
ChatGPTKnow <- round(mean(df$KnowledgeChatGPT == "Yes")*100, 1)
df$UsageChatGPT[is.na(df$UsageChatGPT)] <- "Never"
ChatGPTUse <- round(mean(df$UsageChatGPT != "Never", na.rm=TRUE)*100, 1) 

N_total_turing <- length(unique(df_t$SubjectID))
Mean_age_t <- round(mean(df_t$Age), 1)
Mean_man_t <- round(mean(df_t$Gender == 1)*100, 1)
Mean_woman_t <- round(mean(df_t$Gender == 2)*100, 1)
Mean_nonbinary_t <- round(mean(df_t$Gender == 3)*100, 1)
Mean_college_t <- round(mean(df_t$Education >= 4)*100, 1)
Med_duration_t <- round(median(df_t$DurationInSec/60), 1)
ChatGPTKnow_t <- round(mean(df_t$KnowledgeChatGPT == 1)*100, 1)
ChatGPTUse_t <- round(mean(!(df_t$UsageChatGPT %in% c("","1")), na.rm=TRUE)*100, 1)

# Sample descriptives reported in methods
descriptives <- rbind(c(N_total, Mean_age, Mean_man, Mean_woman, Mean_nonbinary, 
                        Mean_college, Med_duration, ChatGPTKnow, ChatGPTUse),
                      c(N_total_turing, Mean_age_t, Mean_man_t, Mean_woman_t, Mean_nonbinary_t, 
                        Mean_college_t, Med_duration_t, ChatGPTKnow_t, ChatGPTUse_t))
rownames(descriptives) <- c("subjects", "human raters")
colnames(descriptives) <- c("N", "Mean age (years)", "Male (%)", "Female (%)", 
                            "Non-binary (%)", "College/university (%)", 
                            "Median duration (min)", "Know ChatGPT (%)", "Use ChatGPT (%)")
sink(paste0("./tables/Descriptive_statistics.txt"))
knitr::kable(descriptives)
sink()

################################################################################
# Table 1: Key findings of the main experiment
################################################################################
## Hypothesis Tests
data <- df
order_table1 <- c(1,8,3,5,2,7,4,6)

data$RANDOM          <- ifelse(data$Treatment == "TransparentRandom", TRUE, FALSE)
data$PERSONALIZATION <- ifelse(data$PersonalizedTreatment == 1, TRUE, FALSE)
data$TRANSPARENT     <- ifelse(data$Treatment == "OpaqueDelegation", FALSE, TRUE)

data$role[data$Scenario == "AISupport"]                                 <- "support" 
data$role[data$Scenario == "NoAISupport" & data$Case == "AgainstAI"]    <- "AI" 
data$role[data$Scenario == "NoAISupport" & data$Case == "AgainstHuman"] <- "human" 
data$role[data$Scenario == "NoAISupport" & data$Case == "Opaque"]       <- "unknown" 

data <- data %>% rename(UG_S = UGProposer, UG_R = UGResponder, TG_S = TGSender, TG_T = TGReceiver, 
                        D_UG_S = UGProposer_Delegation, D_UG_R = UGResponder_Delegation, 
                        D_TG_S = TGSender_Delegation, D_TG_T = TGReceiver_Delegation,
                        D_PD = PD_Delegation, D_SH = SH_Delegation, D_C = C_Delegation)

data$C <- ifelse(data$C == 0, 1, 0)

# normalize decisions
data$UG_S <- data$UG_S/5
data$UG_R <- data$UG_R/5
data$TG_T <- data$TG_T/6
data$rUG_R <- (1-data$UG_R)

# indices
data$WI = apply(cbind(data$UG_S,data$UG_R,data$TG_S,data$TG_T,data$PD,data$SH,data$C), 1, mean, na.rm = TRUE) # Welfare index
data$PSI = apply(cbind(data$UG_S,data$rUG_R,data$TG_S,data$TG_T,data$PD,data$SH), 1, mean, na.rm = TRUE) # Prosociality index
data$DI = apply(data[,grep("D_UG_S", names(data)):grep("D_C", names(data))], 1, mean, na.rm = TRUE) # Delegation index

# sub-indices
data$PM = data$C  # Predictability measure
data$FM = data$UG_S # Equality measure
data$KI = apply(cbind(data$TG_S,data$PD,data$SH), 1, mean, na.rm = TRUE) # Kindness index
data$II = apply(cbind(data$UG_R,data$TG_T), 1, mean, na.rm = TRUE) # Intentions index

data$WI_noUG = apply(cbind(data$TG_S,data$TG_T,data$PD,data$SH,data$C), 1, mean, na.rm = TRUE) # Welfare index without UG

# comparisons
variables    <- c(       "WI",     "WI",      "WI",      "WI",    "PSI",     "PSI",           "DI",           "DI")
l_roles      <- c(       "AI",     "AI", "unknown", "unknown",     "AI",      "AI",      "support",      "support")
r_roles      <- c(    "human",     "AI",   "human", "unknown",     "AI",      "AI",      "support",      "support")
l_treatments <- list(c("TRU"), c("TRP"),  c("ODU"),  c("ODP"), c("TDU"),  c("TDP"), c("TDP","ODP"), c("ODU","ODP"))
r_treatments <- list(c("TRU"), c("TRU"),  c("TRU"),  c("ODU"), c("TRU"),  c("TDU"), c("TDU","ODU"), c("TDU","TDP"))
alternatives <- c(     "less","greater",    "less", "greater",   "less", "greater",      "greater",      "greater")
paired       <- c(TRUE, rep(FALSE, 7))

mean_values <- sapply(1:8, function(t)  c(mean(data[[variables[t]]][data$role == l_roles[t] & data$TreatmentCode %in% l_treatments[[t]]]), 
                                          mean(data[[variables[t]]][data$role == r_roles[t] & data$TreatmentCode %in% r_treatments[[t]]])))

sd_values <- sapply(1:8, function(t)  c(sd(data[[variables[t]]][data$role == l_roles[t] & data$TreatmentCode %in% l_treatments[[t]]]), 
                                        sd(data[[variables[t]]][data$role == r_roles[t] & data$TreatmentCode %in% r_treatments[[t]]])))

main_tests <- lapply(1:8, function(t)  t.test(data[[variables[t]]][data$role == l_roles[t] & data$TreatmentCode %in% l_treatments[[t]]], 
                                              data[[variables[t]]][data$role == r_roles[t] & data$TreatmentCode %in% r_treatments[[t]]], 
                                              alternative = alternatives[t], paired = paired[t]))

p_values <- sapply(1:8, function(t) main_tests[[t]]$p.value)
degree_of_freedom <- sapply(1:8, function(t) main_tests[[t]]$parameter)
t_statistics <- sapply(1:8, function(t) main_tests[[t]]$statistic)

# Holm Bonferroni correction
p_tests <- p.adjust(p_values, method = "holm")

# compile table 1
ordered_means <- t(round(mean_values,2))[order_table1,]
ordered_means[1:4,] <- ordered_means[1:4, c(2,1)]
ordered_p_values <- round(p_tests[order_table1], 3)
table1_questions <- c("Does interacting with AI decrease payoffs?",
                      "Is opaque delegation more frequent?",
                      "Does opaque interaction decrease payoffs?",
                      "Does delegation crowd-out prosociality?",
                      "Does personalizing the AI restore payoffs?",
                      "Does personalizing the AI increase delegation?",
                      "Does personalizing the AI change payoffs",
                      "Does personalizing the AI change prosocial behavior?")
table1_answers <- ifelse(ordered_p_values < 0.05, "Yes", "No")
table1_variables <- rep(c("payoff index", "delegation frequency", "payoff index", "prosociality index"), 2)
Table1 <- cbind(table1_questions, table1_answers, table1_variables,
                ordered_means, ordered_p_values)
rownames(Table1) <- as.character(1:8)
colnames(Table1) <- c("Research question", "Answer", "Variable", "Cond 1", "Cond 2", "p-val")

sink(paste0("./tables/Table_1.txt"))
knitr::kable(Table1)
sink()

# Main tests statistics reported in paragraphs
ordered_means <- t(round(mean_values,2))[order_table1,]
ordered_means[2,] <- ordered_means[2, c(2,1)]
ordered_sds <- t(round(sd_values,2))[order_table1,]
ordered_sds[2,] <- ordered_sds[2,c(2,1)]
ordered_t <- round(t_statistics[order_table1], 2)
ordered_df <- round(degree_of_freedom[order_table1])
Table1_tests <- cbind(ordered_means[,1], ordered_sds[,1], 
                      ordered_means[,2], ordered_sds[,2],
                      ordered_df, ordered_t, format(ordered_p_values, 3))
colnames(Table1_tests) <- c("Mean 1", "SD 1", "Mean 2", "SD 2", "df", "t-stat", "p-value")
rownames(Table1_tests) <- table1_questions

################################################################################
# Question 1: Tests reported in "Does interacting with AI decrease payoffs?"
################################################################################
variables    <- c(       "WI",     "WI",     "KI",     "PM",     "FM",     "II")
l_roles      <- c(       "AI",     "AI",     "AI",     "AI",     "AI",     "AI")
r_roles      <- c(    "human",  "human",  "human",  "human",  "human",  "human")
l_treatments <- list(c("TDU"), c("TRP"), c("TRU"), c("TRU"), c("TRU"), c("TRU"))
r_treatments <- list(c("TDU"), c("TRP"), c("TRU"), c("TRU"), c("TRU"), c("TRU"))
alternatives <- c(     "less",   "less",   "less",   "less",   "less",   "less")
paired       <- c(       TRUE,     TRUE,     TRUE,     TRUE,     TRUE,     TRUE)

mean_values <- sapply(1:6, function(t)  c(mean(data[[variables[t]]][data$role == l_roles[t] & data$TreatmentCode %in% l_treatments[[t]]]), 
                                          mean(data[[variables[t]]][data$role == r_roles[t] & data$TreatmentCode %in% r_treatments[[t]]])))

sd_values <- sapply(1:6, function(t)  c(sd(data[[variables[t]]][data$role == l_roles[t] & data$TreatmentCode %in% l_treatments[[t]]]), 
                                        sd(data[[variables[t]]][data$role == r_roles[t] & data$TreatmentCode %in% r_treatments[[t]]])))

main_tests <- lapply(1:6, function(t)  t.test(data[[variables[t]]][data$role == l_roles[t] & data$TreatmentCode %in% l_treatments[[t]]], 
                                              data[[variables[t]]][data$role == r_roles[t] & data$TreatmentCode %in% r_treatments[[t]]], 
                                              alternative = alternatives[t], paired = paired[t]))

p_values <- sapply(1:6, function(t) main_tests[[t]]$p.value)
degree_of_freedom <- sapply(1:6, function(t) main_tests[[t]]$parameter)
t_statistics <- sapply(1:6, function(t) main_tests[[t]]$statistic)

ordered_means <- t(round(mean_values,2))
ordered_sds <- t(round(sd_values,2))
ordered_t <- round(t_statistics, 2)
ordered_df <- round(degree_of_freedom)
ordered_p_values <- round(p_values, 3)
Q1_tests <- cbind(ordered_means[,1], ordered_sds[,1], 
                  ordered_means[,2], ordered_sds[,2],
                  ordered_df, ordered_t, format(ordered_p_values, 3))
colnames(Q1_tests) <- c("Mean 1", "SD 1", "Mean 2", "SD 2", "df", "t-stat", "p-value")
rownames(Q1_tests) <- c("Lower payoff index for deliberate delegation",
                        "Lower payoff index for personalized AI",
                        "Lower kindness index",
                        "Lower predictability measure",
                        "Lower equality concerns",
                        "Reciprocity is unaffected")

################################################################################
# Question 1: Regression models controlling for socioeconomic characteristics
################################################################################
data <- data %>% mutate_at(grep("Q", names(.)), 
                           ~as.numeric(recode(.,
                                              "Strongly disagree"=0, 
                                              "Somewhat disagree"=1,
                                              "Neither agree nor disagree"=2,
                                              "Somewhat agree"=3,
                                              "Strongly agree"=4))) 
data <- data %>% mutate(Q_PredictionAIHuman = Q_DifficultyPredictionAI - Q_DifficultyPredictionHuman,
                        Q_TrustworthyAIHuman = Q_AITrustworthy - Q_HumanTrustworthy,
                        Q_EqualityAIHuman = Q_EqualityWithAI - Q_EqualityWithHuman)

# prepare socioeconomic variables
data$Gender[grepl("Prefer", data$Gender)] <- "Non-binary"
data$Gender    <- factor(data$Gender)
data$Education <- factor(data$Education)
data$HigherEducation[grepl("Higher|Secondary|Primary|Prefer", data$Education)] <- 0
data$HigherEducation[grepl("College|Post", data$Education)] <- 1

# prepare other controls
data$KnowledgeChatGPT <- factor(data$KnowledgeChatGPT)
data$UsageChatGPT <- factor(data$UsageChatGPT)
data$role <- factor(data$role)
data$Age <- as.numeric(data$Age)
data$count_ChangedBrowserWindow <- as.numeric(data$count_ChangedBrowserWindow)
data$Case <- factor(data$Case, levels = c("AgainstAI", "AgainstHuman", "Opaque"))
data$UsageChatGPT[is.na(data$UsageChatGPT)] <- "Never"
data$UsageChatGPT_numeric[data$UsageChatGPT == "Never"] <- 0
data$UsageChatGPT_numeric[data$UsageChatGPT == "1 - 5 times"] <- 1
data$UsageChatGPT_numeric[data$UsageChatGPT == "6 - 10 times"] <- 2
data$UsageChatGPT_numeric[data$UsageChatGPT == "11 - 50 times"] <- 3
data$UsageChatGPT_numeric[data$UsageChatGPT == "Over 50 times"] <- 4
data$UsageChatGPT_binary[data$UsageChatGPT_numeric == 0] <- 0
data$UsageChatGPT_binary[data$UsageChatGPT_numeric != 0] <- 1

df_reg <- data %>% filter(Scenario == "NoAISupport" & Treatment != "OpaqueDelegation") %>% pivot_wider(id_cols = c("SubjectID","TreatmentCode","Treatment", "count_ChangedBrowserWindow", "DurationInMin", "KnowledgeChatGPT", "UsageChatGPT_numeric", "UsageChatGPT_binary", "Q_DifficultyPredictionAI", "Q_DifficultyPredictionHuman", "Q_AITrustworthy", "Q_HumanTrustworthy", "Q_AIReflectsHuman", "Q_EqualityWithAI", "Q_EqualityWithHuman", "Q_PredictionAIHuman", "Q_TrustworthyAIHuman", "Q_EqualityAIHuman", "Age", "Gender", "Education", "HigherEducation"), names_from = "Case", values_from = "WI") %>% mutate(Welfare_Loss = -(AgainstAI- AgainstHuman))

#Regression model 1 (robustness)
m1 <- lm(WI ~ Case, data = subset(data,role != "support"))
summ(m1, cluster = "SubjectID", robust = "HC1", digits = 4)
cluster_m1 <- summ(m1, cluster = "SubjectID", robust = "HC1", digits = 4)$coeftable[,2]

#Regression model 2 (robustness)
m2 <- lm(WI ~ Case + Age + Gender + HigherEducation + UsageChatGPT_numeric  + Q_DifficultyPredictionAI + Q_AITrustworthy + Q_EqualityWithAI + Q_AIReflectsHuman, data = subset(data, role != "support"))
summ(m2, cluster = "SubjectID", robust = "HC1", digits = 4)
cluster_m2 <- summ(m2, cluster = "SubjectID", robust = "HC1", digits = 4)$coeftable[,2]

#Regression model 3
m3 <- lm(WI ~ Case + Age + Gender + HigherEducation + UsageChatGPT_numeric + Q_DifficultyPredictionAI + Q_AITrustworthy + Q_EqualityWithAI + Q_AIReflectsHuman + count_ChangedBrowserWindow + DurationInMin, data = subset(data, role != "support"))
summ(m3, cluster = "SubjectID", robust = "HC1", digits = 4)
cluster_m3 <- summ(m3, cluster = "SubjectID", robust = "HC1", digits = 4)$coeftable[,2]

# Regression model 4
m1_diff <- lm(Welfare_Loss ~ Age + Gender + HigherEducation + UsageChatGPT_numeric, data = df_reg)
summ(m1_diff, digits = 4)
cluster_m1_diff <- summ(m1_diff, digits = 4)$coeftable[,2]

# Regression model 5
m2_diff <- lm(Welfare_Loss ~ Age + Gender + HigherEducation + UsageChatGPT_numeric + Q_DifficultyPredictionAI + Q_AITrustworthy + Q_EqualityWithAI + Q_AIReflectsHuman, data = df_reg)
summ(m2_diff, digits = 4)
cluster_m2_diff <- summ(m2_diff, digits = 4)$coeftable[,2]

# Regression model 6
m3_diff <- lm(Welfare_Loss ~ Age + Gender + HigherEducation + UsageChatGPT_numeric + Q_DifficultyPredictionAI + Q_AITrustworthy + Q_EqualityWithAI + Q_AIReflectsHuman  + count_ChangedBrowserWindow + DurationInMin, data = df_reg)
summ(m3_diff, digits = 4)
cluster_m3_diff <- summ(m3_diff, digits = 4)$coeftable[,2]

names(coef(m3))
desired_order <- cbind(names(coef(m3)))

Table_SM1 <- stargazer(m1, m2,m3, m1_diff, m2_diff, m3_diff, 
                       se = list(cluster_m1, 
                                 cluster_m2,
                                 cluster_m3,
                                 cluster_m1_diff,
                                 cluster_m2_diff, 
                                 cluster_m3_diff),
                       type = "text", title = "Welfare Index (base: Against AI)",
                       dep.var.labels=c("Welfare Index"))

sink(paste0("./tables/Table_SM1.txt"))
knitr::kable(Table_SM1)
sink()

################################################################################
# Tests reported in text (in order of appearance)
################################################################################
Q2_8_tests <- NULL
Q2_8_tests_names <- c("Opaque delegation is more appropriate",
                      "Opaque delegation is frequent",
                      "Belief opaque delegation common",
                      "Transparent delegation is appropriate",
                      "Personalization makes AI more humane",
                      "Personalized AI does not adequately represent human")

# opaque delegation is more appropriate
data_short <- data %>% group_by(SubjectID) %>% filter(row_number() == 1) %>% select(SubjectID, TreatmentCode, Treatment, PersonalizedTreatment, grep("Q", names(.)))
test_appropriate <- t.test(data_short$Q_DelegationAppropriate[data_short$Treatment == "TransparentDelegation"],
                           data_short$Q_DelegationAppropriate[data_short$Treatment == "OpaqueDelegation"])

sd_values <- c(sd(data_short$Q_DelegationAppropriate[data_short$Treatment == "TransparentDelegation"]),
               sd(data_short$Q_DelegationAppropriate[data_short$Treatment == "OpaqueDelegation"]))

mean_values <- round(test_appropriate$estimate,2)
sd_values <- round(sd_values,2)
t_df <- round(test_appropriate$parameter)
t_statistic <- round(test_appropriate$statistic, 2)
p_value <- round(test_appropriate$p.value, 3)

Q2_8_tests <- rbind(Q2_8_tests,
                    c(mean_values[1], sd_values[1], 
                      mean_values[2], sd_values[2],
                      t_df, t_statistic, p_value))

# opaque delegation is frequent
mean_opaque_delegation <- round(mean(data$DI[data$Treatment == "OpaqueDelegation"]), 2)
sd_opaque_delegation <- round(sd(data$DI[data$Treatment == "OpaqueDelegation"]), 2)

Q2_8_tests <- rbind(Q2_8_tests,
                    c(mean_opaque_delegation, sd_opaque_delegation, 
                      "", "", "", "", ""))

# belief opaque delegation common
mean_belief_opaque_delegation <- round(mean(data$Q_DelegationBelief[data$Treatment == "OpaqueDelegation"]), 2)
sd_belief_opaque_delegation <- round(sd(data$Q_DelegationBelief[data$Treatment == "OpaqueDelegation"]), 2)

Q2_8_tests <- rbind(Q2_8_tests,
                    c(mean_belief_opaque_delegation, sd_belief_opaque_delegation, 
                      "", "", "", "", ""))

# transparent delegation is appropriate
test_transparent_delegation_appropriate <- t.test(data_short$Q_DelegationAppropriate[data_short$Treatment == "TransparentDelegation"], mu = 2)
sd_transparent_delegation_appropriate <- sd(data_short$Q_DelegationAppropriate[data_short$Treatment == "TransparentDelegation"])

mean_value <- round(test_transparent_delegation_appropriate$estimate, 2)
sd_value <- round(sd_transparent_delegation_appropriate, 2)
t_df <- test_transparent_delegation_appropriate$parameter
t_statistic <- round(test_transparent_delegation_appropriate$statistic, 2)
p_value <- round(test_transparent_delegation_appropriate$p.value, 3)

Q2_8_tests <- rbind(Q2_8_tests,
                    c(mean_value, sd_value, 
                      "", "", t_df, t_statistic, p_value))

# personalization makes AI more humane
test_personalization_alignment <- t.test(data_short$Q_AIReflectsHuman[data_short$TreatmentCode %in% c("ORU","TRU","ODU","TDU")],
                                         data_short$Q_AIReflectsHuman[data_short$TreatmentCode %in% c("ORP","TRP","ODP","TDP")])
sd_personalization_alignment <- c(sd(data_short$Q_AIReflectsHuman[data_short$TreatmentCode %in% c("ORU","TRU","ODU","TDU")]),
                                  sd(data_short$Q_AIReflectsHuman[data_short$TreatmentCode %in% c("ORP","TRP","ODP","TDP")]))
mean_values <- round(test_personalization_alignment$estimate,2)
sd_values <- round(sd_personalization_alignment,2)
t_df <- round(test_personalization_alignment$parameter)
t_statistic <- round(test_personalization_alignment$statistic, 2)
p_value <- round(test_personalization_alignment$p.value, 3)

Q2_8_tests <- rbind(Q2_8_tests,
                    c(mean_values[1], sd_values[1], 
                      mean_values[2], sd_values[2],
                      t_df, t_statistic, p_value))

# personalized AI does not adequately represent human
test_personalization_adequate <- t.test(data_short$Q_AIReflectsHuman[data_short$TreatmentCode %in% c("ORP","TRP","ODP","TDP")], mu = 2, alternative = "greater")
sd_personalization_adequate <- sd(data_short$Q_AIReflectsHuman[data_short$TreatmentCode %in% c("ORP","TRP","ODP","TDP")])

mean_value <- round(test_personalization_adequate$estimate,2)
sd_value <- round(sd_personalization_adequate, 2)
t_df <- round(test_personalization_adequate$parameter)
t_statistic <- round(test_personalization_adequate$statistic, 2)
p_value <- round(test_personalization_adequate$p.value, 3)

# combine tests for Q2-Q8
Q2_8_tests <- rbind(Q2_8_tests,
                    c(mean_value, sd_value, 
                      "", "", t_df, t_statistic, p_value))
rownames(Q2_8_tests) <- Q2_8_tests_names
colnames(Q2_8_tests) <- colnames(Table1_tests)

################################################################################
# Tests for Figure 1
################################################################################
decision_cols <- c("TGSender", "TGReceiver", "PD", "SH", "Earth", "UGProposer", "UGResponder")
df$Earth <- as.numeric(df$C == 0)

xs <- lapply(decision_cols, function(x) 
      sapply(c("AgainstAI","AgainstHuman"),
             function(vs) df[df$Scenario == "NoAISupport" & df$Case == vs & 
                            (df$Treatment == "TransparentDelegation" | df$Treatment == "TransparentRandom"),x][[1]]))
names(xs) <- decision_cols
t_tests <- lapply(decision_cols, function(x) t.test(xs[[x]][,"AgainstAI"],
                                                    xs[[x]][,"AgainstHuman"], paired=TRUE))
names(t_tests) <- decision_cols

Figure1_tests <- t(sapply(decision_cols, function(x) c(round(mean(xs[[x]][,"AgainstAI"],na.rm=TRUE),2),
                                                     round(sd(xs[[x]][,"AgainstAI"],na.rm=TRUE),2),
                                                     round(mean(xs[[x]][,"AgainstHuman"],na.rm=TRUE),2),
                                                     round(sd(xs[[x]][,"AgainstHuman"],na.rm=TRUE),2),
                                                     round(t_tests[[x]]$parameter[1]), 
                                                     round(t_tests[[x]]$statistic,2), 
                                                     round(t_tests[[x]]$p.value,3))))

rownames(Figure1_tests) <- c("Less trust in LLM","Less trustworthy against LLM",
                             "Less cooperation in PD","Less cooperation in SH",
                             "Less predictability in C","Lower offers against LLM",
                             "Tolarate less inequality")
colnames(Figure1_tests) <- colnames(Table1_tests)

################################################################################
# Tests for post-experimental questions related to Figure 1
################################################################################
Figure1_PQ_tests <- NULL
Figure1_PQ_tests_names <- c("AI is believed to be less trustworthy",
                           "Equality concerns AI",
                           "Equality concerns human",
                           "AI is believed to be less equality-concerned")

# Trustworthiness comparison AI vs. Human
test_trustworthiness_AI_Human <- t.test(data_short$Q_TrustworthyAIHuman)
sd_trustworthiness_AI_Human <- sd(data_short$Q_TrustworthyAIHuman)

mean_value <- round(test_trustworthiness_AI_Human$estimate,2)
sd_value <- round(sd_trustworthiness_AI_Human, 2)
t_df <- round(test_trustworthiness_AI_Human$parameter)
t_statistic <- round(test_trustworthiness_AI_Human$statistic, 2)
p_value <- round(test_trustworthiness_AI_Human$p.value, 3)

# add test
Figure1_PQ_tests <- rbind(Figure1_PQ_tests,
                    c(mean_value, sd_value, 
                      "", "", t_df, t_statistic, p_value))

# Equality concerns comparison AI vs. Human
mean_value <- round(mean(data_short$Q_EqualityWithAI), 2)
sd_value <- round(sd(data_short$Q_EqualityWithAI), 2)

# add test
Figure1_PQ_tests <- rbind(Figure1_PQ_tests,
                          c(mean_value, sd_value, 
                            "", "", "", "", ""))

mean_value <- round(mean(data_short$Q_EqualityWithHuman), 2)
sd_value <- round(sd(data_short$Q_EqualityWithHuman), 2)

# add test
Figure1_PQ_tests <- rbind(Figure1_PQ_tests,
                          c(mean_value, sd_value, 
                            "", "", "", "", ""))

test_equality_AI_Human <- t.test(data_short$Q_EqualityAIHuman)
sd_equality_AI_Human <- sd(data_short$Q_EqualityAIHuman)

mean_value <- round(test_equality_AI_Human$estimate,2)
sd_value <- round(sd_equality_AI_Human, 2)
t_df <- round(test_equality_AI_Human$parameter)
t_statistic <- round(test_equality_AI_Human$statistic, 2)
p_value <- round(test_equality_AI_Human$p.value, 3)

# add test
Figure1_PQ_tests <- rbind(Figure1_PQ_tests,
                         c(mean_value, sd_value, 
                           "", "", t_df, t_statistic, p_value))

rownames(Figure1_PQ_tests) <- Figure1_PQ_tests_names
colnames(Figure1_PQ_tests) <- colnames(Table1_tests)

################################################################################
# Tests for delegation
################################################################################
Delegation_tests <- NULL
Delegation_tests_names <- c("Belief in delegation")

test_belief_in_delegation <- t.test(data_short$Q_DelegationBelief, mu = 2)
sd_belief_in_delegation <- sd(data_short$Q_DelegationBelief, na.rm=TRUE)

mean_value <- round(test_belief_in_delegation$estimate,2)
sd_value <- round(sd_belief_in_delegation, 2)
t_df <- round(test_belief_in_delegation$parameter)
t_statistic <- round(test_belief_in_delegation$statistic, 2)
p_value <- round(test_belief_in_delegation$p.value, 3)

# add test
Delegation_tests <- rbind(Delegation_tests,
                          c(mean_value, sd_value, 
                          "", "", t_df, t_statistic, p_value))
rownames(Delegation_tests) <- Delegation_tests_names
colnames(Delegation_tests) <- colnames(Table1_tests)

################################################################################
# Tests of delegation frequency in different games
################################################################################
# frequency of delegation
variables    <- c(        "D_UG_S", "D_UG_R", "D_TG_S", "D_TG_T",   "D_PD",   "D_SH",    "D_C")
l_roles      <- c(        "support","support","support","support","support","support","support")
r_roles      <- c(        "support","support","support","support","support","support","support")
l_treatments <- lapply(1:7, function(x) c("ODU","ODP"))
r_treatments <- lapply(1:7, function(x) c("TDU","TDP"))
alternatives <- c(        "greater","greater","greater","greater","greater","greater","greater","greater")
paired       <- c(            FALSE,    FALSE,    FALSE,    FALSE,    FALSE,    FALSE,    FALSE,    FALSE)

mean_values <- sapply(1:7, function(t)  c(mean(data[[variables[t]]][data$role == l_roles[t] & data$TreatmentCode %in% l_treatments[[t]]]),
                                          mean(data[[variables[t]]][data$role == r_roles[t] & data$TreatmentCode %in% r_treatments[[t]]])))

sd_values <- sapply(1:7, function(t)  c(sd(data[[variables[t]]][data$role == l_roles[t] & data$TreatmentCode %in% l_treatments[[t]]]),
                                        sd(data[[variables[t]]][data$role == r_roles[t] & data$TreatmentCode %in% r_treatments[[t]]])))

main_tests <- lapply(1:7, function(t)  t.test(data[[variables[t]]][data$role == l_roles[t] & data$TreatmentCode %in% l_treatments[[t]]],
                                              data[[variables[t]]][data$role == r_roles[t] & data$TreatmentCode %in% r_treatments[[t]]],
                                              alternative = alternatives[t], paired = paired[t]))

p_values <- sapply(1:7, function(t) main_tests[[t]]$p.value)
degree_of_freedom <- sapply(1:7, function(t) main_tests[[t]]$parameter)
t_statistics <- sapply(1:7, function(t) main_tests[[t]]$statistic)

ordered_means <- t(round(mean_values,2))
ordered_sds <- t(round(sd_values,2))
ordered_t <- round(t_statistics, 2)
ordered_df <- round(degree_of_freedom)
ordered_p_values <- round(p_values, 3)
Delegation_tests_games <- cbind(ordered_means[,1], ordered_sds[,1],
                                ordered_means[,2], ordered_sds[,2],
                                ordered_df, ordered_t, format(ordered_p_values, 3))
colnames(Delegation_tests_games) <- c("Mean 1", "SD 1", "Mean 2", "SD 2", "df", "t-stat", "p-value")
rownames(Delegation_tests_games) <- c("Ultimatium Game Sender delegtion frequency",
                                      "Ultimatium Game Receiver delegtion frequency",
                                      "Trust Game Sender delegtion frequency",
                                      "Trust Game Trustee delegtion frequency",
                                      "PD Game delegtion frequency",
                                      "SH Game delegtion frequency",
                                      "Coordination game delegtion frequency")

################################################################################
# Tests for detectability
################################################################################
Detectability_behavior <- data.frame(df_t %>% filter(Decision == "Behavior") %>% group_by(Decision, Situation) %>% 
          summarise(Mean1 = round(mean(RatingIsAI_adjusted), 2),
                    SD1 = round(sd(RatingIsAI_adjusted), 2), 
                    Mean2 = "",
                    SD2 = "", 
                    df = t.test(RatingIsAI_adjusted, mu = 0.5, alternative = "greater")$parameter, 
                    t = round(t.test(RatingIsAI_adjusted, mu = 0.5, alternative = "greater")$statistic, 2),
                    p_value = round(t.test(RatingIsAI_adjusted, mu = 0.5, alternative = "greater")$p.value, 3)))

Detectability_behavior_rownames <- paste("Detectability behavior", Detectability_behavior$Situation)
Detectability_behavior <- Detectability_behavior[,3:9]
rownames(Detectability_behavior) <- Detectability_behavior_rownames
colnames(Detectability_behavior) <- colnames(Table1_tests)

Detectability_justification <- data.frame(df_t %>% filter(Decision == "Justification") %>% group_by(Decision, Situation) %>% 
                                       summarise(Mean1 = round(mean(RatingIsAI_adjusted), 2),
                                                 SD1 = round(sd(RatingIsAI_adjusted), 2), 
                                                 Mean2 = "",
                                                 SD2 = "", 
                                                 df = t.test(RatingIsAI_adjusted, mu = 0.5, alternative = "greater")$parameter, 
                                                 t = round(t.test(RatingIsAI_adjusted, mu = 0.5, alternative = "greater")$statistic, 2),
                                                 p_value = round(t.test(RatingIsAI_adjusted, mu = 0.5, alternative = "greater")$p.value, 3)))

Detectability_justification_rownames <- paste("Detectability justification", Detectability_justification$Situation)
Detectability_justification <- Detectability_justification[,3:9]
rownames(Detectability_justification) <- Detectability_justification_rownames
colnames(Detectability_justification) <- colnames(Table1_tests)

################################################################################
# Combine all tests reported in text (in order of appearance)
################################################################################
Tests_reported_in_text <- rbind(Table1_tests, Q1_tests, Q2_8_tests, 
                                Figure1_tests, Figure1_PQ_tests,
                                Delegation_tests, Delegation_tests_games,
                                Detectability_behavior,
                                Detectability_justification)
# order of appearance
Tests_reported_in_text_order <- c(1,9,10,11,12,13,14,2,15,3,16,17,4,18,5,7,6,19:53)
Tests_reported_in_text <- Tests_reported_in_text[Tests_reported_in_text_order, ]

sink(paste0("./tables/Tests_reported_in_text.txt"))
knitr::kable(Tests_reported_in_text)
sink()

################################################################################
## Preparing data for plots
################################################################################

## Data for Figure 1
data_plot <- df %>% filter(Scenario == "NoAISupport") %>% select(SubjectID, TreatmentCode, Case, UGProposer:C)

data_plot$C_recoded <- data_plot$C
data_plot$C_recoded[data_plot$C != 0] <- 0
data_plot$C_recoded[data_plot$C == 0] <- 1

data_plot <- data_plot %>% select(-C)

data_plot$Case_adjusted <- ""
data_plot$Case_adjusted <- ifelse(grepl("TR", data_plot$TreatmentCode), "Random", ifelse(grepl("TD", data_plot$TreatmentCode), "Delegation", ""))

data_plot$Case <- paste0(data_plot$Case, data_plot$Case_adjusted)

data_plot <- data_plot %>% dplyr::select(-Case_adjusted)

data_plot_short <- data_plot %>% group_by(Case) %>% summarise(N = n(), mean_UG_S = mean(UGProposer, na.rm = TRUE), 
                                                              mean_UG_R = mean(UGResponder, na.rm = TRUE), mean_TG_S = mean(TGSender, na.rm = TRUE),
                                                              mean_TG_R = mean(TGReceiver, na.rm = TRUE), mean_PD = mean(PD, na.rm = TRUE),
                                                              mean_SH = mean(SH, na.rm = TRUE), mean_C = mean(C_recoded, na.rm = TRUE))

data_plot_short$Case[data_plot_short$Case == "AgainstAIRandom"] <- "AI_R"
data_plot_short$Case[data_plot_short$Case == "AgainstAIDelegation"] <- "AI_D"
data_plot_short$Case[data_plot_short$Case == "AgainstHumanRandom"] <- "Human_R"
data_plot_short$Case[data_plot_short$Case == "AgainstHumanDelegation"] <- "Human_D"
data_plot_short$Case[data_plot_short$Case == "Opaque"] <- "Unknown"

data_plot_short$Case <- factor(data_plot_short$Case, levels = c("AI_R", "AI_D", "Human_R", "Human_D", "Unknown"))

data_plot_short <- data_plot_short %>% pivot_longer(cols = mean_UG_S:mean_C, names_to = "Situation", names_prefix = "mean_", values_to = "mean")

data_plot_short$PlotNumber[grepl("UG", data_plot_short$Situation)] <- "Fairness (UG)"
data_plot_short$PlotNumber[grepl("TG", data_plot_short$Situation)] <- "Trust (TG)"
data_plot_short$PlotNumber[grepl("PD|SH", data_plot_short$Situation)] <- "Cooperation (PD/SH)"
data_plot_short$PlotNumber[grepl("C", data_plot_short$Situation)] <- "Coordination (C)"

data_plot_short$Situation[grepl("UG_S", data_plot_short$Situation)] <- "Proposer"
data_plot_short$Situation[grepl("UG_R", data_plot_short$Situation)] <- "Responder"
data_plot_short$Situation[grepl("TG_S", data_plot_short$Situation)] <- "Sender"
data_plot_short$Situation[grepl("TG_R", data_plot_short$Situation)] <- "Receiver"

data_plot_short$Situation <- factor(data_plot_short$Situation, levels = c("Proposer", "Responder", "Sender", "Receiver", "PD", "SH", "C"))
data_plot_short$PlotNumber <- factor(data_plot_short$PlotNumber, levels = c("Fairness (UG)", "Trust (TG)", "Cooperation (PD/SH)", "Coordination (C)"))

## Bootstrapped CI for main plot 1 (game results)
df_BS_to_sample.opaque <- data_plot %>% filter(Case == "Opaque")
df_BS_to_sample.human_r <- data_plot %>% filter(Case == "AgainstHumanRandom")
df_BS_to_sample.human_d <- data_plot %>% filter(Case == "AgainstHumanDelegation")
df_BS_to_sample.AI_r <- data_plot %>% filter(Case == "AgainstAIRandom")
df_BS_to_sample.AI_d <- data_plot %>% filter(Case == "AgainstAIDelegation")

df_BS.opaque <- data.frame(matrix(ncol = 7, nrow = BS, dimnames=list(c(), c("UG_P_BS_mean","UG_R_BS_mean", "TG_R_BS_mean","TG_S_BS_mean", "PD_BS_mean", "SH_BS_mean", "C_BS_mean"))))
df_BS.human_r <- data.frame(matrix(ncol = 7, nrow = BS, dimnames=list(c(), c("UG_P_BS_mean","UG_R_BS_mean", "TG_R_BS_mean","TG_S_BS_mean", "PD_BS_mean", "SH_BS_mean", "C_BS_mean"))))
df_BS.human_d <- data.frame(matrix(ncol = 7, nrow = BS, dimnames=list(c(), c("UG_P_BS_mean","UG_R_BS_mean", "TG_R_BS_mean","TG_S_BS_mean", "PD_BS_mean", "SH_BS_mean", "C_BS_mean"))))
df_BS.AI_r <- data.frame(matrix(ncol = 7, nrow = BS, dimnames=list(c(), c("UG_P_BS_mean","UG_R_BS_mean", "TG_R_BS_mean", "TG_S_BS_mean", "PD_BS_mean", "SH_BS_mean", "C_BS_mean"))))
df_BS.AI_d <- data.frame(matrix(ncol = 7, nrow = BS, dimnames=list(c(), c("UG_P_BS_mean","UG_R_BS_mean", "TG_R_BS_mean", "TG_S_BS_mean", "PD_BS_mean", "SH_BS_mean", "C_BS_mean"))))


for(i in 1:BS){
  df_BS.opaque[i,] <- sapply(data.frame(sapply(df_BS_to_sample.opaque[,4:10], function(x) sample(x, length(x), replace = TRUE))),mean)
  df_BS.human_r[i,] <- sapply(data.frame(sapply(df_BS_to_sample.human_r[,4:10], function(x) sample(x, length(x), replace = TRUE))),mean)
  df_BS.human_d[i,] <- sapply(data.frame(sapply(df_BS_to_sample.human_d[,4:10], function(x) sample(x, length(x), replace = TRUE))),mean)
  df_BS.AI_r[i,] <- sapply(data.frame(sapply(df_BS_to_sample.AI_r[,4:10], function(x) sample(x, length(x), replace = TRUE))),mean)
  df_BS.AI_d[i,] <- sapply(data.frame(sapply(df_BS_to_sample.AI_d[,4:10], function(x) sample(x, length(x), replace = TRUE))),mean)
  
}

df_BS.opaque.CI <- data.frame(matrix(ncol = 5, nrow = 7, dimnames=list(c(), c("Game","mean","CI_upper","CI_lower","sd"))))
df_BS.opaque.CI$Situation <- c("UG_P","UG_R", "TG_R","TG_S", "PD", "SH", "C")

df_BS.opaque.CI[,2] <- sapply(df_BS.opaque,mean)
df_BS.opaque.CI[,3] <- sapply(df_BS.opaque,function(x) quantile(x,0.975))
df_BS.opaque.CI[,4] <- sapply(df_BS.opaque,function(x) quantile(x,0.025))
df_BS.opaque.CI[,5] <- sapply(df_BS.opaque, sd)

df_BS.opaque.CI$Case <- "Unknown"


df_BS.human_r.CI <- data.frame(matrix(ncol = 5, nrow = 7, dimnames=list(c(), c("Game","mean", "CI_upper","CI_lower", "sd"))))
df_BS.human_r.CI$Situation <- c("UG_P","UG_R", "TG_R","TG_S", "PD", "SH", "C")

df_BS.human_r.CI[,2] <- sapply(df_BS.human_r,mean)
df_BS.human_r.CI[,3] <- sapply(df_BS.human_r,function(x) quantile(x,0.975))
df_BS.human_r.CI[,4] <- sapply(df_BS.human_r,function(x) quantile(x,0.025))
df_BS.human_r.CI[,5] <- sapply(df_BS.human_r, sd)

df_BS.human_r.CI$Case <- "Human_R"


df_BS.human_d.CI <- data.frame(matrix(ncol = 5, nrow = 7, dimnames=list(c(), c("Game","mean", "CI_upper","CI_lower", "sd"))))
df_BS.human_d.CI$Situation <- c("UG_P","UG_R", "TG_R","TG_S", "PD", "SH", "C")

df_BS.human_d.CI[,2] <- sapply(df_BS.human_d,mean)
df_BS.human_d.CI[,3] <- sapply(df_BS.human_d,function(x) quantile(x,0.975))
df_BS.human_d.CI[,4] <- sapply(df_BS.human_d,function(x) quantile(x,0.025))
df_BS.human_d.CI[,5] <- sapply(df_BS.human_d, sd)

df_BS.human_d.CI$Case <- "Human_D"


df_BS.AI_r.CI <- data.frame(matrix(ncol = 5, nrow = 7, dimnames=list(c(), c("Game","mean", "CI_upper","CI_lower", "sd"))))
df_BS.AI_r.CI$Situation <- c("UG_P","UG_R", "TG_R","TG_S", "PD", "SH", "C")

df_BS.AI_r.CI[,2] <- sapply(df_BS.AI_r,mean)
df_BS.AI_r.CI[,3] <- sapply(df_BS.AI_r,function(x) quantile(x,0.975))
df_BS.AI_r.CI[,4] <- sapply(df_BS.AI_r,function(x) quantile(x,0.025))
df_BS.AI_r.CI[,5] <- sapply(df_BS.AI_r, sd)

df_BS.AI_r.CI$Case <- "AI_R"


df_BS.AI_d.CI <- data.frame(matrix(ncol = 5, nrow = 7, dimnames=list(c(), c("Game","mean", "CI_upper","CI_lower", "sd"))))
df_BS.AI_d.CI$Situation <- c("UG_P","UG_R", "TG_R","TG_S", "PD", "SH", "C")

df_BS.AI_d.CI[,2] <- sapply(df_BS.AI_d,mean)
df_BS.AI_d.CI[,3] <- sapply(df_BS.AI_d,function(x) quantile(x,0.975))
df_BS.AI_d.CI[,4] <- sapply(df_BS.AI_d,function(x) quantile(x,0.025))
df_BS.AI_d.CI[,5] <- sapply(df_BS.AI_d, sd)

df_BS.AI_d.CI$Case <- "AI_D"


df_BS.CI <- full_join(df_BS.opaque.CI, df_BS.human_r.CI)
df_BS.CI <- full_join(df_BS.CI, df_BS.human_d.CI)
df_BS.CI <- full_join(df_BS.CI, df_BS.AI_r.CI)
df_BS.CI <- full_join(df_BS.CI, df_BS.AI_d.CI)

df_BS.CI <- df_BS.CI %>% select(-c(mean))

df_BS.CI$Situation[grepl("UG_P", df_BS.CI$Situation)] <- "Proposer"
df_BS.CI$Situation[grepl("UG_R", df_BS.CI$Situation)] <- "Responder"
df_BS.CI$Situation[grepl("TG_S", df_BS.CI$Situation)] <- "Sender"
df_BS.CI$Situation[grepl("TG_R", df_BS.CI$Situation)] <- "Receiver"

data_plot_short <- full_join(data_plot_short, df_BS.CI)

data_plot_short$Situation <- factor(data_plot_short$Situation, levels = c("Proposer", "Responder", "Sender", "Receiver", "PD", "SH", "C"))
data_plot_short$PlotNumber <- factor(data_plot_short$PlotNumber, levels = c("Fairness (UG)", "Trust (TG)", "Cooperation (PD/SH)", "Coordination (C)"))

# Preparation plot 2 (delegation)
df_plot_delegation <- df %>% filter(Treatment != "TransparentRandom" & Scenario == "AISupport")

df_plot_delegation <- df_plot_delegation %>% rename("UG Proposer_Delegation" = "UGProposer_Delegation", "UG Responder_Delegation" = "UGResponder_Delegation", "TG Sender_Delegation" = "TGSender_Delegation", "TG Receiver_Delegation" = "TGReceiver_Delegation")


df_plot_delegation <- df_plot_delegation %>% select(Treatment, SubjectID, "UG Proposer_Delegation":C_Delegation) %>% pivot_longer("UG Proposer_Delegation":C_Delegation, names_to = "Game", values_to = "Delegation")

df_plot_delegation$Game <- gsub("_Delegation", "", df_plot_delegation$Game)

df_BS_to_sample.delegation <- df_plot_delegation 

df_plot_delegation <- df_plot_delegation %>% group_by(Treatment, Game) %>% mutate(N_Total = n()) %>% group_by(Treatment, Game, Delegation) %>% summarise(n = n(), Share = n / N_Total) %>% filter(row_number() == 1)

df_plot_delegation$Game <- factor(df_plot_delegation$Game, levels = c("UG Proposer", "UG Responder", "TG Sender", "TG Receiver", "PD", "SH", "C"))

df_plot_delegation$Treatment[df_plot_delegation$Treatment == "OpaqueDelegation"] <- "Opaque"
df_plot_delegation$Treatment[df_plot_delegation$Treatment == "TransparentDelegation"] <- "Transparent"

## Bootstrapped CI for plot 2 (delegation)
df_BS_to_sample.delegation_opaque      <- df_BS_to_sample.delegation %>% filter(Treatment == "OpaqueDelegation") %>% spread(Game, Delegation) %>% select(c("Treatment", "SubjectID", "UG Proposer", "UG Responder", "TG Sender", "TG Receiver", "PD", "SH", "C"))
df_BS_to_sample.delegation_transparent <- df_BS_to_sample.delegation %>% filter(Treatment == "TransparentDelegation") %>% spread(Game, Delegation) %>% select(c("Treatment", "SubjectID", "UG Proposer", "UG Responder", "TG Sender", "TG Receiver", "PD", "SH", "C"))

df_BS.delegation_opaque <- data.frame(matrix(ncol = 7, nrow = BS, dimnames=list(c(), c("UG_P_BS_mean","UG_R_BS_mean", "TG_S_BS_mean", "TG_R_BS_mean", "PD_BS_mean", "SH_BS_mean", "C_BS_mean"))))
df_BS.delegation_transparent <- data.frame(matrix(ncol = 7, nrow = BS, dimnames=list(c(), c("UG_P_BS_mean","UG_R_BS_mean", "TG_S_BS_mean", "TG_R_BS_mean", "PD_BS_mean", "SH_BS_mean", "C_BS_mean"))))

for(i in 1:BS){
  df_BS.delegation_opaque[i,] <- sapply(data.frame(sapply(df_BS_to_sample.delegation_opaque[,3:9], function(x) sample(x, length(x), replace = TRUE))),mean)
  df_BS.delegation_transparent[i,] <- sapply(data.frame(sapply(df_BS_to_sample.delegation_transparent[,3:9], function(x) sample(x, length(x), replace = TRUE))),mean)
}

df_BS.delegation_opaque.CI <- data.frame(matrix(ncol = 4, nrow = 7, dimnames=list(c(), c("Game","mean", "CI_upper","CI_lower"))))
df_BS.delegation_opaque.CI$Game <- c("UG Proposer", "UG Responder", "TG Sender", "TG Receiver", "PD", "SH", "C")

df_BS.delegation_opaque.CI[,2] <- sapply(df_BS.delegation_opaque,mean)
df_BS.delegation_opaque.CI[,3] <- sapply(df_BS.delegation_opaque,function(x) quantile(x,0.975))
df_BS.delegation_opaque.CI[,4] <- sapply(df_BS.delegation_opaque,function(x) quantile(x,0.025))

df_BS.delegation_opaque.CI$Treatment <- "Opaque"

df_BS.delegation_transparent.CI <- data.frame(matrix(ncol = 4, nrow = 7, dimnames=list(c(), c("Game","mean", "CI_upper","CI_lower"))))
df_BS.delegation_transparent.CI$Game <- c("UG Proposer", "UG Responder", "TG Sender", "TG Receiver", "PD", "SH", "C")

df_BS.delegation_transparent.CI[,2] <- sapply(df_BS.delegation_transparent,mean)
df_BS.delegation_transparent.CI[,3] <- sapply(df_BS.delegation_transparent,function(x) quantile(x,0.975))
df_BS.delegation_transparent.CI[,4] <- sapply(df_BS.delegation_transparent,function(x) quantile(x,0.025))

df_BS.delegation_transparent.CI$Treatment <- "Transparent"

df_BS.CI_delegation <- full_join(df_BS.delegation_opaque.CI, df_BS.delegation_transparent.CI)

df_plot_delegation <- full_join(df_plot_delegation, df_BS.CI_delegation) %>% filter(Delegation == 1)

## Preparation plot 3 (turing)
df_t_plot <- df_t 

df_t_plot$Game <- df_t$Situation

df_t_plot <- df_t_plot %>% group_by(Game, Decision) %>% mutate(n = n(), MeanRatingPerSituation = mean(RatingIsAI_adjusted)) %>%
  select(SubjectID, Treatment, Decision, Game, RatingIsAI_adjusted, MeanRatingPerSituation, n) 

df_t_plot_short <- df_t_plot %>% group_by(Game, Decision) %>% filter(row_number() == 1) %>% select(Game, MeanRatingPerSituation, n)

df_t_plot_short$Game <- factor(df_t_plot_short$Game, levels = c("UG_P", "UG_R", "TG_S", "TG_R", "PD", "SH", "C"))

## Bootstrapped CI for plot 3 (turing)
df_BS_to_sample.behavior <- df_t_plot %>% filter(Decision == "Behavior") %>% select(SubjectID, Game, Decision, RatingIsAI_adjusted) %>% spread(Game, RatingIsAI_adjusted) %>% 
  select(SubjectID, Decision, UG_P, UG_R, TG_S, TG_R, PD, SH, C)
df_BS_to_sample.justification <- df_t_plot %>% filter(Decision == "Justification") %>% select(SubjectID, Game, Decision, RatingIsAI_adjusted) %>% spread(Game, RatingIsAI_adjusted) %>%
  select(SubjectID, Decision, UG_P, UG_R, TG_S, TG_R, PD, SH, C)

df_BS.behavior <- data.frame(matrix(ncol = 7, nrow = BS, dimnames=list(c(), c("UG_P_BS_mean","UG_R_BS_mean", "TG_S_BS_mean", "TG_R_BS_mean", "PD_BS_mean", "SH_BS_mean", "C_BS_mean"))))
df_BS.justification <- data.frame(matrix(ncol = 7, nrow = BS, dimnames=list(c(), c("UG_P_BS_mean","UG_R_BS_mean", "TG_S_BS_mean", "TG_R_BS_mean", "PD_BS_mean", "SH_BS_mean", "C_BS_mean"))))

for(i in 1:BS){
  df_BS.behavior[i,] <- sapply(data.frame(sapply(df_BS_to_sample.behavior[,3:9], function(x) sample(x, length(x), replace = TRUE))),mean)
  df_BS.justification[i,] <- sapply(data.frame(sapply(df_BS_to_sample.justification[,3:9], function(x) sample(x, length(x), replace = TRUE))),mean)
}

df_BS.behavior.CI <- data.frame(matrix(ncol = 4, nrow = 7, dimnames=list(c(), c("Game","mean", "CI_upper","CI_lower"))))
df_BS.behavior.CI$Game <- c("UG_P","UG_R","TG_S", "TG_R", "PD", "SH", "C")

df_BS.behavior.CI[,2] <- sapply(df_BS.behavior,mean)
df_BS.behavior.CI[,3] <- sapply(df_BS.behavior,function(x) quantile(x,0.975))
df_BS.behavior.CI[,4] <- sapply(df_BS.behavior,function(x) quantile(x,0.025))

df_BS.behavior.CI$Decision <- "Behavior"

df_BS.justification.CI <- data.frame(matrix(ncol = 4, nrow = 7, dimnames=list(c(), c("Game","mean", "CI_upper","CI_lower"))))
df_BS.justification.CI$Game <- c("UG_P","UG_R","TG_S", "TG_R", "PD", "SH", "C")

df_BS.justification.CI[,2] <- sapply(df_BS.justification,mean)
df_BS.justification.CI[,3] <- sapply(df_BS.justification,function(x) quantile(x,0.975))
df_BS.justification.CI[,4] <- sapply(df_BS.justification,function(x) quantile(x,0.025))

df_BS.justification.CI$Decision <- "Justification"

df_BS.CI2 <- full_join(df_BS.behavior.CI, df_BS.justification.CI)

df_t_plot_short <- full_join(df_t_plot_short, df_BS.CI2)

df_t_plot_short$Game2 <- df_t_plot_short$Game

df_t_plot_short$Game2[grepl("UG_P", df_t_plot_short$Game2)] <- "UG Proposer"
df_t_plot_short$Game2[grepl("UG_R", df_t_plot_short$Game2)] <- "UG Responder"
df_t_plot_short$Game2[grepl("TG_S", df_t_plot_short$Game2)] <- "TG Sender"
df_t_plot_short$Game2[grepl("TG_R", df_t_plot_short$Game2)] <- "TG Receiver"

df_t_plot_short$Game2 <- factor(df_t_plot_short$Game2, levels = c("UG Proposer", "UG Responder", "TG Sender", "TG Receiver", "PD", "SH", "C"))

################################################################################
# Figure 1
################################################################################
color_scheme1 <- c("#B8BCC1", "#73787E", "#71D1CC", "#067E79", "#59C7EB")
top_margin <- 0.5
text_size <- 3
scaleTG <- 4

data_plot_short$Case[data_plot_short$Case == "AI_D"] <- "AI (D)"
data_plot_short$Case[data_plot_short$Case == "AI_R"] <- "AI (R)"
data_plot_short$Case[data_plot_short$Case == "Human_D"] <- "Human (D)"
data_plot_short$Case[data_plot_short$Case == "Human_R"] <- "Human (R)"
data_plot_short$mean_barTG[data_plot_short$PlotNumber == "Trust (TG)"] <- data_plot_short$mean[data_plot_short$PlotNumber == "Trust (TG)"]
data_plot_short$CIupper_barTG[data_plot_short$PlotNumber == "Trust (TG)"] <- data_plot_short$CI_upper[data_plot_short$PlotNumber == "Trust (TG)"]
data_plot_short$CIlower_barTG[data_plot_short$PlotNumber == "Trust (TG)"] <- data_plot_short$CI_lower[data_plot_short$PlotNumber == "Trust (TG)"]
data_plot_short$mean_barTG[data_plot_short$Situation == "Sender"] <- data_plot_short$mean_barTG[data_plot_short$Situation == "Sender"]* scaleTG
data_plot_short$CIupper_barTG[data_plot_short$Situation == "Sender"] <- data_plot_short$CIupper_barTG[data_plot_short$Situation == "Sender"]* scaleTG
data_plot_short$CIlower_barTG[data_plot_short$Situation == "Sender"] <- data_plot_short$CIlower_barTG[data_plot_short$Situation == "Sender"]* scaleTG

TG <- ggplot(data = subset(data_plot_short, PlotNumber == "Trust (TG)")) +
  geom_bar(aes(x = Situation, y = mean_barTG, fill = Case, group = Case), stat = "identity", position = position_dodge2(width = 0.9)) +
  geom_text(aes(x = Situation, y = mean_barTG, group = Case, label = sprintf("%0.2f", round(mean, digits = 2))), position = position_dodge2(width = 0.9), vjust = 5, colour = "black", size = text_size) +
  geom_errorbar(aes(x = Situation, y = mean_barTG, ymin = CIupper_barTG, ymax = CIlower_barTG, group = Case),width = 0.5, position = position_dodge(width = 0.9)) +
  #geom_segment(x=0.5, xend=1.5, y=4, yend=4, linetype = 2) +
  #geom_segment(x=1.5, xend=2.5, y=4, yend=4, linetype = 2) +
  scale_fill_manual("Interaction partner", values=color_scheme1) +
  scale_y_continuous(limits = c(0,4), breaks=c(0.0, 2.0, 4.0), name = "Trust", labels=function(x)sprintf("%.1f", x/scaleTG), sec.axis = sec_axis(~. *1, name = "Trustworthiness", breaks=c(0.0, 2.0, 4.0), labels=function(x)sprintf("%.1f", x)),  expand = c(0,0)) +
  labs(title = bquote(~~~~ bold('A') ~~ "Trust (TG)"), x = "", y = "") +
  theme_classic() +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_line( size=.1, color="grey" ),
        panel.grid.major.y = element_line( size=.1, color="grey" ),
        legend.position ="bottom",
        plot.title.position = "plot",
        plot.margin = unit(c(top_margin, 0, 0, 0), "cm"), 
        text = element_text(size=16),
        plot.title = element_text(size=16))

PD_SH <- ggplot(data = subset(data_plot_short, PlotNumber == "Cooperation (PD/SH)")) +
  geom_bar(aes(x = Situation, y = mean, fill = Case, group = Case), stat = "identity", position = position_dodge2(width = 0.9)) +
  geom_text(aes(x = Situation, y = mean, group = Case, label = sprintf("%0.2f", round(mean, digits = 2))), position = position_dodge2(width = 0.9), vjust = 5, colour = "black", size = text_size) +
  geom_errorbar(aes(x = Situation, y = mean, ymin = CI_lower, ymax = CI_upper, group = Case), width = 0.5, position = position_dodge(width = 0.9)) +
  #geom_hline(yintercept = 1, linetype = "dashed") +
  scale_fill_manual("Interaction partner", values=color_scheme1) +
  scale_y_continuous(limits = c(0,1), breaks=c(0, 0.5, 1), expand = c(0,0)) +
  labs(title = bquote(~~~~ bold('B') ~~ "Cooperation (PD/SH)"), x = "", y = "") +
  theme_classic() +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_line( size=.1, color="grey" ),
        panel.grid.major.y = element_line( size=.1, color="grey" ),
        legend.position ="bottom",
        plot.title.position = "plot",
        plot.margin = unit(c(top_margin, 0, 0, 0), "cm"), 
        text = element_text(size=16),
        plot.title = element_text(size=16))

C <- ggplot(data = subset(data_plot_short, PlotNumber == "Coordination (C)")) +
  geom_bar(aes(x = Situation, y = mean, fill = Case, group = Case), stat = "identity", position = position_dodge2(width = 0.9), width = 1) +
  geom_text(aes(x = Situation, y = mean, group = Case, label = sprintf("%0.2f", round(mean, digits = 2))), position = position_dodge2(width = 1), vjust = 5, colour = "black", size = text_size) +
  geom_errorbar(aes(x = Situation, y = mean, ymin = CI_lower, ymax = CI_upper, group = Case), width = 0.5, position = position_dodge(width = 1)) +
  #geom_hline(yintercept = 1, linetype = "dashed") +
  scale_fill_manual("Interaction partner", values=color_scheme1) +
  scale_y_continuous(limits = c(0,1), breaks=c(0, 0.5, 1), expand = c(0,0)) +
  labs(title = bquote(~~~~ bold('C') ~~ "Coordination (C)"), x = "", y = "") +
  theme_classic() +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_line( size=.1, color="grey" ),
        panel.grid.major.y = element_line( size=.1, color="grey" ),
        legend.position ="bottom",
        plot.title.position = "plot",
        plot.margin = unit(c(top_margin, 0, 0, 0), "cm"), 
        text = element_text(size=16),
        plot.title = element_text(size=16))

UG <- ggplot(data = subset(data_plot_short, PlotNumber == "Fairness (UG)")) +
  geom_bar(aes(x = Situation, y = mean, fill = Case, group = Case), stat = "identity", position = position_dodge2(width = 0.9)) +
  geom_text(aes(x = Situation, y = mean, group = Case, label = sprintf("%0.2f", round(mean, digits = 2))), position = position_dodge2(width = 0.9), vjust = 5, colour = "black", size = text_size) +
  geom_errorbar(aes(x = Situation, y = mean, ymin = CI_lower, ymax = CI_upper, group = Case), width = 0.5, position = position_dodge(width = 0.9)) +
  #geom_hline(yintercept = 5, linetype = "dashed") +
  scale_fill_manual("Interaction partner", values=color_scheme1) +
  scale_y_continuous(limits = c(0,5), breaks=c(0, 2.5, 5), expand = expansion(mult = c(0, 0))) +
  labs(title = bquote(~~~~ bold('D') ~~ "Fairness (UG)"), x = "", y = "") +
  theme_classic() +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_line( size=.1, color="grey" ),
        panel.grid.major.y = element_line( size=.1, color="grey" ),
        legend.position ="bottom", 
        plot.title.position = "plot",
        plot.margin = unit(c(top_margin, 0, 0, 0), "cm"), 
        text = element_text(size=16),
        plot.title = element_text(size=16))

Figure_1 <- ggarrange(TG, PD_SH, C, UG, ncol = 4, nrow = 1,
                      common.legend = TRUE,
                      legend="bottom", widths = c(1,1,0.5,1),
                      font.label=list(size=15))

# save plot
ggsave(file="./figures/Figure_1.pdf", Figure_1, width = 15.5, height = 4.5)

################################################################################
# Figure 3
################################################################################
color_scheme2 <- c("#6AAAB7", "#035F72")
text_size2 <- 4.5

df_plot_delegation$Game <- factor(df_plot_delegation$Game, levels = c("UG Proposer", "UG Responder", "TG Sender", "TG Receiver", "PD", "SH", "C"))

DelegationPlot <- ggplot(data = subset(df_plot_delegation, Delegation == 1)) +
  geom_bar(aes(x = Game, y = Share, fill = Treatment, group = Treatment), stat = "identity", position = position_dodge2(width = 0.9)) +
  geom_text(aes(x = Game, y = Share, label = sprintf("%0.2f", round(Share, digits = 2))), position = position_dodge2(width = 0.9), hjust = 2, colour = "white", size = text_size2) +
  geom_errorbar(aes(x = Game, y = mean, ymin = CI_lower, ymax = CI_upper, group = Treatment), width = 0.5, position = position_dodge(width = 0.9)) +
  coord_flip() +
  scale_fill_manual(values=color_scheme2) +
  labs(title = "Share of Delegation \n", x = "", y = "") +
  scale_x_discrete(limits = rev) +
  scale_y_continuous(limits = c(0,0.8)) +
  guides(fill = guide_legend(reverse = TRUE, nrow = 2)) +
  theme_classic() +
  theme(panel.grid.minor.x = element_line( size=.1, color="grey" ),
        panel.grid.major.x = element_line( size=.1, color="grey" ),
        panel.grid.major.y = element_blank(),
        legend.position="bottom", text = element_text(size=16))

## Mean Rating in opaque by game & decision
color_scheme3 <- c("#586BA4", "#324376")

TuringPlot <- df_t_plot_short %>%  ggplot() +
  geom_bar(aes(x = Game2, y = MeanRatingPerSituation, fill = Decision, group = Decision), position = position_dodge2(width = 0.9), stat = "identity") +
  geom_text(aes(x = Game2, y = MeanRatingPerSituation, label = sprintf("%0.2f", round(MeanRatingPerSituation, digits = 2))), hjust = 3.38, colour = "white", position = position_dodge2(width = 0.9), size = text_size2) +
  geom_errorbar(aes(x = Game2, y = MeanRatingPerSituation, ymin = CI_lower, ymax = CI_upper, group = Decision), width = 0.5, position = position_dodge(width = 0.9)) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  scale_fill_manual(values=color_scheme3) +
  coord_flip() +
  scale_x_discrete(limits = rev) +
  scale_y_continuous(limits = c(0,0.8)) +
  labs(title = "Share of correctly \nidentified AI decisions", x = "", y = "") +
  guides(fill = guide_legend(reverse = TRUE, nrow = 2)) +
  theme_classic()+
  theme(panel.grid.minor.x = element_line( size=.1, color="grey" ),
        panel.grid.major.x = element_line( size=.1, color="grey" ),
        panel.grid.major.y = element_blank(),
        legend.position="bottom", text = element_text(size=16))

Figure3 <- ggarrange(DelegationPlot +
                  theme(plot.margin = margin(r = 1) ), 
                TuringPlot + 
                  theme(plot.margin = margin(l = 1, r = 1) ),
                nrow = 1,
                labels= c("A", "B"),
                font.label=list(size=18))

ggsave(file="./figures/Figure_3.pdf", Figure3, width = 9, height = 6)

################################################################################
# Figure SM1
################################################################################
color_scheme1 <- c("#9AA0A7", "#c994c7")
top_margin <- 0.5
text_size <- 5
scaleTG <- 4

data_plot_short <- data.frame(
  PlotNumber = factor(c(rep("Coordination (C)", 2), rep("Trust (TG)", 4), rep("Fairness (UG)", 4), rep("Cooperation (PD/SH)", 4))),
  Situation = factor(rep(c( "C", "Sender", "Receiver", "Proposer", "Responder", "PD", "SH"), each=2), levels=c( "C", "Sender", "Receiver", "Proposer", "Responder", "PD", "SH")),
  Case = factor(rep(c("personalized", "unpersonalized"), 7), levels=c("unpersonalized","personalized")),
  mean = c(sapply(names(decisions_list), function(x) c(mean(decisions_list[[x]][Type=="personalized"]), mean(decisions_list[[x]][Type=="unpersonalized"])))),
  CI_lower = c(sapply(names(decisions_list), function(x) c(quantile(sapply(1:BS, function(y) mean(sample(decisions_list[[x]][Type=="personalized"], replace=TRUE))), 0.025),
                                                           quantile(sapply(1:BS, function(y) mean(sample(decisions_list[[x]][Type=="unpersonalized"], replace=TRUE))), 0.025)))),
  CI_upper = c(sapply(names(decisions_list), function(x) c(quantile(sapply(1:BS, function(y) mean(sample(decisions_list[[x]][Type=="personalized"], replace=TRUE))), 0.975),
                                                           quantile(sapply(1:BS, function(y) mean(sample(decisions_list[[x]][Type=="unpersonalized"], replace=TRUE))), 0.975))))
)
data_plot_short$mean_barTG[data_plot_short$PlotNumber == "Trust (TG)"] <- data_plot_short$mean[data_plot_short$PlotNumber == "Trust (TG)"]
data_plot_short$CIlower_barTG[data_plot_short$PlotNumber == "Trust (TG)"] <- data_plot_short$CI_lower[data_plot_short$PlotNumber == "Trust (TG)"]
data_plot_short$CIupper_barTG[data_plot_short$PlotNumber == "Trust (TG)"] <- data_plot_short$CI_upper[data_plot_short$PlotNumber == "Trust (TG)"]
data_plot_short$mean_barTG[data_plot_short$Situation == "Sender"] <- data_plot_short$mean[data_plot_short$Situation == "Sender"]* scaleTG
data_plot_short$CIlower_barTG[data_plot_short$Situation == "Sender"] <- data_plot_short$CI_lower[data_plot_short$Situation == "Sender"]* scaleTG
data_plot_short$CIupper_barTG[data_plot_short$Situation == "Sender"] <- data_plot_short$CI_upper[data_plot_short$Situation == "Sender"]* scaleTG

TG_plot <- ggplot(data = subset(data_plot_short, PlotNumber == "Trust (TG)")) +
  geom_bar(aes(x = Situation, y = mean_barTG, fill = Case, group = Case), stat = "identity", position = position_dodge2(width = 0.9)) +
  geom_text(aes(x = Situation, y = mean_barTG, group = Case, label = sprintf("%0.2f", round(mean, digits = 2))), position = position_dodge2(width = 0.9), vjust = 4, colour = "black", size = text_size) +
  geom_errorbar(aes(x = Situation, y = mean_barTG, ymin = CIupper_barTG, ymax = CIlower_barTG, group = Case),width = 0.5, position = position_dodge(width = 0.9)) +
  geom_segment(x=0.5, xend=1.5, y=4, yend=4, linetype = 2) +
  geom_segment(x=1.5, xend=2.5, y=4, yend=4, linetype = 2) +
  scale_fill_manual("AI type", values=color_scheme1) +
  scale_y_continuous(limits = c(0,4), breaks=c(0.0, 2.0, 4.0), name = "Trust", 
                     labels=function(x)sprintf("%.1f", x/scaleTG), 
                     sec.axis = sec_axis(~. *1, name = "Trustworthiness", 
                                         breaks=c(0.0, 2.0, 4.0), 
                                         labels=function(x)sprintf("%.1f", x)),  
                     expand = c(0,0)) +
  labs(title = bquote(~~~~ bold('A') ~~ "Trust (TG)"), x = "", y = "") +
  theme_classic() +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_line( size=.1, color="grey" ),
        panel.grid.major.y = element_line( size=.1, color="grey" ),
        legend.position ="bottom",
        plot.title.position = "plot",
        plot.margin = unit(c(top_margin, 0, 0, 0), "cm"), 
        text = element_text(size=16),
        plot.title = element_text(size=16))

PD_SH_plot <- ggplot(data = subset(data_plot_short, PlotNumber == "Cooperation (PD/SH)")) +
  geom_bar(aes(x = Situation, y = mean, fill = Case, group = Case), stat = "identity", position = position_dodge2(width = 0.9)) +
  geom_text(aes(x = Situation, y = mean, group = Case, label = sprintf("%0.2f", round(mean, digits = 2))), position = position_dodge2(width = 0.9), vjust = 5, colour = "black", size = text_size) +
  geom_errorbar(aes(x = Situation, y = mean, ymin = CI_lower, ymax = CI_upper, group = Case), width = 0.5, position = position_dodge(width = 0.9)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  scale_fill_manual("AI type", values=color_scheme1) +
  scale_y_continuous(limits = c(0,1), breaks=c(0, 0.5, 1), expand = c(0,0)) +
  labs(title = bquote(~~~~ bold('B') ~~ "Cooperation (PD/SH)"), x = "", y = "") +
  theme_classic() +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_line( size=.1, color="grey" ),
        panel.grid.major.y = element_line( size=.1, color="grey" ),
        legend.position ="bottom",
        plot.title.position = "plot",
        plot.margin = unit(c(top_margin, 0, 0, 0), "cm"), 
        text = element_text(size=16),
        plot.title = element_text(size=16))

C_plot <- ggplot(data = subset(data_plot_short, PlotNumber == "Coordination (C)")) +
  geom_bar(aes(x = Situation, y = mean, fill = Case, group = Case), stat = "identity", 
           position = position_dodge(width = 1), width = 1) +
  geom_text(aes(x = Situation, y = mean, group = Case, 
                label = sprintf("%0.2f", round(mean, digits = 2))), 
            position = position_dodge2(width = 1), vjust = -3, 
            colour = "black", size = text_size) +
  geom_errorbar(aes(x = Situation, y = mean, ymin = CI_lower, ymax = CI_upper, 
                    group = Case), width = 0.5, position = position_dodge(width = 1)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  scale_fill_manual("AI type", values=color_scheme1) +
  scale_y_continuous(limits = c(0,1), breaks=c(0, 0.5, 1), expand = c(0,0)) +
  labs(title = bquote(~~~~ bold('C') ~~ "Coordination (C)"), x = "", y = "") +
  theme_classic() +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_line( size=.1, color="grey" ),
        panel.grid.major.y = element_line( size=.1, color="grey" ),
        legend.position ="bottom",
        plot.title.position = "plot",
        plot.margin = unit(c(top_margin, 0, 0, 0), "cm"), 
        text = element_text(size=16),
        plot.title = element_text(size=16))

UG_plot <- ggplot(data = subset(data_plot_short, PlotNumber == "Fairness (UG)")) +
  geom_bar(aes(x = Situation, y = mean, fill = Case, group = Case), stat = "identity", position = position_dodge2(width = 0.9)) +
  geom_text(aes(x = Situation, y = mean, group = Case, label = sprintf("%0.2f", round(mean, digits = 2))), position = position_dodge2(width = 0.9), vjust = c(5,5,-3,-3), colour = "black", size = text_size) +
  geom_errorbar(aes(x = Situation, y = mean, ymin = CI_lower, ymax = CI_upper, group = Case), width = 0.5, position = position_dodge(width = 0.9)) +
  geom_hline(yintercept = 5, linetype = "dashed") +
  scale_fill_manual("AI type", values=color_scheme1) +
  scale_y_continuous(limits = c(0,5.2), breaks=c(0, 2.5, 5), expand = expansion(mult = c(0, 0))) +
  labs(title = bquote(~~~~ bold('D') ~~ "Fairness (UG)"), x = "", y = "") +
  theme_classic() +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_line( size=.1, color="grey" ),
        panel.grid.major.y = element_line( size=.1, color="grey" ),
        legend.position ="bottom",
        plot.title.position = "plot",
        plot.margin = unit(c(top_margin, 0, 0, 0), "cm"), 
        text = element_text(size=16),
        plot.title = element_text(size=16))
Figure_SM1 <- ggarrange(TG_plot, PD_SH_plot, C_plot, UG_plot, ncol = 4, nrow = 1,
                        widths = c(1,1,0.5,1),
                        common.legend = TRUE, legend = "bottom",
                        font.label=list(size=15))

# save figure
ggsave("./figures/Figure_SM1.pdf", Figure_SM1, width = 15.5, height = 4.5)

################################################################################
# Figure SM2
################################################################################
sbjs_decisions_list <- list(C=df_short$C, Sender=df_short$TGSender, 
                            Receiver=df_short$TGReceiver, Proposer = df_short$UGProposer,
                            Responder=df_short$UGResponder, PD=df_short$PD,
                            SH=df_short$SH)

Q1 = as.numeric(df_short$IntuitionThoughtfulness) - 1
Q2 = as.numeric(df_short$IntroversionExtraverison) - 1
Q3 = as.numeric(df_short$FairnessEfficiency) - 1
Q4 = as.numeric(df_short$ChaosBoredom) - 1
Q5 = as.numeric(df_short$SelfishnessAltruism) - 1
Q6 = as.numeric(df_short$NoveltyReliability) - 1
Q7 = as.numeric(df_short$TruthHarmony) - 1
AI_flavour <- 64*Q1 + 32*Q2 + 16*Q3 + 8*Q4 + 4*Q5 + 2*Q6 + Q7 + 1

matches_personalized <- lapply(c("C","Sender","PD","SH"), 
                               function(x) sapply(1:nrow(df_short), 
                                                  function(y) sbjs_decisions_list[[x]][y] ==
                                                    decisions_list[[x]][AI_flavour[y]]))

matches_unpersonalized <- lapply(c("C","Sender","PD","SH"), 
                                 function(x) sapply(1:nrow(df_short), 
                                                    function(y) sbjs_decisions_list[[x]][y] ==
                                                      decisions_list[[x]][sample(129:192, 1)]))


diffs_personalized <- lapply(c("Receiver","Proposer","Responder"), 
                             function(x) sapply(1:nrow(df_short), 
                                                function(y) abs(sbjs_decisions_list[[x]][y] -
                                                                  decisions_list[[x]][AI_flavour[y]])))

diffs_unpersonalized <- lapply(c("Receiver","Proposer","Responder"), 
                               function(x) sapply(1:nrow(df_short), 
                                                  function(y) abs(sbjs_decisions_list[[x]][y] -
                                                                    decisions_list[[x]][sample(129:192, 1)])))

# plot matches
plot_data_matches <- data.frame(
  Situation = factor(rep(c("C","TG Sender","PD","SH"), each = 2), levels=c("C","TG Sender","PD","SH")),
  Case = factor(rep(c("unpersonalized", "personalized"), 4), levels=c("unpersonalized", "personalized")),
  Freq = c(sapply(1:4, function(x) c(mean(matches_unpersonalized[[x]], na.rm=TRUE), 
                                     mean(matches_personalized[[x]], na.rm=TRUE)))),
  CI_lower = c(sapply(1:4, function(x) c(quantile(sapply(1:BS, function(y) mean(sample(matches_unpersonalized[[x]], replace=TRUE), na.rm=TRUE)), 0.025), 
                                         quantile(sapply(1:BS, function(y) mean(sample(matches_personalized[[x]], replace=TRUE), na.rm=TRUE)), 0.025)))),
  CI_upper = c(sapply(1:4, function(x) c(quantile(sapply(1:BS, function(y) mean(sample(matches_unpersonalized[[x]], replace=TRUE), na.rm=TRUE)), 0.975), 
                                         quantile(sapply(1:BS, function(y) mean(sample(matches_personalized[[x]], replace=TRUE), na.rm=TRUE)), 0.975))))
)

plot_matches <- ggplot(plot_data_matches, aes(x = Situation, y = Freq, fill = Case)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_errorbar(aes(x = Situation, y = Freq, ymin = CI_lower, ymax = CI_upper, group = Case), width = 0.5, position = position_dodge(width = 0.9)) +
  labs(title = "Matches of AI and human decisions", x = "", y = "Relative Frequency") +
  scale_fill_manual(values=color_scheme1) +
  scale_y_continuous(limits = c(0,1), breaks=seq(0, 1, 0.25), expand = c(0,0)) +
  theme_classic() +
  guides(fill=guide_legend(title="AI type")) +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_line( size=.1, color="grey" ),
        panel.grid.major.y = element_line( size=.1, color="grey" ),
        legend.position ="bottom",
        plot.margin = unit(c(top_margin, 0, 0, 0), "cm"), text = element_text(size=16))

# plot continuous differences
plot_data_diffs <- data.frame(
  Situation = factor(rep(c("TG Receiver","UG Proposer","UG Responder"), each = 2), levels=c("TG Receiver","UG Proposer","UG Responder")),
  Case = factor(rep(c("unpersonalized", "personalized"), 3), levels=c("unpersonalized", "personalized")),
  Difference = c(sapply(1:3, function(x) c(mean(diffs_unpersonalized[[x]], na.rm=TRUE), 
                                           mean(diffs_personalized[[x]], na.rm=TRUE)))),
  CI_lower = c(sapply(1:3, function(x) c(quantile(sapply(1:BS, function(y) mean(sample(diffs_unpersonalized[[x]], replace=TRUE), na.rm=TRUE)), 0.025), 
                                         quantile(sapply(1:BS, function(y) mean(sample(diffs_personalized[[x]], replace=TRUE), na.rm=TRUE)), 0.025)))),
  CI_upper = c(sapply(1:3, function(x) c(quantile(sapply(1:BS, function(y) mean(sample(diffs_unpersonalized[[x]], replace=TRUE), na.rm=TRUE)), 0.975), 
                                         quantile(sapply(1:BS, function(y) mean(sample(diffs_personalized[[x]], replace=TRUE), na.rm=TRUE)), 0.975))))
)

plot_diffs <- ggplot(plot_data_diffs, aes(x = Situation, y = Difference, fill = Case)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_errorbar(aes(x = Situation, y = Difference, ymin = CI_lower, ymax = CI_upper, group = Case), width = 0.5, position = position_dodge(width = 0.9)) +
  labs(title = "Differences to human desisions", x = "", y = "Absolute Difference (ECU)") +
  scale_fill_manual(values=color_scheme1) +
  scale_y_continuous(limits = c(0,3.5), breaks=seq(0, 3.5, 1), expand = c(0,0)) +
  theme_classic() +
  guides(fill=guide_legend(title="AI type")) +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_line( size=.1, color="grey" ),
        panel.grid.major.y = element_line( size=.1, color="grey" ),
        legend.position ="bottom",
        plot.margin = unit(c(top_margin, 0, 0, 0), "cm"), text = element_text(size=16))

Figure_SM2 <- ggarrange(plot_matches, plot_diffs, nrow = 1, ncol = 2, 
                        labels =c("A","B"), common.legend = TRUE, legend="bottom",
                        font.label=list(size=15))

# save figure
ggsave("./figures/Figure_SM2.pdf", Figure_SM2, width = 12, height = 5)

################################################################################
# Payoffs simulation
################################################################################
names(df)[names(df) == "UGProposer"] <- "Proposer"
names(df)[names(df) == "UGResponder"] <- "Responder"
names(df)[names(df) == "TGReceiver"] <- "Receiver"
names(df)[names(df) == "TGSender"] <- "Sender"
names(df)[names(df) == "UGProposer_Delegation"] <- "Proposer_Delegation"
names(df)[names(df) == "UGResponder_Delegation"] <- "Responder_Delegation"
names(df)[names(df) == "TGReceiver_Delegation"] <- "Receiver_Delegation"
names(df)[names(df) == "TGSender_Delegation"] <- "Sender_Delegation"

# change C decisions back to original choices (not dummy for mode)
decisions_list$C <- gpt_data$C2

# prepare some objects
decision_cols <- names(decisions_list)
payoffs <- list()
roles <- c("NoAISupport","AISupport")
vs_scenarios <- c("AgainstAI", "AgainstHuman", "Delegation")

payoffs <- list()
# loop to calculate payoffs for all potential matches
for( L1 in c("T","O")){                                                         # transparent or opaque
  for( L2 in c("D","R")){                                                       # delegation or random
    if( L1 == "T" | (L1 == "O" & L2 == "D") ){                                  # restriction: opaque only exists with delegation
      for( L3 in c("P","U")){                                                   # personalized or unpersonalized
        T_ids <- unique(df$SubjectID[df$TreatmentCode == paste0(L1,L2,L3)])     # individual ids in treatment
        roles_list <- list()                                                    # to store results of different roles
        for( role in roles ){                                                   # loop through both roles (with/without AI support)
          VS_list <- list()                                                     # to store vs results
          if( L2 == "D" ) vs_s <- "Delegation"                                  # scenario for delegation
          if( L2 == "R" ) vs_s <- c("AgainstAI", "AgainstHuman")                # scenarions for random
          for( VS in vs_s ){                                                    # choices vs. AI, human or result of delegation
            expected_payoffs <- NULL                                            # to store payoff of id in all matches            
            print(paste(L1,L2,L3,role,VS, collapse = " "))
            for( id in T_ids ){                                                 # loop through ids in treatment
              payoffs_id <- NULL                                                # to store payoff of id in all matches
              TO_ids <- T_ids[T_ids != id]                                      # retrieve other individuals in same treatment
              for( o_id in TO_ids ){                                            # loop through potential matches
                id_dec_vec <- rep(NA, 7)                                        # vector to store id decisions
                o_id_dec_vec <- rep(NA, 7)                                      # vector to store others decisions
                if( L2 == "R" ){                                                # restriction: if random payoff of both choices
                  if( VS != "Delegation"){
                    if( role == "NoAISupport" ){
                      
                      # retrieve id's decision
                      id_dec_vec <- sapply(decision_cols, function(col) as.numeric(df[df$SubjectID == id & 
                                                                                        df$Scenario == role & 
                                                                                        df$Case == VS, col]))
                      
                      # retrieve o_id's decisions
                      if( VS == "AgainstAI" ){
                        if( L3 == "U" ){
                          o_id_dec_vec <- sapply(decision_cols, function(col) decisions_list[[col]][sample(129:192, 1)])
                        }else if( L3 == "P" ){
                          o_id_flavor <- unique(df$AI_flavour[df$SubjectID == o_id])
                          o_id_dec_vec <- sapply(decision_cols, function(col) decisions_list[[col]][o_id_flavor])
                        }
                      }else if( VS == "AgainstHuman" ){
                        o_id_dec_vec <- sapply(decision_cols, function(col) as.numeric(df[df$SubjectID == o_id & 
                                                                                            df$Scenario == roles[roles != role] & 
                                                                                            df$Case == "AgainstHuman", col]))
                      }
                    }else if( role == "AISupport" ){
                      # retrieve o_id's decision
                      o_id_dec_vec <- sapply(decision_cols, function(col) as.numeric(df[df$SubjectID == o_id & 
                                                                                          df$Scenario == roles[roles != role] & 
                                                                                          df$Case == VS, col]))
                      
                      # retrieve o_id's decisions
                      if( VS == "AgainstAI" ){
                        if( L3 == "U" ){
                          id_dec_vec <- sapply(decision_cols, function(col) decisions_list[[col]][sample(129:192, 1)])
                        }else if( L3 == "P" ){
                          id_flavor <- unique(df$AI_flavour[df$SubjectID == id])
                          id_dec_vec <- sapply(decision_cols, function(col) decisions_list[[col]][id_flavor])
                        }
                      }else if( VS == "AgainstHuman" ){
                        id_dec_vec <- sapply(decision_cols, function(col) as.numeric(df[df$SubjectID == id & 
                                                                                          df$Scenario == role & 
                                                                                          df$Case == "AgainstHuman", col]))
                      }
                    }
                  }
                }else if( L2 == "D" ){
                  if( role == "NoAISupport" ){
                    for( c in (1:length(decision_cols)) ){                        # retrieve delegation decision
                      o_id_del <- df[df$SubjectID == o_id & df$Scenario == "AISupport", paste0(decision_cols[c],"_Delegation")]
                      if( o_id_del == 1 ){
                        if( L1 == "O" ){
                          id_dec_vec[c] <- as.numeric(df[df$SubjectID == id & 
                                                           df$Scenario == role & 
                                                           df$Case == "Opaque", 
                                                         decision_cols[c]])
                        }else if( L1 == "T" ){
                          id_dec_vec[c] <- as.numeric(df[df$SubjectID == id & 
                                                           df$Scenario == role & 
                                                           df$Case == "AgainstAI", 
                                                         decision_cols[c]])
                        }
                        if( L3 == "U" ){
                          o_id_dec_vec[c] <- decisions_list[[decision_cols[c]]][sample(129:192, 1)]
                        }else if( L3 == "P" ){
                          o_id_flavor <- unique(df$AI_flavour[df$SubjectID == o_id])
                          o_id_dec_vec[c] <- decisions_list[[decision_cols[c]]][o_id_flavor]
                        }
                      }else if( o_id_del == 0 ){
                        if( L1 == "O" ){
                          id_dec_vec[c] <- as.numeric(df[df$SubjectID == id & 
                                                           df$Scenario == role & 
                                                           df$Case == "Opaque", 
                                                         decision_cols[c]])
                        }else if( L1 == "T" ){
                          id_dec_vec[c] <- as.numeric(df[df$SubjectID == id & 
                                                           df$Scenario == role & 
                                                           df$Case == "AgainstHuman", 
                                                         decision_cols[c]])
                        }
                        o_id_dec_vec[c] <- as.numeric(df[df$SubjectID == o_id & 
                                                           df$Scenario == roles[roles != role] & 
                                                           df$Case == "AgainstHuman", 
                                                         decision_cols[c]])
                      }
                    }
                  }else if( role == "AISupport" ){
                    for( c in (1:length(decision_cols)) ){                        # retrieve delegation decision
                      id_del <- df[df$SubjectID == id & df$Scenario == "AISupport", paste0(decision_cols[c],"_Delegation")]
                      if( id_del == 1 ){
                        if( L1 == "O" ){
                          o_id_dec_vec[c] <- as.numeric(df[df$SubjectID == o_id & 
                                                             df$Scenario == roles[roles!=role] & 
                                                             df$Case == "Opaque", 
                                                           decision_cols[c]])
                        }else if( L1 == "T" ){
                          o_id_dec_vec[c] <- as.numeric(df[df$SubjectID == o_id & 
                                                             df$Scenario == roles[roles!=role] & 
                                                             df$Case == "AgainstAI", 
                                                           decision_cols[c]])
                        }
                        if( L3 == "U" ){
                          id_dec_vec[[c]] <- decisions_list[[decision_cols[c]]][sample(129:192, 1)]
                        }else if( L3 == "P" ){
                          id_flavor <- unique(df$AI_flavour[df$SubjectID == id])
                          id_dec_vec[c] <- decisions_list[[decision_cols[c]]][id_flavor]
                        }
                      }else if( id_del == 0 ){
                        if( L1 == "O" ){
                          o_id_dec_vec[c] <- as.numeric(df[df$SubjectID == o_id & 
                                                             df$Scenario == roles[roles!=role] & 
                                                             df$Case == "Opaque", 
                                                           decision_cols[c]])
                        }else if( L1 == "T" ){
                          o_id_dec_vec[c] <- as.numeric(df[df$SubjectID == o_id & 
                                                             df$Scenario == roles[roles!=role] & 
                                                             df$Case == "AgainstHuman", 
                                                           decision_cols[c]])
                        }
                        id_dec_vec[c] <- as.numeric(df[df$SubjectID == id & 
                                                         df$Scenario == role & 
                                                         df$Case == "AgainstHuman", 
                                                       decision_cols[c]])
                      }
                    }
                  }
                }
                # payoffs for match
                payoffs_match <- rep(NA, 7)
                payoffs_match[1] <- 2 + 3*as.numeric(id_dec_vec[1] == o_id_dec_vec[1])                # C
                payoffs_match[2] <- ifelse(id_dec_vec[2] == 1, mean(5+o_id_dec_vec[3]), 5)            # Sender
                payoffs_match[3] <- ifelse(o_id_dec_vec[2] == 1, 11-id_dec_vec[3], 5 )                # Receiver
                payoffs_match[4] <- as.numeric(o_id_dec_vec[5] <= id_dec_vec[4])*(10-id_dec_vec[4])   # Proposer                    
                payoffs_match[5] <- as.numeric(id_dec_vec[5] <= o_id_dec_vec[4])*(10-o_id_dec_vec[4]) # Responder      
                payoffs_match[6] <- c(3,8,1,5)[1 + 2*id_dec_vec[6] + o_id_dec_vec[6]]                 # PD
                payoffs_match[7] <- c(4,5,1,8)[1 + 2*id_dec_vec[7] + o_id_dec_vec[7]]                 # SH
                payoffs_id <- rbind(payoffs_id, payoffs_match)
              } # loop through o_ids
              expected_payoffs_id <- c(id, apply(payoffs_id, 2, mean, na.rm=TRUE))
              expected_payoffs <- rbind(expected_payoffs, expected_payoffs_id)
            } # loop through ids
            colnames(expected_payoffs) <- c("id", decision_cols)
            VS_list[[VS]] <- expected_payoffs 
          }
          roles_list[[role]] <- VS_list
        }
        # append payoffs to list
        payoffs[[paste0(L1,L2,L3)]] <- roles_list
      }
    }
  }
}

# combine both roles
payoffs$TDP$BothRoles <- list(Delegation = (payoffs$TDP$NoAISupport$Delegation + payoffs$TDP$AISupport$Delegation)/2)
payoffs$TDU$BothRoles <- list(Delegation = (payoffs$TDU$NoAISupport$Delegation + payoffs$TDU$AISupport$Delegation)/2)
payoffs$TRP$BothRoles <- list(AgainstHuman = (payoffs$TRP$NoAISupport$AgainstHuman + payoffs$TRP$AISupport$AgainstHuman)/2,
                              AgainstAI = (payoffs$TRP$NoAISupport$AgainstAI + payoffs$TRP$AISupport$AgainstAI)/2)
payoffs$TRU$BothRoles <- list(AgainstHuman = (payoffs$TRU$NoAISupport$AgainstHuman + payoffs$TRU$AISupport$AgainstHuman)/2,
                              AgainstAI = (payoffs$TRU$NoAISupport$AgainstAI + payoffs$TRU$AISupport$AgainstAI)/2)
payoffs$ODP$BothRoles <- list(Delegation = (payoffs$ODP$NoAISupport$Delegation + payoffs$ODP$AISupport$Delegation)/2)
payoffs$ODU$BothRoles <- list(Delegation = (payoffs$ODU$NoAISupport$Delegation + payoffs$ODU$AISupport$Delegation)/2)

# compile cases
payoffs$TR <- list(NoAISupport = list(AgainstHuman = rbind(payoffs$TRP$NoAISupport$AgainstHuman, payoffs$TRU$NoAISupport$AgainstHuman),
                                      AgainstAI = rbind(payoffs$TRP$NoAISupport$AgainstAI, payoffs$TRU$NoAISupport$AgainstAI)),
                   AISupport = list(AgainstHuman = rbind(payoffs$TRP$AISupport$AgainstHuman, payoffs$TRU$AISupport$AgainstHuman),
                                    AgainstAI = rbind(payoffs$TRP$AISupport$AgainstAI, payoffs$TRU$AISupport$AgainstAI)),
                   BothRoles = list(AgainstHuman = rbind(payoffs$TRP$BothRoles$AgainstHuman, payoffs$TRU$BothRoles$AgainstHuman),
                                    AgainstAI = rbind(payoffs$TRP$BothRoles$AgainstAI, payoffs$TRU$BothRoles$AgainstAI)))
payoffs$TD <- list(NoAISupport = rbind(payoffs$TDP$NoAISupport$Delegation, payoffs$TDU$NoAISupport$AgainstHuman),
                   AISupport = rbind(payoffs$TDP$AISupport$Delegation, payoffs$TDU$AISupport$Delegation),
                   BothRoles = rbind(payoffs$TDP$BothRoles$Delegation, payoffs$TDU$BothRoles$Delegation))
payoffs$OD <- list(NoAISupport = rbind(payoffs$ODP$NoAISupport$Delegation, payoffs$ODU$NoAISupport$AgainstHuman),
                   AISupport = rbind(payoffs$ODP$AISupport$Delegation, payoffs$ODU$AISupport$Delegation),
                   BothRoles = rbind(payoffs$ODP$BothRoles$Delegation, payoffs$ODU$BothRoles$Delegation))

# cases
benchmark <- apply(payoffs$TR$BothRoles$AgainstHuman[,2:8], 2, mean, na.rm=TRUE)
AIR <- list()
TD <- list()
OD <- list()
AIR$both <- rbind(apply(payoffs$TR$BothRoles$AgainstAI[,2:8], 2, mean, na.rm=TRUE),
                  apply(sapply(1:BS, function(x) apply(payoffs$TR$BothRoles$AgainstAI[sample(1:nrow(payoffs$TR$BothRoles$AgainstAI), replace=TRUE),2:8], 2, mean, na.rm=TRUE)), 1, quantile, c(0.025, 0.975)))
AIR$no <- rbind(apply(payoffs$TR$NoAISupport$AgainstAI[,2:8], 2, mean, na.rm=TRUE),
                apply(sapply(1:BS, function(x) apply(payoffs$TR$NoAISupport$AgainstAI[sample(1:nrow(payoffs$TR$NoAISupport$AgainstAI), replace=TRUE),2:8], 2, mean, na.rm=TRUE)), 1, quantile, c(0.025, 0.975)))
AIR$yes <- rbind(apply(payoffs$TR$AISupport$AgainstAI[,2:8], 2, mean, na.rm=TRUE),
                 apply(sapply(1:BS, function(x) apply(payoffs$TR$AISupport$AgainstAI[sample(1:nrow(payoffs$TR$AISupport$AgainstAI), replace=TRUE),2:8], 2, mean, na.rm=TRUE)), 1, quantile, c(0.025, 0.975)))
TD$both  <- rbind(apply(payoffs$TD$BothRoles[,2:8], 2, mean, na.rm=TRUE),
                  apply(sapply(1:BS, function(x) apply(payoffs$TD$BothRoles[sample(1:nrow(payoffs$TD$BothRoles), replace=TRUE),2:8], 2, mean, na.rm=TRUE)), 1, quantile, c(0.025, 0.975)))
TD$no <- rbind(apply(payoffs$TD$NoAISupport[,2:8], 2, mean, na.rm=TRUE),
               apply(sapply(1:BS, function(x) apply(payoffs$TD$NoAISupport[sample(1:nrow(payoffs$TD$NoAISupport), replace=TRUE),2:8], 2, mean, na.rm=TRUE)), 1, quantile, c(0.025, 0.975)))
TD$yes <- rbind(apply(payoffs$TD$AISupport[,2:8], 2, mean, na.rm=TRUE),
                apply(sapply(1:BS, function(x) apply(payoffs$TD$AISupport[sample(1:nrow(payoffs$TD$AISupport), replace=TRUE),2:8], 2, mean, na.rm=TRUE)), 1, quantile, c(0.025, 0.975)))
OD$both  <- rbind(apply(payoffs$OD$BothRoles[,2:8], 2, mean, na.rm=TRUE),
                  apply(sapply(1:BS, function(x) apply(payoffs$OD$BothRoles[sample(1:nrow(payoffs$OD$BothRoles), replace=TRUE),2:8], 2, mean, na.rm=TRUE)), 1, quantile, c(0.025, 0.975)))
OD$no <- rbind(apply(payoffs$OD$NoAISupport[,2:8], 2, mean, na.rm=TRUE),
               apply(sapply(1:BS, function(x) apply(payoffs$OD$NoAISupport[sample(1:nrow(payoffs$OD$NoAISupport), replace=TRUE),2:8], 2, mean, na.rm=TRUE)), 1, quantile, c(0.025, 0.975)))
OD$yes <- rbind(apply(payoffs$OD$AISupport[,2:8], 2, mean, na.rm=TRUE),
                apply(sapply(1:BS, function(x) apply(payoffs$OD$AISupport[sample(1:nrow(payoffs$OD$AISupport), replace=TRUE),2:8], 2, mean, na.rm=TRUE)), 1, quantile, c(0.025, 0.975)))

# construct data frames
plot_dat <- list()
for( i in c("both", "no", "yes")){
  dat <- (rbind(AIR[[i]][1,]/benchmark,TD[[i]][1,]/benchmark, OD[[i]][1,]/benchmark) - 1)*100
  rownames(dat) <- c("AIR", "TD", "OD")
  dat <- reshape2::melt(dat, c("Case", "Game"), value.name = "mean_payoff_diff")
  dat_CI_lower <- (rbind(AIR[[i]][2,]/benchmark,TD[[i]][2,]/benchmark, OD[[i]][2,]/benchmark) - 1)*100
  rownames(dat_CI_lower) <- c("AIR", "TD", "OD")
  dat$CI_lower <- reshape2::melt(dat_CI_lower, c("Case", "Game"), value.name = "CI_lower")$CI_lower
  dat_CI_upper <- (rbind(AIR[[i]][3,]/benchmark,TD[[i]][3,]/benchmark, OD[[i]][3,]/benchmark) - 1)*100
  rownames(dat_CI_upper) <- c("AIR", "TD", "OD")
  dat$CI_upper <- reshape2::melt(dat_CI_upper, c("Case", "Game"), value.name = "CI_upper")$CI_upper
  dat$Game <- factor(dat$Game, 
                     levels = c("Sender","Receiver","PD","SH",
                                "C","Proposer","Responder"))
  dat$PlotNumber <- factor(c(rep("Coordination (C)", 3),
                             rep("Trust (TG)", 6), 
                             rep("Fairness (UG)", 6),
                             rep("Cooperation (PD/SH)", 6)),
                           levels = c("Trust (TG)","Cooperation (PD/SH)",
                                      "Coordination (C)","Fairness (UG)"))
  levels(dat$Case) <- c("Transparent Random (AI)", "Transparent Delegation", "Opaque Delegation")
  plot_dat[[i]] <- dat
}

################################################################################
# Figures 2 & SM2
################################################################################
color_scheme6 <- c("#73787E", "#B8BCC1", "#59C7EB")
payoff_plots <- list()
titles <- c("Relative Payoff Differences Compared to Interaction with Human",
            "Relative Payoff Differences (Player without AI Support)",
            "Relative Payoff Differences (Player with AI Support)")
lower_limits <- c(-40,-50,-50)
upper_limits <- c(30,65,65)

for( i in 1:length(c("both", "no", "yes"))){
  payoff_plots[[i]] <- ggplot(data = plot_dat[[i]]) +
    geom_bar(aes(x = Game, y = mean_payoff_diff, fill = Case, group = Game), stat = "identity", position = position_dodge2(width = 0.5), width = 0.9) +
    geom_text(aes(x = Game, y = ifelse(mean_payoff_diff > 0, CI_upper, CI_lower), group = Case, label = paste(sprintf("%0.1f", round(mean_payoff_diff, digits = 1)), "%", sep = ""), vjust = ifelse(mean_payoff_diff > 0, -1.5, 1.5)), position = position_dodge2(width = 0.9), colour = "black", size = 3) +
    geom_errorbar(aes(x = Game, y = mean_payoff_diff, ymin = CI_lower, ymax = CI_upper, group = Case), width = 0.4, position = position_dodge(width = 0.85)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    facet_grid(.~PlotNumber, scales = "free_x", space = "free") +
    scale_fill_manual(values=color_scheme6) +
    scale_y_continuous(limits = c(lower_limits[i], upper_limits[i]),  expand = expansion(mult = c(0, .05))) +
    guides(fill=guide_legend(title="Interaction with: ")) +
    labs(title = titles[i], x = "", y = "Payoff Difference (%)") +
    theme_ds4psy() +
    theme(panel.grid.major.x = element_blank(),
          panel.grid.minor.y = element_line( size=.1, color="grey" ),
          panel.grid.major.y = element_line( size=.1, color="grey" ),
          legend.position ="bottom") + 
    guides(fill = guide_legend("Condition"))
}

Figure2 <- payoff_plots[[1]]
Figure_SM3 <- ggarrange(payoff_plots[[2]], payoff_plots[[3]], nrow = 2, labels = c("A", "B"), common.legend = TRUE)

# save figures
ggsave("./figures/Figure_2.pdf", Figure2, width = 10, height = 5)
ggsave("./figures/Figure_SM3.pdf", Figure_SM3, width = 9, height = 7)

