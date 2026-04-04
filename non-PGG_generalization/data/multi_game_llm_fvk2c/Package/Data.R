################################################################################
# Data Preparation Script
# Adverse Reactions to the Use of Large Language Models in Social Interactions
# F. Dvorak, R. Stumpf, S. Fehrler & U. Fischbacher
# PNAS Nexus
################################################################################
library(tidyverse)
library(dplyr)
library(ds4psy)
library(reshape2)
library(stringr)
library(readr)
library(readxl)

rm(list = ls())                                         # clear workspace
setwd(dirname(sys.frame(1)$ofile))                      # set working directory

################################################################################
# Load and prepare main raw data 
################################################################################
df <- read_csv("./data/MainDataRawClean.csv", col_names = TRUE, col_types = NULL)

# Treatment identifier
df$TreatmentCode[df$Treatment == "TransparentRandom" & df$PersonalizedTreatment == 1]     <- "TRP"
df$TreatmentCode[df$Treatment == "TransparentRandom" & df$PersonalizedTreatment == 0]     <- "TRU"
df$TreatmentCode[df$Treatment == "TransparentDelegation" & df$PersonalizedTreatment == 1] <- "TDP"
df$TreatmentCode[df$Treatment == "TransparentDelegation" & df$PersonalizedTreatment == 0] <- "TDU"
df$TreatmentCode[df$Treatment == "OpaqueDelegation" & df$PersonalizedTreatment == 1]      <- "ODP"
df$TreatmentCode[df$Treatment == "OpaqueDelegation" & df$PersonalizedTreatment == 0]      <- "ODU"

df_JUS_raw <- df
df <- df %>% dplyr::select(-grep("JUS", names(df)))

## Combine all decisions
df$UGProposer_11 <- apply(cbind(df[,grep("UGProposer_11", names(df))]), 1,
                          function(x) paste(x[!is.na(x)], collapse = ""))
df$UGProposer_21 <- apply(cbind(df[,grep("UGProposer_21", names(df))]), 1,
                          function(x) paste(x[!is.na(x)], collapse = ""))
df$UGProposer_22 <- apply(cbind(df[,grep("UGProposer_22", names(df))]), 1,
                          function(x) paste(x[!is.na(x)], collapse = ""))
df$UGProposer_2 <- apply(cbind(df[,grep("OD_UGProposer_2", names(df))]), 1,
                         function(x) paste(x[!is.na(x)], collapse = ""))
df$UGProposer_Delegation <- NA
df$UGProposer_Delegation[df$Treatment != "TransparentRandom" & df$UGProposer_11 == ""] <- 1
df$UGProposer_Delegation[df$Treatment != "TransparentRandom" & df$UGProposer_11 != ""] <- 0

df$UGResponder_11 <- apply(cbind(df[,grep("UGResponder_11", names(df))]), 1,
                           function(x) paste(x[!is.na(x)], collapse = ""))
df$UGResponder_21 <- apply(cbind(df[,grep("UGResponder_21", names(df))]), 1,
                           function(x) paste(x[!is.na(x)], collapse = ""))
df$UGResponder_22 <- apply(cbind(df[,grep("UGResponder_22", names(df))]), 1,
                           function(x) paste(x[!is.na(x)], collapse = ""))
df$UGResponder_2 <- apply(cbind(df[,grep("OD_UGResponder_2", names(df))]), 1,
                          function(x) paste(x[!is.na(x)], collapse = ""))
df$UGResponder_Delegation <- NA
df$UGResponder_Delegation[df$Treatment != "TransparentRandom" & df$UGResponder_11 == ""] <- 1
df$UGResponder_Delegation[df$Treatment != "TransparentRandom" & df$UGResponder_11 != ""] <- 0


df$TGReceiver_11 <- apply(cbind(df[,grep("TGReceiver_11", names(df))]), 1,
                          function(x) paste(x[!is.na(x)], collapse = ""))
df$TGReceiver_21 <- apply(cbind(df[,grep("TGReceiver_21", names(df))]), 1,
                          function(x) paste(x[!is.na(x)], collapse = ""))
df$TGReceiver_22 <- apply(cbind(df[,grep("TGReceiver_22", names(df))]), 1,
                          function(x) paste(x[!is.na(x)], collapse = ""))
df$TGReceiver_2 <- apply(cbind(df[,grep("OD_TGReceiver_2", names(df))]), 1,
                         function(x) paste(x[!is.na(x)], collapse = ""))
df$TGReceiver_Delegation <- NA
df$TGReceiver_Delegation[df$Treatment != "TransparentRandom" & df$TGReceiver_11 == ""] <- 1
df$TGReceiver_Delegation[df$Treatment != "TransparentRandom" & df$TGReceiver_11 != ""] <- 0

df$TGSender_11 <- apply(cbind(df[,grep("TGSender_11", names(df))]), 1,
                        function(x) paste(x[!is.na(x)], collapse = ""))
df$TGSender_21 <- apply(cbind(df[,grep("TGSender_21", names(df))]), 1,
                        function(x) paste(x[!is.na(x)], collapse = ""))
df$TGSender_22 <- apply(cbind(df[,grep("TGSender_22", names(df))]), 1,
                        function(x) paste(x[!is.na(x)], collapse = ""))
df$TGSender_2 <- apply(cbind(df[,grep("OD_TGSender_2", names(df))]), 1,
                       function(x) paste(x[!is.na(x)], collapse = ""))
df$TGSender_Delegation <- NA
df$TGSender_Delegation[df$Treatment != "TransparentRandom" & df$TGSender_11 == ""] <- 1
df$TGSender_Delegation[df$Treatment != "TransparentRandom" & df$TGSender_11 != ""] <- 0

df$PD_11 <- apply(cbind(df[,grep("PD_11", names(df))]), 1,
                  function(x) paste(x[!is.na(x)], collapse = ""))
df$PD_21 <- apply(cbind(df[,grep("PD_21", names(df))]), 1,
                  function(x) paste(x[!is.na(x)], collapse = ""))
df$PD_22 <- apply(cbind(df[,grep("PD_22", names(df))]), 1,
                  function(x) paste(x[!is.na(x)], collapse = ""))
df$PD_2 <- apply(cbind(df[,grep("OD_PD_2", names(df))]), 1,
                 function(x) paste(x[!is.na(x)], collapse = ""))

df$PD_Delegation <- NA
df$PD_Delegation[df$Treatment != "TransparentRandom" & df$PD_11 == ""] <- 1
df$PD_Delegation[df$Treatment != "TransparentRandom" & df$PD_11 != ""] <- 0


df$SH_11 <- apply(cbind(df[,grep("SH_11", names(df))]), 1,
                  function(x) paste(x[!is.na(x)], collapse = ""))
df$SH_21 <- apply(cbind(df[,grep("SH_21", names(df))]), 1,
                  function(x) paste(x[!is.na(x)], collapse = ""))
df$SH_22 <- apply(cbind(df[,grep("SH_22", names(df))]), 1,
                  function(x) paste(x[!is.na(x)], collapse = ""))
df$SH_2 <- apply(cbind(df[,grep("OD_SH_2", names(df))]), 1,
                 function(x) paste(x[!is.na(x)], collapse = ""))

df$SH_Delegation <- NA
df$SH_Delegation[df$Treatment != "TransparentRandom" & df$SH_11 == ""] <- 1
df$SH_Delegation[df$Treatment != "TransparentRandom" & df$SH_11 != ""] <- 0


df$C_11 <- apply(cbind(df[,grep("C_11", names(df))]), 1,
                 function(x) paste(x[!is.na(x)], collapse = ""))
df$C_21 <- apply(cbind(df[,grep("C_21", names(df))]), 1,
                 function(x) paste(x[!is.na(x)], collapse = ""))
df$C_22 <- apply(cbind(df[,grep("C_22", names(df))]), 1,
                 function(x) paste(x[!is.na(x)], collapse = ""))
df$C_2 <- apply(cbind(df[,grep("OD_C_2", names(df))]), 1,
                function(x) paste(x[!is.na(x)], collapse = ""))

df$C_Delegation <- NA
df$C_Delegation[df$Treatment != "TransparentRandom" & df$C_11 == ""] <- 1
df$C_Delegation[df$Treatment != "TransparentRandom" & df$C_11 != ""] <- 0

## Convert variables into correct class
firstcol <- which(colnames(df)=="UGProposer_11")
lastcol  <- which(colnames(df)=="TGReceiver_Delegation")

cols.num <- c(firstcol:lastcol)
df[cols.num] <- sapply(df[cols.num],as.numeric)
df <- df %>% mutate_at(c('TGSender_Delegation','PD_Delegation', 'SH_Delegation', 'C_Delegation'), as.numeric)

df_for_turing <- df

suppressWarnings(df <- df %>% mutate_at(grep("TGSender", names(df)), funs(recode(., 'Yes' = 1, 'No' = 0))))
suppressWarnings(df <- df %>% mutate_at(grep("PD", names(df)), funs(recode(., 'A' = 1, 'B' = 0))))
suppressWarnings(df <- df %>% mutate_at(grep("SH", names(df)), funs(recode(., 'X' = 1, 'Y' = 0))))
suppressWarnings(df <- df %>% mutate_at(grep("C_", names(df)), funs(recode(., 'mercury' = -2, 'venus' = -1, 'earth' = 0, 'mars' = 1, 'saturn' = 2))))

df1 <- df %>% dplyr::select(c(Treatment:C_Delegation))

df2 <- df1 %>% pivot_longer(grep("(UG)(.*_11|_21|_22|_2)", names(df1) ), names_to = "scenario")

df2 <- df1 %>% pivot_longer(grep("_11|_21|_22|_2", names(df1)), names_to = "scenario")
df2 <- df2 %>% separate(scenario, c("Game", "Scenario"), sep = "_")
df2 <- df2 %>% group_by(Game) %>% pivot_wider(names_from = Game, values_from = value)

df2$ScenarioCode <- df2$Scenario

df2$Scenario <- ""
df2$Scenario[df2$ScenarioCode == 11] <- "AISupport"
df2$Scenario[df2$ScenarioCode == 21] <- "NoAISupport"
df2$Scenario[df2$ScenarioCode == 22] <- "NoAISupport"
df2$Scenario[df2$ScenarioCode == 2]  <- "NoAISupport"

df2$Case <- ""
df2$Case[df2$ScenarioCode == 11] <- "AgainstHuman"
df2$Case[df2$ScenarioCode == 21] <- "AgainstHuman"
df2$Case[df2$ScenarioCode == 22] <- "AgainstAI"
df2$Case[df2$ScenarioCode == 2]  <- "Opaque"

df2 <- df2 %>% filter(Treatment %in% c("TransparentRandom", "TransparentDelegation") & Case != "Opaque" | Treatment == "OpaqueDelegation" & ScenarioCode %in% c('11','2'))

df_short <- df %>% select(SubjectID, TreatmentCode, `Duration (in seconds)`:Personalization_7, PersonalizedTreatment, count_ChangedBrowserWindow, PostExStatementsTR_1:Exclusion_DisturbTR, PostExStatementsTDOD_1:Exclusion_DisturbTD)

df_short$Q_DifficultyPredictionAI <- apply(cbind(df_short[,grep("PostExStatementsTR_1|PostExStatementsTDOD_1", names(df_short))]), 1,
                                           function(x) paste(x[!is.na(x)], collapse = ""))
df_short$Q_DifficultyPredictionHuman <- apply(cbind(df_short[,grep("PostExStatementsTR_2|PostExStatementsTDOD_2", names(df_short))]), 1,
                                              function(x) paste(x[!is.na(x)], collapse = ""))
df_short$Q_AITrustworthy <- apply(cbind(df_short[,grep("PostExStatementsTR_3|PostExStatementsTDOD_3", names(df_short))]), 1,
                                  function(x) paste(x[!is.na(x)], collapse = ""))
df_short$Q_HumanTrustworthy <- apply(cbind(df_short[,grep("PostExStatementsTR_4|PostExStatementsTDOD_4", names(df_short))]), 1,
                                     function(x) paste(x[!is.na(x)], collapse = ""))
df_short$Q_AIReflectsHuman <- apply(cbind(df_short[,grep("PostExStatementsTR_5|PostExStatementsTDOD_5", names(df_short))]), 1,
                                    function(x) paste(x[!is.na(x)], collapse = ""))
df_short$Q_DelegationBelief <- apply(cbind(df_short[,grep("PostExStatementsTDOD_6", names(df_short))]), 1,
                                     function(x) paste(x[!is.na(x)], collapse = ""))
df_short$Q_DelegationAppropriate <- apply(cbind(df_short[,grep("PostExStatementsTDOD_7", names(df_short))]), 1,
                                          function(x) paste(x[!is.na(x)], collapse = ""))
df_short$Q_EqualityWithAI <- apply(cbind(df_short[,grep("PostExStatementsTR_6|PostExStatementsTDOD_8", names(df_short))]), 1,
                                   function(x) paste(x[!is.na(x)], collapse = ""))
df_short$Q_EqualityWithHuman <- apply(cbind(df_short[,grep("PostExStatementsTR_7|PostExStatementsTDOD_9", names(df_short))]), 1,
                                      function(x) paste(x[!is.na(x)], collapse = ""))
df_short$Age <- apply(cbind(df_short[,grep("Age", names(df_short))]), 1,
                      function(x) paste(x[!is.na(x)], collapse = ""))
df_short$Gender <- apply(cbind(df_short[,grep("Gender", names(df_short))]), 1,
                         function(x) paste(x[!is.na(x)], collapse = ""))
df_short$Education <- apply(cbind(df_short[,grep("Education", names(df_short))]), 1,
                            function(x) paste(x[!is.na(x)], collapse = ""))
df_short$Exclusion_Disturb <- apply(cbind(df_short[,grep("Exclusion_Disturb", names(df_short))]), 1,
                                    function(x) paste(x[!is.na(x)], collapse = ""))
df_short$ExclusionUsageGPT <- apply(cbind(df_short[,grep("ExclusionUsageGPT", names(df_short))]), 1,
                                    function(x) paste(x[!is.na(x)], collapse = ""))
# String to remove
string_to_remove <- "\n"

# Loop through columns and remove the specified string
for (col in grep("Q_",names(df_short))) {
  df_short[[col]] <- gsub(string_to_remove, "", df_short[[col]])
}

df_short$personality_string[df_short$PersonalizedTreatment == 1] <- paste0(df_short$Personalization_1[df_short$PersonalizedTreatment == 1],df_short$Personalization_2[df_short$PersonalizedTreatment == 1],df_short$Personalization_3[df_short$PersonalizedTreatment == 1],df_short$Personalization_4[df_short$PersonalizedTreatment == 1],df_short$Personalization_5[df_short$PersonalizedTreatment == 1],df_short$Personalization_6[df_short$PersonalizedTreatment == 1],df_short$Personalization_7[df_short$PersonalizedTreatment == 1],sep="")


df_short <- df_short %>% rename(IntuitionThoughtfulness = Personalization_1, IntroversionExtraverison = Personalization_2,
                                FairnessEfficiency = Personalization_3, ChaosBoredom = Personalization_4,
                                SelfishnessAltruism = Personalization_5, NoveltyReliability = Personalization_6,
                                TruthHarmony = Personalization_7,KnowledgeChatGPT = Q1367, UsageChatGPT = Q1369, DurationInSec = `Duration (in seconds)`)

df_short <- df_short %>% dplyr::select(SubjectID, TreatmentCode, DurationInSec, KnowledgeChatGPT:count_ChangedBrowserWindow, Q_DifficultyPredictionAI:ExclusionUsageGPT, personality_string)


df <- full_join(df2, df_short)

df <- df %>% select(SubjectID, TreatmentCode, Treatment, PersonalizedTreatment, Scenario, Case, UGProposer:C, UGProposer_Delegation:C_Delegation, count_ChangedBrowserWindow, DurationInSec:ExclusionUsageGPT, personality_string)
write_csv(df,"./data/MainData.csv")

################################################################################
# Load and prepare turing raw data 
################################################################################
df_turing <- read_csv("./data/TuringDataRawClean.csv", show_col_types = FALSE)

## Combining all decisions
df_turing$O_UGProposer1_Cert <- apply(cbind(df_turing[,grep("UGProposer_Cert1", names(df_turing))]), 1, 
                                 function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$O_UGProposer2_Cert <- apply(cbind(df_turing[,grep("UGProposer_Cert2", names(df_turing))]), 1, 
                                 function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$T_UGProposer_Cert <- apply(cbind(df_turing[,grep("T_UGProp_Cert", names(df_turing))]), 1, 
                                function(x) paste(x[!is.na(x)], collapse = ""))

df_turing$O_UGResponder1_Cert <- apply(cbind(df_turing[,grep("UGResp_Cert1", names(df_turing))]), 1, 
                                      function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$O_UGResponder2_Cert <- apply(cbind(df_turing[,grep("UGResp_Cert2", names(df_turing))]), 1, 
                                      function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$T_UGResponder_Cert <- apply(cbind(df_turing[,grep("T_UGResp_Cert", names(df_turing))]), 1, 
                                     function(x) paste(x[!is.na(x)], collapse = ""))

df_turing$O_TGSender1_Cert <- apply(cbind(df_turing[,grep("TGSender_Cert1", names(df_turing))]), 1, 
                                       function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$O_TGSender2_Cert <- apply(cbind(df_turing[,grep("TGSender_Cert2", names(df_turing))]), 1, 
                                       function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$T_TGSender_Cert <- apply(cbind(df_turing[,grep("T_TGSender_Cert", names(df_turing))]), 1, 
                                      function(x) paste(x[!is.na(x)], collapse = ""))

df_turing$O_TGReceiver1_Cert <- apply(cbind(df_turing[,grep("TGReceiver_Cert1", names(df_turing))]), 1, 
                                    function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$O_TGReceiver2_Cert <- apply(cbind(df_turing[,grep("TGReceiver_Cert2", names(df_turing))]), 1, 
                                    function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$T_TGReceiver_Cert <- apply(cbind(df_turing[,grep("T_TGReceiver_Cert", names(df_turing))]), 1, 
                                   function(x) paste(x[!is.na(x)], collapse = ""))

df_turing$O_PD1_Cert <- apply(cbind(df_turing[,grep("PD_Cert1", names(df_turing))]), 1, 
                                      function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$O_PD2_Cert <- apply(cbind(df_turing[,grep("PD_Cert2", names(df_turing))]), 1, 
                                      function(x) paste(x[!is.na(x)], collapse = ""))

df_turing$O_SH1_Cert <- apply(cbind(df_turing[,grep("SH_Cert1", names(df_turing))]), 1, 
                              function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$O_SH2_Cert <- apply(cbind(df_turing[,grep("SH_Cert2", names(df_turing))]), 1, 
                              function(x) paste(x[!is.na(x)], collapse = ""))

df_turing$O_C1_Cert <- apply(cbind(df_turing[,grep("C_Cert1", names(df_turing))]), 1, 
                              function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$O_C2_Cert <- apply(cbind(df_turing[,grep("C_Cert2", names(df_turing))]), 1, 
                              function(x) paste(x[!is.na(x)], collapse = ""))


df_turing$T_Cert1 <- apply(cbind(df_turing[,grep("T_Cert1", names(df_turing))]), 1, 
                                     function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$T_Cert2 <- apply(cbind(df_turing[,grep("T_Cert2", names(df_turing))]), 1, 
                           function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$T_Cert3 <- apply(cbind(df_turing[,grep("T_Cert3", names(df_turing))]), 1, 
                           function(x) paste(x[!is.na(x)], collapse = ""))

df_turing <- df_turing %>% dplyr::select(-grep("ODP_UGProposer_Cert|ODN_UGProposer_Cert|ODP_UGResp_Cert|ODN_UGResp_Cert|ODP_TGSender_Cert|ODN_TGSender_Cert|ODP_TGReceiver_Cert|ODN_TGReceiver_Cert|ODP_PD_Cert|ODN_PD_Cert|ODP_SH_Cert|ODN_SH_Cert|ODP_C_Cert|ODN_C_Cert|_T_UGProp_Cert|_T_UGResp_Cert|_T_TGSender_Cert|_T_TGReceiver_Cert|_T_Cert1|_T_Cert2|_T_Cert3|Cert_Final", names(df_turing)))

df_turing$O_UGProposer1 <- apply(cbind(df_turing[,grep("UGProposer1$", names(df_turing))]), 1, 
                                  function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$O_UGProposer2 <- apply(cbind(df_turing[,grep("UGProposer2$", names(df_turing))]), 1, 
                               function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$T_UGProposer <- apply(cbind(df_turing[,grep("_T_UGProposer", names(df_turing))]), 1, 
                               function(x) paste(x[!is.na(x)], collapse = ""))

df_turing$O_UGResponder1 <- apply(cbind(df_turing[,grep("UGResponder1$", names(df_turing))]), 1, 
                                 function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$O_UGResponder2 <- apply(cbind(df_turing[,grep("UGResponder2$", names(df_turing))]), 1, 
                                 function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$T_UGResponder <- apply(cbind(df_turing[,grep("_T_UGResponder", names(df_turing))]), 1, 
                                function(x) paste(x[!is.na(x)], collapse = ""))

df_turing$O_TGSender1 <- apply(cbind(df_turing[,grep("TGSender1$", names(df_turing))]), 1, 
                                  function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$O_TGSender2 <- apply(cbind(df_turing[,grep("TGSender2$", names(df_turing))]), 1, 
                                  function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$T_TGSender <- apply(cbind(df_turing[,grep("_T_TGSender", names(df_turing))]), 1, 
                                 function(x) paste(x[!is.na(x)], collapse = ""))


df_turing$O_TGReceiver1 <- apply(cbind(df_turing[,grep("TGReceiver1$", names(df_turing))]), 1, 
                               function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$O_TGReceiver2 <- apply(cbind(df_turing[,grep("TGReceiver2$", names(df_turing))]), 1, 
                               function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$T_TGReceiver <- apply(cbind(df_turing[,grep("_T_TGReceiver", names(df_turing))]), 1, 
                              function(x) paste(x[!is.na(x)], collapse = ""))

df_turing$O_PD1 <- apply(cbind(df_turing[,grep("PD1$", names(df_turing))]), 1, 
                                 function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$O_PD2 <- apply(cbind(df_turing[,grep("PD2$", names(df_turing))]), 1, 
                                  function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$T_PD <- apply(cbind(df_turing[,grep("_T_PD", names(df_turing))]), 1, 
                                function(x) paste(x[!is.na(x)], collapse = ""))

df_turing$O_SH1 <- apply(cbind(df_turing[,grep("SH1$", names(df_turing))]), 1, 
                         function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$O_SH2 <- apply(cbind(df_turing[,grep("SH2$", names(df_turing))]), 1, 
                         function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$T_SH <- apply(cbind(df_turing[,grep("_T_SH", names(df_turing))]), 1, 
                        function(x) paste(x[!is.na(x)], collapse = ""))

df_turing$O_C1 <- apply(cbind(df_turing[,grep("C1$", names(df_turing))]), 1, 
                         function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$O_C2 <- apply(cbind(df_turing[,grep("C2$", names(df_turing))]), 1, 
                         function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$T_C <- apply(cbind(df_turing[,grep("_T_C", names(df_turing))]), 1, 
                        function(x) paste(x[!is.na(x)], collapse = ""))

df_turing <- df_turing %>% dplyr::select(-grep("ODP_UGProposer|ODN_UGProposer|ODP_UGResponder|ODN_UGResponder|ODP_TGSender|ODN_TGSender|ODP_TGReceiver|ODN_TGReceiver|ODP_PD|ODN_PD|ODP_SH|ODN_SH|ODP_C|ODN_C|_T_UGProposer|_T_UGResponder|_T_TGSender|_T_TGReceiver|_T_|_T_FinalPrediction", names(df_turing)))

df_turing$AttentionCheck <- apply(cbind(df_turing[,grep("AttentionCheck", names(df_turing))]), 1, 
                                   function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$ExclusionDisturb <- apply(cbind(df_turing[,grep("ExclusionDisturb", names(df_turing))]), 1, 
                                   function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$ExclusionChatGPT <- apply(cbind(df_turing[,grep("ExclusionChatGPT", names(df_turing))]), 1, 
                                   function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$GeneralFeedback <- apply(cbind(df_turing[,grep("GeneralFeedback", names(df_turing))]), 1, 
                                   function(x) paste(x[!is.na(x)], collapse = ""))

df_turing <- df_turing %>% select(-grep("_AttentionCheck", names(df_turing)))
df_turing_backup <- df_turing
df_turing <- df_turing_backup

## Preparing decision ordering
df_turing$T_DecisionOdering <- gsub("Q1\\|AttentionCheck\\|","",df_turing$T_DecisionOdering)

df_turing$T_DecisionOdering <- gsub("\\|T_Cert1", "", df_turing$T_DecisionOdering)
df_turing$T_DecisionOdering <- gsub("\\|T_Cert2", "", df_turing$T_DecisionOdering)
df_turing$T_DecisionOdering <- gsub("\\|T_Cert3", "", df_turing$T_DecisionOdering)
df_turing$T_DecisionOdering <- gsub("\\|T_UGProp_Cert", "", df_turing$T_DecisionOdering)
df_turing$T_DecisionOdering <- gsub("\\|T_UGResp_Cert", "", df_turing$T_DecisionOdering)
df_turing$T_DecisionOdering <- gsub("\\|T_TGSender_Cert", "", df_turing$T_DecisionOdering)
df_turing$T_DecisionOdering <- gsub("\\|T_TGReceiver_Cert", "", df_turing$T_DecisionOdering)
df_turing$T_DecisionOdering <- gsub("T_", "", df_turing$T_DecisionOdering)

df_turing$ODP_DecisionOrdering <- gsub("ODP:", "", df_turing$ODP_DecisionOrdering)
df_turing$ODP_DecisionOrdering <- gsub("UG", "UGProposer\\|UGResponder", df_turing$ODP_DecisionOrdering)
df_turing$ODP_DecisionOrdering <- gsub("TG", "TGSender\\|TGReceiver", df_turing$ODP_DecisionOrdering)

df_turing$ODN_DecisionOrdering <- gsub("ODN:", "", df_turing$ODN_DecisionOrdering)
df_turing$ODN_DecisionOrdering <- gsub("UG", "UGProposer\\|UGResponder", df_turing$ODN_DecisionOrdering)
df_turing$ODN_DecisionOrdering <- gsub("TG", "TGSender\\|TGReceiver", df_turing$ODN_DecisionOrdering)

df_turing <- df_turing %>%
  separate(T_DecisionOdering, into = paste0("T_DecisionOrdering", 1:8), sep = "\\|", fill = "right") %>%
  separate(ODP_DecisionOrdering, into = paste0("ODP_DecisionOrdering", 1:7), sep = "\\|", fill = "right") %>%
  separate(ODN_DecisionOrdering, into = paste0("ODN_DecisionOrdering", 1:7), sep = "\\|", fill = "right")

df_turing$DecisionOrdering1 <- apply(cbind(df_turing[,grep("DecisionOrdering1", names(df_turing))]), 1, 
                                    function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$DecisionOrdering2 <- apply(cbind(df_turing[,grep("DecisionOrdering2", names(df_turing))]), 1, 
                                     function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$DecisionOrdering3 <- apply(cbind(df_turing[,grep("DecisionOrdering3", names(df_turing))]), 1, 
                                     function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$DecisionOrdering4 <- apply(cbind(df_turing[,grep("DecisionOrdering4", names(df_turing))]), 1, 
                                     function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$DecisionOrdering5 <- apply(cbind(df_turing[,grep("DecisionOrdering5", names(df_turing))]), 1, 
                                     function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$DecisionOrdering6 <- apply(cbind(df_turing[,grep("DecisionOrdering6", names(df_turing))]), 1, 
                                     function(x) paste(x[!is.na(x)], collapse = ""))
df_turing$DecisionOrdering7 <- apply(cbind(df_turing[,grep("DecisionOrdering7", names(df_turing))]), 1, 
                                     function(x) paste(x[!is.na(x)], collapse = ""))

df_turing <- df_turing %>% select(-grep("T_DecisionOrdering|ODN_DecisionOrdering|ODP_DecisionOrdering", names(df_turing)))
df_t <- df_turing %>% pivot_longer(cols = grep("Cert", names(df_turing)), names_to = c("Situation","trash"), names_sep = "_Cert",values_to = "Cert", names_prefix = "[OT]_") 

df_t$Situation[df_t$Situation == "Cert1"] <- df_t$DecisionOrdering1[df_t$Situation == "Cert1"]
df_t$Situation[df_t$Situation == "Cert2"] <- df_t$DecisionOrdering4[df_t$Situation == "Cert2"]
df_t$Situation[df_t$Situation == "Cert3"] <- df_t$DecisionOrdering7[df_t$Situation == "Cert3"]

df_t <- df_t %>%
  pivot_longer(cols = c(O_UGProposer1:T_C), names_to = "Situation2", values_to = "RatingIsAI", names_prefix = "[OT]_") %>%
  filter(Situation == Situation2) %>% filter(RatingIsAI != "") %>% rename(DurationInSec = `Duration (in seconds)`)

df_t <- df_t %>%
  pivot_longer(cols = c(DecisionOrdering1:DecisionOrdering7), names_to = "DecisionOrdering", values_to = "Situation3", names_prefix = "DecisionOrdering")

df_t$last_character <- ""
df_t$last_character[grepl("O", df_t$TreatmentCode)] <- substr(df_t$Situation[grepl("O", df_t$TreatmentCode)], nchar(df_t$Situation[grepl("O", df_t$TreatmentCode)]), nchar(df_t$Situation[grepl("O", df_t$TreatmentCode)]))
df_t$Situation4 <- paste(df_t$Situation3, df_t$last_character, sep = "")
df_t <- df_t %>% filter(Situation == Situation4) %>% select(RecordedDate, SubjectID, TreatmentCode, Situation, DecisionOrdering, RatingIsAI, Cert,DurationInSec, KnowledgeChatGPT, UsageChatGPT,
                                                            Age, Gender, Education, grep("StatementID", names(df_t)), SubjectID_Statement, count_ChangedBrowserWindow, AttentionCheck, ExclusionDisturb, ExclusionChatGPT, GeneralFeedback
                                                            )

# ,Prolific_PID

df_t$StatementID[df_t$StatementID == 0 & grepl("UG", df_t$Situation)] <- df_t$StatementID_UG[df_t$StatementID == 0 & grepl("UG", df_t$Situation)]
df_t$StatementID[df_t$StatementID == 0 & grepl("TG", df_t$Situation)] <- df_t$StatementID_TG[df_t$StatementID == 0 & grepl("TG", df_t$Situation)]
df_t$StatementID[df_t$StatementID == 0 & grepl("PD", df_t$Situation)] <- df_t$StatementID_PD[df_t$StatementID == 0 & grepl("PD", df_t$Situation)]
df_t$StatementID[df_t$StatementID == 0 & grepl("SH", df_t$Situation)] <- df_t$StatementID_SH[df_t$StatementID == 0 & grepl("SH", df_t$Situation)]
df_t$StatementID[df_t$StatementID == 0 & grepl("C", df_t$Situation)]  <- df_t$StatementID_C[df_t$StatementID == 0 & grepl("C", df_t$Situation)]

df_t <- df_t %>% select(-c(StatementID_UG:StatementID_C))
df_t[,5:17] <- sapply(df_t[,5:17],as.numeric)


## Preparing statements for turing analysis
file_list_opaqueP  <- list.files(path ="./data/ODP_statements/", pattern = ".csv", full.names = TRUE)
file_list_opaqueNP <- list.files(path = "./data/ODN_statements/", pattern = ".csv", full.names = TRUE)

data_list_statements_opaqueP  <- lapply(file_list_opaqueP, read_csv, show_col_types = FALSE)
data_list_statements_opaqueNP <- lapply(file_list_opaqueNP, read_csv, show_col_types = FALSE)

combined_statements_opaqueP <- data_list_statements_opaqueP %>% reduce(full_join, by = c("StatementID","TreatmentCode"))
combined_statements_opaqueP <- combined_statements_opaqueP %>% select(-grep("SubjectID|personality|StatementID2", names(combined_statements_opaqueP)))

combined_statements_opaqueNP <- data_list_statements_opaqueNP %>% reduce(full_join, by = c("StatementID","TreatmentCode"))
combined_statements_opaqueNP <- combined_statements_opaqueNP %>% select(-grep("SubjectID|ID.|StatementID2", names(combined_statements_opaqueNP)), -ID)

combined_statements_opaque  <- bind_rows(combined_statements_opaqueNP, combined_statements_opaqueP)

combined_statements_transparent <- read_csv("./data/T_statements/TuringStatementsFinal_Transparent.csv", show_col_types = FALSE)
combined_statements_transparent <- combined_statements_transparent %>% select(-c(SubjectID, personality_string, ID))

combined_statements <- bind_rows(combined_statements_opaque, combined_statements_transparent)

combined_statements <- combined_statements %>% rename_with(~paste("ST_", ., sep = ""), 3:30)

combined_statements$ST_C[combined_statements$ST_C == "mercury"] <- -2 
combined_statements$ST_C[combined_statements$ST_C == "venus"]   <- -1 
combined_statements$ST_C[combined_statements$ST_C == "earth"]   <-  0 
combined_statements$ST_C[combined_statements$ST_C == "mars"]    <-  1 
combined_statements$ST_C[combined_statements$ST_C == "saturn"]  <-  2


combined_statements$ST_AI_C[combined_statements$ST_AI_C == "mercury"] <- -2 
combined_statements$ST_AI_C[combined_statements$ST_AI_C == "venus"]   <- -1 
combined_statements$ST_AI_C[combined_statements$ST_AI_C == "earth"]   <-  0 
combined_statements$ST_AI_C[combined_statements$ST_AI_C == "mars"]    <-  1 
combined_statements$ST_AI_C[combined_statements$ST_AI_C == "saturn"]  <-  2 

combined_statements$ST_PD[combined_statements$ST_PD == "A"] <- 1 
combined_statements$ST_PD[combined_statements$ST_PD == "B"] <- 0 

combined_statements$ST_AI_PD[combined_statements$ST_AI_PD == "A"] <- 1 
combined_statements$ST_AI_PD[combined_statements$ST_AI_PD == "B"] <- 0 

combined_statements$ST_SH[combined_statements$ST_SH == "X"] <- 1 
combined_statements$ST_SH[combined_statements$ST_SH == "Y"] <- 0 

combined_statements$ST_AI_SH[combined_statements$ST_AI_SH == "X"] <- 1 
combined_statements$ST_AI_SH[combined_statements$ST_AI_SH == "Y"] <- 0 

combined_statements$ST_TGSender[combined_statements$ST_TGSender == "Yes"] <- 1 
combined_statements$ST_TGSender[combined_statements$ST_TGSender == "No"] <- 0 

combined_statements$ST_AI_TGSender[combined_statements$ST_AI_TGSender == "Yes"] <- 1 
combined_statements$ST_AI_TGSender[combined_statements$ST_AI_TGSender == "yes"] <- 1 
combined_statements$ST_AI_TGSender[combined_statements$ST_AI_TGSender == "No"] <- 0 

combined_statements$ST_C <- as.numeric(combined_statements$ST_C)
combined_statements$ST_AI_C <- as.numeric(combined_statements$ST_AI_C)
combined_statements$ST_PD <- as.numeric(combined_statements$ST_PD)
combined_statements$ST_AI_PD <- as.numeric(combined_statements$ST_AI_PD)
combined_statements$ST_SH <- as.numeric(combined_statements$ST_SH)
combined_statements$ST_AI_SH <- as.numeric(combined_statements$ST_AI_SH)
combined_statements$ST_TGSender <- as.numeric(combined_statements$ST_TGSender)
combined_statements$ST_AI_TGSender <- as.numeric(combined_statements$ST_AI_TGSender)

pattern <- "^ST_AI_.*_JUS$"
combined_statements_long <- combined_statements %>% pivot_longer(cols = grep(pattern, names(combined_statements), value = TRUE), 
                                                                 names_to = "Situation", values_to = "AI_JUS", names_prefix = "ST_AI_")
combined_statements_long$Situation <- gsub("_JUS", "", combined_statements_long$Situation)

pattern <- "^ST_.*_JUS$"
combined_statements_long <- combined_statements_long %>% pivot_longer(cols = grep(pattern, names(combined_statements_long), value = TRUE), 
                                                                 names_to = "Situation2", values_to = "Human_JUS", names_prefix = "ST_")
combined_statements_long$Situation2 <- gsub("_JUS", "", combined_statements_long$Situation2)
combined_statements_long <- combined_statements_long %>% filter(Situation == Situation2) %>% select(-Situation2)

pattern <- "^ST_AI"
combined_statements_long <- combined_statements_long %>% pivot_longer(cols = grep(pattern, names(combined_statements_long), value = TRUE), 
                                                                      names_to = "Situation2", values_to = "AI_DEC", names_prefix = "ST_")
combined_statements_long$Situation2 <- gsub("AI_", "", combined_statements_long$Situation2)
combined_statements_long <- combined_statements_long %>% filter(Situation == Situation2) %>% select(-Situation2)

pattern <- "^ST_"
combined_statements_long <- combined_statements_long %>% pivot_longer(cols = grep(pattern, names(combined_statements_long), value = TRUE), 
                                                                      names_to = "Situation2", values_to = "Human_DEC", names_prefix = "ST_")
combined_statements_long <- combined_statements_long %>% filter(Situation == Situation2) %>% select(-Situation2)
combined_statements_long <- combined_statements_long %>% rename(SituationToMerge = Situation)

df_t$SituationToMerge <- df_t$Situation
df_t$SituationToMerge[grepl("O", df_t$TreatmentCode)] <- substr(df_t$SituationToMerge[grepl("O", df_t$TreatmentCode)], 1, nchar(df_t$SituationToMerge[grepl("O", df_t$TreatmentCode)]) - 1)
df_t <- left_join(df_t, combined_statements_long)
df_t <- df_t %>% select(-SituationToMerge)

write_csv(df_t,"./data/TuringData.csv")
