library(data.table); library(psy); library(nlme); library(lmboot); library(MASS); library(lme4); library(rptR)

StakesList = rep(c(1,2,4,5),4)
ProbaList = rep(c(0.80,0.75,0.70,0.65),each=4)

### DAY 1 ###
datainit_1=as.data.frame(fread('RepeatedEconomicGames/Repeated_trust_game+-+day+1_November+8%2C+2021_13.45.csv',stringsAsFactors=FALSE))
data= datainit_1[3:nrow(datainit_1),]

PID = as.character(data$Q52)
NumTraining = as.numeric(data$NumTraining)

data$Q91_1[data$Q91_1==''] = '0'
data$Q91_2[data$Q91_2==''] = '0'
data$Q88_1[data$Q88_1==''] = '0'
data$Q88_2[data$Q88_2==''] = '0'
data$Q94_1[data$Q94_1==''] = '0'
data$Q94_2[data$Q94_2==''] = '0'
ErrorTraining = rep(0,length(NumTraining))#as.numeric(data$ErrorTraining)
ErrorTraining[as.numeric(data$Q91_1)!= 5 & as.numeric(data$Q91_2)!= 5 & as.numeric(data$Q88_1)!= 0 & as.numeric(data$Q88_2)!= 20 & as.numeric(data$Q94_1)!= 10 & as.numeric(data$Q94_2)!= 10] =1
ErrorTraining[NumTraining==1] = 0

totalGains = as.numeric(data$totalGains)
totalBonus = as.numeric(data$totalBonus)

TrustDecisions_List = unlist(lapply(colnames(data), function(x) length(grep('_Q38',x))))
TrustDecisions = data[, TrustDecisions_List ==1]
TrustDecisions[TrustDecisions=='Not at all'] = '1'
TrustDecisions[TrustDecisions=='Extremely'] = '9'
TrustDecisions=sapply(TrustDecisions, as.numeric)

TrustRT_List = unlist(lapply(colnames(data),function(x) length(grep('Q25_Page Submit',x))))
TrustRT=sapply(data[, TrustRT_List ==1], as.numeric)
RTBelow200 = rowSums(TrustRT<0.200)/16

TrustExplicit_1 = data$Q60
TrustExplicit_1[TrustExplicit_1=='you canât be too careful'] = '0'
TrustExplicit_1[TrustExplicit_1=='most people can be trusted'] = '10'
TrustExplicit_1[TrustExplicit_1=="don't know"] = ''
TrustExplicit_1 =sapply(TrustExplicit_1, as.numeric)

TrustExplicit_2 = data$Q61
TrustExplicit_2[TrustExplicit_2 =='most people would try to take advantage of me'] = '0'
TrustExplicit_2[TrustExplicit_2 =='most people would try to be fair'] = '10'
TrustExplicit_2[TrustExplicit_2 =="don't know"] = ''
TrustExplicit_2 =sapply(TrustExplicit_2, as.numeric)

TrustExplicit_3 = data$Q62
TrustExplicit_3[TrustExplicit_3 =='people mostly look out for themselves'] = '0'
TrustExplicit_3[TrustExplicit_3 =='people mostly try to be helpful'] = '10'
TrustExplicit_3[TrustExplicit_3 =="don't know"] = ''
TrustExplicit_3 =sapply(TrustExplicit_3, as.numeric)

MeanTrustGame = rowMeans(TrustDecisions)

matrix_demo_D1 = data.frame('PID' = PID, 'ErrorTraining_D1' = ErrorTraining, 'NumTraining_D1' = NumTraining, 'totalGains_D1' = totalGains, 'totalBonus_D1' = totalBonus, 'TrustExplicit_1_D1' = TrustExplicit_1,'TrustExplicit_2_D1' = TrustExplicit_2,'TrustExplicit_3_D1' = TrustExplicit_3,'MeanTrustGame_D1'= MeanTrustGame,'RTBelow200_D1'= RTBelow200)
matrix_demo_D1$PID=as.character(matrix_demo_D1$PID)

matrix_demo_D1_long = data.frame('PID' = PID, 'ErrorTraining' = ErrorTraining, 'NumTraining' = NumTraining, 'totalGains' = totalGains, 'totalBonus' = totalBonus, 'TrustExplicit_1' = TrustExplicit_1,'TrustExplicit_2' = TrustExplicit_2,'TrustExplicit_3' = TrustExplicit_3,'Day' = 1)
matrix_demo_D1_long$PID=as.character(matrix_demo_D1_long$PID)

matrix_choice_D1 = data.frame('PID' = rep(PID,length(StakesList)), 'Stakes' = rep(StakesList,each=nrow(data)), 'ProbaList' = rep(ProbaList,each=nrow(data)),'TrustDecisions' = c(TrustDecisions),'TrustRT' = c(TrustRT), 'Day' = 1)
matrix_choice_D1$PID=as.character(matrix_choice_D1$PID)

### DAY 2 ###
datainit_2=as.data.frame(fread('RepeatedEconomicGames/Repeated_trust_game+-+day+2_November+8%2C+2021_13.46.csv',stringsAsFactors=FALSE))
data= datainit_2[3:nrow(datainit_2),]

PID = as.character(data$Q52)

ErrorTraining = as.numeric(data$ErrorTraining)
NumTraining = as.numeric(data$NumTraining)
totalGains = as.numeric(data$totalGains)
totalBonus = as.numeric(data$totalBonus)

TrustDecisions_List = unlist(lapply(colnames(data), function(x) length(grep('_Q38',x))))
TrustDecisions = data[, TrustDecisions_List ==1]
TrustDecisions[TrustDecisions=='Not at all'] = '1'
TrustDecisions[TrustDecisions=='Extremely'] = '9'
TrustDecisions=sapply(TrustDecisions, as.numeric)

MeanTrustGame = rowMeans(TrustDecisions)

TrustRT_List = unlist(lapply(colnames(data),function(x) length(grep('Q25_Page Submit',x))))
TrustRT=sapply(data[, TrustRT_List ==1], as.numeric)
RTBelow200 = rowSums(TrustRT<0.200)/16

matrix_demo_D2 = data.frame('PID' = PID, 'ErrorTraining_D2' = ErrorTraining, 'NumTraining_D2' = NumTraining, 'totalGains_D2' = totalGains, 'totalBonus_D2' = totalBonus,'MeanTrustGame_D2'= MeanTrustGame,'RTBelow200_D2'= RTBelow200)
matrix_demo_D2$PID=as.character(matrix_demo_D2$PID)

matrix_choice_D2 = data.frame('PID' = rep(PID,length(StakesList)), 'Stakes' = rep(StakesList,each=nrow(data)), 'ProbaList' = rep(ProbaList,each=nrow(data)),'TrustDecisions' = c(TrustDecisions), 'TrustRT' = c(TrustRT), 'Day' = 2)
matrix_choice_D2$PID=as.character(matrix_choice_D2$PID)

### DAY 3 ###
datainit_3=as.data.frame(fread('RepeatedEconomicGames/Repeated_trust_game+-+day+3_November+8%2C+2021_13.46.csv',stringsAsFactors=FALSE))
data= datainit_3[3:nrow(datainit_3),]

PID = as.character(data$Q52)

ErrorTraining = as.numeric(data$ErrorTraining)
NumTraining = as.numeric(data$NumTraining)
totalGains = as.numeric(data$totalGains)
totalBonus = as.numeric(data$totalBonus)

TrustDecisions_List = unlist(lapply(colnames(data), function(x) length(grep('_Q38',x))))
TrustDecisions = data[, TrustDecisions_List ==1]
TrustDecisions[TrustDecisions=='Not at all'] = '1'
TrustDecisions[TrustDecisions=='Extremely'] = '9'
TrustDecisions=sapply(TrustDecisions, as.numeric)

MeanTrustGame = rowMeans(TrustDecisions)

TrustRT_List = unlist(lapply(colnames(data),function(x) length(grep('Q25_Page Submit',x))))
TrustRT=sapply(data[, TrustRT_List ==1], as.numeric)
RTBelow200 = rowSums(TrustRT<0.200)/16

matrix_demo_D3 = data.frame('PID' = PID, 'ErrorTraining_D3' = ErrorTraining, 'NumTraining_D3' = NumTraining, 'totalGains_D3' = totalGains, 'totalBonus_D3' = totalBonus,'MeanTrustGame_D3'= MeanTrustGame,'RTBelow200_D3'= RTBelow200)
matrix_demo_D3 $PID=as.character(matrix_demo_D3 $PID)

matrix_choice_D3 = data.frame('PID' = rep(PID,length(StakesList)), 'Stakes' = rep(StakesList,each=nrow(data)), 'ProbaList' = rep(ProbaList,each=nrow(data)),'TrustDecisions' = c(TrustDecisions), 'TrustRT' = c(TrustRT), 'Day' = 3)
matrix_choice_D3 $PID=as.character(matrix_choice_D3 $PID)

### DAY 4 ###
datainit_4=as.data.frame(fread('RepeatedEconomicGames/Repeated_trust_game+-+day+4_November+8%2C+2021_13.47.csv',stringsAsFactors=FALSE))
data= datainit_4[3:nrow(datainit_4),]

PID = as.character(data$Q52)

ErrorTraining = as.numeric(data$ErrorTraining)
NumTraining = as.numeric(data$NumTraining)
totalGains = as.numeric(data$totalGains)
totalBonus = as.numeric(data$totalBonus)

TrustDecisions_List = unlist(lapply(colnames(data), function(x) length(grep('_Q38',x))))
TrustDecisions = data[, TrustDecisions_List ==1]
TrustDecisions[TrustDecisions=='Not at all'] = '1'
TrustDecisions[TrustDecisions=='Extremely'] = '9'
TrustDecisions=sapply(TrustDecisions, as.numeric)

MeanTrustGame = rowMeans(TrustDecisions)

TrustRT_List = unlist(lapply(colnames(data),function(x) length(grep('Q25_Page Submit',x))))
TrustRT=sapply(data[, TrustRT_List ==1], as.numeric)
RTBelow200 = rowSums(TrustRT<0.200)/16

matrix_demo_D4 = data.frame('PID' = PID, 'ErrorTraining_D4' = ErrorTraining, 'NumTraining_D4' = NumTraining, 'totalGains_D4' = totalGains, 'totalBonus_D4' = totalBonus,'MeanTrustGame_D4'= MeanTrustGame,'RTBelow200_D4'= RTBelow200)
matrix_demo_D4 $PID=as.character(matrix_demo_D4 $PID)

matrix_choice_D4 = data.frame('PID' = rep(PID,length(StakesList)), 'Stakes' = rep(StakesList,each=nrow(data)), 'ProbaList' = rep(ProbaList,each=nrow(data)),'TrustDecisions' = c(TrustDecisions), 'TrustRT' = c(TrustRT), 'Day' = 4)
matrix_choice_D4 $PID=as.character(matrix_choice_D4 $PID)

### DAY 5 ###
datainit_5=as.data.frame(fread('RepeatedEconomicGames/Repeated_trust_game+-+day+5_November+8%2C+2021_13.47.csv',stringsAsFactors=FALSE))
data= datainit_5[3:nrow(datainit_5),]
data[data$Q52=="613c9c83c9cd63d09d4ed30 ","Q52"] = "613c9c83c9cd63d09d4ed300"
data[data$IPAddress=='51.9.95.189','Q52']='615c49ef513583533427c961'


PID = as.character(data$Q52)

ErrorTraining = as.numeric(data$ErrorTraining)
NumTraining = as.numeric(data$NumTraining)
totalGains = as.numeric(data$totalGains)
totalBonus = as.numeric(data$totalBonus)

TrustDecisions_List = unlist(lapply(colnames(data), function(x) length(grep('_Q38',x))))
TrustDecisions = data[, TrustDecisions_List ==1]
TrustDecisions[TrustDecisions=='Not at all'] = '1'
TrustDecisions[TrustDecisions=='Extremely'] = '9'
TrustDecisions=sapply(TrustDecisions, as.numeric)

MeanTrustGame = rowMeans(TrustDecisions)

TrustRT_List = unlist(lapply(colnames(data),function(x) length(grep('Q25_Page Submit',x))))
TrustRT=sapply(data[, TrustRT_List ==1], as.numeric)
RTBelow200 = rowSums(TrustRT<0.200)/16

matrix_demo_D5 = data.frame('PID' = PID, 'ErrorTraining_D5' = ErrorTraining, 'NumTraining_D5' = NumTraining, 'totalGains_D5' = totalGains, 'totalBonus_D5' = totalBonus,'MeanTrustGame_D5'= MeanTrustGame,'RTBelow200_D5'= RTBelow200)
matrix_demo_D5 $PID=as.character(matrix_demo_D5 $PID)

matrix_choice_D5 = data.frame('PID' = rep(PID,length(StakesList)), 'Stakes' = rep(StakesList,each=nrow(data)), 'ProbaList' = rep(ProbaList,each=nrow(data)),'TrustDecisions' = c(TrustDecisions), 'TrustRT' = c(TrustRT), 'Day' = 5)
matrix_choice_D5 $PID=as.character(matrix_choice_D5 $PID)

### DAY 6 ###
datainit_6=as.data.frame(fread('RepeatedEconomicGames/Repeated_trust_game+-+day+6_November+8%2C+2021_13.47.csv',stringsAsFactors=FALSE))
data= datainit_6[3:nrow(datainit_6),]

PID = as.character(data$Q52)

ErrorTraining = as.numeric(data$ErrorTraining)
NumTraining = as.numeric(data$NumTraining)
totalGains = as.numeric(data$totalGains)
totalBonus = as.numeric(data$totalBonus)

TrustDecisions_List = unlist(lapply(colnames(data), function(x) length(grep('_Q38',x))))
TrustDecisions = data[, TrustDecisions_List ==1]
TrustDecisions[TrustDecisions=='Not at all'] = '1'
TrustDecisions[TrustDecisions=='Extremely'] = '9'
TrustDecisions=sapply(TrustDecisions, as.numeric)

MeanTrustGame = rowMeans(TrustDecisions)

TrustRT_List = unlist(lapply(colnames(data),function(x) length(grep('Q25_Page Submit',x))))
TrustRT=sapply(data[, TrustRT_List ==1], as.numeric)
RTBelow200 = rowSums(TrustRT<0.200)/16

matrix_demo_D6 = data.frame('PID' = PID, 'ErrorTraining_D6' = ErrorTraining, 'NumTraining_D6' = NumTraining, 'totalGains_D6' = totalGains, 'totalBonus_D6' = totalBonus,'MeanTrustGame_D6'= MeanTrustGame,'RTBelow200_D6'= RTBelow200)
matrix_demo_D6 $PID=as.character(matrix_demo_D6 $PID)

matrix_choice_D6 = data.frame('PID' = rep(PID,length(StakesList)), 'Stakes' = rep(StakesList,each=nrow(data)), 'ProbaList' = rep(ProbaList,each=nrow(data)),'TrustDecisions' = c(TrustDecisions), 'TrustRT' = c(TrustRT), 'Day' = 6)
matrix_choice_D6 $PID=as.character(matrix_choice_D6 $PID)

### DAY 7 ###
datainit_7=as.data.frame(fread('RepeatedEconomicGames/Repeated_trust_game+-+day+7_November+8%2C+2021_13.49.csv',stringsAsFactors=FALSE))
data= datainit_7[3:nrow(datainit_7),]

PID = as.character(data$Q52)

ErrorTraining = as.numeric(data$ErrorTraining)
NumTraining = as.numeric(data$NumTraining)
totalGains = as.numeric(data$totalGains)
totalBonus = as.numeric(data$totalBonus)

TrustDecisions_List = unlist(lapply(colnames(data), function(x) length(grep('_Q38',x))))
TrustDecisions = data[, TrustDecisions_List ==1]
TrustDecisions[TrustDecisions=='Not at all'] = '1'
TrustDecisions[TrustDecisions=='Extremely'] = '9'
TrustDecisions=sapply(TrustDecisions, as.numeric)

MeanTrustGame = rowMeans(TrustDecisions)


TrustRT_List = unlist(lapply(colnames(data),function(x) length(grep('Q25_Page Submit',x))))
TrustRT=sapply(data[, TrustRT_List ==1], as.numeric)
RTBelow200 = rowSums(TrustRT<0.200)/16

matrix_demo_D7 = data.frame('PID' = PID, 'ErrorTraining_D7' = ErrorTraining, 'NumTraining_D7' = NumTraining, 'totalGains_D7' = totalGains, 'totalBonus_D7' = totalBonus,'MeanTrustGame_D7'= MeanTrustGame,'RTBelow200_D7'= RTBelow200)
matrix_demo_D7 $PID=as.character(matrix_demo_D7 $PID)

matrix_choice_D7 = data.frame('PID' = rep(PID,length(StakesList)), 'Stakes' = rep(StakesList,each=nrow(data)), 'ProbaList' = rep(ProbaList,each=nrow(data)),'TrustDecisions' = c(TrustDecisions), 'TrustRT' = c(TrustRT), 'Day' = 7)
matrix_choice_D7 $PID=as.character(matrix_choice_D7 $PID)

### DAY 8 ###
datainit_8=as.data.frame(fread('RepeatedEconomicGames/Repeated_trust_game+-+day+8_November+8%2C+2021_13.48.csv',stringsAsFactors=FALSE))
data= datainit_8[3:nrow(datainit_8),]

PID = as.character(data$Q52)

ErrorTraining = as.numeric(data$ErrorTraining)
NumTraining = as.numeric(data$NumTraining)
totalGains = as.numeric(data$totalGains)
totalBonus = as.numeric(data$totalBonus)

TrustDecisions_List = unlist(lapply(colnames(data), function(x) length(grep('_Q38',x))))
TrustDecisions = data[, TrustDecisions_List ==1]
TrustDecisions[TrustDecisions=='Not at all'] = '1'
TrustDecisions[TrustDecisions=='Extremely'] = '9'
TrustDecisions=sapply(TrustDecisions, as.numeric)

MeanTrustGame = rowMeans(TrustDecisions)


TrustRT_List = unlist(lapply(colnames(data),function(x) length(grep('Q25_Page Submit',x))))
TrustRT=sapply(data[, TrustRT_List ==1], as.numeric)
RTBelow200 = rowSums(TrustRT<0.200)/16

matrix_demo_D8 = data.frame('PID' = PID, 'ErrorTraining_D8' = ErrorTraining, 'NumTraining_D8' = NumTraining, 'totalGains_D8' = totalGains, 'totalBonus_D8' = totalBonus,'MeanTrustGame_D8'= MeanTrustGame,'RTBelow200_D8'= RTBelow200)
matrix_demo_D8 $PID=as.character(matrix_demo_D8 $PID)

matrix_choice_D8 = data.frame('PID' = rep(PID,length(StakesList)), 'Stakes' = rep(StakesList,each=nrow(data)), 'ProbaList' = rep(ProbaList,each=nrow(data)),'TrustDecisions' = c(TrustDecisions), 'TrustRT' = c(TrustRT), 'Day' = 8)
matrix_choice_D8 $PID=as.character(matrix_choice_D8 $PID)

### DAY 9 ###
datainit_9=as.data.frame(fread('RepeatedEconomicGames/Repeated_trust_game+-+day+9_November+8%2C+2021_13.48.csv',stringsAsFactors=FALSE))
data= datainit_9[3:nrow(datainit_9),]

PID = as.character(data$Q52)

ErrorTraining = as.numeric(data$ErrorTraining)
NumTraining = as.numeric(data$NumTraining)
totalGains = as.numeric(data$totalGains)
totalBonus = as.numeric(data$totalBonus)

TrustDecisions_List = unlist(lapply(colnames(data), function(x) length(grep('_Q38',x))))
TrustDecisions = data[, TrustDecisions_List ==1]
TrustDecisions[TrustDecisions=='Not at all'] = '1'
TrustDecisions[TrustDecisions=='Extremely'] = '9'
TrustDecisions=sapply(TrustDecisions, as.numeric)

MeanTrustGame = rowMeans(TrustDecisions)

TrustRT_List = unlist(lapply(colnames(data),function(x) length(grep('Q25_Page Submit',x))))
TrustRT=sapply(data[, TrustRT_List ==1], as.numeric)
RTBelow200 = rowSums(TrustRT<0.200)/16

matrix_demo_D9 = data.frame('PID' = PID, 'ErrorTraining_D9' = ErrorTraining, 'NumTraining_D9' = NumTraining, 'totalGains_D9' = totalGains, 'totalBonus_D9' = totalBonus,'MeanTrustGame_D9'= MeanTrustGame,'RTBelow200_D9'= RTBelow200)
matrix_demo_D9 $PID=as.character(matrix_demo_D9 $PID)

matrix_choice_D9 = data.frame('PID' = rep(PID,length(StakesList)), 'Stakes' = rep(StakesList,each=nrow(data)), 'ProbaList' = rep(ProbaList,each=nrow(data)),'TrustDecisions' = c(TrustDecisions), 'TrustRT' = c(TrustRT), 'Day' = 9)
matrix_choice_D9 $PID=as.character(matrix_choice_D9 $PID)

### DAY 10 ###
datainit_10=as.data.frame(fread('RepeatedEconomicGames/Repeated_trust_game+-+day+10_November+8%2C+2021_13.48.csv',stringsAsFactors=FALSE))
data= datainit_10[3:nrow(datainit_10),]
data= data[data$Q52!='',]

PID = as.character(data$Q52)

NumTraining = as.numeric(data$NumTraining)

data$Q91_1[data$Q91_1==''] = '0'
data$Q91_2[data$Q91_2==''] = '0'
data$Q88_1[data$Q88_1==''] = '0'
data$Q88_2[data$Q88_2==''] = '0'
data$Q94_1[data$Q94_1==''] = '0'
data$Q94_2[data$Q94_2==''] = '0'
ErrorTraining = rep(0,length(NumTraining))#as.numeric(data$ErrorTraining)
ErrorTraining[as.numeric(data$Q91_1)!= 5 & as.numeric(data$Q91_2)!= 5 & as.numeric(data$Q88_1)!= 0 & as.numeric(data$Q88_2)!= 20 & as.numeric(data$Q94_1)!= 10 & as.numeric(data$Q94_2)!= 10] =1
ErrorTraining[NumTraining==1] = 0

totalGains = as.numeric(data$totalGains)
totalBonus = as.numeric(data$totalBonus)

TrustDecisions_List = unlist(lapply(colnames(data), function(x) length(grep('_Q38',x))))
TrustDecisions = data[, TrustDecisions_List ==1]
TrustDecisions[TrustDecisions=='Not at all'] = '1'
TrustDecisions[TrustDecisions=='Extremely'] = '9'
TrustDecisions=sapply(TrustDecisions, as.numeric)

TrustRT_List = unlist(lapply(colnames(data),function(x) length(grep('Q25_Page Submit',x))))
TrustRT=sapply(data[, TrustRT_List ==1], as.numeric)
RTBelow200 = rowSums(TrustRT<0.200)/16

TrustExplicit_1 = data$Q60
TrustExplicit_1[TrustExplicit_1=='you canât be too careful'] = '0'
TrustExplicit_1[TrustExplicit_1=='most people can be trusted'] = '10'
TrustExplicit_1[TrustExplicit_1=="don't know"] = ''
TrustExplicit_1 =sapply(TrustExplicit_1, as.numeric)

TrustExplicit_2 = data$Q61
TrustExplicit_2[TrustExplicit_2 =='most people would try to take advantage of me'] = '0'
TrustExplicit_2[TrustExplicit_2 =='most people would try to be fair'] = '10'
TrustExplicit_2[TrustExplicit_2 =="don't know"] = ''
TrustExplicit_2 =sapply(TrustExplicit_2, as.numeric)

TrustExplicit_3 = data$Q62
TrustExplicit_3[TrustExplicit_3 =='people mostly look out for themselves'] = '0'
TrustExplicit_3[TrustExplicit_3 =='people mostly try to be helpful'] = '10'
TrustExplicit_3[TrustExplicit_3 =="don't know"] = ''
TrustExplicit_3 =sapply(TrustExplicit_3, as.numeric)

Gender = data$Q49
Age = as.numeric(data$Q50)
ChildhoodResources = data[,c('Q53_1','Q53_2','Q53_3')] #Q53
ChildhoodResources[ChildhoodResources=='Strongly disagree'] = 1
ChildhoodResources[ChildhoodResources=='Strongly agree'] = 7
ChildhoodResources_mat =sapply(ChildhoodResources, as.numeric)
ChildhoodResources = rowSums(ChildhoodResources_mat)

ChildhoodPredictability = data[,c('Q53_4','Q53_5','Q53_6')]
ChildhoodPredictability[ChildhoodPredictability =='Strongly disagree'] = 1
ChildhoodPredictability[ChildhoodPredictability =='Strongly agree'] = 7
ChildhoodPredictability_mat =sapply(ChildhoodPredictability, as.numeric)
ChildhoodPredictability = rowSums(ChildhoodPredictability_mat)

AdultResources = data[,c('Q56_1','Q56_2','Q56_3')]
AdultResources[AdultResources =='Strongly disagree'] = 1
AdultResources[AdultResources =='Strongly agree'] = 7
AdultResources_mat =sapply(AdultResources, as.numeric)
AdultResources = rowSums(AdultResources_mat)

Health = data$Q57
Health[Health=='Bad']='1'
Health[Health=='Acceptable']='2'
Health[Health=='Good']='3'
Health[Health=='Excellent']='4'
Health=as.numeric(Health)

EffortHealthSafety = as.numeric(data$Q58)

DataSharing = as.numeric(data$Q39=='Yes')
ExtraHelp = as.numeric(data$Q41=='Yes, I can take a look')
SumExtraHelp = DataSharing+ ExtraHelp
CommentHelp=data$Q42

MeanTrustGame = rowMeans(TrustDecisions)

matrix_demo_D10 = data.frame('PID' = PID, 'ErrorTraining_D10' = ErrorTraining, 'NumTraining_D10' = NumTraining, 'totalGains_D10' = totalGains, 'totalBonus_D10' = totalBonus, 'TrustExplicit_1_D10' = TrustExplicit_1,'TrustExplicit_2_D10' = TrustExplicit_2,'TrustExplicit_3_D10' = TrustExplicit_3,'Gender'=Gender,'Age'=Age, 'ChildhoodResources'= ChildhoodResources,'ChildhoodPredictability' = ChildhoodPredictability, 'AdultResources'= AdultResources,'Health'= Health, 'EffortHealthSafety' = EffortHealthSafety, 'DataSharing'= DataSharing, 'ExtraHelp'= ExtraHelp, 'SumExtraHelp'= SumExtraHelp,'CommentHelp' = CommentHelp,'MeanTrustGame_D10' = MeanTrustGame,'RTBelow200_D10'= RTBelow200)
matrix_demo_D10 $PID=as.character(matrix_demo_D10 $PID)

matrix_demo_D10_long = data.frame('PID' = PID, 'ErrorTraining' = ErrorTraining, 'NumTraining' = NumTraining, 'totalGains' = totalGains, 'totalBonus' = totalBonus, 'TrustExplicit_1' = TrustExplicit_1,'TrustExplicit_2' = TrustExplicit_2,'TrustExplicit_3' = TrustExplicit_3,'Day' = 10)
matrix_demo_D10_long $PID=as.character(matrix_demo_D10_long $PID)

matrix_choice_D10 = data.frame('PID' = rep(PID,length(StakesList)), 'Stakes' = rep(StakesList,each=nrow(data)), 'ProbaList' = rep(ProbaList,each=nrow(data)),'TrustDecisions' = c(TrustDecisions), 'TrustRT' = c(TrustRT), 'Day' = 10)
matrix_choice_D10 $PID=as.character(matrix_choice_D10 $PID)


## COMBINE
matrix_demo_D1init = merge(matrix_demo_D1, matrix_demo_D10[,c('PID','ChildhoodResources','ChildhoodPredictability','AdultResources','Health','EffortHealthSafety')],by='PID',all.x=TRUE)
matrix_demo_D1init= matrix_demo_D1init[matrix_demo_D1init$PID!='',]
matrix_demo_D1init$Questionnaire = rowMeans(matrix_demo_D1init[,c('TrustExplicit_1_D1','TrustExplicit_2_D1','TrustExplicit_3_D1')],na.rm=TRUE)

t.test(Questionnaire~is.na(ChildhoodPredictability), matrix_demo_D1init,var.eq=TRUE)
t.test(MeanTrustGame_D1 ~ is.na(ChildhoodResources), matrix_demo_D1init,var.eq=TRUE)

matrixRT = merge(matrix_demo_D1[,c('PID','RTBelow200_D1')], matrix_demo_D2[,c('PID','RTBelow200_D2')],by='PID')
matrixRT = merge(matrixRT, matrix_demo_D3[,c('PID','RTBelow200_D3')],by='PID')
matrixRT = merge(matrixRT, matrix_demo_D4[,c('PID','RTBelow200_D4')],by='PID')
matrixRT = merge(matrixRT, matrix_demo_D5[,c('PID','RTBelow200_D5')],by='PID')
matrixRT = merge(matrixRT, matrix_demo_D6[,c('PID','RTBelow200_D6')],by='PID')
matrixRT = merge(matrixRT, matrix_demo_D7[,c('PID','RTBelow200_D7')],by='PID')
matrixRT = merge(matrixRT, matrix_demo_D8[,c('PID','RTBelow200_D8')],by='PID')
matrixRT = merge(matrixRT, matrix_demo_D9[,c('PID','RTBelow200_D9')],by='PID')
matrixRT = merge(matrixRT, matrix_demo_D10[,c('PID','RTBelow200_D10')],by='PID')

matrixRT = data.frame('PID'= matrixRT$PID,'RTBelow200' = rowMeans(matrixRT[,2:11]))
matrixRT[,'ExclusionRT'] = matrixRT[,'RTBelow200']>.9

matrix_demo_D1 = merge(matrix_demo_D1, matrix_demo_D10[,c('PID','ErrorTraining_D10')],by='PID')
matrix_demo_D1= matrix_demo_D1[matrix_demo_D1$ErrorTraining_D1==0 & matrix_demo_D1$ErrorTraining_D10==0,]

matrix_demo_D10 = merge(matrix_demo_D10, matrix_demo_D1[,c('PID','ErrorTraining_D1')],by='PID')
matrix_demo_D10= matrix_demo_D10[matrix_demo_D10 $ErrorTraining_D1==0 & matrix_demo_D10$ErrorTraining_D10==0,]


matrix_choice = as.data.frame(rbind(matrix_choice_D1,matrix_choice_D2,matrix_choice_D3,matrix_choice_D4,matrix_choice_D5,matrix_choice_D6,matrix_choice_D7,matrix_choice_D8,matrix_choice_D9,matrix_choice_D10))
matrix_choice =  merge(matrix_choice, matrix_demo_D10[,c('PID','ErrorTraining_D10')],by='PID')

matrix_choice_D1 =  merge(matrix_choice_D1, matrix_demo_D10[,c('PID','ErrorTraining_D10')],by='PID')
matrix_choice_D1= matrix_choice_D1[matrix_choice_D1$PID!='',]

ForTest = merge(matrix_demo_D1[,c('PID','TrustExplicit_1_D1','TrustExplicit_2_D1','TrustExplicit_3_D1','MeanTrustGame_D1')],matrix_demo_D2[,c('PID','MeanTrustGame_D2')],by='PID')
ForTest = merge(ForTest,matrix_demo_D3[,c('PID','MeanTrustGame_D3')],by='PID')
ForTest = merge(ForTest,matrix_demo_D4[,c('PID','MeanTrustGame_D4')],by='PID')
ForTest = merge(ForTest,matrix_demo_D5[,c('PID','MeanTrustGame_D5')],by='PID')
ForTest = merge(ForTest,matrix_demo_D6[,c('PID','MeanTrustGame_D6')],by='PID')
ForTest = merge(ForTest,matrix_demo_D7[,c('PID','MeanTrustGame_D7')],by='PID')
ForTest = merge(ForTest,matrix_demo_D8[,c('PID','MeanTrustGame_D8')],by='PID')
ForTest = merge(ForTest,matrix_demo_D9[,c('PID','MeanTrustGame_D9')],by='PID')
ForTest = merge(ForTest,matrix_demo_D10[,c('PID','TrustExplicit_1_D10','TrustExplicit_2_D10','TrustExplicit_3_D10','MeanTrustGame_D10','SumExtraHelp',"ExtraHelp","DataSharing", 'ChildhoodResources','ChildhoodPredictability','AdultResources','Health','EffortHealthSafety','Age','Gender')],by='PID')


matrix_mean = data.frame('PID' = c(matrix_demo_D1$PID,matrix_demo_D2$PID,matrix_demo_D3$PID,matrix_demo_D4$PID,matrix_demo_D5$PID,matrix_demo_D6$PID,matrix_demo_D7$PID,matrix_demo_D8$PID,matrix_demo_D9$PID,matrix_demo_D10$PID),
                         'MeanTrustGame'= c(matrix_demo_D1$MeanTrustGame_D1,matrix_demo_D2$MeanTrustGame_D2,matrix_demo_D3$MeanTrustGame_D3,matrix_demo_D4$MeanTrustGame_D4,matrix_demo_D5$MeanTrustGame_D5,matrix_demo_D6$MeanTrustGame_D6,matrix_demo_D7$MeanTrustGame_D7,matrix_demo_D8$MeanTrustGame_D8,matrix_demo_D9$MeanTrustGame_D9,matrix_demo_D10$MeanTrustGame_D10),
                         'Day' = c(rep(1,nrow(matrix_demo_D1)),rep(2,nrow(matrix_demo_D2)),rep(3,nrow(matrix_demo_D3)),rep(4,nrow(matrix_demo_D4)),rep(5,nrow(matrix_demo_D5)),rep(6,nrow(matrix_demo_D6)),rep(7,nrow(matrix_demo_D7)),rep(8,nrow(matrix_demo_D8)),rep(9,nrow(matrix_demo_D9)),rep(10,nrow(matrix_demo_D10))))

matrix_mean=merge(matrix_mean,
                  data.frame('PID'=ForTest[,'PID'],
                             'Questionnaire'=rowMeans(ForTest[,c('TrustExplicit_1_D1','TrustExplicit_2_D1','TrustExplicit_3_D1','TrustExplicit_1_D10','TrustExplicit_2_D10','TrustExplicit_3_D10')],na.rm=TRUE),
                             'QuestionnaireD1' = rowMeans(ForTest[,c('TrustExplicit_1_D1','TrustExplicit_2_D1','TrustExplicit_3_D1')],na.rm=TRUE),
                             'QuestionnaireD10' = rowMeans(ForTest[,c('TrustExplicit_1_D10','TrustExplicit_2_D10','TrustExplicit_3_D10')],na.rm=TRUE),
                             'SumExtraHelp'= ForTest$SumExtraHelp,
                             'ExtraHelp'= ForTest$ExtraHelp,
                             'DataSharing'= ForTest$DataSharing,
                             'ChildhoodResources'= ForTest$ChildhoodResources,
                             'ChildhoodPredictability'= ForTest$ChildhoodPredictability,
                             'AdultResources'= ForTest$AdultResources,
                             'Health'= ForTest$Health,
                             'EffortHealthSafety'= ForTest$EffortHealthSafety, 
                             'Age'= ForTest$Age,
                             'Gender'= ForTest$Gender),
                  by='PID')

matrixAll1 = matrix_mean[matrix_mean[,'Day']==1,]
matrixAll10 = aggregate(MeanTrustGame~PID, matrix_mean,function(x) mean(x,na.rm=TRUE))
matrixAll10[,'Day'] = 2
matrixAll10=merge(matrixAll10, subset(matrixAll1,select=-c(MeanTrustGame,Day)),by='PID')
matrixAll = data.frame(rbind(matrixAll1,matrixAll10))
# matrixAll = data.frame(rbind(matrix_mean[matrix_mean$Day==1,c("PID",
#                                                               "MeanTrustGame",
#                                                               "Day",
#                                                               "Questionnaire",
#                                                               "QuestionnaireD1",
#                                                               "QuestionnaireD10",                                                              "SumExtraHelp",
#                                                               "SumExtraHelp",
#                                                               "ExtraHelp",
#                                                               "DataSharing",
#                                                               'ChildhoodResources',
#                                                               'ChildhoodPredictability',
#                                                               'AdultResources',
#                                                               'Health',
#                                                               'EffortHealthSafety',
#                                                               'Age',
#                                                               'Gender')],
#                                                                aggregate(.~PID, matrix_mean,function(x) mean(x,na.rm=TRUE))))
matrixAll$Day = as.numeric(matrixAll$Day>1)
matrixAll$MeanTrustGame = scale(matrixAll$MeanTrustGame)
matrixAll$Questionnaire = scale(matrixAll$Questionnaire)
matrixAll$Age = scale(matrixAll$Age)
matrixAll$Gender = as.factor(matrixAll$Gender)

# mat2Comp = merge(matrixAll[matrixAll$Day==FALSE,],matrixAll[matrixAll$Day==TRUE,c('PID','MeanTrustGame')],by='PID')

# PARTICIPANT SAMPLE
mean(matrix_demo_D1$MeanTrustGame_D1)
sd(matrix_demo_D1$MeanTrustGame_D1)
summary(matrix_demo_D10)

t.test(Questionnaire~is.na(ChildhoodPredictability), matrix_demo_D1init,var.eq=TRUE)
t.test(MeanTrustGame_D1 ~ is.na(ChildhoodResources), matrix_demo_D1init,var.eq=TRUE)

# QUALITY CHECKS
shapiro.test(aggregate(TrustDecisions~ PID, matrix_choice_D1,mean)[,2])
shapiro.test(aggregate(TrustDecisions~PID, aggregate(TrustDecisions~ PID+Day, matrix_choice,mean),mean)[,2])

shapiro.test(rowMeans(matrix_demo_D1[,c('TrustExplicit_1_D1','TrustExplicit_2_D1','TrustExplicit_3_D1')],na.rm=TRUE))
shapiro.test(rowMeans(matrix_demo_D10[,c('TrustExplicit_1_D10','TrustExplicit_2_D10','TrustExplicit_3_D10')],na.rm=TRUE))

cronbach(matrix_demo_D1[,c('TrustExplicit_1_D1','TrustExplicit_2_D1','TrustExplicit_3_D1')])
cronbach(matrix_demo_D10[,c('TrustExplicit_1_D10','TrustExplicit_2_D10','TrustExplicit_3_D10')])

# 1. Intraclass correlation coefficient
# solution with 'ordinal' package, using the clmm2 function
model1.ordered <- clmm2(as.ordered(TrustDecisions) ~ Stakes * ProbaList + Day, random=as.factor(PID), data= matrix_choice, Hess = TRUE, nAGQ = 3)
model1.ordered
model1.ordered.sum<-summary(model1.ordered)
model1.ordered.sum
# ICC formula for ordered logistic model (Liu, 2015)
ICC.ord <- model1.ordered.sum$varMat[1,1]/(model1.ordered.sum$varMat[1,1]+(pi^2))
ICC.ord


## 2. Spearman correlations
cor.test(rowMeans(ForTest[,c('TrustExplicit_1_D1','TrustExplicit_2_D1','TrustExplicit_3_D1')]),rowMeans(ForTest[,c('TrustExplicit_1_D10','TrustExplicit_2_D10','TrustExplicit_3_D10')]), method = "spearman") #0.27, 0.02

### 3. Mixed model
#matrixAll<-matrixAll[matrixAll$Gender!='Non-binary',]
#matrix_mean<-matrix_mean[matrix_mean$Gender!='Non-binary',]

summary(lme((MeanTrustGame) ~(Questionnaire) * Day +(Age), random = (~1|PID), data = na.omit(matrixAll), method = 'ML'))
summary(lme((MeanTrustGame) ~(Questionnaire) * Day * I(Gender!='Male') +(Age), random = (~1|PID), data = na.omit(matrixAll), method = 'ML'))

summary(lm((MeanTrustGame) ~(Questionnaire) +(Age), data =(matrixAll[matrixAll$Day==0,])))
summary(lm((MeanTrustGame) ~(Questionnaire) +(Age), data =(matrixAll[matrixAll$Day==1,])))
summary(lme((MeanTrustGame) ~(Questionnaire)*Day +(Age),random=(~1|PID), data =(matrixAll)))

summary(lm((MeanTrustGame) ~(Questionnaire)* I(Gender!='Male') +(Age), data = na.omit(matrixAll[matrixAll$Day==1,])))
summary(lm((MeanTrustGame) ~(Questionnaire) * I(Gender!='Male')+(Age), data = na.omit(matrixAll[matrixAll$Day==10,])))


ggplot(matrix_mean, aes(Day, MeanTrustGame, color=Gender, group = Gender)) +
        geom_point() + 
        geom_smooth(method=lm, se=FALSE, fullrange=TRUE)

ggplot(matrixAll, aes(Day, MeanTrustGame, color=Gender, group = Gender)) +
        geom_point() + 
        geom_smooth(method=lm, se=FALSE, fullrange=TRUE)


#### 4. Exploratory analyses
summary(lm((MeanTrustGame) ~ I(SumExtraHelp>0) +(Age), data =(matrixAll[matrixAll$Day==0,])))
summary(lm((MeanTrustGame) ~ I(SumExtraHelp>0) +(Age), data =(matrixAll[matrixAll$Day==1,])))

summary(lm((MeanTrustGame) ~ I(ExtraHelp) +(Age), data =(matrixAll[matrixAll$Day==0,])))
summary(lm((MeanTrustGame) ~ I(ExtraHelp) +(Age), data =(matrixAll[matrixAll$Day==1,])))


summary(lm((MeanTrustGame) ~ I(SumExtraHelp>0)*I(Gender!='Male') +(Age), data = na.omit(matrixAll[matrixAll$Day==1,])))
summary(lm((MeanTrustGame) ~ I(SumExtraHelp>0)*I(Gender!='Male') +(Age), data = na.omit(matrixAll[matrixAll$Day==10,])))
summary(lm((Questionnaire) ~ I(SumExtraHelp>0)*I(Gender!='Male') +(Age), data = na.omit(matrixAll[matrixAll$Day==1,])))

table(matrixAll$SumExtraHelp, matrixAll$Gender) # AUCUNE FEMME A UN HELP DE 0

summary(lm(QuestionnaireD10 ~ I(SumExtraHelp>0) + Age, matrixAll[matrixAll $Day==1,]))
summary(lm(QuestionnaireD1 ~ I(SumExtraHelp>0) + Age, matrixAll[matrixAll $Day==TRUE,]))

summary(lm(I(QuestionnaireD1+ QuestionnaireD10)/2 ~ I(ExtraHelp) +(Age), data =(matrixAll[matrixAll$Day==1,])))

summary(lm((Questionnaire) ~ I(ExtraHelp) +(Age), data =(matrixAll[matrixAll$Day==1,])))
summary(lm((Questionnaire) ~ I(SumExtraHelp>0) +(Age), data =(matrixAll[matrixAll$Day==1,])))

