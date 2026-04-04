library(dplyr)
library(ggplot2)
library(purrr)
library(tidyr)
library(stringr)
library(simsem)
library(lavaan)
library(fitdistrplus)
library(psych)
library(tidyverse)
library(knitr)
library(devtools)
library(car)
library(PerformanceAnalytics)
library(foreign)
library(gdata)
library(popbio)
library(dplyr)
library(mitools)
library(foreach)
library(VIM)
library(data.table)
library(psy)
library(nlme)
library(lmboot)
library(MASS)
library(rptR)
library(cowplot) 
library(lme4) 
library(sjPlot)
library(sjmisc) 
library(effects)
library(sjstats)
library(lcmm)
library(NormPsy)
library(Rmisc)
library(matrixStats)

# setwd("RepeatedEconomicGames") 
# setwd("RepeatedEconomicGames") 
setwd("RepeatedEconomicGames") 

matrix_mean <- read.csv('matrix_mean.csv', header = T, sep = ',')
df <-matrix_mean[order(matrix_mean$PID,matrix_mean$Day),]

##### Change the time unit in Day
df$Day<-replace(df$Day, df$Day==1, 11)
df$Day<-replace(df$Day, df$Day==2, 22)
df$Day<-replace(df$Day, df$Day==3, 33)
df$Day<-replace(df$Day, df$Day==4, 44)
df$Day<-replace(df$Day, df$Day==5, 55)
df$Day<-replace(df$Day, df$Day==6, 66)
df$Day<-replace(df$Day, df$Day==7, 77)
df$Day<-replace(df$Day, df$Day==8, 88)
df$Day<-replace(df$Day, df$Day==9, 99)
df$Day<-replace(df$Day, df$Day==10, 100)

df$Day<-replace(df$Day, df$Day==11, 1)
df$Day<-replace(df$Day, df$Day==22, 3)
df$Day<-replace(df$Day, df$Day==33, 5)
df$Day<-replace(df$Day, df$Day==44, 7)
df$Day<-replace(df$Day, df$Day==55, 9)
df$Day<-replace(df$Day, df$Day==66, 11)
df$Day<-replace(df$Day, df$Day==77, 13)
df$Day<-replace(df$Day, df$Day==88, 15)
df$Day<-replace(df$Day, df$Day==99, 17)
df$Day<-replace(df$Day, df$Day==100, 19)

##### Create columnwise df
df.wide <- reshape(df, v.names=c("MeanTrustGame"), idvar="PID",         
                   timevar="Day", direction="wide") 

gender <- read.csv('gender.csv', header = T, sep = ';')

df.wide$NID <- 1:nrow(df.wide)
df.wide$gender <- gender$Gender

describe(df.wide[,12:21])

hist(df.wide$MeanTrustGame.1, breaks = 100)
hist(df.wide$MeanTrustGame.3, breaks = 100)
hist(df.wide$MeanTrustGame.5, breaks = 100)
hist(df.wide$MeanTrustGame.7, breaks = 100)
hist(df.wide$MeanTrustGame.9, breaks = 100)
hist(df.wide$MeanTrustGame.11, breaks = 100)
hist(df.wide$MeanTrustGame.13, breaks = 100)
hist(df.wide$MeanTrustGame.15, breaks = 100)
hist(df.wide$MeanTrustGame.17, breaks = 100)
hist(df.wide$MeanTrustGame.19, breaks = 100)

df.wide$ind.SD <- rowSds(as.matrix(df.wide[,c(12:21)]))
hist(df.wide$ind.SD, breaks = 100)
median(df.wide$ind.SD)

##### Create rowwise df
df.long <- reshape(df.wide, direction='long', 
                   idvar = "NID",
                   ids = 1:NROW(df.wide),
                   varying=c(12:21), 
                   timevar='Day',
                   times=c(1,3,5,7,9,11,13,15,17,19),
                   v.names=c('MeanTrustGame'))

df.long$MeanTrustGame = scale(df.long$MeanTrustGame)

library(lattice)
color <- df.long$NID
xyplot(MeanTrustGame ~ Day, df.long, groups = NID, col=color, lwd=1, type="l")

ggplot(df.long, aes(Day, MeanTrustGame)) +
  geom_point() +
  stat_smooth(method = "lm", formula = y ~ x + I(x^2) + I(x^3), size = 1)


############################################################
########## IDENTiFYING THE BEST NUMBER OF CLASSES ##########
############################################################

m1.lin <- hlme(MeanTrustGame ~ Day, random = ~1+Day, subject = 'NID', data = df.long) # ng=1
m2.lin <- gridsearch(hlme(MeanTrustGame ~ Day, random = ~1+Day, subject = 'NID', data=df.long, ng = 2, mixture=~Day), rep=100, maxiter=30, minit=m1.lin)
m3.lin <- gridsearch(hlme(MeanTrustGame ~ Day, random = ~1+Day, subject = 'NID', data=df.long, ng = 3, mixture=~Day), rep=100, maxiter=30, minit=m1.lin)
m4.lin <- gridsearch(hlme(MeanTrustGame ~ Day, random = ~1+Day, subject = 'NID', data=df.long, ng = 4, mixture=~Day), rep=100, maxiter=30, minit=m1.lin)
m5.lin <- gridsearch(hlme(MeanTrustGame ~ Day, random = ~1+Day, subject = 'NID', data=df.long, ng = 5, mixture=~Day), rep=100, maxiter=30, minit=m1.lin)
m6.lin <- gridsearch(hlme(MeanTrustGame ~ Day, random = ~1+Day, subject = 'NID', data=df.long, ng = 6, mixture=~Day), rep=100, maxiter=30, minit=m1.lin)

summarytable(m1.lin,m2.lin,m3.lin,m4.lin,m5.lin,m6.lin, which = c("G", "loglik", "conv", "npm", "AIC", "BIC", "SABIC", "entropy","ICL", "%class"))
summaryplot(m1.lin,m2.lin,m3.lin,m4.lin,m5.lin,m6.lin, which = c("AIC","BIC","SABIC","entropy","ICL"))

m1.quad <- hlme(MeanTrustGame ~ Day+I(Day^2), random = ~1+Day+I(Day^2), subject = 'NID', data = df.long) # ng=1
m2.quad <- gridsearch(hlme(MeanTrustGame ~ Day+I(Day^2), random = ~1+Day+I(Day^2), subject = 'NID', data=df.long, ng = 2, mixture=~Day+I(Day^2)), rep=100, maxiter=30, minit=m1.quad)
m3.quad <- gridsearch(hlme(MeanTrustGame ~ Day+I(Day^2), random = ~1+Day+I(Day^2), subject = 'NID', data=df.long, ng = 3, mixture=~Day+I(Day^2)), rep=100, maxiter=30, minit=m1.quad)
m4.quad <- gridsearch(hlme(MeanTrustGame ~ Day+I(Day^2), random = ~1+Day+I(Day^2), subject = 'NID', data=df.long, ng = 4, mixture=~Day+I(Day^2)), rep=100, maxiter=30, minit=m1.quad)
m5.quad <- gridsearch(hlme(MeanTrustGame ~ Day+I(Day^2), random = ~1+Day+I(Day^2), subject = 'NID', data=df.long, ng = 5, mixture=~Day+I(Day^2)), rep=100, maxiter=30, minit=m1.quad)
m6.quad <- gridsearch(hlme(MeanTrustGame ~ Day+I(Day^2), random = ~1+Day+I(Day^2), subject = 'NID', data=df.long, ng = 6, mixture=~Day+I(Day^2)), rep=100, maxiter=30, minit=m1.quad)

summarytable(m1.quad,m2.quad,m3.quad,m4.quad,m5.quad,m6.quad, which = c("G", "loglik", "conv", "npm", "AIC", "BIC", "SABIC", "entropy","ICL", "%class"))
summaryplot(m1.quad,m2.quad,m3.quad,m4.quad,m5.quad,m6.quad, which = c("AIC","BIC","SABIC","entropy","ICL"))

# Check models outputs
summary(m1.lin)
summary(m2.lin)
summary(m3.lin)
summary(m4.lin)
summary(m5.lin)
summary(m6.lin)

summary(m1.quad)
summary(m2.quad)
summary(m3.quad)
summary(m4.quad)
summary(m5.quad)
summary(m6.quad)

# Store models outputs
# setwd("RepeatedEconomicGames/LCA_Outputs") 
setwd("RepeatedEconomicGames/LCA_Outputs") 
# setwd("RepeatedEconomicGames/LCA_Outputs") 
# save(m1.lin, file="m1.lin.rda")
# save(m2.lin, file="m2.lin.rda")
# save(m3.lin, file="m3.lin.rda")
# save(m4.lin, file="m4.lin.rda")
# save(m5.lin, file="m5.lin.rda")
# save(m6.lin, file="m6.lin.rda")
# 
# save(m1.quad, file="m1.quad.rda")
# save(m2.quad, file="m2.quad.rda")
# save(m3.quad, file="m3.quad.rda")
# save(m4.quad, file="m4.quad.rda")
# save(m5.quad, file="m5.quad.rda")
# save(m6.quad, file="m6.quad.rda")

# Load models outputs
load("m1.lin.rda")
load("m2.lin.rda")
load("m3.lin.rda")
load("m4.lin.rda")
load("m5.lin.rda")
load("m6.lin.rda")

load("m1.quad.rda")
load("m2.quad.rda")
load("m3.quad.rda")
load("m4.quad.rda")
load("m5.quad.rda")
load("m6.quad.rda")

# average latent class posterior probabilities
post.prob.lin<-postprob(m4.lin) 
post.prob.quad<-postprob(m4.quad) 

# posterior probabilities of class membership and logit transformation
class.mem<-m4.lin$pprob
logit.class<-logit(class.mem[,3:6], percents=max(class.mem[,3:6], na.rm = TRUE) > 1)
class.mem.logit.weighted<-cbind(class.mem[,1:2],logit.class)

# plot predicted class trajectories
data_pred1.quest <- data.frame(Day=seq(1,19,length.out=100))
pred1.quest <- predictY(m4.lin, data_pred1.quest, var.time = "Day",draws=TRUE)
plot(pred1.quest, col=c(3,4,5,6), lty=1, lwd=c(96,1,50,7), ylab="Willingness to play", legend=NULL, main="Predicted trajectories for WTP", ylim=c(-5,2), shades=TRUE)
legend(x="topleft",legend=c("class1 WTP Slow Decrease, N=96",
                            "class2 WTP Null, N=1",
                            "class3 WTP Increase, N=50",
                            "class4 WTP Steep Decrease, N=7"),
       col=c(3,4,5,6), lwd=c(2), lty=c(1,1), ncol=2, bty="n", cex = 0.8)

WTP_pred <- data.frame(pred1.quest[["pred"]])

# Weighted subject-specific predictions #
plot(m4.lin, which="fit", var.time="Day", marg=FALSE, shades = TRUE, ylim=c(-5,3),col=c(3,4,5,6))
# Inspection of residuals #
plot(m4.lin, cex.main=0.8)
# Inspection of posterior proabilities distributions #
plot(m4.lin, which="postprob")


##### Multinomial logistic regressions predicting class membership by mean score on social trust questionnaire #####
library(nnet)
library(caret)
library(MASS)
library(effects)
library(questionr)
library(stargazer)
library(ggpubr)
library(MNLpred)
library(ggplot2)
library(scales)

df.wide.class <- df.wide
df.wide.class <- cbind(df.wide.class, class.mem.logit.weighted)
df.wide.class<-df.wide.class[,-c(22)]
df.wide.class = df.wide.class %>% filter((class)!=2)

# df.wide.class$class<-replace(df.wide.class$class, df.wide.class$class==2, "WTP Null")
df.wide.class$class<-replace(df.wide.class$class, df.wide.class$class==3, "WTP Increase")
df.wide.class$class<-replace(df.wide.class$class, df.wide.class$class==1, "WTP Slow Decrease")
df.wide.class$class<-replace(df.wide.class$class, df.wide.class$class==4, "WTP Steep Decrease")

df.wide.class$class <- as.factor(df.wide.class$class)

df.wide.class$class <- relevel(df.wide.class$class, ref = "WTP Slow Decrease")

df.wide.class$high.post.prob<-pmax(df.wide.class$prob1,
                                   # df.wide.class$prob2,
                                   df.wide.class$prob3,
                                   df.wide.class$prob4)

df.wide.class[,c(2,8:11)] = scale(df.wide.class[,c(2,8:11)])

df.wide.class$gender<-replace(df.wide.class$gender, df.wide.class$gender=='Male', 0)
df.wide.class$gender<-replace(df.wide.class$gender, df.wide.class$gender=='Female', 1)
df.wide.class$gender<-as.numeric(df.wide.class$gender)
df.wide.class = df.wide.class %>% filter(!is.na(gender))

# Compute t-test to measure the effect of gender on questionnaire
t.test(Questionnaire ~ gender, data = df.wide.class)
t.test(MeanTrustGame.1 ~ gender, data = df.wide.class)
t.test(MeanTrustGame.19 ~ gender, data = df.wide.class)
t.test(ind.SD ~ gender, data = df.wide.class)
t.test(ExtraHelp ~ gender, data = df.wide.class)
t.test(DataSharing ~ gender, data = df.wide.class)

ggboxplot(df.wide.class, x = 'gender', y = 'Questionnaire', add='jitter')
ggboxplot(df.wide.class, x = 'gender', y = 'MeanTrustGame.1', add='jitter')
ggboxplot(df.wide.class, x = 'gender', y = 'MeanTrustGame.19', add='jitter')
ggboxplot(df.wide.class, x = 'gender', y = 'ind.SD', add='jitter')
ggboxplot(df.wide.class, x = 'gender', y = 'ExtraHelp', add='jitter')
ggboxplot(df.wide.class, x = 'gender', y = 'DataSharing', add='jitter')

##### Univariate models

multinom.model.null <- multinom(class ~ +1, data=df.wide.class, weights=high.post.prob) # multinom Model
summary(multinom.model.null)

multinom.model.quest <- multinom(class ~ Questionnaire, data=df.wide.class, weights=high.post.prob) # multinom Model
Anova(multinom.model.quest, type="II") # Anova table of likelihood ratio tests (Baseline model = intercept)
summary (multinom.model.quest)
z.quest <- summary(multinom.model.quest)$coefficients/summary(multinom.model.quest)$standard.errors
p.quest <- (1 - pnorm(abs(z.quest), 0, 1)) * 2
odds.ratio(multinom.model.quest, level = 0.95)

multinom.model.help <- multinom(class ~ ExtraHelp, data=df.wide.class, weights=high.post.prob) # multinom Model
Anova(multinom.model.help, type="II") # Anova table of likelihood ratio tests
summary (multinom.model.help)
z.help <- summary(multinom.model.help)$coefficients/summary(multinom.model.help)$standard.errors
p.quest <- (1 - pnorm(abs(z.help), 0, 1)) * 2
odds.ratio(multinom.model.help, level = 0.95)

multinom.model.share<- multinom(class ~ DataSharing, data=df.wide.class, weights=high.post.prob) # multinom Model
Anova(multinom.model.share, type="II") # Anova table of likelihood ratio tests
summary (multinom.model.share)
z.share <- summary(multinom.model.share)$coefficients/summary(multinom.model.share)$standard.errors
p.share <- (1 - pnorm(abs(z.share), 0, 1)) * 2
odds.ratio(multinom.model.share, level = 0.95)

multinom.model.age <- multinom(class ~ Age, data=df.wide.class, weights=high.post.prob) # multinom Model
Anova(multinom.model.age, type="II") # Anova table of likelihood ratio tests
summary (multinom.model.age)
z.age <- summary(multinom.model.age)$coefficients/summary(multinom.model.age)$standard.errors
p.age <- (1 - pnorm(abs(z.age), 0, 1)) * 2
odds.ratio(multinom.model.age, level = 0.95)

multinom.model.gender <- multinom(class ~ gender, data=df.wide.class, weights=high.post.prob) # multinom Model
Anova(multinom.model.gender, type="II") # Anova table of likelihood ratio tests
summary (multinom.model.gender)
z.gender <- summary(multinom.model.gender)$coefficients/summary(multinom.model.gender)$standard.errors
p.gender <- (1 - pnorm(abs(z.gender), 0, 1)) * 2
odds.ratio(multinom.model.gender, level = 0.95)

multinom.model.adult.res <- multinom(class ~ AdultResources, data=df.wide.class, weights=high.post.prob) # multinom Model
Anova(multinom.model.adult.res, type="II") # Anova table of likelihood ratio tests
summary (multinom.model.adult.res)
z.adult.res <- summary(multinom.model.adult.res)$coefficients/summary(multinom.model.adult.res)$standard.errors
p.adult.res <- (1 - pnorm(abs(z.adult.res), 0, 1)) * 2
odds.ratio(multinom.model.adult.res, level = 0.95)

multinom.model.child.res <- multinom(class ~ ChildhoodResources, data=df.wide.class, weights=high.post.prob) # multinom Model
Anova(multinom.model.child.res, type="II") # Anova table of likelihood ratio tests
summary (multinom.model.child.res)
z.child.res <- summary(multinom.model.child.res)$coefficients/summary(multinom.model.child.res)$standard.errors
p.child.res <- (1 - pnorm(abs(z.child.res), 0, 1)) * 2
odds.ratio(multinom.model.child.res, level = 0.95)

multinom.model.child.pred <- multinom(class ~ ChildhoodPredictability, data=df.wide.class, weights=high.post.prob) # multinom Model
Anova(multinom.model.child.pred, type="II") # Anova table of likelihood ratio tests
summary (multinom.model.child.pred)
z.child.pred <- summary(multinom.model.child.pred)$coefficients/summary(multinom.model.child.pred)$standard.errors
p.child.pred <- (1 - pnorm(abs(z.child.pred), 0, 1)) * 2
odds.ratio(multinom.model.child.pred, level = 0.95)

##### Multivariate models
multinom.model.gen <- multinom(class ~ Questionnaire + gender + Age, data=df.wide.class, weights=high.post.prob, Hess = TRUE) # multinom Model
Anova(multinom.model.gen, type="II") # Anova table of likelihood ratio tests
summary(multinom.model.gen)
z.gen <- summary(multinom.model.gen)$coefficients/summary(multinommodel.gen)$standard.errors
p.gen <- (1 - pnorm(abs(z.gen), 0, 1)) * 2
odds.ratio(multinom.model.gen, level = 0.95)

multinom.model.gen.final <- multinom(class ~ Questionnaire + gender, data=df.wide.class, weights=high.post.prob, Hess = TRUE) # multinom Model
Anova(multinom.model.gen.final, type="II") # Anova table of likelihood ratio tests
summary(multinom.model.gen.final)
z.gen.final <- summary(multinom.model.gen.final)$coefficients/summary(multinom.model.gen.final)$standard.errors
p.gen.final <- (1 - pnorm(abs(z.gen.final), 0, 1)) * 2
odds.ratio(multinom.model.gen.final, level = 0.95)

# Derive predicted probabilities
preds <- mnl_pred_ova(model = multinom.model.gen.final,
                      data = df.wide.class,
                      x = "Questionnaire",
                      by = 0.01,
                      seed = "random", 
                      nsim = 1000, # default
                      probs = c(0.025, 0.975))
preds$plotdata %>% head()

# determine how long before predicted probability of WTP slow decrease <= 50%
y.50 <- which(preds$plotdata[1:483,3] <= 0.50) # y vals <= 50%
x.50 <- first(y.50) # index of x val where first y val <= 50%
x.val <- preds$plotdata[x.50,1] # corresponding x val

# determine how long before predicted probability of WTP Increase >= 50%
y.50.increase <- which(preds$plotdata[484:966,3] >= 0.50) # y vals >= 50%
x.50.increase <- first(y.50.increase) # index of x val where first y val >= 50%
x.val.increase <- preds$plotdata[x.50.increase,1] # corresponding x val

# plot predicted probabilities across WTP classes as a function of questionnaire scores
ggplot(data = preds$plotdata, aes(x = Questionnaire, 
                                  y = mean,
                                  ymin = lower, ymax = upper)) +
  geom_ribbon(alpha = 0.1) + # Confidence intervals
  geom_line() + # mean
  facet_wrap(.~ class, ncol = 3) +
  geom_segment(aes(x = x.val, y = .5, xend = -2.22, yend = .5), 
               linetype = "dashed",data = preds$plotdata[1:483,])+
  geom_segment(aes(x = x.val, y = .5, xend = x.val, yend = 0), 
               linetype = "dashed",data = preds$plotdata[1:483,])+
  geom_segment(aes(x = x.val.increase, y = .5, xend = -2.22, yend = .5), 
               linetype = "dashed",data = preds$plotdata[484:966,])+
  geom_segment(aes(x = x.val.increase, y = .5, xend = x.val.increase, yend = 0), 
               linetype = "dashed",data = preds$plotdata[484:966,])+
  scale_y_continuous(labels = percent_format(accuracy = 1)) + # % labels
  scale_x_continuous(breaks = c(-3:3)) +
  theme_bw() +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=12),
        strip.text = element_text(size=12),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        strip.background = element_blank(),
        panel.border = element_rect(colour = "black"))+
  labs(y = "Predicted class probabilities",
       x = "Questionnaire scores")

### Diff in probabilities between the lowest and the highest questionnaire score, by classes
fdif1 <- mnl_fd2_ova(model = multinom.model.gen.final,
                     data = df.wide.class,
                     x = "Questionnaire",
                     value1 = min(df.wide.class$Questionnaire),
                     value2 = max(df.wide.class$Questionnaire),
                     nsim = 1000)

ggplot(fdif1$plotdata_fd, aes(x = categories, 
                              y = mean,
                              ymin = lower, ymax = upper)) +
  geom_pointrange() +
  geom_hline(yintercept = 0) +
  scale_y_continuous(labels = percent_format()) +
  theme_bw() +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=12),
        strip.text = element_text(size=12),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        strip.background = element_blank(),
        panel.border = element_rect(colour = "black"))+
  labs(y = "Predicted probabilities",
       x = "Class")

### Predictied probability curve or each questionnaire score, classe and gender
fdif2 <- mnl_fd_ova(model = multinom.model.gen.final,
                    data = df.wide.class,
                    x = "Questionnaire",
                    by = 0.01,
                    z = "gender",
                    z_values = c(0,1),
                    nsim = 1000)
fdif2$plotdata_fd %>% head()

ggplot(data = fdif2$plotdata, aes(x = Questionnaire, 
                                  y = mean,
                                  ymin = lower, ymax = upper,
                                  group = as.factor(gender),
                                  linetype = as.factor(gender))) +
  geom_ribbon(alpha = 0.1) +
  geom_line() +
  facet_wrap(. ~ class, ncol = 3) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) + # % labels
  scale_x_continuous(breaks = c(-3:3)) +
  scale_linetype_discrete(name = "Gender",
                          breaks = c(0, 1),
                          labels = c("Male", "Female")) +
  theme_bw() +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=12),
        strip.text = element_text(size=12),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        strip.background = element_blank(),
        panel.border = element_rect(colour = "black"))+
  labs(y = "Predicted probabilities",
       x = "Questionnaire score") 

ggplot(data = fdif2$plotdata_fd, aes(x = Questionnaire, 
                                     y = mean,
                                     ymin = lower, ymax = upper)) +
  geom_ribbon(alpha = 0.1) +
  geom_line() +
  geom_hline(yintercept = 0) +
  facet_wrap(. ~ class, ncol = 3) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) + # % labels
  scale_x_continuous(breaks = c(-3:3)) +
  theme_bw() +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=12),
        strip.text = element_text(size=12),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        strip.background = element_blank(),
        panel.border = element_rect(colour = "black"))+
  labs(y = "Predicted probabilities",
       x = "Questionnaire score") 

### predicited probability Gender

pred2 <- mnl_pred_ova(model = multinom.model.gen.final,
                      data = df.wide.class,
                      x = "gender",
                      by = 1,
                      seed = "random", # default
                      nsim = 1000, # faster
                      probs = c(0.025, 0.975))
pred2$plotdata %>% head()

ggplot(data = pred2$plotdata, aes(x = gender, 
                                  y = mean,
                                  ymin = lower, ymax = upper)) +
  geom_ribbon(alpha = 0.1) + # Confidence intervals
  geom_line() + # Mean
  facet_wrap(.~ class, ncol = 3) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) + # % labels
  scale_x_continuous(breaks = c(0:1)) +
  theme_bw() +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=12),
        strip.text = element_text(size=12),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        strip.background = element_blank(),
        panel.border = element_rect(colour = "black"))+
  labs(y = "Predicted class probabilities",
       x = "Gender")

fdif1.g <- mnl_fd2_ova(model = multinom.model.gen.final,
                     data = df.wide.class,
                     x = "gender",
                     value1 = min(df.wide.class$gender),
                     value2 = max(df.wide.class$gender),
                     nsim = 1000)

ggplot(fdif1.g$plotdata_fd, aes(x = categories, 
                              y = mean,
                              ymin = lower, ymax = upper)) +
  geom_pointrange() +
  geom_hline(yintercept = 0) +
  scale_y_continuous(labels = percent_format()) +
  theme_bw() +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=12),
        strip.text = element_text(size=12),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        strip.background = element_blank(),
        panel.border = element_rect(colour = "black"))+
  labs(y = "Predicted probabilities",
       x = "Class")


