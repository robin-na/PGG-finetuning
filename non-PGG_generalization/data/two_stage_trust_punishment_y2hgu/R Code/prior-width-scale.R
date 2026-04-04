## Calculations for Bayesian Analyses

#The prior width will be designed such that only one-third of the prior mass on 
#each side of zero is larger than the desired effect (i.e., the relevant 
#coefficient observed in Jordan et al.)
#Specifically, for any desired effect, one-third of the prior mass on each side
#of zero will be more extreme than the absolute value of the desired effect. 
#The total prior mass smaller than the desired effect is calculated as 
#0.5 + 0.5 * 2/3 = 0.83333 (e.g. assuming an effect of 5.6 would lead to an 
#effect prior with scale of 5.09). 

#The defined function "myt" calculates the absolute difference between the 
#desired effect size and the quantile of the t-distribution with 4 degrees of 
#freedom and zero mean, corresponding to the prior mass of 0.5 + 0.5*2/3. 
#The "optimize" function in R then finds the value of the scale parameter that 
#minimizes the absolute difference between the observed effect size and the 
#quantile of the t-distribution. 

#The prior_width_scale value will be used as the width of our prior distributions.
  
#Code to calculate the scale of the prior width for a given desired effect:

###########################
# H1.1, H1.2a, H5.1, H5.2 #
###########################

desiredEffect <- 0.49  
myt <- function(x) { abs(extraDistr::qlst(0.5 + 0.5 * 2/3 , df = 4, mu = 0, sigma = x) - desiredEffect) } 
calc_scale <- optimize(myt, interval = c(0, 20)) 
prior_width_scale <- calc_scale$minimum
## [1] 0.4456985

extraDistr::plst(desiredEffect, df = 4, mu = 0, sigma = prior_width_scale)
#0.8333


###########################
# H2.1, H2.2a, H6.1, H6.2 #
###########################

desiredEffect <- 5.67  
myt <- function(x) { abs(extraDistr::qlst(0.5 + 0.5 * 2/3 , df = 4, mu = 0, sigma = x) - desiredEffect) } 
calc_scale <- optimize(myt, interval = c(0, 20)) 
prior_width_scale <- calc_scale$minimum
## [1] 5.157565


##########
# H2.2b #
#########

desiredEffect <- 9.84   
myt <- function(x) { abs(extraDistr::qlst(0.5 + 0.5 * 2/3 , df = 4, mu = 0, sigma = x) - desiredEffect) } 
calc_scale <- optimize(myt, interval = c(0, 20)) 
prior_width_scale <- calc_scale$minimum
## [1] 8.950664


#############################
# H7.1, H7.2a, H11.1, H11.2 #
#############################

desiredEffect <- 2.06  
myt <- function(x) { abs(extraDistr::qlst(0.5 + 0.5 * 2/3 , df = 4, mu = 0, sigma = x) - desiredEffect) } 
calc_scale <- optimize(myt, interval = c(0, 20)) 
prior_width_scale <- calc_scale$minimum
## [1] 1.873815


##########
# H7.2b #
#########

desiredEffect <- 4.90
myt <- function(x) { abs(extraDistr::qlst(0.5 + 0.5 * 2/3 , df = 4, mu = 0, sigma = x) - desiredEffect) } 
calc_scale <- optimize(myt, interval = c(0, 20)) 
prior_width_scale <- calc_scale$minimum
## [1] 4.457163


#############################
# H8.1, H8.2a, H12.1, H12.2 #
#############################

desiredEffect <- 7.73 # interaction coefficient 
myt <- function(x) { abs(extraDistr::qlst(0.5 + 0.5 * 2/3 , df = 4, mu = 0, sigma = x) - desiredEffect) } 
calc_scale <- optimize(myt, interval = c(0, 20)) 
prior_width_scale <- calc_scale$minimum
## [1] 7.031387


##########
# H8.2b #
#########

desiredEffect <- 14.74 # interaction coefficient 
myt <- function(x) { abs(extraDistr::qlst(0.5 + 0.5 * 2/3 , df = 4, mu = 0, sigma = x) - desiredEffect) } 
calc_scale <- optimize(myt, interval = c(0, 20)) 
prior_width_scale <- calc_scale$minimum
## [1] 13.40785

########################
# smallest interaction #
#######################

#this is based on Jordan et al.'s model predicting percentage returned as a
#function of helping decision, log-transformed decision time, their interaction,
#and log-transformed general comprehension speed 

desiredEffect <- 7.69 # interaction coefficient 
myt <- function(x) { abs(extraDistr::qlst(0.5 + 0.5 * 2/3 , df = 4, mu = 0, sigma = x) - desiredEffect) } 
calc_scale <- optimize(myt, interval = c(0, 20)) 
prior_width_scale <- calc_scale$minimum
## [1] 6.995011


###################################
# Scales for Sensitivity Analyses #
###################################

#H1.1, H1.2a, H5.1, H5.2: 
0.4456985 * 0.5 # 0.22
0.4456985 * 1.5 # 0.67
  
#H2.1, H2.2a, H6.1, H6.2
0.7654834 * 0.5 # 0.38
0.7654834 * 1.5 # 1.15

#H2.2b: 
8.951 * 0.5 # 4.48
8.951 * 1.5 # 13.43

#H7.1, H7.2a, H11.1, H11.2
1.873815 * 0.5 #0.94
1.873815 * 1.5 #2.81

#H7.2b
4.457163 * 0.5 #2.23
4.457163 * 1.5 # 6.69

#H8.1, H8.2a, H12.1, H12.2
7.031387 * 0.5 # 3.52
7.031387 * 1.5 # 10.55

#H8.2b
13.40785 * 0.5 # 6.70
13.40785 * 1.5 # 20.11

#smallest interaction
6.995011 * 0.5 # 3.50
6.995011 * 1.5 # 10.49
