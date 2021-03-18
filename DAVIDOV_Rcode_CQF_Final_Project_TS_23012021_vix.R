rm(list=ls())

library(quantmod)
library(ggplot2)
library(magrittr)
library(broom)
library(urca)
library(vars)
library(Hmisc)
library(dynlm)
library(tseries)
library(graphics)
library(forecast)
library(PerformanceAnalytics)
library(Rfast)
library(dplyr)
library(data.table)
library(lmtest)



###############################
#RW and 1 lag Diff illustration
###############################

mu<-0.01
sigma<-0.05
dt<-0.10

U1<-runif(1000)
U2<-runif(1000)

N_0_1<-sqrt(-2*log(U1))*cos(2*pi*U2)
W<-c()
W<-N_0_1*sqrt(dt)
Log<-c()
Log_T_1<-c()
Log[1]<-1
for ( i in 2:length(W)) {
  Log[i]<-Log[i-1]*exp((mu-0.5*sigma^2)*dt+sigma*W[i-1])
}

Log_T_1<-ts(diff(Log, lag = 1))
Log<-ts(Log)
ts.plot(Log, Log_T_1, gpars = list(col = c("blue", "green")),type="l",ylab=expression(italic(Y[t])),xlab="Time",
        main="Random Walk and its First-Differenced Path" )
legend("topleft",c('Random Walk','Stationary Diff'),
       col=c("blue", "green" ),lty=1,lwd=1, cex=0.8);



## setup plot area
par(mfrow = c(1, 2))
## plot line
plot.ts(Log, ylab = expression(italic(x[t])))
## plot ACF, typical for random-walk the significance of the lags decays progressively
(acf(Log, plot = TRUE))
par(mfrow = c(1, 1))
# pacf suggests it's AR(1) process
pacf(Log)

###################
#Ornstein-Uhlenbeck
###################

#theta is mean reversion speed, mu is long-term value
ou_process <- function(T,n,mu,theta,sigma,x0){
  dw  <- rnorm(n, 0, sqrt(T/n))
  dt  <- T/n
  x <- c(x0)
  for (i in 2:(n+1)) {
    x[i]  <-  x[i-1] + theta*(mu-x[i-1])*dt + sigma*dw[i-1]
  }
  return(x);
}

ou_res_1<- ou_process(1000,1000,0.05,0.005,0.2,-20)
ou_res_2<- ou_process(1000,1000,0.05,0.01,0.2,0)
ou_res_3<- ou_process(1000,1000,0.05,0.05,0.2,20)

par(mfrow=c(1,2))
plot(ou_res_1,col="blue",type="l",ylab=expression(italic(Y[t])), xlab = "Time", ylim=c(-25,25) )
lines(ou_res_2,col="red")
lines(ou_res_3,col="green")
legend("topleft",c("theta=.005","theta=0.1","theta=0.05"),
       col=c("blue","red","green"),lty=c(1,1,1),lwd=c(1,1,1), cex=0.8);



ou_res_1_diff<- diff(ou_res_1, lag = 1)
ou_res_2_diff<- diff(ou_res_2, lag = 1)
ou_res_3_diff<- diff(ou_res_3, lag = 1)
plot(ou_res_1_diff,col="blue",type="l", ylab=expression(italic(dY[t])), xlab = "Time", ylim=c(-1,1) )
lines(ou_res_2_diff,col="red")
lines(ou_res_3_diff,col="green")
legend("topleft",c("theta=.005","theta=0.1","theta=0.05"),
       col=c("blue","red","green"),lty=c(1,1,1),lwd=c(1,1,1), cex=0.8);
par(mfrow=c(1,1))
title(main="Ornstein-Uhlenbeck process with different mean-reversion speed")

####################################################
# Loading real data
####################################################


#Loading data with quantmod
# 
#start = as.Date("2018-02-01") 
#end = as.Date("2020-11-11")
#
# getSymbols(c("VIXM", "VXZ"), src = "yahoo", from = start, to = end)
# etfs = as.xts(data.frame(VIXM = `VIXM`[, "VIXM.Adjusted"], VXZ = `VXZ`[, "VXZ.Adjusted"]))
# write.zoo(etfs, "D:\\cqf\\etf_data.csv", quote = FALSE, sep = ",")


#reading data and converting to xts format 
data_etf<-read.csv( "D:\\cqf\\etf_data.csv", header= TRUE, sep = ",")


#clean data
data_etf<-na.omit(data_etf) 

date <- as.Date(data_etf$Index,"%Y-%m-%d")


etfs<-as.xts(data_etf[,c("VIXM","VXZ")], date)




#preparing data set for plot
names(etfs) = c("VIXM", "VXZ")
index(etfs) = as.Date(index(etfs))

#spread between time series 
etfs$diff <- etfs$VIXM - etfs$VXZ


etfs_alldata<-etfs


#choose time window 
# we split the in-sample to out-sample in ratio 75/25
# in-sample period spanning one year



# etfs_insample = window(etfs, start=as.Date("2018-12-01"), end=as.Date("2019-12-01")) 
# etfs_outsample = window(etfs, start=as.Date("2019-12-01"), end=as.Date("2020-03-01")) 


etfs_insample = window(etfs, start=as.Date("2019-01-01"), end=as.Date("2020-01-01")) 
etfs_outsample = window(etfs, start=as.Date("2020-01-01"), end=as.Date("2020-03-01"))

#illustrates regime changes with including March 2020 in the test period
#etfs_insample = window(etfs, start=as.Date("2019-01-01"), end=as.Date("2020-01-01")) 
#etfs_outsample = window(etfs, start=as.Date("2020-01-01"), end=as.Date("2020-04-01"))

# #good in sample results, lots of signals but there are periods of stale data
# etfs_insample = window(etfs, start=as.Date("2018-02-01"), end=as.Date("2019-02-01")) 
# etfs_outsample = window(etfs, start=as.Date("2019-02-01"), end=as.Date("2019-08-01")) 


#regime change
#non-stationary even after first differencing, nevertheless interesting results, small half-life, very active trading
# etfs_insample = window(etfs, start=as.Date("2019-06-01"), end=as.Date("2020-06-01")) 
# etfs_outsample = window(etfs, start=as.Date("2020-06-01"), end=as.Date("2020-11-01")) 


# #covering the covid crisis period - interesting to see when there is regime change 
# etfs_insample = window(etfs, start=as.Date("2019-02-01"), end=as.Date("2020-02-01")) 
# etfs_outsample = window(etfs, start=as.Date("2020-02-01"), end=as.Date("2020-06-01")) 


#zoo format data to be used later
etfs.zoo<-zoo(etfs_insample, order.by = index(etfs_insample))


#plot all data
etfs_series = tidy(etfs_alldata) %>% 
  
  ggplot(aes(x=index,y=value, color=series)) + geom_line() + 
  
  labs(title = "VIXM vs VXZ: Daily Prices Evolution Feb 2018 - Nov 2020",
       
       subtitle = "End of Day Adjusted Prices",
       caption = " Source: Yahoo Finance") +
  
  labs(y="Price", x = "Date")

etfs_series

####################
# Stationarity tests
####################

#Bayesian Information Criterion as alternative to AIC
my_BIC <- function(model) {
  
  ssr <- sum(model$residuals^2)
  len_et <- length(model$residuals)
  ndeg <- length(model$coef)
  
  return(
    round(c("p" = ndeg - 1,
            "BIC" = log(ssr/len_et) + ndeg * log(len_et)/len_et,
            "R^2" = summary(model)$r.squared), 4)
  )
}

#lags to test for AR
lags <- 1:30

#
BICs <- sapply(lags, function(x) 
  "AR" = my_BIC(dynlm(etfs.zoo[,"VIXM"] ~ L(etfs.zoo[,"VIXM"], 1:x))))

BICs




#check for stationarity , no trend, no const
adf.test_nc = ur.df(etfs_insample[,"VIXM"], type = "none")
print(summary(adf.test_nc))

#check for stationarity, const, no trend
adf.test_drift = ur.df(etfs_insample[,"VIXM"], type = "drift")
print(summary(adf.test_drift))

#check for stationarity, const and trend
adf.test_trend = ur.df(etfs_insample[,"VIXM"], type = "trend")
print(summary(adf.test_trend))


#repeat the tests for the other leg 
adf.test = ur.df(etfs_insample[,"VXZ"], type = "none")
print(summary(adf.test))

adf.test_drift = ur.df(etfs_insample[,"VXZ"], type = "drift")
print(summary(adf.test_drift))


adf.test_trend = ur.df(etfs_insample[,"VXZ"], type = "trend")
print(summary(adf.test_trend))


#now check for the first difference of the two time series
#delete the first observation which is NA after taking first differences
# VXZ
y_t1<- diff(etfs_insample[,"VXZ"],1)
adf.test_nc = ur.df(y_t1[-1], type = "none") 
print(summary(adf.test_nc))

adf.test_drift = ur.df(y_t1[-1], type = "drift")
print(summary(adf.test_drift))


adf.test_trend = ur.df(y_t1[-1], type = "trend")
print(summary(adf.test_trend))



# VIXM
x_t1<- diff(etfs_insample[,"VIXM"],1)
adf.test_nc = ur.df(x_t1[-1], type = "none")
print(summary(adf.test_nc))

adf.test_drift = ur.df(etfs_insample$diff, type = "drift")
print(summary(adf.test_drift))


adf.test_trend = ur.df(etfs_insample$diff, type = "trend")
print(summary(adf.test_trend))




#now check for the difference of the two time series 
adf.test_nc = ur.df(etfs_insample$diff, type = "none")
print(summary(adf.test_nc))

adf.test_drift = ur.df(etfs_insample$diff, type = "drift")
print(summary(adf.test_drift))


adf.test_trend = ur.df(etfs_insample$diff, type = "trend")
print(summary(adf.test_trend))




#optimal lag selection with PACF
par(mfrow=c(2,1))
pacf(etfs_insample[,"VIXM"], main = "VIX Mid-Term Futures ETF (VIXM)")
pacf(etfs_insample[,"VXZ"], main = "iPath Series B S&P 500® VIX Mid-Term Futures ETN (VXZ)" )
par(mfrow=c(1,1))

delta_VIXM <-etfs_insample[-1,"VIXM"] - Lag(etfs_insample[,"VIXM"], 1)
delta_VXZ <-etfs_insample[-1,"VXZ"] - Lag(etfs_insample[,"VXZ"], 1)

par(mfrow=c(2,1))
pacf(delta_VIXM, main = "VIXM First Differenced PACF" )
pacf(delta_VXZ, main = "VXZ First Differenced PACF" )
par(mfrow=c(1,1))


######################################
# Naive Cointregating Equation
######################################


#############################################################
# function to calculate OLS regression, 
# when X<-cbind(intercept = 1, X)
# this is in fact  alternative to r function lm, when column of 1s added to the ind.variable X  

my_LM <- function(Y,X) {
  #X<-cbind(intercept = 1, X)
  M <- solve(t(X)%*%X)    # (X'X)^-1
  G <- (t(X)%*%Y)
  beta_est <- M %*% G  # [(X'X)^-1]X'Y , i.e. beta_hat.
  
  resid_est <- Y - (X %*% beta_est) # e = Y - X * beta_est
  
  nobs <- nrow(as.matrix(X))
  rank <- ncol(as.matrix(X))   #degrees of freedom 
  
  resid_df <- nobs - rank 
  ssr <-  (t(resid_est)) %*% resid_est # sum of squares regression
  sigma2 <- ssr / nobs                 # equivalent to sigma2 in lm 
  ols_est <- ssr / resid_df        # to be used to compute cov matrix
  beta_est_cov <- (M %x% ols_est) # cov matrix of est. betas
  se <- sqrt(diag(beta_est_cov)) # standard error
  tstat <- (beta_est / se)        # t value for the ADF
  log_lhood <- -0.5*nobs*log(pi*2)-0.5*nobs*log(sigma2) -0.5*nobs
  eigenvalues <- polyroot(c(1, -beta_est))
  roots <- eigenvalues^(-1)
  
  
  list(X=X, 
       Y=Y,
       beta_est = beta_est,
       resid_est = resid_est,
       nobs = nobs,
       rank = rank,
       resid_df = resid_df,
       ssr = ssr,
       sigma2 = sigma2,
       ols_est = ols_est,
       beta_est_cov = beta_est_cov,
       se = se,
       tstat = tstat,
       log_lhood = log_lhood,
       roots = roots)
  
}


#let's assert the results are the same as from lm function  
X_VIXM<-cbind(intercept = 1, etfs_insample[,"VIXM"])

myRegResults<-my_LM(etfs_insample[,"VXZ"],X_VIXM)
myRegResults$beta_est

coint.reg = lm(etfs_insample[,"VXZ"] ~ etfs_insample[,"VIXM"])
coint.reg$coefficients


#checking stability condition
Xt1<-Lag(etfs_insample[,"VXZ"], shift = -1)
Xt1<-Xt1[-length(etfs_insample[,"VXZ"])]

regResults<-my_LM(etfs_insample[,"VXZ"][-length(etfs_insample[,"VXZ"])],Xt1)
stabilityCheck<-abs(regResults$roots)<1
stabilityCheck
#FALSE

#now stability condition for first difference
#important is to make sure we have time series of equal length after transforming
leg_VXZ<- diff(etfs_insample[,"VXZ"])
leg_VXZ<- leg_VXZ[-1]
#head(leg_VXZ)
#tail(leg_VXZ)

leg_VXZ_Xt1<-Lag(leg_VXZ, shift = -1)
leg_VXZ<-leg_VXZ[-length(leg_VXZ)]
leg_VXZ_Xt1<-leg_VXZ_Xt1[-length(leg_VXZ_Xt1)]
#head(leg_VXZ_Xt1)
#tail(leg_VXZ_Xt1)


regResults<-my_LM(leg_VXZ,leg_VXZ_Xt1)
stabilityCheck<-abs(regResults$roots)<1
stabilityCheck
#TRUE

##################################
# naive approach 
##################################

coint.reg = lm(etfs_insample[,"VXZ"] ~ etfs_insample[,"VIXM"])
print(summary(coint.reg))
#We conclude that the cointegration spread should be
#et_hat <- y - int_hat - beta_hat*x


# plot of the cointegration regression
x_range<-seq(min(etfs_insample[,"VIXM"]), max(etfs_insample[,"VIXM"]), length.out=400)
int_hat <- coint.reg$coefficients[1]
beta_hat <- coint.reg$coefficients[2]

int_hat 
beta_hat

y_hat<- int_hat + x_range*beta_hat
plot(etfs.zoo[,"VIXM"],etfs.zoo[,"VXZ"],xlab="VIXM", ylab="VXZ" , main = "Cointegration Regression VIXM vs VXZ")
lines(x_range,y_hat, col = "red")
#abline(int_hat, beta_hat)

#plot(coint.reg)
#CADF TEST ON RESIDUAL

cadf.test = ur.df(residuals(coint.reg), type = "none") # CADF because ADF test applies to cointegrated residual
print(summary(cadf.test))

adf.test(coint.reg$residuals,k=1)

#Phillips-Ouliaris Cointegration Test
phi_oui_test <- ca.po(cbind(etfs_insample[,'VXZ'], etfs_insample[,'VIXM']), demean = "constant", lag = "short", type = "Pu")
print(summary(phi_oui_test))


######################################
# plot the spread
######################################


et_hat_insample <- etfs_insample[,"VXZ"] - int_hat - beta_hat*etfs_insample[,"VIXM"]
plot(et_hat_insample)
et_hat_matrix<-et_hat_insample
et_hat_matrix$mu<-mean(et_hat_insample)
lines(et_hat_matrix[,"mu"], col="blue", lwd=3, lty=2)


pacf(et_hat_insample, 25,  main = "Residuals Time Series PACF")
#we can confirm that apart from significant first lag there is no memory in the residuals




#preparing out of sample data 
et_hat_outsample <- etfs_outsample[,"VXZ"] - int_hat - beta_hat*etfs_outsample[,"VIXM"]
plot(et_hat_outsample)


et_hat_combined <- rbind(et_hat_insample,et_hat_outsample)


#spread for the whole data
plot(et_hat_combined, main = expression(paste("Test/Train Combined data - Residual ", hat(e))))

events <- xts("Out-of-sample period",
              as.Date("2020-02-01" ))
addEventLines(events, srt=90, pos=2, lwd = 5, lty = 3, col = "blue")



##################################
#EC EQUATION taking diff
##################################


leg_VXZ<- diff(etfs[,"VXZ"])[-1]
leg_VIXM<-diff(etfs[,"VIXM"])[-1]

#testing first VIXM as leading variable
coint.reg = lm(leg_VXZ ~ leg_VIXM)

#ec_term.lag = lag(residuals(coint.reg), k = -1) #due to some indexing issue we suggest using the Lag function from Himsc package

ec_term.lag = Lag(residuals(coint.reg), shift = -1)



ecm.reg = lm(leg_VXZ ~ leg_VIXM + ec_term.lag + 0) # we want to avoid the extra constant outside of ec_term.lag
print(summary(ecm.reg))

#EC with an augment Delta Y_t-1 
# ecm_aug.reg = lm(leg_VXZ[-1] ~ (lag(leg_VXZ, k = -1)[-length(leg_VXZ)]) + leg_VIXM[-1] + ec_term.lag + 0)
# print(summary(ecm_aug.reg)) # not significant by t statistic



#testing VXZ as leading variable
coint.reg1 = lm( leg_VIXM ~ leg_VXZ )


ec_term.lag1 = Lag(residuals(coint.reg1), shift = -1)

ecm.reg1 = lm(leg_VIXM ~ leg_VXZ + ec_term.lag1 + 0) # we want to avoid the extra constant outside of ec_term.lag
print(summary(ecm.reg1))

#We conclude that VIXM is the leading variable due to slightly higher t-statistic

cadf.test = ur.df(residuals(coint.reg), type = "none") 
print(summary(cadf.test))


cadf.test1 = ur.df(residuals(coint.reg1), type = "none") 
print(summary(cadf.test1))

#######################
#Granger Causality Test
#######################

#Null: VIXM doesn't cause VXZ
grangertest(leg_VIXM ~ leg_VXZ)

#Null: - VXZ doesn't cause VIXM
grangertest(leg_VXZ ~ leg_VIXM)
#conclusion - It's more probable that VIXM is leading and VXZ lagging variable

########
# we continue analysis with VIXM as leading variable

leg_VXZ<- diff(etfs_insample[,"VXZ"])[-1]
leg_VIXM<-diff(etfs_insample[,"VIXM"])[-1]

coint.reg = lm(leg_VXZ ~ leg_VIXM)


# plot of the estimated Y
int_hat <- coint.reg$coefficients[1]
beta_hat <- coint.reg$coefficients[2]
y_hat<- int_hat + etfs_insample$VIXM*beta_hat
plot(etfs_insample$VXZ, ylab="VXZ" , main = "Real vs Fitted VXZ")
lines(y_hat, col = "red",  lwd=2)

addLegend(legend.loc = "topright", legend.names = c("Real VXZ", "Fitted VXZ") , col = c("black","red"),
          bg=c("black","red" ), ncol = 1, on = 1,lwd =1:2, cex=0.8)




#################################################################
# Determining boundaries for trading and other important features 
#################################################################

###############################
# implementation of AR(n) model

my_AR <- function(Y, lags, trend = F) {
  Y_cut =Y[-(1:lags),]  # remove first n elements from the dependent variable 
  Y_t =  head(Y,-lags)    # independent variable
  if (trend != 0) {
    Y_t = cbind(1, Y_t) }
    output = my_LM(Y_cut,Y_t)
    output$lags = lags
    output$aic= log(output$sigma2)+2*(1+output$rank)/output$nobs
      
  return(output)    
}


#fitting AR(1) model with our function
my_res <-my_AR(et_hat_insample, 1, trend='c')

my_res$beta_est[1]
my_res$beta_est[2]

#checking results with existing r package
#coefficients up to 3 decimals matching with those from our output
#the r package seems to use one observation more, which could explain the small difference

ar1_et_hat1=arima(et_hat_insample, order=c(1,0,0))
ar1_et_hat1$coef

#checking for stability - we can see that the characteristic roots lies within the unit circle
#stability condition is fulfilled
fit <- Arima((et_hat_insample), order = c(1, 0, 0), include.drift = T)
plot(fit)
autoplot(fit)


#######################
#Trading Strategy 
######################

#Fitting to OU process
tau = 1 / 252
my_C = my_res$beta_est[1]
my_B = my_res$beta_est[2]
my_theta = - log(my_B) / tau
my_mu_e = my_C / (1 - my_B)
my_sigma_ou = sqrt((2 * my_theta / (1 - exp(-2 * my_theta * tau))) * my_res$sigma2)
my_sigma_e = my_sigma_ou / sqrt(2 * my_theta)
my_halflife = log(2) / my_theta

#normalize spread
spread_norm <- (et_hat_insample -my_mu_e) / as.numeric(my_sigma_e)
names(spread_norm)<-"norm_spread"
plot(spread_norm)

et_hat_matrix$zUP<-1
et_hat_matrix$zDOWN<--1

lines(et_hat_matrix[, "zUP"], col = "red",lwd=3, lty=3)
lines(et_hat_matrix[, "zDOWN"], col = "red",lwd=3, lty=3)
lines(et_hat_matrix[, "mu"], col = "green",lwd=3, lty=4)

#merged insample data of normalized spread and ETFs' prices
trading_data_is<-merge(spread_norm, etfs_insample)
names(trading_data_is)<-c('norm_spread', 'X', 'Y', 'diff')


#convert to data.frame
data<-data.frame(date=index(trading_data_is), coredata(trading_data_is))

###############################
# simulate trading 
# function to compute hypothetical profits
# when followed rules @ buy Y sell X if z reaches 1
#                     @ buy X sell Y if z reaches -1
#                     @ exit position (when existing) if z reaches 0
# result is dataframe holding the complete trading history for the period

trade_simulation <-function(data, threshold=1) {
  
  data$mm_position <-c()
  data$cash <-  rep(0, length(data[,1]))
  data$shares_Y <- rep(0, length(data[,1]))
  data$shares_X <- rep(0, length(data[,1]))
  data$mm_position[1] <- 100000    #starting capital
  data$cash[1] <- 100000
  data$Y <- as.numeric(data[,'Y'])
  data$X <- as.numeric(data[,'X'])
  data$action<-"do nothing"
  existing_Position <- FALSE
  short_selling_cash <-0
  for (i in (2:length(data[,1]))) {
    if ((existing_Position == FALSE) & (data[,'norm_spread'][i-1] < threshold && data[,'norm_spread'][i] > threshold)) 
      #|
      #(data[,'norm_spread'][i-1] < 1 && data[,'norm_spread'][i] > 1)) 
    {
      data$action[i] <- "short Y and buy X"
      existing_Position <- TRUE
      data$shares_Y[i] <- -(data$cash[i-1] %/% data$Y[i])
      short_selling_cash <- -data$shares_Y[i]*data$Y[i]
      data$shares_X[i] <- (data$cash[i-1]+short_selling_cash)%/%data$X[i]
      data$cash[i] <- data$cash[i-1]+short_selling_cash -data$shares_X[i]*data$X[i]
      data$mm_position[i] <- data$cash[i]+data$shares_Y[i]*data$Y[i]+data$shares_X[i]*data$X[i]
    }
    else 
    {
      if ((existing_Position == FALSE) & (data[,'norm_spread'][i-1] > -threshold && data[,'norm_spread'][i] < -threshold)) 
        #|
        #(data[,'norm_spread'][i-1] < -1 && data[,'norm_spread'][i-1] > -1))
      {
        data$action[i] <- "short X and buy Y"
        existing_Position <- TRUE
        data$shares_X[i] <- -(data$cash[i-1] %/% data$X[i])
        short_selling_cash <- -data$shares_X[i]*data$X[i]
        data$shares_Y[i] <- (data$cash[i-1]+short_selling_cash)%/%data$Y[i]
        data$cash[i] <- data$cash[i-1]+short_selling_cash -data$shares_Y[i]*data$Y[i]
        data$mm_position[i] <- data$cash[i]+data$shares_Y[i]*data$Y[i]+data$shares_X[i]*data$X[i]
      }
      else 
      {
        if ((existing_Position == TRUE) & ((data[,'norm_spread'][i-1] > 0 && data[,'norm_spread'][i] < 0) |
                                           (data[,'norm_spread'][i-1] < 0 && data[,'norm_spread'][i] > 0))) 
        {
          data$action[i] <- "take profit"
          data$cash[i] <- data$cash[i-1]+data$shares_Y[i-1]*data$Y[i]+data$shares_X[i-1]*data$X[i]
          data$mm_position[i] <- data$cash[i]
          data$shares_X[i] <- 0
          data$shares_Y[i] <- 0
          existing_Position <- FALSE
        }
        else
        {
          data$shares_X[i] <- data$shares_X[i-1] 
          data$shares_Y[i] <- data$shares_Y[i-1] 
          data$cash[i] <- data$cash[i-1]
          data$mm_position[i] <- data$cash[i-1]+data$shares_Y[i]*data$Y[i]+data$shares_X[i]*data$X[i]
        }
        
      }
      
    }
  }
  
  print(paste("The Return of the strategy for the period ", paste(data[,"date"][1]), " - ",
              paste(data[,"date"][length(data[,"date"])]), " is ",
              paste(round((data[,"mm_position"][length(data[,"date"])]/1000-100),3)), "%", sep =""))
  
  return(data)
}

#optimize
pnl_optim<-data.frame(threshold=integer(length(seq(0.3,2, by = 0.05))),
                      res = double(length(seq(0.3,2, by = 0.05))), stringsAsFactors=FALSE)
j <-1
for (i in (seq(0.3,2, by = 0.05))) {
  pnl <-c()
  print(paste("When threshold =", i))
  pnl <-invisible((trade_simulation(data, i)))
  pnl_optim$threshold[j] <-i
  pnl_optim$res[j] <-round((pnl[nrow(pnl),"mm_position"]/1000-100),3)
  j <- j+1
} 

#threshold for entering position for which highest profit is obtained
threshold <- pnl_optim$threshold[which.max(pnl_optim$res)]

#result of the in-sample period
pnl<-trade_simulation(data,threshold)


pnl_xts <- xts(pnl[,-1], order.by=pnl[,1])


#plot the captured signals
et_hat_matrix$optimUP<-threshold
et_hat_matrix$optimDOWN<--threshold

lines(et_hat_matrix[, "optimUP"], col = "red",lwd=3, lty=3)
lines(et_hat_matrix[, "optimDOWN"], col = "red",lwd=3, lty=3)


plot(spread_norm, main = "VIXM vs. VXZ  in-sample",
     cex.main = 0.8,
     cex.lab = 0.8,
     cex.axis = 0.8)
#lines(et_hat_matrix[, "zUP"], col = "purple",lwd=3, lty=4)
#lines(et_hat_matrix[, "zDOWN"], col = "purple",lwd=3, lty=4)
lines(et_hat_matrix[, "optimUP"], col = "purple",lwd=3, lty=2)
lines(et_hat_matrix[, "optimDOWN"], col = "purple",lwd=3, lty=2)
lines(et_hat_matrix[, "mu"], col = "green",lwd=3, lty=4)
point_type_buy <- rep(NA, nrow(pnl_xts))
point_type_sell <- rep(NA, nrow(pnl_xts))
buy_index <- which(pnl_xts$action == "short Y and buy X")
sell_index <- which(pnl_xts$action == "short X and buy Y")
point_type_buy[buy_index] <- 25
point_type_sell[sell_index] <- 24
points(pnl_xts$norm_spread, col = "black" ,pch = point_type_buy, cex =3, lwd = 2, bg="blue")
points(pnl_xts$norm_spread, col = "black" ,pch = point_type_sell, cex =3, lwd = 2, bg="red")

legend( x="topright", 
        legend=c("Red line","blue points","Green line","purple points"),
        col=c("red","blue","green","purple"), lwd=1, lty=c(1,NA,2,NA),
        pch=c(NA,15,NA,17) )

addLegend(legend.loc = "topright", legend.names = c("short VXZ and buy VIXM", "short VIXM and buy VXZ") , col = c("red","blue"),
          bg=c("red", "blue"), ncol = 1, on = 1, pch=25:24, cex=0.8)


#count number of positions
table(pnl_xts$action)


#performance evaluation

#calculating simple daily return Rt = (Pt-P(t-1))/P(t-1)
n <- nrow(pnl)
pnl$daily_ret<-rep(0, n)
pnl$daily_ret[2:n]<-((pnl$mm_position[2:n] - pnl$mm_position[1:(n-1)])/pnl$mm_position[1:(n-1)])

#Plot cumulative return and drawdown
rt_xts <-  xts(pnl$daily_ret, order.by=as.Date(pnl[,1], format="%d-%m-%Y"))
charts.PerformanceSummary(rt_xts)

#return over the whole period
final_ret<-round((pnl[,"mm_position"][length(pnl[,"date"])]/100000-1),3)
#annualized standard deviation
std_annual <-sd(rt_xts)*sqrt(250)
#Sharpe ratio
final_ret/std_annual

#Sortino ratio
final_ret/(DownsideDeviation(rt_xts)*sqrt(250))

#RoMaD
final_ret/maxDrawdown(rt_xts)

#statistic summary of downside risk
table.DownsideRisk(rt_xts,Rf=.03/12)


###################################################################
# testing out-of-sample
###################################################################

# 
# et_hat_outsample <- etfs_outsample[,"VXZ"] - int_hat - beta_hat*etfs_outsample[,"VIXM"]
# plot(et_hat_outsample)

#normalize spread
spread_norm_outsample <- (et_hat_outsample -my_mu_e) / as.numeric(my_sigma_e)
names(spread_norm_outsample)<-"norm_spread"
plot(spread_norm_outsample)



#merged insample data of normalized spread and ETFs' prices
trading_data_out<-merge(spread_norm_outsample, etfs_outsample)
names(trading_data_out)<-c('norm_spread', 'X', 'Y', 'diff')

#convert to data.frame
data_out<-data.frame(date=index(trading_data_out), coredata(trading_data_out))

pnl<-trade_simulation(data_out,threshold)


pnl_xts <- xts(pnl[,-1], order.by=pnl[,1])

#show output
as.data.table(pnl_xts)

#count number of positions
table(pnl_xts$action)


#plot the out-of-sample trading
plot(spread_norm_outsample, main = "VIXM vs. VXZ  out-of-sample",
     cex.main = 0.8,
     cex.lab = 0.8,
     cex.axis = 0.8)


spread_norm_outsample$optimUP<-threshold
spread_norm_outsample$optimDOWN<--threshold

lines(spread_norm_outsample[, "optimUP"], col = "purple",lwd=3, lty=3)
lines(spread_norm_outsample[, "optimDOWN"], col = "purple",lwd=3, lty=3)
lines(et_hat_matrix[, "mu"], col = "green",lwd=3, lty=4)
point_type_buy <- rep(NA, nrow(pnl_xts))
point_type_sell <- rep(NA, nrow(pnl_xts))
buy_index <- which(pnl_xts$action == "short Y and buy X")
sell_index <- which(pnl_xts$action == "short X and buy Y")
point_type_buy[buy_index] <- 25
point_type_sell[sell_index] <- 24
points(pnl_xts$norm_spread, col = "black" ,pch = point_type_buy, cex =3, lwd = 2, bg="blue")
points(pnl_xts$norm_spread, col = "black" ,pch = point_type_sell, cex =3, lwd = 2, bg="red")

legend( x="topright", 
        legend=c("Red line","blue points","Green line","purple points"),
        col=c("red","blue","green","purple"), lwd=1, lty=c(1,NA,2,NA),
        pch=c(NA,15,NA,17) )

addLegend(legend.loc = "topright", legend.names = c("short VXZ and buy VIXM", "short VIXM and buy VXZ") , col = c("red","blue"),
          bg=c("red", "blue"), ncol = 1, on = 1, pch=25:24, cex=0.8)



#highlight interesting time period
#regimeChange <- window(spread_norm_outsample, start=as.Date("2020-03-01"), end=as.Date("2020-04-01"))
regimeChange <- window(spread_norm_outsample, start=as.Date("2020-02-01"), end=as.Date("2020-02-28"))
nr <- nrow(regimeChange)
shade <- cbind(upper = rep(range(regimeChange)[2], nr), lower = rep(range(regimeChange)[1], nr))
shade <- xts(shade, index(regimeChange))
 
addPolygon(shade, col = "lightpink", on = -1)



#evaluate out-of-sample performance
n <- nrow(pnl)
pnl$daily_ret<-rep(0, n)
pnl$daily_ret[2:n]<-((pnl$mm_position[2:n] - pnl$mm_position[1:(n-1)])/pnl$mm_position[1:(n-1)])

#Plot cumulative return and drawdown
rt_xts <-  xts(pnl$daily_ret, order.by=as.Date(pnl[,1], format="%d-%m-%Y"))
charts.PerformanceSummary(rt_xts)


#statistic summary of downside risk
table.DownsideRisk(rt_xts,Rf=.03/12)

#return over the out-of-sample period
final_ret<-round((pnl[,"mm_position"][length(pnl[,"date"])]/100000-1),3)
#annualized standard deviation
std_annual <-sd(rt_xts)*sqrt(40)
#Sharpe ratio
final_ret/std_annual

#Sortino ratio
final_ret/(DownsideDeviation(rt_xts)*sqrt(40))

#RoMaD
final_ret/maxDrawdown(rt_xts)


###############################################################
#Johansen Procedure 
###############################################################


#reading data and converting to xts format 
data_etf_all<-read.csv( "D:\\cqf\\etf_data_vix.csv", header= TRUE, sep = ";")


#clean data
data_etf_all<-na.omit(data_etf_all) 

date <- as.Date(data_etf_all$Date,"%d/%m/%Y")


etfs_all<-as.xts(data_etf_all[,c("VXX","VXZ","VIXM","VIXY")], date)



names(etfs_all) =c("VXX","VXZ","VIXM","VIXY")
index(etfs_all) = as.Date(index(etfs_all))


#choose time window 
etfs_all_insample = window(etfs_all, start=as.Date("2019-01-01"), end=as.Date("2020-01-01")) 
etfs_all_outsample = window(etfs_all, start=as.Date("2020-01-01"), end=as.Date("2020-03-01")) 


etfs_alldata<-etfs_all


etfs_all.zoo<-zoo(etfs_all_insample, order.by = index(etfs_all_insample))

#plot all data
etfs_all_series = tidy(etfs_alldata) %>% 
  
  ggplot(aes(x=index,y=value, color=series)) + geom_line() + 
  
  labs(title = "Short Term vs Mid Term Volatility",
       
       subtitle = "End of Day Adjusted Prices of ETFs/ETNs",
       caption = " Source: Yahoo Finance") +
  
  labs(y="Price", x = "Date")

etfs_all_series


#first let's test only VIXM as leading and VXZ as lagging variables
#this should be similar result to the naive OLS regression run earlier 
johansen.test_VIXMVXZ = ca.jo(etfs_all.zoo[,colnames(etfs_all.zoo) %in% c("VIXM","VXZ")], ecdet = "const", type="eigen", K=2, spec="longrun")
cajorls(johansen.test_VIXMVXZ) # OLS regression of restricted VECM -- EC-term (long run) and past differences (short run)

#now test all with all four assets
johansen.test = ca.jo(etfs_all.zoo, ecdet = "const", type="eigen", K=2, spec="longrun")
cajorls(johansen.test) # OLS regression of restricted VECM -- EC-term (long run) and past differences (short run)


#we see that there are at most 2 independent time_series
#equivalently one needs 2 assets to form stationary portfolio  
print(summary(ca.jo(etfs_all.zoo, type="eigen", ecdet="const", K=2, spec="longrun")))

#testing for up to 5 lags has not revealed any other cointegrated pairs
#hence one can keep k=2
for(k in seq(2,5)) {
  
  print("######################################")
  print(paste("#              Lag = ",k,"               #",sep=""))
  print("######################################")
  
  # Run Johansen Maximum Eigen Statistic Test on prices with trend and lag of k
  print(summary(ca.jo(etfs_all.zoo, type="eigen", ecdet="trend", K=k)))
  
  # Run Johansen Trace Statistic Test on prices with trend and lag of k
  print(summary(ca.jo(etfs_all.zoo, type="trace", ecdet="trend", K=k)))
  
}


