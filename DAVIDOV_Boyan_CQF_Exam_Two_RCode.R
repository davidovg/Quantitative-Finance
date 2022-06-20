#Real price of a Call from BS is exp(-r * T) * N(d2) and the 
# probability of being in-the-money is just d2

d2<-(log(100/100)+(0.05-0.5*(0.2^2))*(1-0))/0.2*sqrt(1)

d2

Nd2<-pnorm(d2)

Nd2

exp(-0.05)*Nd2 # value of call 

#Function to calculate the theoretical (BS) price
#cash-or-nothing -> 1$ or nothing
# Type = 1 for call, Type=-1 for put

#T - time to maturity in years
#S0 - initial price
#K - execution price or strike
#sigma - annual volatility

BS_digital<-function(S0, K, sigma, r, T, Type, show_price=TRUE)
{
  d2 = (log(S0/K) + (r - 0.5 * sigma^2) * T)/ (sigma*sqrt(T))
  Put_or_Call<-c()
  ifelse(Type==1, Put_or_Call<-"Call", Put_or_Call<-"Put")
  price<-exp(-r * T) * pnorm(Type*d2)
  if(show_price==TRUE) {print(paste("The BS price of the digital ", paste(Put_or_Call), " is ",
                                    round(price,4),sep=""))} else {
                                      print(paste("The BS probability of ending ITM of the digital" , paste(Put_or_Call), "is ", 
                                                  round(pnorm(Type*d2),4),sep=""))
                                    }
 return(price) 
}

BS_binary_call_theoretical_price<-BS_digital(100,100,0.2,0.05,1,1)
BS_binary_call_theoretical_price

BS_binary_put_theoretical_price<-BS_digital(100,100,0.2,0.05,1,-1)
BS_binary_put_theoretical_price

# this is the payoff of the option
# just an indicator function
digital_call_payoff<-function(S_T,K)
{
  if (S_T >= K) {
    return(1)} else {
      return(0)}
}
# a plot illustrates
plot(sapply((0:150),digital_call_payoff,100),
     main="Payoff of a digital call option",col="red",
     xlab="Stock price",ylab='Payoff')


# run simulation using Euler - Maruyama Scheme

monte_carlo_sim_euler<-function(simulations, days, init_price,annual_vol, risk_free) {
  price_matrix<-matrix(0,nrow=(days+1),ncol=simulations)
  dt<-1/days
  for (i in 1:simulations) {
  price_matrix[1,i]<-init_price
    for (j in 2:(days+1)) {
      price_matrix[j,i]=price_matrix[j-1,i]*(1+risk_free*dt+annual_vol*sqrt(dt)*rnorm(1))
    } 
  } 
  matplot(price_matrix, main="MC - Euler-Maruyama Scheme", ylab="Underlying Price", xlab="Time", type="l")
  price_matrix
return(price_matrix)
}

#error analysis of the simple method

##############
#1000 paths
##############

mc_1000<-monte_carlo_sim_euler(1000,250,100,0.2,0.05)
call_payoffs_1000<-c(mc_1000[nrow(mc_1000),]>100)
binary_call_price_1000<-mean(call_payoffs_1000)*exp(-0.05*1)
binary_call_price_1000

absolute_error_call<-(binary_call_price_1000-BS_binary_call_theoretical_price)
relative_error_call<-absolute_error_call/BS_binary_call_theoretical_price
relative_error_call

put_payoffs_1000<-c(mc_1000[251,]<100)
binary_put_price_1000<-mean(put_payoffs_1000)*exp(-0.05*1)
binary_put_price_1000

absolute_error_put<-(binary_put_price_1000-BS_binary_put_theoretical_price)
relative_error_put<-absolute_error_put/BS_binary_put_theoretical_price
relative_error_put

##############
#5000 paths
##############

mc_5000<-monte_carlo_sim_euler(5000,250,100,0.2,0.05)
call_payoffs_5000<-c(mc_5000[nrow(mc_5000),]>100)
binary_call_price_5000<-mean(call_payoffs_5000)*exp(-0.05*1)
binary_call_price_5000

absolute_error_call<-(binary_call_price_5000-BS_binary_call_theoretical_price)
relative_error_call<-absolute_error_call/BS_binary_call_theoretical_price
relative_error_call

put_payoffs_5000<-c(mc_5000[251,]<100)
binary_put_price_5000<-mean(put_payoffs_5000)*exp(-0.05*1)
binary_put_price_5000


absolute_error_put<-(binary_put_price_5000-BS_binary_put_theoretical_price)
relative_error_put<-absolute_error_put/BS_binary_put_theoretical_price
relative_error_put

#The code has been refined to accomodate more simulated paths with greater speed
#only the final price is relevant and only payoffs are stored, rather than the whole path

binary_option_value<-function(simulations, days, init_price, strike, annual_vol, risk_free) {
  call_payoffs<-0
  put_payoffs<-0
  call_value<-0
  put_value<-0
  value<-0
  dt<-1/days
  price_vector<-c()
  price_vector[1]<-init_price
  for (i in 1:simulations) {
    price_vector<-c()
    price_vector[1]<-init_price
    for (j in 2:(days+1)) {
      price_vector[j]=price_vector[j-1]*(1+risk_free*dt+annual_vol*sqrt(dt)*rnorm(1))
    }
    if (price_vector[length(price_vector)]>strike) {
      call_payoffs<-call_payoffs+1 } else { put_payoffs<-put_payoffs+1}
  }
  call_value<-exp(-risk_free)*(call_payoffs/simulations)
  put_value<-exp(-risk_free)*(put_payoffs/simulations)
return(list("Put" = put_value, "Call" =call_value))    
}

#########################################
#Error using 1000 paths & 10 repetitions
#########################################

system.time(binary_option_value(1000,250,100,100,0.2,0.05))

x_1000<-replicate(10,binary_option_value(1000,250,100,100,0.2,0.05),TRUE)

x_1000_call<-Reduce("+",x_1000["Call",])/10

x_1000_put<-Reduce("+",x_1000["Put",])/10

error_call<-(x_1000_call-BS_binary_call_theoretical_price)/BS_binary_call_theoretical_price
error_call

error_put<-(x_1000_put-BS_binary_put_theoretical_price)/BS_binary_put_theoretical_price
error_put


#########################################
#Error using 5000 paths & 10 repetitions
#########################################

#we check the time needed for one replication
system.time(binary_option_value(5000,250,100,100,0.2,0.05))

x_5000<-replicate(10,binary_option_value(5000,250,100,100,0.2,0.05),TRUE)

x_5000_call<-Reduce("+",x_5000["Call",])/10

x_5000_put<-Reduce("+",x_5000["Put",])/10

error_call<-(x_5000_call-BS_binary_call_theoretical_price)/BS_binary_call_theoretical_price
error_call

error_put<-(x_5000_put-BS_binary_put_theoretical_price)/BS_binary_put_theoretical_price
error_put


##########################################
#Error using 10000 paths & 10 repetitions
##########################################

x_10000<-replicate(10,binary_option_value(10000,250,100,100,0.2,0.05),TRUE)

x_10000_call<-Reduce("+",x_10000["Call",])/10

x_10000_put<-Reduce("+",x_10000["Put",])/10

error_call<-(x_10000_call-BS_binary_call_theoretical_price)/BS_binary_call_theoretical_price
error_call

error_put<-(x_10000_put-BS_binary_put_theoretical_price)/BS_binary_put_theoretical_price
error_put

##########################################
#Error using 100000 paths & 10 repetitions
##########################################

x_100000<-replicate(10,binary_option_value(100000,250,100,100,0.2,0.05),TRUE)

x_100000_call<-Reduce("+",x_100000["Call",])/10

x_100000_put<-Reduce("+",x_100000["Put",])/10

error_call<-(x_100000_call-BS_binary_call_theoretical_price)/BS_binary_call_theoretical_price
error_call

error_put<-(x_100000_put-BS_binary_put_theoretical_price)/BS_binary_put_theoretical_price
error_put


######################################################
# graph for the % error vs #simulations in Euler method
######################################################
par(mfrow=c(1,2))
xx<-c(1000,5000,10000,50000,100000,500000,1000000)
x<-c("1","5","10","50","100","500","1000")
y1<-c(1.18,0.57, 0.43,0.32,0.24,0.12,0.06)
plot(x,y1,col="blue",type="l",ylab="%Error (abs.value)",xlab="# of Simulations in '000",main = "%Error vs. #Paths - Calls ")
points(x,y1, col="red", pch=15)
text(x=45, y=1.18, "1",col="black", font=2, cex=0.8)
text(x=55, y=0.57, "5",col="black", font=2, cex=0.8)
text(x=70, y=0.43, "10",col="black", font=2, cex=0.8)
text(x=100, y=0.32, "50",col="black", font=2, cex=0.8)
text(x=170, y=0.26, "100",col="black", font=2, cex=0.8)
text(x=580, y=0.15, "500",col="black", font=2, cex=0.8)
text(x=950, y=0.11, "1000",col="black", font=2, cex=0.8)
legend("topright","#Paths in '000",
       col="red",pch=15,cex=0.8);
x1<-c(1,5,10,50,100,500,1000) 
y2<-c(1.50,0.72,0.55,0.41,0.30,0.12,0.08)
plot(x,y2,col="blue",type="l",ylab="%Error (abs.value)",xlab="# of Simulations in '000",main = "%Error vs. #Paths - Puts ")
points(x,y2, col="red", pch=15)
text(x=45, y=1.50, "1",col="black", font=2, cex=0.8)
text(x=55, y=0.72, "5",col="black", font=2, cex=0.8)
text(x=70, y=0.55, "10",col="black", font=2, cex=0.8)
text(x=100, y=0.41, "50",col="black", font=2, cex=0.8)
text(x=170, y=0.3, "100",col="black", font=2, cex=0.8)
text(x=580, y=0.15, "500",col="black", font=2, cex=0.8)
text(x=950, y=0.13, "1000",col="black", font=2, cex=0.8)
legend("topright","#Paths in '000",
       col="red",pch=15,cex=0.8)

#############################################
#error analysis based on different time steps
#############################################
error_call<-c()
error_put<-c()
for (i in seq(10,250,by=10)) {
  error_call[i/10]<-(binary_option_value(10000,i,100,100,0.2,0.05)$Call
                     -BS_binary_call_theoretical_price)/-BS_binary_call_theoretical_price
  error_put[i/10]<-(binary_option_value(10000,i,100,100,0.2,0.05)$Put
                    -BS_binary_put_theoretical_price)/-BS_binary_put_theoretical_price
}

plot(seq(10,250,by=10),error_call, type="l",col="blue",xlab="Number of Time steps",
     ylab="Error", main="Call: Error vs #Time Steps")


plot(seq(10,250,by=10),error_put, type="l",col="blue",xlab="Number of Time steps",
     ylab="Error", main="Put: Error vs #Time Steps")


## This is to compare the simulated underlying path
## USING MILSTEIN, EULER AND THE REAL GBM
## Random numbers generated using the Box-Muller Method

mu<-0.05
sigma<-0.2
dt<-0.01

U1<-runif(251)
U2<-runif(251)

N_0_1<-sqrt(-2*log(U1))*cos(2*pi*U2)
W<-c()
W<-N_0_1*sqrt(dt)
Euler<-Milstein<-Log<-c()
Euler[1]<-Milstein[1]<-Log[1]<-100
for ( i in 2:length(W))
{
  Euler[i]<-Euler[i-1]+mu*Euler[i-1]*dt+sigma*Euler[i-1]*W[i-1]
  Milstein[i]<-Milstein[i-1]+mu*Milstein[i-1]*dt+sigma*Milstein[i-1]*W[i-1]+
    0.5*(W[i-1]^2-dt)*Milstein[i-1]*sigma^2
  Log[i]<-Log[i-1]*exp((mu-0.5*sigma^2)*dt+sigma*W[i-1])
}
R<-cbind(Euler,Milstein,Log)
colnames(R)<-c('Euler','Milstein','Log')
plot(Euler,col="blue",type="l",ylab="Value", main = "Path Simulation, dt=0.01")
lines(Milstein,col="red")
lines(Log,col="green")
legend("topleft",c('Euler','Milstein','Log'),
       col=c("blue","red","green"),lty=c(1,1,1),cex=0.8);

#random numbers using rnorm function from r 
par(mfrow=c(1,2))
R1<-rnorm(251)
qqnorm(R1, main = "RNORM",
        xlab = "Theoretical Quantiles", ylab = "Sample Quantiles",
        plot.it = TRUE, datax = FALSE)
qqline(R1, col = 2)


#random numbers generated by Box-Muller method
U1<-runif(251)
U2<-runif(251)

N_0_1<-sqrt(-2*log(U1))*cos(2*pi*U2)
qqnorm(N_0_1, main = "Box-Müller",
       xlab = "Theoretical Quantiles", ylab = "Sample Quantiles",
       plot.it = TRUE, datax = FALSE)
qqline(N_0_1, col = 2)

####################################
# Using Milstein method
####################################

binary_option_value_Milstein<-function(simulations, days, init_price, strike, sigma, mu) {
  call_payoffs<-0
  put_payoffs<-0
  call_value<-0
  put_value<-0
  dt<-1/days
  Milstein<-c()
  Milstein[1]<-init_price
  for (i in 1:simulations) {
    Milstein<-c()
    Milstein[1]<-strike
    for (j in 2:(days+1)) {
      Milstein[j]<-Milstein[j-1]*(1+mu*dt+sigma*rnorm(1)*sqrt(dt)+
        0.5*(rnorm(1)^2-1)*dt*sigma^2)}
    if (Milstein[length(Milstein)]>strike) {
      call_payoffs<-call_payoffs+1 } else { put_payoffs<-put_payoffs+1}
  }
  call_value<-exp(-mu)*(call_payoffs/simulations)
  put_value<-exp(-mu)*(put_payoffs/simulations)
  return(list("Put" = put_value, "Call" =call_value))
}


#######################################################
#Error using 1000 paths under Milstein
#######################################################

x_1000<-binary_option_value_Milstein(1000,250,100,100,0.2,0.05)

error_call<-(x_1000$Call-BS_binary_call_theoretical_price)/BS_binary_call_theoretical_price
error_call

error_put<-(x_1000$Put-BS_binary_put_theoretical_price)/BS_binary_put_theoretical_price
error_put


#######################################################
#Error using 5000 paths under Milstein
#######################################################

x_5000<-binary_option_value_Milstein(5000,250,100,100,0.2,0.05)

error_call<-(x_5000$Call-BS_binary_call_theoretical_price)/BS_binary_call_theoretical_price
error_call

error_put<-(x_5000$Put-BS_binary_put_theoretical_price)/BS_binary_put_theoretical_price
error_put

#######################################################
#Error using 1000 paths & 10 repetitions under Milstein
#######################################################
    
x_1000<-replicate(10,binary_option_value_Milstein(1000,250,100,100,0.2,0.05),TRUE)
    
x_1000_call<-Reduce("+",x_1000["Call",])/10
x_1000_call
    
x_1000_put<-Reduce("+",x_1000["Put",])/10
x_1000_put
    
error_call<-(x_1000_call-BS_binary_call_theoretical_price)/BS_binary_call_theoretical_price
error_call
    
error_put<-(x_1000_put-BS_binary_put_theoretical_price)/BS_binary_put_theoretical_price
error_put


#######################################################
#Error using 50000 paths under Milstein
#######################################################

x_50000<-binary_option_value_Milstein(50000,250,100,100,0.2,0.05)

error_call<-(x_50000$Call-BS_binary_call_theoretical_price)/BS_binary_call_theoretical_price
error_call

error_put<-(x_50000$Put-BS_binary_put_theoretical_price)/BS_binary_put_theoretical_price
error_put


## This is to compare the simulated underlying path
## USING MILSTEIN_Heston, EULER AND THE REAL GBM

kappa<-1.5
theta<-0.09
xi<-0.25
rho<-(-0.5)

mu<-0.05
sigma<-0.2
dt<-0.004


W1<-rnorm(251)
W2<-rnorm(251)


W2 <- rho*W1 + sqrt(1 - rho^2)*W2;
W1<-W1*sqrt(dt)
annual_var<-c()
annual_var[1]<-0.04
Euler<-Milstein<-Log<-c()
Euler[1]<-Milstein[1]<-Log[1]<-100
for ( i in 2:length(W1))
{
  Euler[i]<-Euler[i-1]+mu*Euler[i-1]*dt+sigma*Euler[i-1]*W1[i-1]
  annual_var[i] = annual_var[i-1] + kappa*(theta - max(annual_var[i-1],0))*dt + xi*
    W2*sqrt(max(annual_var[i-1],0)*dt)+0.25*xi*xi*(W2*W2 - 1)*dt
  Milstein[i]<-Milstein[i-1]+mu*Milstein[i-1]*dt+(sqrt(annual_var[i]))*Milstein[i-1]*W1[i-1]+
    0.5*(W1[i-1]^2-dt)*Milstein[i-1]*annual_var[i]
  Log[i]<-Log[i-1]*exp((mu-0.5*sigma^2)*dt+sigma*W1[i-1])
}
R<-cbind(Euler,Milstein,Log)
colnames(R)<-c('Euler','Milstein_Heston','Log')
plot(Euler,col="blue",type="l",ylab="Value", main = "Path Simulation, dt=0.004", ylim=c(50,150))
lines(Milstein,col="red")
lines(Log,col="green")
legend("topleft",c('Euler','Milstein','Log'),
       col=c("blue","red","green"),lty=c(1,1,1),cex=0.8);


###########################################
# Heston Stochastic Volatility and Milstein
###########################################

binary_option_value_Milstein_Heston<-function(simulations, days, init_price, strike, 
                                    sigma, mu, kappa=1.2,theta=0.0225,xi=0.20,rho=-0.5) {
  call_payoffs<-0
  put_payoffs<-0
  call_value<-0
  put_value<-0
  dt<-1/days
  annual_var<-c()
annual_var[1]<-sigma^2
  for (i in 1:simulations) {
    Milstein<-c()                        #free memory
    Milstein[1]<-init_price 
    annual_var<-c()
    annual_var[1]<-sigma^2
      for (j in 2:(days+1)) {
      W1 <- rnorm(1);                   #generate random numbers
      W2 <- rnorm(1); 
      W2 <- rho*W1 + sqrt(1 - rho^2)*W2;   #generate correlated brownian motions
      annual_var[j] = annual_var[j-1] + kappa*(theta - max(annual_var[j-1],0))*dt + xi*
      W2*sqrt(max(annual_var[j-1],0)*dt)+0.25*xi*xi*(W2*W2 - 1)*dt
      Milstein[j]<-Milstein[j-1]*(1+mu*dt+(sqrt(annual_var[j]))*W1*sqrt(dt)+
                                    0.5*(W1^2-1)*dt*annual_var[j])}
    if (Milstein[length(Milstein)]>strike) {
      call_payoffs<-call_payoffs+1 } else { put_payoffs<-put_payoffs+1}
  }
  call_value<-exp(-mu)*(call_payoffs/simulations)
  put_value<-exp(-mu)*(put_payoffs/simulations)
  return(list("Put" = put_value, "Call" =call_value))
}

#######################################################
#Error using 1000 paths under Milstein & Heston
#######################################################

x_1000<-binary_option_value_Milstein_Heston(10000,250,100,100,0.2,0.05, kappa=1.05, xi=0.15,theta=0.0225)

x_1000$Call
error_call<-(x_1000$Call-BS_binary_call_theoretical_price)/BS_binary_call_theoretical_price
error_call

x_1000$Put
error_put<-(x_1000$Put-BS_binary_put_theoretical_price)/BS_binary_put_theoretical_price
error_put

#######################################################
#Error using 5000 paths under Milstein & Heston
#######################################################

x_5000<-binary_option_value_Milstein_Heston(5000,250,100,100,0.2,0.05)

x_5000$Call
error_call<-(x_5000$Call-BS_binary_call_theoretical_price)/BS_binary_call_theoretical_price
error_call

x_5000$Put
error_put<-(x_5000$Put-BS_binary_put_theoretical_price)/BS_binary_put_theoretical_price
error_put

#######################################################
#Error using 10000 paths under Milstein & Heston
#######################################################

x_10000<-binary_option_value_Milstein_Heston(10000,250,100,100,0.2,0.05)

x_10000$Call
error_call<-(x_10000$Call-BS_binary_call_theoretical_price)/BS_binary_call_theoretical_price
error_call

x_10000$Put
error_put<-(x_10000$Put-BS_binary_put_theoretical_price)/BS_binary_put_theoretical_price
error_put

#######################################################
#Error using 50000 paths under Milstein & Heston
#######################################################

x_50000<-binary_option_value_Milstein_Heston(50000,250,100,100,0.2,0.05)

x_50000$Call
error_call<-(x_50000$Call-BS_binary_call_theoretical_price)/BS_binary_call_theoretical_price
error_call

x_50000$Put
error_put<-(x_50000$Put-BS_binary_put_theoretical_price)/BS_binary_put_theoretical_price
error_put

#######################################################
# Binary Asset-or-Nothing option value
#######################################################

binary_option_value_Asset<-function(simulations, days, init_price, strike, annual_vol, risk_free) {
  call_payoffs<-0
  put_payoffs<-0
  call_value<-0
  put_value<-0
  value<-0
  dt<-1/days
  price_vector<-c()
  price_vector[1]<-init_price
  for (i in 1:simulations) {
    price_vector<-c()
    price_vector[1]<-init_price
    for (j in 2:(days+1)) {
      price_vector[j]=price_vector[j-1]*(1+risk_free*dt+annual_vol*sqrt(dt)*rnorm(1))
    }
    if (price_vector[length(price_vector)]>strike) {
      call_payoffs<-c(call_payoffs,price_vector[j])} else
      { put_payoffs<-c(put_payoffs,price_vector[j])}
  }
  call_value<-exp(-risk_free)*(sum(call_payoffs)/simulations)
  put_value<-exp(-risk_free)*(sum(put_payoffs)/simulations)
  return(list("Put" = put_value, "Call" =call_value))    
}

#######################################################
#Error of the Asset-or-Nothing value 
#######################################################

x_1000<-binary_option_value_Asset(1000,250,100,100,0.2,0.05)
x_1000$Call
Asset_BS_binary_call<-100*BS_binary_call_theoretical_price
error_call<-(x_1000$Call-Asset_BS_binary_call)/Asset_BS_binary_call
error_call

x_1000$Put
Asset_BS_binary_put<-100*BS_binary_put_theoretical_price
error_put<-(x_1000$Put-Asset_BS_binary_put)/Asset_BS_binary_put
error_put
