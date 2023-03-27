
#########################
#Question1
######################### 

asset.names <- c("A", "B", "C", "D")

mu.vec = c(0.02, 0.07, 0.15, 0.2)

names(mu.vec) = asset.names

ro.mat = matrix(c(1, 0.3, 0.3, 0.3,
                  0.3, 1, 0.6, 0.6,
                  0.3, 0.6, 1, 0.6,
                  0.3, 0.6, 0.6, 1),
                  nrow=4, ncol=4)



stdev.mat = matrix(c(0.05, 0, 0, 0,
                     0, 0.12, 0, 0,
                     0, 0, 0.17, 0,
                     0, 0, 0, 0.25),
                    nrow=4, ncol=4)

sigma.mat <- stdev.mat%*%ro.mat%*%stdev.mat

dimnames(sigma.mat) = list(asset.names, asset.names)

ones_vector<-rep(1,4)

A<-as.numeric(t(ones_vector)%*%(solve(sigma.mat))%*%ones_vector)
B<-as.numeric(t(mu.vec)%*%(solve(sigma.mat))%*%ones_vector)
C<-as.numeric(t(mu.vec)%*%(solve(sigma.mat))%*%mu.vec)



rf<-0.005

lambda<-(A*m-B)/(A*C-B^2)
gamma<-(C-B*m)/(A*C-B^2)

w_optim<-solve(sigma.mat)%*%(lambda*mu.vec+gamma*ones_vector)

w_optim

portfolio_risk<-sqrt(as.numeric(t(w_optim)%*%sigma.mat%*%w_optim))

portfolio_risk
@

#########
#stressed*1.25 
#########

ro.mat_stressed1<-ro.mat*1.25-0.25*diag(diag(ro.mat))

ro.mat_stressed1

sigma.mat_stressed1 <- stdev.mat%*%ro.mat_stressed1%*%stdev.mat

A<-as.numeric(t(ones_vector)%*%(solve(sigma.mat_stressed1))%*%ones_vector)
B<-as.numeric(t(mu.vec)%*%(solve(sigma.mat_stressed1))%*%ones_vector)
C<-as.numeric(t(mu.vec)%*%(solve(sigma.mat_stressed1))%*%mu.vec)

lambda<-(A*m-B)/(A*C-B^2)
gamma<-(C-B*m)/(A*C-B^2)

w_optim_stressed1<-solve(sigma.mat_stressed1)%*%(lambda*mu.vec+gamma*ones_vector)

w_optim_stressed1

portfolio_risk_stressed1<-sqrt(as.numeric(t(w_optim_stressed1)%*%sigma.mat_stressed1%*%w_optim_stressed1))

portfolio_risk_stressed1

#########
#stressed*1.5 
#########

ro.mat_stressed2<-ro.mat*1.5-0.5*diag(diag(ro.mat))

ro.mat_stressed2

sigma.mat_stressed2 <- stdev.mat%*%ro.mat_stressed2%*%stdev.mat

A<-as.numeric(t(ones_vector)%*%(solve(sigma.mat_stressed2))%*%ones_vector)
B<-as.numeric(t(mu.vec)%*%(solve(sigma.mat_stressed2))%*%ones_vector)
C<-as.numeric(t(mu.vec)%*%(solve(sigma.mat_stressed2))%*%mu.vec)

lambda<-(A*m-B)/(A*C-B^2)
gamma<-(C-B*m)/(A*C-B^2)

w_optim_stressed2<-solve(sigma.mat_stressed2)%*%(lambda*mu.vec+gamma*ones_vector)

w_optim_stressed2

portfolio_risk_stressed2<-sqrt(as.numeric(t(w_optim_stressed2)%*%sigma.mat_stressed2%*%w_optim_stressed2))

portfolio_risk_stressed2





####################
#probes

mu.vec1 = c(0.08, 0.10, 0.10, 0.14)
ro.mat1 = matrix(c(1, 0.2, 0.5, 0.3,
                  0.2, 1, 0.7, 0.4,
                  0.5, 0.7, 1, 0.9,
                  0.3, 0.4, 0.9, 1),
                nrow=4, ncol=4)
stdev.mat1 = matrix(c(0.12, 0, 0, 0,
                     0, 0.12, 0, 0,
                     0, 0, 0.15, 0,
                     0, 0, 0, 0.20),
                   nrow=4, ncol=4)

sigma.mat1 <- stdev.mat1%*%ro.mat1%*%stdev.mat1

ones_vector<-rep(1,4)

A<-as.numeric(t(ones_vector)%*%(solve(sigma.mat1))%*%ones_vector)
B<-as.numeric(t(mu.vec1)%*%(solve(sigma.mat1))%*%ones_vector)
C<-as.numeric(t(mu.vec1)%*%(solve(sigma.mat1))%*%mu.vec1)

m<-0.1
lambda<-(A*m-B)/(A*C-B^2)
gamma<-(C-B*m)/(A*C-B^2)

lambda

gamma

w_optim<-solve(sigma.mat1)%*%(lambda*mu.vec1+gamma*ones_vector)

w_optim




opt<-function(mu0,mu,CovMatrix,minw,maxw,fixed){
  minVarPortfolio<-function(mu0,fixed){
    Constr <- makeConstr(fixed)
    dvec <- rep(0,times=nrow(CovMatrix))
    # Constraints matrix (add the constraint mu=mu0)
    A <- rbind(matrix(mu,nrow=1),Constr$Constr)
    bvec <- rbind(mu0,Constr$bvec)
    xx <- solve.QP(2*CovMatrix,dvec,t(A),meq=2+sum(fixed==0),bvec=bvec)
    return(xx)
  }
  
  makeConstr <- function(fixed){
    # this is the portfolio constraint
    Constr <- matrix(1,ncol=length(mu),nrow=1)
    bvec <- matrix(1,nrow=1)
    # first those that are fixed to w=0, i.e., fixed == 0;
    sel <- (1:length(mu))[fixed==0]
    for (i in sel){
      Constr <- rbind(Constr,0)
      Constr[nrow(Constr),i] <- 1
      bvec <- rbind(bvec,0)
    }
    # then those with weights between minx and maxw, fixed = 1;
    sel <-  (1:length(fixed))[fixed==1]
    for (i in sel){
      # the maximum constraint
      Constr <- rbind(Constr,0)
      Constr[nrow(Constr),i] <- -1
      bvec <- rbind(bvec,-maxw)
      # the minimum constraint
      Constr <- rbind(Constr,0)
      Constr[nrow(Constr),i] <- 1
      bvec <- rbind(bvec,minw)
    }
    # undetermined = free, fixed = -1;
    sel = (1:length(fixed))[fixed==-1]
    for (i in sel){
      # the maximum constraint
      Constr <- rbind(Constr,0)
      Constr[nrow(Constr),i] <- -1
      bvec <- rbind(bvec,-maxw)
      # the minimum constraint
      Constr <- rbind(Constr,0)
      Constr[nrow(Constr),i] <- 1
      bvec <- rbind(bvec,0)
    }
    return(list(Constr=Constr,bvec=bvec))
  }
  sol <- try(minVarPortfolio(mu0,fixed),silent=TRUE)
  if((is(sol))[1]=="try-error"){
    "infeasible"
  } else {
    sol
  }
  return(list(Weights=sol$solution,Objective=sol$value))
}



   
   
   mu_random_portfolio<-rep(0,2001)
   sig_random_portfolio<-rep(0,2001)
   for (i in 1:2001) {
   w<-0
   w_random<-c()
   w_random<-runif(4,-1,1)   #generating a vector containing random numbers between -1 and 1
   w_random<-w_random/sum(w_random) # this is to ensure that the weights sum up to 1 
   mu_random_portfolio[i]<-crossprod(w_random,mu.vec) # this is the return of portfolio i 
   sig_random_portfolio[i]<-sqrt(t(w_random)%*%sigma.mat%*%w_random) # this is the volatility of portfolio i
   plot(sig_random_portfolio, mu_random_portfolio, ylim=c(0, 0.8), xlim=c(0, 0.8),
        pch=16,col="blue",ylab=expression(mu[p]),
        xlab=expression(sigma[p]))
   }

   
   mu0<-seq(0.01,2, by=0.1)
   solution<-c()
   risk_vector<-rep(0,length(mu0))
   returns<-rep(0,length(mu0))
   for (i in 1:length(mu0)) {
   solution<-opt(mu0[i],mu.vec,stdev.vector, ro.mat ,minw,maxw,1)
   risk_vector[i]<-solution$Risk_St_Deviation
   returns[i]<-solution$Return
   plot(returns, risk_vector, ylim=c(0, 2), xlim=c(0, 2),
        pch=20,col="blue",ylab=expression(mu[p]),
        xlab=expression(sigma[p]))
   }
   
   
 #########################
   #Question2
######################### 
   
   
   #case1 rf=50bps
   
   rf<-0.005
   
   ones_vector<-rep(1,4)
   
   A<-as.numeric(t(ones_vector)%*%(solve(sigma.mat))%*%ones_vector)
   
   A
   
   B<-as.numeric(t(mu.vec)%*%(solve(sigma.mat))%*%ones_vector)
   
   B
   
   #case1 rf=50bps
   
   rf<-0.005
   
   w_tangency1<-(solve(sigma.mat)%*%(mu.vec-rf*ones_vector))/(B-A*rf)
   
   w_tangency1
   
   return_portfolio1<-as.numeric(crossprod(mu.vec,w_tangency1))
   
   return_portfolio1
   
   sigma_portfolio1<-as.numeric(sqrt(t(w_tangency1)%*%sigma.mat%*%w_tangency1))
   
   sigma_portfolio1
   
   #case2 rf=100bps
   
   rf<-0.01
   
   w_tangency2<-(solve(sigma.mat)%*%(mu.vec-rf*ones_vector))/(B-A*rf)
   
   w_tangency2
   
   return_portfolio2<-as.numeric(crossprod(mu.vec,w_tangency2))
   
   return_portfolio2
   
   sigma_portfolio2<-as.numeric(sqrt(t(w_tangency2)%*%sigma.mat%*%w_tangency2))
   
   sigma_portfolio2
   
   #case3 rf=150bps
   
   rf<-0.015
   
   w_tangency3<-(solve(sigma.mat)%*%(mu.vec-rf*ones_vector))/(B-A*rf)
   
   w_tangency3
   
   return_portfolio3<-as.numeric(crossprod(mu.vec,w_tangency3))
   
   return_portfolio3
   
   sigma_portfolio3<-as.numeric(sqrt(t(w_tangency3)%*%sigma.mat%*%w_tangency3))
   
   sigma_portfolio3
   
   
   #case3 rf=175bps
   
   
   rf<-0.0175
   
   w_tangency4<-(solve(sigma.mat)%*%(mu.vec-rf*ones_vector))/(B-A*rf)
   
   w_tangency4
   
   return_portfolio4<-as.numeric(crossprod(mu.vec,w_tangency4))
   
   return_portfolio4
   
   sigma_portfolio4<-as.numeric(sqrt(t(w_tangency4)%*%sigma.mat%*%w_tangency4))
   
   sigma_portfolio4
   
   
   
   
   table_results <- cbind(c(return_portfolio1,return_portfolio2,return_portfolio3,return_portfolio4),
                          c(sigma_portfolio1,sigma_portfolio2,sigma_portfolio3,sigma_portfolio4))
   colnames(talbe_results) <- c("Return","Standard Deviation")
   rownames(table_results) <- c("rf=50bps","rf=100bps","rf=150bps", "rf=175bps")
   results <- as.table(table_results)
   results
   
   
   ones_vector<-rep(1,4)
   
   A<-as.numeric(t(ones_vector)%*%(solve(sigma.mat))%*%ones_vector)
   
   A
   
   B<-as.numeric(t(mu.vec)%*%(solve(sigma.mat))%*%ones_vector)
   
   B
   
   rf<-0.005
   
   w_tangency<-(solve(sigma.mat)%*%(mu.vec-rf*ones_vector))/(B-A*rf)
   
   
   Sweave("E://cqf/exam1/2.Rnw")