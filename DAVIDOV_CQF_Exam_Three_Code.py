# -*- coding: utf-8 -*-
"""
DAVIDOV

CQF Exam 3 - codes 
"""

"""
Separate sections A1-3, B1-3 can be run, though 
as a first run, execute the first 250 lines of this code
to load data, features, function etc.
"""
###############################
#Import packages and load data

import pandas as pd
import numpy as np
from pylab import plt
plt.style.use('seaborn')
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")


from scipy import stats
import statsmodels.api as sm

import statistics as st

import yfinance as yf 

EBS_data = yf.download("EBS.VI", start="2014-01-01", end="2020-02-28")

RBI_data = yf.download("RBI.VI", start="2014-01-01", end="2020-02-28")


EBS = EBS_data['Adj Close']
RBI = RBI_data['Adj Close']

plt.plot(EBS, 'r');
plt.plot(RBI, 'b');
plt.title("Erste vs Raiffeisen",fontweight="bold")
plt.show()

np.corrcoef(EBS,RBI)

#altough from same industry and similar size not always very correlated
plt.plot(EBS[300:600], 'r');
plt.plot(RBI[300:600], 'b');
plt.title("Erste vs Raiffeisen",fontweight="bold")
plt.show()

np.corrcoef(EBS[300:600],RBI[300:600])


fig, axes = plt.subplots(nrows=1, ncols=2)
sm.qqplot(EBS,fit= True, line='45')
plt.title("Erste Bank",fontweight="bold")

sm.qqplot(RBI,fit= True, line='45')
plt.title("Raiffeisen Bank",fontweight="bold")

sm.qqplot(EBS[300:600],fit= True)
###########################################
# Compute lagged log returns

def getLaggedReturns(prices,lags = 5, window_length = 1):
    #df: pandas Dataframe type variable, and column 'Close' should be included
    #lags: how many lagged returns do you want
    #window_length: the window length we use to compute each return
    laggedReturns =pd.DataFrame(index = prices.index)
    laggedReturns['Adj Close'] = prices
    laggedReturns['ret_0'] = np.log(laggedReturns['Adj Close']/laggedReturns['Adj Close'].shift(window_length))
    
        
    for lag in range( 1,lags + 1 ):
        col = 'ret_%d'%lag
        laggedReturns[col] = laggedReturns['ret_0'].shift(lag)
    laggedReturns.dropna(inplace = True)
    
    return laggedReturns

EBS_laggedRet=getLaggedReturns(EBS)
RBI_laggedRet=getLaggedReturns(RBI)

EBS_laggedRet.tail(3)

fig, axes = plt.subplots(nrows=1, ncols=2)
sm.qqplot(EBS_laggedRet['ret_0'],fit= True, line='45')
plt.title("Erste Bank",fontweight="bold")

sm.qqplot(RBI_laggedRet['ret_0'],fit= True, line='45')
plt.title("Raiffeisen Bank",fontweight="bold")


#############################################
#Set of Features to predict the prices

#starting point always lagged returns
EBS_features=getLaggedReturns(EBS)
RBI_features=getLaggedReturns(RBI)
 
#shows whether return is negative or positive       
cols =['ret_0','ret_1','ret_2','ret_3','ret_4','ret_5']
for idx,col in enumerate(cols):
        EBS_features['Sign'+str(idx)] = np.sign(EBS_features[col])

EBS_features.head()

#computes momentum for different time intervals
def GetMomentum(targetDF,sourceDF,time_intervals=[5,7,13,21]):
    for time_interval in time_intervals:
        momentum = (sourceDF.shift(time_interval)['Adj Close']-sourceDF['Adj Close'])/sourceDF['Adj Close']
        targetDF = pd.DataFrame(targetDF.iloc[time_interval:,:])
        targetDF['MOM'+str(time_interval)]=momentum
    
    targetDF = targetDF.dropna()
    return targetDF


EBS_features=GetMomentum(EBS_features,EBS_features)
RBI_features=GetMomentum(RBI_features,RBI_features)


#Compute EWMA for 5,7,13,21 days and make crossover strategies
EBS_features['EWMA5'] = pd.Series(EBS_features['Adj Close']).ewm(span = 5).mean()
EBS_features['EWMA7'] = pd.Series(EBS_features['Adj Close']).ewm(span = 7).mean()
EBS_features['EWMA13'] = pd.Series(EBS_features['Adj Close']).ewm(span = 13).mean()
EBS_features['EWMA21'] = pd.Series(EBS_features['Adj Close']).ewm(span = 21).mean()
EBS_features['EWMA5-7'] = EBS_features['EWMA5'] - EBS_features['EWMA7']
EBS_features['EWMA7-13'] = EBS_features['EWMA7'] - EBS_features['EWMA13']
EBS_features['EWMA13-21'] = EBS_features['EWMA13'] - EBS_features['EWMA21']

RBI_features['EWMA5'] = pd.Series(RBI_features['Adj Close']).ewm(span = 5).mean()
RBI_features['EWMA7'] = pd.Series(RBI_features['Adj Close']).ewm(span = 7).mean()
RBI_features['EWMA13'] = pd.Series(RBI_features['Adj Close']).ewm(span = 13).mean()
RBI_features['EWMA21'] = pd.Series(RBI_features['Adj Close']).ewm(span = 21).mean()
RBI_features['EWMA5-7'] = RBI_features['EWMA5'] - RBI_features['EWMA7']
RBI_features['EWMA7-13'] = RBI_features['EWMA7'] - RBI_features['EWMA13']
RBI_features['EWMA13-21'] = RBI_features['EWMA13'] - RBI_features['EWMA21']

#Compute Simple MA for 5,7,13,21 days and make crossover strategies
EBS_features['MA5'] = pd.Series(EBS_features['Adj Close']).rolling(5).mean()
EBS_features['MA7'] = pd.Series(EBS_features['Adj Close']).rolling(7).mean()
EBS_features['MA13'] = pd.Series(EBS_features['Adj Close']).rolling(13).mean()
EBS_features['MA21'] = pd.Series(EBS_features['Adj Close']).rolling(21).mean()
EBS_features['MA5-7'] = EBS_features['MA5'] - EBS_features['MA7']
EBS_features['MA7-13'] = EBS_features['MA7'] - EBS_features['MA13']
EBS_features['MA13-21'] = EBS_features['MA13'] - EBS_features['MA21']

RBI_features['MA5'] = pd.Series(RBI_features['Adj Close']).rolling(5).mean()
RBI_features['MA7'] = pd.Series(RBI_features['Adj Close']).rolling(7).mean()
RBI_features['MA13'] = pd.Series(RBI_features['Adj Close']).rolling(13).mean()
RBI_features['MA21'] = pd.Series(RBI_features['Adj Close']).rolling(21).mean()
RBI_features['MA5-7'] = RBI_features['MA5'] - RBI_features['MA7']
RBI_features['MA7-13'] = RBI_features['MA7'] - RBI_features['MA13']
RBI_features['MA13-21'] = RBI_features['MA13'] - RBI_features['MA21']

#standard deviations on 21-days rolling windows
EBS_features['stdev21'] = pd.Series(EBS_features['Adj Close']).rolling(21).std()
RBI_features['stdev21'] = pd.Series(RBI_features['Adj Close']).rolling(21).std()


#set a column to indicate up or down move 
#from "ret_0" where
#returns >= 0 will be assigned 1, else 0.

EBS_features['I'] = EBS_features['ret_0']
EBS_features['I'][EBS_features['I']>0] = 1
EBS_features['I'][EBS_features['I']<=0] = -1

RBI_features['I'] = RBI_features['ret_0']
RBI_features['I'][RBI_features['I']>0] = 1
RBI_features['I'][RBI_features['I']<=0] = -1

EBS_features['Sign']= np.sign(EBS_features['ret_0'])
EBS_features['Sign'][EBS_features['Sign']<0] =0

RBI_features['Sign']= np.sign(RBI_features['ret_0'])
RBI_features['Sign'][RBI_features['Sign']<0] =0

#let's see all features created 
EBS_features.columns

EBS_features=EBS_features.dropna()
RBI_features=RBI_features.dropna()

##################################
# Decide which features 

#start with all
cols=['ret_1', 'ret_2', 'ret_3', 'ret_4', 'ret_5',
      'Sign1', 'Sign2', 'Sign3', 'Sign4', 'Sign5', 'MOM5', 'MOM7',
       'MOM13', 'MOM21', 'EWMA5', 'EWMA7', 'EWMA13', 'EWMA21', 'EWMA5-7',
       'EWMA7-13', 'EWMA13-21', 'MA5', 'MA7', 'MA13', 'MA21', 'MA5-7',
       'MA7-13', 'MA13-21', 'stdev21']

#Accuracy with logit regression: ~ 0.97
#overfitted model

#python warns of multicollinearity
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

EBS_features_reg = EBS_features
model = sm.OLS(EBS_features_reg['ret_0'], EBS_features[cols]).fit()

print(model.summary())

#Multicolinearity emerges when three or more variables,
#which are highly correlated, are included within a model.

# remove the ret_1 as the information is contained in Sign
# and execute lines 231-250
cols=['Sign1', 'Sign2', 'Sign3', 'Sign4', 'Sign5', 'MOM5', 'MOM7',
       'MOM13', 'MOM21', 'EWMA5', 'EWMA7', 'EWMA13', 'EWMA21', 'EWMA5-7',
       'EWMA7-13', 'EWMA13-21', 'MA5', 'MA7', 'MA13', 'MA21', 'MA5-7',
       'MA7-13', 'MA13-21', 'stdev21']

cols=[ 'MOM5', 'MOM7',
       'MOM13', 'MOM21', 'EWMA5', 'EWMA7', 'EWMA13', 'EWMA21', 'EWMA5-7',
       'EWMA7-13', 'EWMA13-21', 'MA5', 'MA7', 'MA13', 'MA21', 'MA5-7',
       'MA7-13', 'MA13-21', 'stdev21']

# Leave one from strongest indicators types given the coefficients
cols=['Sign1', 'MOM5', 'EWMA5-7','EWMA7-13', 'MA7']
#Accuracy with logit regression: 0.63
#Accuracy values can vary in new run due to the shuffle=TRUE parameter

cols=['MOM5', 'EWMA5-7','EWMA7-13', 'MA7']
#Accuracy with logit regression: 0.61

cols=['MOM5', 'MOM7', 'EWMA5-7','EWMA7-13']
#Accuracy with logit regression: 0.63

#dropping certain feauture can sometimes also increase accuracy
cols=['MOM5', 'EWMA5-7','EWMA7-13']
#Accuracy with logit regression: 0.65

#this is our optimal choice
cols=['MOM5','EWMA5-7','ret_1']
#Accuracy with logit regression: 0.68

#let's check the linear regression summary
lm = sm.OLS(EBS_features_reg['ret_0'], EBS_features[cols]).fit()
print(lm.summary())
# not a strong result but we will run more sophisticated model
# to improve that

#at least we are sure there is no multicollinearity
variables = lm.model.exog
vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif 


from sklearn import model_selection
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
#EBS_features = StandardScaler().fit_transform(EBS_features)

EBS_features=EBS_features.dropna()
X_Train_EBS, X_Test_EBS, Y_Train_EBS, Y_Test_EBS = model_selection.train_test_split(EBS_features[cols],
                                                                                    EBS_features['I'] ,
                                                                                    test_size=0.25, shuffle=True)
from sklearn import linear_model
logit_EBS = linear_model.LogisticRegression(C = 1e15,solver ='liblinear',penalty = 'l2')
logit_EBS.fit(X_Train_EBS,Y_Train_EBS)
Y_EBS_pred = logit_EBS.predict(X_Test_EBS)


print("Accuracy with logit regression:",metrics.accuracy_score( Y_Test_EBS, Y_EBS_pred))

EBS_coef = pd.Series(logit_EBS.coef_.ravel(),index = cols)
print("EBS Regression Model Coefficient Analysis")
EBS_coef.sort_values()

# #KNN to confirm
# #k=5, number of neighbours                                                          
# knn5 = KNeighborsClassifier(n_neighbors=5)        

# knn5.fit(X_Train_EBS,Y_Train_EBS)
# Y_EBS_P = knn5.predict(X_Test_EBS)

# print("KNN Accuracy with k=5:",metrics.accuracy_score( Y_Test_EBS, Y_EBS_P))


#######################################################
#######################################################
# A1 a) logistic regression
#######################################################
#######################################################


#import sklearn module for logistic regression
from sklearn import linear_model
#Create a list to store independent variables' keys
#Please remember that this list does not include 'ret_0' because our target is to predict 'ret_0' 
cols = ['ret_1']
cols = ['ret_1','ret_2', 'ret_3', 'ret_4', 'ret_5']


#LogisticRegression Type Object of Sklearn are created
lm_EBS = linear_model.LogisticRegression(C = 1e6,solver='liblinear',multi_class='ovr',penalty='l2')
lm_RBI = linear_model.LogisticRegression(C = 1e6,solver='liblinear',multi_class='ovr',penalty='l2')

EBS_features['Sign']= np.sign(EBS_features['ret_0'])
EBS_features['Sign'][EBS_features['Sign']<0] =0

RBI_features['Sign']= np.sign(RBI_features['ret_0'])
RBI_features['Sign'][RBI_features['Sign']<0] =0

###Fit our data 
lm_EBS.fit(EBS_features[cols],EBS_features['Sign'])
lm_RBI.fit(RBI_features[cols], RBI_features['Sign'])

###Now we using our model to do prediction
EBS_features['Logit_Predict'] = lm_EBS.predict(EBS_features[cols])
RBI_features['Logit_Predict'] = lm_RBI.predict(RBI_features[cols])

EBS_features.head(5)

EBS_features['Logit_Predict'][EBS_features['Logit_Predict']==0] = -1
RBI_features['Logit_Predict'][RBI_features['Logit_Predict']==0] = -1

EBS_features['Logit_Returns'] = EBS_features['Logit_Predict'] * EBS_features['ret_0']
RBI_features['Logit_Returns'] = RBI_features['Logit_Predict'] * RBI_features['ret_0']

EBS_features.head(5)

#Compute the performance of the model
EBS_features[['ret_0', 'Logit_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Erste Bank",fontweight="bold")
RBI_features[['ret_0', 'Logit_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Raiffeisen Bank",fontweight="bold")
#Interesting observation - the model seems to match better the RBI stock

from sklearn.model_selection import cross_val_score
crossval_EBS = cross_val_score(lm_EBS,EBS_features[cols],EBS_features['Sign'],scoring = 'accuracy')
print("Accuracy For EBS Logit Model : %.4f"%crossval_EBS.mean())
crossval_RBI = cross_val_score(lm_RBI,RBI_features[cols],RBI_features['Sign'],scoring = 'accuracy')
print("Accuracy For RBI Logit Model : %.4f"%crossval_RBI.mean())


#let's check the coefficients
print('Erste Bank - Logit Regression Coefficients: ')
print(lm_EBS.coef_.ravel())
print('Raiffeisen Bank - Logit Regression Coefficients : ')
print(lm_RBI.coef_.ravel())

#here we can see that the stocks are to some extent correlated 

########################################
#Same regression but using L1 penalty
cols = ['ret_1','ret_2', 'ret_3', 'ret_4', 'ret_5']
lm_EBS = linear_model.LogisticRegression(C = 1e6,solver='liblinear',multi_class='ovr',penalty='l1')
lm_RBI = linear_model.LogisticRegression(C = 1e6,solver='liblinear',multi_class='ovr',penalty='l1')

#create column "Sign" from "ret_0" where
#returns >= 0 will be assigned 1, else 0.

EBS_features['Sign']= np.sign(EBS_features['ret_0'])
EBS_features['Sign'][EBS_features['Sign']<0] =0

RBI_features['Sign']= np.sign(RBI_features['ret_0'])
RBI_features['Sign'][RBI_features['Sign']<0] =0

EBS_features.tail(3)


###Fit our data using Lagged ret_1, ret_2, ret_3, ret_4, ret_5 with Output Sign +1 or -1
lm_EBS.fit(EBS_features[cols],EBS_features['Sign'])
lm_RBI.fit(RBI_features[cols], RBI_features['Sign'])

###Now we using our model to do prediction
EBS_features['Logit_Predict'] = lm_EBS.predict(EBS_features[cols])
RBI_features['Logit_Predict'] = lm_RBI.predict(RBI_features[cols])

EBS_features.head(5)

EBS_features['Logit_Predict'][EBS_features['Logit_Predict']==0] = -1
RBI_features['Logit_Predict'][RBI_features['Logit_Predict']==0] = -1

EBS_features['Logit_Returns'] = EBS_features['Logit_Predict'] * EBS_features['ret_0']
RBI_features['Logit_Returns'] = RBI_features['Logit_Predict'] * RBI_features['ret_0']

EBS_features.head(5)

#Compute the performance of the model
EBS_features[['ret_0', 'Logit_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Erste Bank",fontweight="bold")
RBI_features[['ret_0', 'Logit_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Raiffeisen Bank",fontweight="bold")


#visually not much of difference, lets make more sophisticated comparison
#let's check the coefficients
print('Erste Bank - Logit Regression Coefficients: ')
print(lm_EBS.coef_.ravel())
print('Raiffeisen Bank - Logit Regression Coefficients : ')
print(lm_RBI.coef_.ravel())

#In the same regression if we put C=0.1 the L1 will give only 0 coeff.

###########
#Comparison
#Different parameters C, L1,L2 penalties
cols=['ret_1', 'ret_2', 'ret_3', 'ret_4', 'ret_5',
      'Sign1', 'MOM5', 'MOM7',
       'MOM13', 'MOM21', 'EWMA5', 'EWMA7', 'EWMA13', 'EWMA21', 'EWMA5-7',
       'EWMA7-13', 'EWMA13-21', 'MA5', 'MA7', 'MA13', 'MA21', 'MA5-7',
       'MA7-13', 'MA13-21', 'stdev21']


fig, axes = plt.subplots(3, 2)

# Set regularization parameter
# C - smaller values specify stronger regularization.
for i, (C, axes_row) in enumerate(zip((1, 0.1, 0.01), axes)):
    # turn down tolerance for short training time
    clf_l1_LR = linear_model.LogisticRegression(C=C, penalty='l1', tol=0.01, solver='liblinear')
    clf_l2_LR = linear_model.LogisticRegression(C=C, penalty='l2', tol=0.01, solver='liblinear')
    
    # EBS_features=getLaggedReturns(EBS,lags=24)
     
    EBS_features['Sign']= np.sign(EBS_features['ret_0'])
    EBS_features['Sign'][EBS_features['Sign']<0] =0
    
    clf_l1_LR.fit(EBS_features[cols],EBS_features['Sign'])
    clf_l2_LR.fit(EBS_features[cols],EBS_features['Sign'])
    
    coef_l1_LR = clf_l1_LR.coef_.ravel()
    coef_l2_LR = clf_l2_LR.coef_.ravel()

    # coef_l1_LR contains zeros due to the
    # L1 sparsity inducing norm

    sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
    sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100

    print("C=%.2f" % C)
    print("{:<40} {:.2f}%".format("Sparsity with L1 penalty:", sparsity_l1_LR))
    print("{:<40} {:.2f}%".format("Sparsity with L2 penalty:", sparsity_l2_LR))
    print("{:<40} {:.2f}".format("Score with L1 penalty:",
                                 clf_l1_LR.score(EBS_features[cols],EBS_features['Sign'])))
    print("{:<40} {:.2f}".format("Score with L2 penalty:",
                                 clf_l2_LR.score(EBS_features[cols],EBS_features['Sign'])))

    if i == 0:
        axes_row[0].set_title("L1 penalty")
        axes_row[1].set_title("L2 penalty")

    for ax, coefs in zip(axes_row, [coef_l1_LR, coef_l2_LR]):
        ax.imshow(np.abs(coefs.reshape(5, 5)), interpolation='nearest',
                  cmap='binary', vmax=1, vmin=0)
        ax.set_xticks(())
        ax.set_yticks(())

    axes_row[0].set_ylabel('C = %s' % C)

plt.show()

#C sets the freedom of the model. 
#Higher values of C give more leeway to the model 
#whereas, smaller values constrain it
#In the L1 penalty case, this leads to sparser solutions (percentage of zero coeff. )

#############################################
#Let's use more sophisticated model
cols=['MOM5', 'EWMA5-7','EWMA7-13']

lm_EBS = linear_model.LogisticRegression(C = 1e6,solver='liblinear',multi_class='ovr',penalty='l2')
lm_RBI = linear_model.LogisticRegression(C = 1e6,solver='liblinear',multi_class='ovr',penalty='l2')

EBS_features['Sign']= np.sign(EBS_features['ret_0'])
EBS_features['Sign'][EBS_features['Sign']<0] =0

RBI_features['Sign']= np.sign(RBI_features['ret_0'])
RBI_features['Sign'][RBI_features['Sign']<0] =0

###Fit our data 
lm_EBS.fit(EBS_features[cols],EBS_features['Sign'])
lm_RBI.fit(RBI_features[cols], RBI_features['Sign'])

###Now we using our model to do prediction
EBS_features['Logit_Predict'] = lm_EBS.predict(EBS_features[cols])
RBI_features['Logit_Predict'] = lm_RBI.predict(RBI_features[cols])

EBS_features.head(5)

EBS_features['Logit_Predict'][EBS_features['Logit_Predict']==0] = -1
RBI_features['Logit_Predict'][RBI_features['Logit_Predict']==0] = -1

EBS_features['Logit_Returns'] = EBS_features['Logit_Predict'] * EBS_features['ret_0']
RBI_features['Logit_Returns'] = RBI_features['Logit_Predict'] * RBI_features['ret_0']

EBS_features.head(5)

#Compute the performance of the model
EBS_features[['ret_0', 'Logit_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
RBI_features[['ret_0', 'Logit_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));

#Amazing. This seems like perfect match


#let's check the coefficients
print('Erste Bank - Logit Regression Coefficients: ')
print(lm_EBS.coef_.ravel())
print('Raiffeisen Bank - Logit Regression Coefficients : ')
print(lm_RBI.coef_.ravel())

#the accuracy would go to 1
#however, more important is out-of-sample performance
#i.e. after model is trained how would it work in real trading  
# we will therefore split data to train and test

############################################
#Split data into training set and validation
#using 75% of the observations to train and 25% to test 

#the timeseries for both stocks have same length
cols=['MOM5', 'EWMA5-7','EWMA7-13']
training_split = int(0.75*EBS.shape[0])

EBS_training = EBS_features.iloc[0:training_split,:]
RBI_training = RBI_features.iloc[0:training_split,:]

EBS_validate = EBS_features.iloc[training_split:,:]
RBI_validate = RBI_features.iloc[training_split:,:]

#fit again
lm_EBS.fit(EBS_training[cols],EBS_training['Sign'])
lm_RBI.fit(RBI_training[cols],RBI_training['Sign'])

#predict using the validation set of data
EBS_validate['Logit_Predict'] = lm_EBS.predict(EBS_validate[cols])
RBI_validate['Logit_Predict'] = lm_RBI.predict(RBI_validate[cols])


EBS_validate['Logit_Predict'][EBS_validate['Logit_Predict']==0] = -1
RBI_validate['Logit_Predict'][RBI_validate['Logit_Predict']==0] = -1

EBS_validate['Logit_Returns'] =EBS_validate['ret_0']*EBS_validate['Logit_Predict']
RBI_validate['Logit_Returns'] =RBI_validate['ret_0']*RBI_validate['Logit_Predict']

EBS_validate.tail(3)

EBS_validate[['ret_0', 'Logit_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Erste Bank - Logit Reg - Test -Refined Model",fontweight="bold")
RBI_validate[['ret_0', 'Logit_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Raiffeisen - Logit Reg - Test -Refined Model",fontweight="bold")


#Conclusion: We can confirm that the model works well in out-of-sample data set


#let's see our old model in out-of-sample
cols = ['ret_1','ret_2', 'ret_3', 'ret_4', 'ret_5']
training_split = int(0.75*EBS.shape[0])

EBS_training = EBS_features.iloc[0:training_split,:]
RBI_training = RBI_features.iloc[0:training_split,:]

EBS_validate = EBS_features.iloc[training_split:,:]
RBI_validate = RBI_features.iloc[training_split:,:]

#fit again
lm_EBS.fit(EBS_training[cols],EBS_training['Sign'])
lm_RBI.fit(RBI_training[cols],RBI_training['Sign'])

#predict using the validation set of data
EBS_validate['Logit_Predict'] = lm_EBS.predict(EBS_validate[cols])
RBI_validate['Logit_Predict'] = lm_RBI.predict(RBI_validate[cols])


EBS_validate['Logit_Predict'][EBS_validate['Logit_Predict']==0] = -1
RBI_validate['Logit_Predict'][RBI_validate['Logit_Predict']==0] = -1

EBS_validate['Logit_Returns'] =EBS_validate['ret_0']*EBS_validate['Logit_Predict']
RBI_validate['Logit_Returns'] =RBI_validate['ret_0']*RBI_validate['Logit_Predict']

EBS_validate.tail(3)

EBS_validate[['ret_0', 'Logit_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Erste Bank - Logit Reg - Test",fontweight="bold")
RBI_validate[['ret_0', 'Logit_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Raiffeisen Bank - Logit Reg - Test",fontweight="bold")


#Conclusion: good in-sample fit doesn't guarantee good out-of-sample fit 



# Naive Bayes Classifier using GaussianNB
from sklearn.naive_bayes import GaussianNB

gaussianBayes_EBS = GaussianNB()
gaussianBayes_RBI = GaussianNB()

gaussianBayes_EBS.fit(EBS_training[cols],EBS_training['Sign'])
gaussianBayes_RBI.fit(RBI_training[cols],RBI_training['Sign'])

#predict using the validation set of data
EBS_validate['GaussBayes_Predict'] = gaussianBayes_EBS.predict(EBS_validate[cols])
RBI_validate['GaussBayes_Predict'] = gaussianBayes_RBI.predict(RBI_validate[cols])


EBS_validate['GaussBayes_Predict'][EBS_validate['GaussBayes_Predict']==0] = -1
RBI_validate['GaussBayes_Predict'][RBI_validate['GaussBayes_Predict']==0] = -1

EBS_validate['GaussBayes_Returns'] =EBS_validate['ret_0']*EBS_validate['GaussBayes_Predict']
RBI_validate['GaussBayes_Returns'] =RBI_validate['ret_0']*RBI_validate['GaussBayes_Predict']
                                   

EBS_validate[['ret_0', 'GaussBayes_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Erste Bank - GaussBayes - Test",fontweight="bold")
RBI_validate[['ret_0', 'GaussBayes_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Raiffeisen Bank - GaussBayes - Test",fontweight="bold")
# this is not better than the logit regression


# Naive Bayes Classifier using BernoulliNB
from sklearn.naive_bayes import BernoulliNB

bernulliBayes_EBS = BernoulliNB()
bernulliBayes_RBI = BernoulliNB()

#make a backup on features data
EBS_features_Binary=EBS_features
RBI_features_Binary=RBI_features

#make the relevant factors in binary form
for factor in cols:
       EBS_features_Binary[factor][EBS_features_Binary[factor]>0] = 1
       EBS_features_Binary[factor][EBS_features_Binary[factor]<=0] = 0
       RBI_features_Binary[factor][RBI_features_Binary[factor]>0] = 1
       RBI_features_Binary[factor][RBI_features_Binary[factor]<=0] = 0
        
training_split = int(0.75*EBS.shape[0])

EBS_training = EBS_features_Binary.iloc[0:training_split,:]
RBI_training = RBI_features_Binary.iloc[0:training_split,:]

EBS_validate = EBS_features_Binary.iloc[training_split:,:]
RBI_validate = RBI_features_Binary.iloc[training_split:,:]        

bernulliBayes_EBS.fit(EBS_training[cols],EBS_training['Sign'])
bernulliBayes_RBI.fit(RBI_training[cols],RBI_training['Sign'])

#predict using the validation set of data
EBS_validate['BernulliBayes_Predict'] = bernulliBayes_EBS.predict(EBS_validate[cols])
RBI_validate['BernulliBayes_Predict'] = bernulliBayes_RBI.predict(RBI_validate[cols])


EBS_validate['BernulliBayes_Predict'][EBS_validate['BernulliBayes_Predict']==0] = -1
RBI_validate['BernulliBayes_Predict'][RBI_validate['BernulliBayes_Predict']==0] = -1

EBS_validate['BernulliBayes_Returns'] =EBS_validate['ret_0']*EBS_validate['BernulliBayes_Predict']
RBI_validate['BernulliBayes_Returns'] =RBI_validate['ret_0']*RBI_validate['BernulliBayes_Predict']
                                   

EBS_validate[['ret_0', 'BernulliBayes_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Erste Bank - BernulliBayes - Test",fontweight="bold")
RBI_validate[['ret_0', 'BernulliBayes_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Raiffeisen Bank - BernulliBayes - Test",fontweight="bold")
# this is better than Gaussian but not the logit regression

#######################################################
#######################################################
# A1 b) Model Validation using K-fold crossvalidation
#######################################################
#######################################################

cols = ['ret_1','ret_2', 'ret_3', 'ret_4', 'ret_5']

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#We first create a KFold object, and we define the number of sections to be 5
#Here shuffle= True means we will shuffle our data, or in other words, randomly sort it, before our split 
kfold = KFold(n_splits = 5,random_state=7,shuffle = True)
print(kfold)

###Now we get the number of splitting iterations in the cross-validator
kfold.get_n_splits(EBS_features[cols])
kfold.get_n_splits(RBI_features[cols])

#to cross-validate our EBS logit model
logit_EBS = linear_model.LogisticRegression(C=1e5,solver ='liblinear',multi_class='auto',penalty='l2')
logit_EBS.fit(EBS_features[cols],EBS_features['Sign'])
crossval_EBS = cross_val_score(logit_EBS,EBS_features[cols],EBS_features['Sign'],cv=kfold,scoring = 'accuracy')
print("Accuracy For EBS Logit Model with 5-Fold Cross Validation: %.4f"%crossval_EBS.mean())

#to cross-validate our RBI logit model
logit_RBI = linear_model.LogisticRegression(C=1e5,solver ='liblinear',multi_class='auto',penalty='l2')
logit_RBI.fit(RBI_features[cols],RBI_features['Sign'])
crossval_RBI = cross_val_score(logit_RBI,RBI_features[cols],RBI_features['Sign'],cv=kfold,scoring = 'accuracy')
print("Accuracy For RBI Logit Model with 5-Fold Cross Validation: %.4f"%crossval_RBI.mean())

#the model seems to gets improved with 5-Fold Cross Validation
#further investigation can be looked at with different 
#number of splits and changing the random section

#######################################################
#######################################################
# A2 a) Support Vector Machines - soft vs. hard margins
#######################################################
#######################################################


#the margins refer to the tolerance wrt to outliers misclassifictaion
#more flexible model (lower C) will tolerate outliers misclass., this we speak of soft margin
#the higher the C the harder the margin of the model. For the math notation see report


#######################################################
#######################################################
# A2 b) Support Vector Machines-using Momentum and Ret1
#######################################################
#######################################################

from sklearn.svm import LinearSVC,SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn import model_selection

#########################################
# SVM Using only return with soft margins
# linear kernel

cols = [ 'ret_1' ]
training_split = int(0.75*EBS.shape[0])
EBS_training = EBS_features.iloc[0:training_split,:]
RBI_training = RBI_features.iloc[0:training_split,:]

EBS_validate = EBS_features.iloc[training_split:,:]
RBI_validate = RBI_features.iloc[training_split:,:]

SVM_EBS =SVC(C=1e5,probability=True)
SVM_RBI =SVC(C=1e5,probability=True)
#fit again
SVM_EBS.fit(EBS_training[cols],EBS_training['Sign'])
SVM_RBI.fit(RBI_training[cols],RBI_training['Sign'])

#predict using the validation set of data
EBS_validate['SVM_Predict'] = SVM_EBS.predict(EBS_validate[cols])
RBI_validate['SVM_Predict'] = SVM_RBI.predict(RBI_validate[cols])


EBS_validate['SVM_Predict'][EBS_validate['SVM_Predict']==0] = -1
RBI_validate['SVM_Predict'][RBI_validate['SVM_Predict']==0] = -1

EBS_validate['SVM_Returns'] =EBS_validate['ret_0']*EBS_validate['SVM_Predict']
RBI_validate['SVM_Returns'] =RBI_validate['ret_0']*RBI_validate['SVM_Predict']

EBS_validate.tail(3)

EBS_validate[['ret_0', 'SVM_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Raiffeisen Bank",fontweight="bold")
RBI_validate[['ret_0', 'SVM_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Raiffeisen Bank",fontweight="bold")


#########################################
# SVM comparing Ret1 and Momentum with soft margins

cols1 = [ 'ret_1' ]
cols2 = [ 'MOM5' ]
training_split = int(0.75*EBS.shape[0])
EBS_training = EBS_features.iloc[0:training_split,:]
RBI_training = RBI_features.iloc[0:training_split,:]

EBS_validate = EBS_features.iloc[training_split:,:]
RBI_validate = RBI_features.iloc[training_split:,:]

SVM_EBS =SVC(C=1e5,probability=True, kernel="linear")
SVM_RBI =SVC(C=1e5,probability=True, kernel="linear")
#fit again
SVM_EBS.fit(EBS_training[cols1],EBS_training['I'])
SVM_RBI.fit(RBI_training[cols1],RBI_training['I'])

SVM_EBS.fit(EBS_training[cols2],EBS_training['I'])
SVM_RBI.fit(RBI_training[cols2],RBI_training['I'])

#predict using the validation set of data
EBS_validate['SVM_Pred_Ret1'] = SVM_EBS.predict(EBS_validate[cols1])
RBI_validate['SVM_Pred_Ret1'] = SVM_RBI.predict(RBI_validate[cols1])

EBS_validate['SVM_Pred_Mom5'] = SVM_EBS.predict(EBS_validate[cols2])
RBI_validate['SVM_Pred_Mom5'] = SVM_RBI.predict(RBI_validate[cols2])


EBS_validate['SVM_Pred_Ret1'][EBS_validate['SVM_Pred_Ret1']==0] = -1
RBI_validate['SVM_Pred_Ret1'][RBI_validate['SVM_Pred_Ret1']==0] = -1


EBS_validate['SVM_Pred_Mom5'][EBS_validate['SVM_Pred_Mom5']==0] = -1
RBI_validate['SVM_Pred_Mom5'][RBI_validate['SVM_Pred_Mom5']==0] = -1


EBS_validate['SVM_Returns_Ret1'] =EBS_validate['ret_0']*EBS_validate['SVM_Pred_Ret1']
RBI_validate['SVM_Returns_Ret1'] =RBI_validate['ret_0']*RBI_validate['SVM_Pred_Ret1']

EBS_validate['SVM_Returns_Mom5'] =EBS_validate['ret_0']*EBS_validate['SVM_Pred_Mom5']
RBI_validate['SVM_Returns_Mom5'] =RBI_validate['ret_0']*RBI_validate['SVM_Pred_Mom5']


#compare the two models Ret1 vs MOM5
EBS_validate[['ret_0', 'SVM_Returns_Ret1', 'SVM_Returns_Mom5']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Erste Bank",fontweight="bold")
RBI_validate[['ret_0', 'SVM_Returns_Ret1', 'SVM_Returns_Mom5']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Raiffeisen Bank",fontweight="bold")
plt.show()

#Now include in the comparison also the old logit model
EBS_validate[['ret_0', 'SVM_Returns_Ret1', 'SVM_Returns_Mom5','Logit_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Erste Bank",fontweight="bold")
RBI_validate[['ret_0', 'SVM_Returns_Ret1', 'SVM_Returns_Mom5','Logit_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Raiffeisen Bank",fontweight="bold")
plt.show()


crossval_EBS = cross_val_score(SVM_EBS,EBS_features[cols],EBS_features['Sign'],scoring = 'accuracy')
print("Accuracy For EBS SVM Soft Margins Model : %.4f"%crossval_EBS.mean())
crossval_RBI = cross_val_score(SVM_RBI,RBI_features[cols],RBI_features['Sign'],scoring = 'accuracy')
print("Accuracy For RBI SVM Soft Margins Model : %.4f"%crossval_RBI.mean())




#########################################
# SVM comparing Ret1 and Momentum with hard margins

cols1 = [ 'ret_1' ]
cols2 = [ 'MOM5' ]
training_split = int(0.75*EBS.shape[0])
EBS_training = EBS_features.iloc[0:training_split,:]
RBI_training = RBI_features.iloc[0:training_split,:]

EBS_validate = EBS_features.iloc[training_split:,:]
RBI_validate = RBI_features.iloc[training_split:,:]

SVM_EBS_hard =SVC(C=1,probability=True, kernel="linear")
SVM_RBI_hard =SVC(C=1,probability=True, kernel="linear")
#fit again
SVM_EBS_hard.fit(EBS_training[cols1],EBS_training['I'])
SVM_RBI_hard.fit(RBI_training[cols1],RBI_training['I'])

SVM_EBS_hard.fit(EBS_training[cols2],EBS_training['I'])
SVM_RBI_hard.fit(RBI_training[cols2],RBI_training['I'])

#predict using the validation set of data
EBS_validate['SVM_Pred_Ret1'] = SVM_EBS_hard.predict(EBS_validate[cols1])
RBI_validate['SVM_Pred_Ret1'] = SVM_RBI_hard.predict(RBI_validate[cols1])

EBS_validate['SVM_Pred_Mom5'] = SVM_EBS_hard.predict(EBS_validate[cols2])
RBI_validate['SVM_Pred_Mom5'] = SVM_RBI_hard.predict(RBI_validate[cols2])


EBS_validate['SVM_Pred_Ret1'][EBS_validate['SVM_Pred_Ret1']==0] = -1
RBI_validate['SVM_Pred_Ret1'][RBI_validate['SVM_Pred_Ret1']==0] = -1


EBS_validate['SVM_Pred_Mom5'][EBS_validate['SVM_Pred_Mom5']==0] = -1
RBI_validate['SVM_Pred_Mom5'][RBI_validate['SVM_Pred_Mom5']==0] = -1


EBS_validate['SVM_Returns_Ret1'] =EBS_validate['ret_0']*EBS_validate['SVM_Pred_Ret1']
RBI_validate['SVM_Returns_Ret1'] =RBI_validate['ret_0']*RBI_validate['SVM_Pred_Ret1']

EBS_validate['SVM_Returns_Mom5'] =EBS_validate['ret_0']*EBS_validate['SVM_Pred_Mom5']
RBI_validate['SVM_Returns_Mom5'] =RBI_validate['ret_0']*RBI_validate['SVM_Pred_Mom5']


EBS_validate[['ret_0', 'SVM_Returns_Ret1', 'SVM_Returns_Mom5']].cumsum().apply(np.exp).plot(figsize=(10, 6));
RBI_validate[['ret_0', 'SVM_Returns_Ret1', 'SVM_Returns_Mom5']].cumsum().apply(np.exp).plot(figsize=(10, 6));

crossval_EBS = cross_val_score(SVM_EBS_hard,EBS_features[cols],EBS_features['Sign'],scoring = 'accuracy')
print("Accuracy For EBS SVM Hard Margins Model : %.4f"%crossval_EBS.mean())
crossval_RBI = cross_val_score(SVM_RBI_hard,RBI_features[cols],RBI_features['Sign'],scoring = 'accuracy')
print("Accuracy For RBI SVM Hard Margins Model : %.4f"%crossval_RBI.mean())


#########################################
# graph for SVM 

cols = [ 'ret_1', 'MOM5' ]
training_split = int(0.75*EBS.shape[0])
EBS_training = EBS_features.iloc[0:training_split,:]
RBI_training = RBI_features.iloc[0:training_split,:]

EBS_validate = EBS_features.iloc[training_split:,:]
RBI_validate = RBI_features.iloc[training_split:,:]

SVM_EBS =SVC(C=1e5,probability=True,kernel='linear')
SVM_RBI =SVC(C=1e5,probability=True,kernel='linear')
#fit again
SVM_EBS.fit(EBS_training[cols],EBS_training['Sign'])
SVM_RBI.fit(RBI_training[cols],RBI_training['Sign'])

#predict using the validation set of data
EBS_validate['SVM_Predict'] = SVM_EBS.predict(EBS_validate[cols])
RBI_validate['SVM_Predict'] = SVM_RBI.predict(RBI_validate[cols])


EBS_validate['SVM_Predict'][EBS_validate['SVM_Predict']==0] = -1
RBI_validate['SVM_Predict'][RBI_validate['SVM_Predict']==0] = -1

EBS_validate['SVM_Returns'] =EBS_validate['ret_0']*EBS_validate['SVM_Predict']
RBI_validate['SVM_Returns'] =RBI_validate['ret_0']*RBI_validate['SVM_Predict']

EBS_validate.tail(3)

EBS_validate[['ret_0', 'SVM_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Erste Bank",fontweight="bold")
RBI_validate[['ret_0', 'SVM_Returns']].cumsum().apply(np.exp).plot(figsize=(10, 6));
plt.title("Raiffeisen Bank",fontweight="bold")

#print the support vectors
print('         Support Vectors for EBS')
df = pd.DataFrame(SVM_EBS.support_vectors_)
df.columns = cols
df.head(3)


#I reference this function to PythonDataScienceHandbook
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
plt.scatter(EBS_validate['ret_1'], EBS_validate['MOM5'], c=EBS_validate['Sign'], s=len(EBS_validate["ret_1"]), cmap='autumn')
plot_svc_decision_function(SVM_EBS);

plt.scatter(RBI_validate['ret_1'], RBI_validate['MOM5'], c=RBI_validate['Sign'], s=len(RBI_validate["ret_1"]), cmap='autumn')
plot_svc_decision_function(SVM_RBI);


#######################################################
#######################################################
# A3 a) KNN Neighbors Analysis - Default
#######################################################
#######################################################

from sklearn.preprocessing import StandardScaler
#EBS_features[cols] = StandardScaler().fit_transform(EBS_features[cols])

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


#EBS
knn.fit(EBS_training[cols1],EBS_training['I'])

EBS_validate['KNN_Pred_Ret1'] = knn.predict(EBS_validate[cols1])

EBS_validate['KNN_Pred_Ret1'][EBS_validate['KNN_Pred_Ret1']==0] = -1

EBS_validate['KNN_Returns_Ret1'] =EBS_validate['ret_0']*EBS_validate['KNN_Pred_Ret1']

EBS_validate[['ret_0', 'KNN_Returns_Ret1']].cumsum().apply(np.exp).plot(figsize=(10, 6));

#RBI
knn.fit(RBI_training[cols1],RBI_training['I'])

RBI_validate['KNN_Pred_Ret1'] = knn.predict(RBI_validate[cols1])

RBI_validate['KNN_Pred_Ret1'][RBI_validate['KNN_Pred_Ret1']==0] = -1

RBI_validate['KNN_Returns_Ret1'] =RBI_validate['ret_0']*RBI_validate['KNN_Pred_Ret1']

RBI_validate[['ret_0', 'KNN_Returns_Ret1']].cumsum().apply(np.exp).plot(figsize=(10, 6));

#######################################################
#######################################################
# A3 b) Report on sensible values,comparsion etc.
#######################################################
#######################################################

#instead of repeating the previous procedure  
#we make use of the model_selection function
#start with different number of neighbours k

from sklearn import model_selection
from sklearn import metrics

EBS_features['I'] = EBS_features['ret_0']
EBS_features['I'][EBS_features['I']>0] = 1
EBS_features['I'][EBS_features['I']<=0] = -1

cols1=["MOM5","ret_1"]
#plot_logit_ROC(default_ind,Y_test,X_test,logit,Y_response,X_features)
X_Train_EBS, X_Test_EBS, Y_Train_EBS, Y_Test_EBS = model_selection.train_test_split(EBS_features[cols1],
                                                                                    EBS_features['I'] ,
                                                                                    test_size=0.25, shuffle=True)
#k=5, number of neighbours                                                          
knn5 = KNeighborsClassifier(n_neighbors=5)        

knn5.fit(X_Train_EBS,Y_Train_EBS)
Y_EBS_P = knn5.predict(X_Test_EBS)

print("Accuracy with k=5:",metrics.accuracy_score( Y_Test_EBS, Y_EBS_P))

#k=7, number of neighbours 
knn7 = KNeighborsClassifier(n_neighbors=7)        

knn7.fit(X_Train_EBS,Y_Train_EBS)
Y_EBS_P = knn7.predict(X_Test_EBS)

print("Accuracy with k=7:",metrics.accuracy_score( Y_Test_EBS, Y_EBS_P))

#generally,increased number of neighbors leads to more accuracy
#but not always

#k=13, number of neighbours 
knn13 = KNeighborsClassifier(n_neighbors=13)        

knn13.fit(X_Train_EBS,Y_Train_EBS)
Y_EBS_P = knn13.predict(X_Test_EBS)

print("Accuracy with k=13:",metrics.accuracy_score( Y_Test_EBS, Y_EBS_P))
#accuracy fell in fact


#let's find with a loop the optimal number for k
for k in range(3,20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_Train_EBS,Y_Train_EBS)
    Y_EBS_P = knn.predict(X_Test_EBS)  
    print("Accuracy with k=:{} is".format(k),metrics.accuracy_score( Y_Test_EBS, Y_EBS_P))


#######################################################
#######################################################
# A3 c) Plot decision boundary.
#######################################################
#######################################################

#easily-visualize-scikit-learn-models-decision-boundaries
#similar procedures can seen in the examples of the official package documentation    
def plot_decision_boundaries(X, y, model_class, **model_params):
    """
    Function to plot the decision boundaries of a classification model.
    This uses just the first two columns of the data for fitting 
    the model as we need to find the predicted value for every point in 
    scatter plot.
    Arguments:
            X: Feature data as a NumPy-type array.
            y: Label data as a NumPy-type array.
            model_class: A Scikit-learn ML estimator class 
            e.g. GaussianNB (imported from sklearn.naive_bayes) or
            LogisticRegression (imported from sklearn.linear_model)
            **model_params: Model parameters to be passed on to the ML estimator
    
    """
    try:
        X = np.array(X)
        y = np.array(y).flatten()
    except:
        print("Coercing input data to NumPy arrays failed")
    # Reduces to the first two columns of data
    reduced_data = X[:, :2]
    # Instantiate the model object
    model = model_class(**model_params)
    # Fits the model with the reduced data
    model.fit(reduced_data, y)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].    

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 0.1, reduced_data[:, 0].max() + 0.1
    y_min, y_max = reduced_data[:, 1].min() - 0.1, reduced_data[:, 1].max() + 0.1
    # Meshgrid creation
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh using the model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])    

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() -0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predictions to obtain the classification results
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plotting
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel("MOM5",fontsize=15)
    plt.ylabel("Ret_1",fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return plt
#This function is referenced to towardsdatascience.com 



#now let's compare the distance metric
for dist in ["manhattan", "euclidean", "mahalanobis"]:
    if (dist=="mahalanobis"):   
        knn = KNeighborsClassifier(n_neighbors=5, metric=dist, metric_params={'V': np.cov(X_Train_EBS, rowvar=False)})
    else: 
        knn = KNeighborsClassifier(n_neighbors=5, metric=dist)
    knn.fit(X_Train_EBS,Y_Train_EBS)
    Y_EBS_P = knn.predict(X_Test_EBS)  
    print("Accuracy with {} distance metric is".format(dist),metrics.accuracy_score( Y_Test_EBS, Y_EBS_P))
    plt.figure()
    plt.title("KNN decision boundary with distance: {}".format(dist),fontsize=16)
    if (dist=="mahalanobis"):
        plot_decision_boundaries(X_Train_EBS,Y_Train_EBS,KNeighborsClassifier,n_neighbors=5,metric=dist,metric_params={'V': np.cov(X_Train_EBS, rowvar=False)}) 
    else:    
        plot_decision_boundaries(X_Train_EBS,Y_Train_EBS,KNeighborsClassifier,n_neighbors=5)
    plt.show()
        
#the mahalanobis distance was the most accurate according the results
#the main difference comes how outliers are considered
#eg. the most significant outlier ret_1=12% is not in the dark area    

#plot decision boundaries
for k in [3,5,13]:
    plt.figure()
    plt.title("KNN decision boundary with neighbours: {}".format(k),fontsize=16)
    plot_decision_boundaries(X_Train_EBS,Y_Train_EBS,KNeighborsClassifier,n_neighbors=k)
    plt.show()


#######################################################
#######################################################
# B. Prediction Quality and Bias
#######################################################
# B.1 Investigaion using confusion matrix & ROC curve
#######################################################
#######################################################

#module we use for auc
from sklearn.metrics import roc_curve,roc_auc_score,auc
#module we will use for confusion matrix analysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


#we first define a function to plot ROC/AUC
def plot_logit_ROC(default_ind,Y_test,X_test,logit,Y_response,X_features):
    #logit: We pass a defined LogisticRegression Object here
    #Y-Response: Y Population
    #X_Features: X Population
    #Y_test: testing set Y
    #X_test: testing set X
    
    # (1) Calculate AUC on testing set
    logit_roc_aucT = roc_auc_score(Y_test, logit.predict(X_test))
    # (2) calculate False Positive Rate and True Positive Rate on test set
    fprT,tprT,thresholdsT = roc_curve(Y_test,logit.predict_proba(X_test)[:,1],pos_label = default_ind)
    
    #(3)Calculate AUC on population testing set
    logit_roc_aucP = roc_auc_score(Y_response,logit.predict(X_features))
    #(4)calculate False Positive Rate and True Positive Rate on Population
    fprP,tprP,threasholdsP = roc_curve(Y_response,logit.predict_proba(X_features)[:,1],pos_label=default_ind)
    
    fig,ax = plt.subplots(figsize=(10,8))
    
    #Plot diagnoal line
    ax.plot([0,1],[0,1],'r--',label = "Random Classifier")
    
    #Plot ROC for predictions on testing set
    ax.plot(fprT,tprT,label = 'Train/Test Regression (area = %0.2f)'%logit_roc_aucT)
    
    #Plot ROC curve for the prediction on full set
    ax.plot(fprP,tprP,label = 'Population Regression (area = %0.2f)'%logit_roc_aucP)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.legend(loc="lower right", fontsize=14)
    plt.show()
    
    return ax

#from sklearn import model_selection

cols = ['ret_1','ret_2', 'ret_3', 'ret_4', 'ret_5']

X_Train_EBS, X_Test_EBS, Y_Train_EBS, Y_Test_EBS = model_selection.train_test_split(EBS_features[cols],
                                                                                    EBS_features['I'] ,
                                                                                    test_size=0.25, shuffle=True)

lm_EBS = linear_model.LogisticRegression(C = 1e15,solver ='liblinear',penalty = 'l2')
lm_EBS.fit(X_Train_EBS,Y_Train_EBS)


plot_logit_ROC(1,Y_Test_EBS,X_Test_EBS,lm_EBS,EBS_features['I'],EBS_features[cols])    


#print confusion matrix
Y_EBS_pred = lm_EBS.predict(X_Test_EBS)
confusion_matrix_EBS = confusion_matrix(Y_Test_EBS,Y_EBS_pred)
print("Confusion Matrix")
print(confusion_matrix_EBS)
#this is clearly bad model, a random model could have been better

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
print('Accuracy Score : ' + str(accuracy_score(Y_Test_EBS,Y_EBS_pred)))
print('Precision Score : ' + str(precision_score(Y_Test_EBS,Y_EBS_pred)))
print('Recall Score : ' + str(recall_score(Y_Test_EBS,Y_EBS_pred)))
print('F1 Score : ' + str(f1_score(Y_Test_EBS,Y_EBS_pred)))


#let's use the refined model
cols=['MOM5', 'EWMA5-7','EWMA7-13']

X_Train_EBS, X_Test_EBS, Y_Train_EBS, Y_Test_EBS = model_selection.train_test_split(EBS_features[cols],
                                                                                    EBS_features['I'] ,
                                                                                    test_size=0.25, shuffle=True)

lm_EBS = linear_model.LogisticRegression(C = 1e15,solver ='liblinear',penalty = 'l2')
lm_EBS.fit(X_Train_EBS,Y_Train_EBS)


plot_logit_ROC(1,Y_Test_EBS,X_Test_EBS,lm_EBS,EBS_features['I'],EBS_features[cols])    


#print confusion matrix
Y_EBS_pred = lm_EBS.predict(X_Test_EBS)
confusion_matrix_EBS = confusion_matrix(Y_Test_EBS,Y_EBS_pred)
print("Confusion Matrix")
print(confusion_matrix_EBS)
#this is a very good model

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
print('Accuracy Score : ' + str(accuracy_score(Y_Test_EBS,Y_EBS_pred)))
print('Precision Score : ' + str(precision_score(Y_Test_EBS,Y_EBS_pred)))
print('Recall Score : ' + str(recall_score(Y_Test_EBS,Y_EBS_pred)))
print('F1 Score : ' + str(f1_score(Y_Test_EBS,Y_EBS_pred)))


#######################################################
#######################################################
# B.2 Grid Search
#######################################################
#######################################################
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

training_split = int(0.75*EBS.shape[0])
EBS_training = EBS_features.iloc[0:training_split,:]
RBI_training = RBI_features.iloc[0:training_split,:]

EBS_validate = EBS_features.iloc[training_split:,:]
RBI_validate = RBI_features.iloc[training_split:,:]

X_Train_EBS, X_Test_EBS, Y_Train_EBS, Y_Test_EBS = model_selection.train_test_split(EBS_features[cols],
                                                                                    EBS_features['I'] ,
                                                                                    test_size=0.25, shuffle=True)



##############
# Logit Model

lm_EBS = linear_model.LogisticRegression()
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
grid_lm_EBS_acc = GridSearchCV(lm_EBS, param_grid = grid_values,scoring = 'recall')
grid_lm_EBS_acc.fit(X_Train_EBS,Y_Train_EBS)
Y_EBS_pred = grid_lm_EBS_acc.predict(X_Test_EBS)


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
print('Accuracy Score : ' + str(accuracy_score(Y_Test_EBS,Y_EBS_pred)))
print('Precision Score : ' + str(precision_score(Y_Test_EBS,Y_EBS_pred)))
print('Recall Score : ' + str(recall_score(Y_Test_EBS,Y_EBS_pred)))
print('F1 Score : ' + str(f1_score(Y_Test_EBS,Y_EBS_pred)))

#see what parameters to use
print(grid_lm_EBS_acc.best_params_)

##############
# SVM Model

cols1=["MOM5","ret_1"]



parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(EBS_validate[cols], EBS_validate['Sign'])
GridSearchCV(estimator=SVC(),
             param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})

sorted(clf.cv_results_.keys())

print('Accuracy Score : ' + str(accuracy_score(Y_Test_EBS,Y_EBS_pred)))
print('Precision Score : ' + str(precision_score(Y_Test_EBS,Y_EBS_pred)))
print('Recall Score : ' + str(recall_score(Y_Test_EBS,Y_EBS_pred)))
print('F1 Score : ' + str(f1_score(Y_Test_EBS,Y_EBS_pred)))



#######################################################
#######################################################
# B.3 P&L Backtesting. transition probabilities. Kelly
#######################################################
#######################################################

#this is our best set of features in logit regression
cols=['MOM5', 'EWMA5-7','EWMA7-13']
lm_EBS = linear_model.LogisticRegression(C = 1e6,solver='liblinear',multi_class='ovr',penalty='l2')
lm_RBI = linear_model.LogisticRegression(C = 1e6,solver='liblinear',multi_class='ovr',penalty='l2')

training_split = int(0.75*EBS.shape[0])

EBS_training = EBS_features.iloc[0:training_split,:]
RBI_training = RBI_features.iloc[0:training_split,:]

EBS_validate = EBS_features.iloc[training_split:,:]
RBI_validate = RBI_features.iloc[training_split:,:]

#fit again
lm_EBS.fit(EBS_training[cols],EBS_training['Sign'])
lm_RBI.fit(RBI_training[cols],RBI_training['Sign'])

#predict using the validation set of data
EBS_validate['Logit_Predict'] = lm_EBS.predict(EBS_validate[cols])
RBI_validate['Logit_Predict'] = lm_RBI.predict(RBI_validate[cols])

EBS_validate['Logit_Returns'] =EBS_validate['ret_0']*EBS_validate['Logit_Predict']
RBI_validate['Logit_Returns'] =RBI_validate['ret_0']*RBI_validate['Logit_Predict']


Pred_Probs_EBS=lm_EBS.predict_proba(EBS_validate[cols])
Pred_Probs_RBI=lm_RBI.predict_proba(RBI_validate[cols])
#1st column is probability of down move
#2nd column is probability of up move

EBS_validate['real_NAV'] = abs(EBS_validate['ret_0']).cumsum().apply(np.exp)
EBS_validate['predicted_NAV'] = EBS_validate['Logit_Returns'].cumsum().apply(np.exp)
np.corrcoef(EBS_validate['real_NAV'],EBS_validate['predicted_NAV'])


#defining an array of colors  
colors = ['r', 'b']
  #assigns a color to each data point
plt.scatter(Pred_Probs_EBS[:,0], EBS_validate['ret_0'],c=EBS_validate['Sign'], alpha=0.70, cmap='RdYlBu')
plt.title("Erste - Probability of Down Move")
plt.xlabel('Probability')
plt.ylabel('Return')
#negative correlation - this is exactly what we want
#i.e. high probability of large negative returns 

plt.scatter(Pred_Probs_EBS[:,1], EBS_validate['ret_0'],c=EBS_validate['Sign'], alpha=0.70, cmap='RdYlBu')
plt.title("Erste - Probability of Up Move")
plt.xlabel('Probability')
plt.ylabel('Return')
#positive correlation - this is exactly what we want
#i.e. high probability for large positive returns 

#Kelly
#betting only if p>0.55

strategy_PNL=0
PNL_history = []
for i in range(len(Pred_Probs_EBS)):
    daily_PNL=0
    if (Pred_Probs_EBS[i][0]>0.55):
        daily_PNL=(2*Pred_Probs_EBS[i][0]-1)*(-EBS_validate['ret_0'][i])
    else:
        if (Pred_Probs_EBS[i][1]>0.55):
            daily_PNL=(2*Pred_Probs_EBS[i][1]-1)*(EBS_validate['ret_0'][i])
    strategy_PNL=strategy_PNL+daily_PNL
    PNL_history.append(daily_PNL)
    

strategy_PNL    

#if we didn't use Kelly the return is
EBS_validate['Logit_Returns'].cumsum()[-1]

#it turns out that the Kelly criteterion curbed the profit 
# the benefit is though less risk
#f let's check the most negative return
min(PNL_history)
min(EBS_validate['Logit_Returns'])         
