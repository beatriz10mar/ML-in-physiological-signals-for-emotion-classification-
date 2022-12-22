#!/usr/bin/env python
# coding: utf-8

# In[6]:


import statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import pandas as pd
import numpy as np

# In[7]:


class StationarityTests:
    def __init__(self, significance=.05):
        self.SignificanceLevel = significance
        self.pValue = None
        self.isStationary = None


# In[17]:


def ADF_Stationarity_Test(self, timeseries, printResults = True):
        #Dickey-Fuller test:
        adfTest = adfuller(timeseries, autolag='AIC')
        
        self.pValue = adfTest[1]
        
        if (self.pValue<self.SignificanceLevel):
            self.isStationary = True
        else:
            self.isStationary = False
        
        dfResults = pd.Series(adfTest[0:4], index=['ADF Test Statistic','P-Value','# Lags Used','# Observations Used'])
        return dfResults
        
        if printResults:
            dfResults = pd.Series(adfTest[0:4], index=['ADF Test Statistic','P-Value','# Lags Used','# Observations Used'])
            #Add Critical Values
            for key,value in adfTest[4].items():
                dfResults['Critical Value (%s)'%key] = value
            #print('Augmented Dickey-Fuller Test Results:')
            #print(dfResults)

# In[]:

def KPSS_Stationarity_Test(self, timeseries,printResults = True):
    
    
    kpsstest = kpss(np.array(timeseries),nlags="auto")
    self.pValue = kpsstest[1]
    
    if (self.pValue<self.SignificanceLevel):
        self.isStationary = False
    else:
        self.isStationary = True
           
    dfResults_kpss = pd.Series(kpsstest[0:4], index=['KPSS Test Statistic','P-value','#Lags Used','# Critical Values'])

    return dfResults_kpss
# In[4]:


def stationarity (signal,window):
    for i in range(0,len(signal)-window):
        begin=i
        final=i+window
        sTest = StationarityTests()
        ADF_Stationarity_Test(sTest,signal[begin:final])
        #return begin, final, sTest.isStationary
        if sTest.isStationary:
            break   
    return begin, final


# %%
