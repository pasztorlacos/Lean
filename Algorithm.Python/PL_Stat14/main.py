environment = "Lean"
if environment=="Lean":
    import sys
    sys.path.append('C:/Github/Repos/pasztorlacos/Quantconnect/Libraries/')
    sys.path.append('C:/Github/Repos/pasztorlacos/Quantconnect/Strategies/')
    #---------------------------------------------------------------------------------------------

from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Algorithm.Framework")
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Indicators")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
#from QuantConnect.Algorithm.Framework import *
#from QuantConnect.Algorithm.Framework.Alphas import *
#from QuantConnect.Algorithm.Framework.Execution import *
#from QuantConnect.Algorithm.Framework.Portfolio import *
#from QuantConnect.Algorithm.Framework.Risk import *
#from QuantConnect.Algorithm.Framework.Selection import *
from QuantConnect.Orders import *
from QuantConnect.Orders.Fees import *
from QuantConnect.Securities import *
from QuantConnect.Orders.Fills import *
from QuantConnect.Brokerages import  BrokerageName
from QuantConnect import Resolution, SecurityType

from System.Drawing import Color
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
import math
import random
import decimal as d

from pm import MyPositionManager
from pmB import MyPositionManagerB
from var import MyVaR, MyCharts, MyStats
import hp
from sim import MySIMPosition

from eq_st14 import Eq_St14

class MyAlgo(QCAlgorithm):
    '''
    Multistrategy Framework 1.0
    '''
    file = __file__
    def Initialize(self):
        self.Debug(" ---- Initialize() Started") 
        self.enabled = True
        self.twsSynced = False
        self.updateSettings = True if self.LiveMode else False
        self.settingsURL = 'https://www.dropbox.com/s/mhovszuzfwtfp60/QCStrategySettings_2.csv?dl=1'
        self.strategySettings = hp.MyStrategySettings(self)
        '''DEBUG/LOG'''
        self.debug = False
        self.log = 0 #0) only Debug 1) Debug anf Log 2) only Log
        self.debugOrderFill = False #self.debug
        if self.LiveMode: self.debugOrderFill = True
        self.myHelpers = hp.MyHelpers(self)
                
        '''DATA STORAGE'''
        self.myStrategyClassList = []
        self.myVaRList = []
        self.mySymbolDict = {} 
        self.openStopMarketOrders = []   
        self.openLimitOrders = []
        self.openMarketOrders = []     
        self.myVaR = None
        self.foreignVaR = None
        
        '''PositionManager instantiation'''
        self.myPositionManager = MyPositionManager(self)
        self.myPositionManagerB = MyPositionManagerB(self)
        self.consistencyStartUpReleaseTime = self.Time - timedelta(hours=20)
        
        '''DataNormalizationMode for Equities'''
        self.myDataNormalizationMode = DataNormalizationMode.SplitAdjusted  #DataNormalizationMode.Raw, DataNormalizationMode.SplitAdjusted, DataNormalizationMode.Adjusted, DataNormalizationMode.TotalReturn
        #This must be before InstallStrategy() as it resets custom models
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

        '''STARTEGIES (First would inlude benchmark)'''
        self.myHelpers.InstallStrategy(Eq_St14, myAllocation=1.00)

        '''BACKTEST DATES and SETTINGS'''
        #Start Date
        #self.SetStartDate(datetime.now() - timedelta(days=30))  
        self.simStartYear=2005
        self.SetStartDate(self.simStartYear,1,1)

        #End Date
        #self.SetEndDate(2019,10,11) #Last PiData Date
        self.simYears = 20
        self.simEndDate = datetime(self.simStartYear+self.simYears, 1, 1, 0, 0) #Use None if not applicable
        self.SetEndDate(min(self.simEndDate + timedelta(days=30), datetime(2019,10,10)))
        #self.SetEndDate(datetime.now())
        
        #self.Portfolio.SetAccountCurrency("EUR")
        self.SetCash(100000)

        '''Resolution'''
        self.mainResolution = self.myHelpers.MyResolution()
        self.UniverseSettings.Resolution = self.mainResolution
       
        '''WarmUp'''
        self.SetWarmUp(self.myHelpers.WarUpDays()) 
        
        #Add chartSymbol and Set Tradable Property!
        if environment=="Lean":
            self.chartTicker = "SPY"
        else:
            self.chartTicker = "VOO" #VOO: Vanguard S&P 500 ETF
        self.AddEquity(self.chartTicker, self.mainResolution)
        self.chartSymbol = self.Securities[self.chartTicker].Symbol 
        self.Securities[self.chartSymbol].SetDataNormalizationMode(self.myDataNormalizationMode)
        self.myHelpers.AddSymbolDict(self.chartSymbol, self.myStrategyClassList[0], self.myVaR)
        self.mySymbolDict[self.chartSymbol].posEnabled = False
        #Add Benchmark Symbol that is Not Tradable
        self.benchmarkTicker = "MDY" #IWV:iShares Russell 3000 ETF, IWM Russell 2000 ETF: small cap part of R3000, MDY S&P MidCap 400 Index
        self.AddEquity(self.benchmarkTicker, self.mainResolution)
        self.benchmarkSymbol = self.Securities[self.benchmarkTicker].Symbol
        self.myHelpers.AddSymbolDict(self.benchmarkSymbol, self.myStrategyClassList[0], self.myVaR)
        self.mySymbolDict[self.benchmarkSymbol].posEnabled = False
        self.SetBenchmark(self.benchmarkSymbol)

        '''Charts and Stats instantiation'''
        self.myCharts = MyCharts (self, self.chartSymbol, backtestUpdateHours=1)
        self.myStats = MyStats (self)
        
        self.MyDebug(" ---- Initialize() Finished") 
        return
    
    '''
    AFTER WARMUP
    '''
    def OnWarmupFinished (self):
        self.myHelpers.MyOnWarmupFinished()
        return
    '''
    ON DATA
    '''
    def OnData(self, data):
        self.myHelpers.MyOnData(data)
        #if self.LiveMode and not self.IsWarmingUp: self.MyDebug(' pendingFlipPositions:' +str(len(self.myPositionManager.pendingFlipPositions)))
        return
    '''
    ORDEREVENT HANDLER
    '''
    def OnOrderEvent(self, OrderEvent):
        self.myPositionManagerB.MyOrderEventHandler(OrderEvent)
        return

    '''
    AFTER BACKTEST
    '''
    def OnEndOfAlgorithm(self):
        if True and environment=="Lean" and not self.LiveMode: 
            MySIMPosition.SaveData(self)

        self.myStats.PrintStrategyTradeStats()
        return

    '''
    DEBUG
    '''
    def MyDebug(self, debugString):
        message = str(self.Time) + debugString
        if self.log == 0:
            self.Debug(message)
        elif self.log == 1:
            self.Debug(message)
            self.Log(message) 
        elif self.log == 2:
            self.Log(message)