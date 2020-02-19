environment = "Lean"
if environment=="Lean" and False:
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

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import time
from System.Drawing import Color
import decimal as d
import math

from pm3 import MyPositionManager
from pmB3 import MyPositionManagerB
from var3 import MyVaR, MyCharts, MyStats
import hp3
from eq2_1_pt_1 import Equities2_1_pt_1
from fx2_1_pt_1 import Fx2_1_pt_1

from sim1 import MySIMPosition

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
        self.settingsURL = 'https://www.dropbox.com/s/thxemetfxuisl27/QCStrategySettings.csv?dl=1'
        self.strategySettings = hp3.MyStrategySettings(self)
        '''DEBUG/LOG'''
        self.debug = False
        self.log = 0 #0) only Debug 1) Debug anf Log 2) only Log
        self.debugOrderFill = False #self.debug
        if self.LiveMode: self.debugOrderFill = True
        self.myHelpers = hp3.MyHelpers(self)
                
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
        self.myHelpers.InstallStrategy(Equities2_1_pt_1, myAllocation=1.00*0.01)
        self.myHelpers.InstallStrategy(Fx2_1_pt_1, myAllocation=1.00*0.99)
        #The first strategíy is for benchmarksymbol only, so to speed up warmup
        self.myStrategyClassList[0].warmupcalendardays = self.myStrategyClassList[1].warmupcalendardays

        '''BACKTEST DATES and SETTINGS'''
        #Start Date
        #self.SetStartDate(2000,1,1)
        #self.SetStartDate(2002,1,1)
        #self.SetStartDate(2003,1,1)
        #self.SetStartDate(2004,1,1)
        #self.SetStartDate(2007,1,1)
        #self.SetStartDate(2009,1,1)
        #self.SetStartDate(2016,1,1)
        #self.SetStartDate(2017,1,1)
        #elf.SetStartDate(2018,10,1)
        #self.SetStartDate(2019,6,1)
        #self.SetStartDate(datetime.now() - timedelta(days=30))  
        self.simStartYear=2003
        self.SetStartDate(self.simStartYear,1,1)

        #End Date
        #self.SetEndDate(2003,6,30)
        #self.SetEndDate(2003,12,31)
        #self.SetEndDate(2004,12,31)
        #self.SetEndDate(2006,12,31)
        #self.SetEndDate(2009,12,31)
        #self.SetEndDate(2013,12,31)
        #self.SetEndDate(2012,12,31)
        #self.SetEndDate(2015,6,24)
        #self.SetEndDate(datetime.now())
        #self.SetEndDate(2016,10,11) #Last PiData Date
        self.simYears = 2
        self.simEndDate = datetime(self.simStartYear+self.simYears, 1, 1, 0, 0) #Use None if not applicable
        self.SetEndDate(min(self.simEndDate + timedelta(days=30), datetime(2019,10,10)))

        #self.Portfolio.SetAccountCurrency("EUR")
        self.SetCash(100000)

        '''Resolution'''
        self.mainResolution = self.myHelpers.MyResolution()
        self.UniverseSettings.Resolution = self.mainResolution
       
        '''WarmUp'''
        self.SetWarmUp(self.myHelpers.WarUpDays()) 
        
        #Add chartSymbol and Set Tradable Property!
        self.chartTicker = "SPY" #don't use "QQQ" in Lean envirionment
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
        if environment=="Lean" and True and not self.LiveMode: 
            MySIMPosition.SaveData(self)
        
        #self.mySymbolDict[self.Securities["AMAT"].Symbol].priceNormaliser.PriceDebug()
        if False: self.mySymbolDict[self.Securities["AA"].Symbol].zz1.ListZZ()
        if False: self.mySymbolDict[self.Securities["AA"].Symbol].zz1.listZZPoints()
        
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