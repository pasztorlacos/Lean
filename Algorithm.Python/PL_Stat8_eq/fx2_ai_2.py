### <summary>
### 
### </summary>
from QuantConnect import SecurityType, Resolution
from QuantConnect.Indicators import RollingWindow, ExponentialMovingAverage, SimpleMovingAverage, IndicatorExtensions, AverageTrueRange, DonchianChannel, RelativeStrengthIndex, IndicatorDataPoint
from QuantConnect.Data.Consolidators import TradeBarConsolidator, QuoteBarConsolidator
from QuantConnect.Data.Market import IBaseDataBar, TradeBar

from datetime import datetime, timedelta, date
from ta1 import MyRelativePrice, MyPriceNormaliser, MyZigZag, MyDCHState, MyMAState, MySupportResistance, MyPatterns, MyVolatility
from taB1 import MyBarStrength, MyBarRejection
from sim1 import MySIMPosition
#from m_pt1 import NN_1

import hp3
from pandas import DataFrame
import numpy as np
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

class Fx2_ai_2():
    file = __file__
    '''
    Strategy Implementation
    '''
    '''STRATEGY SETTINGS'''
    enabled = True
    manageOrderConsistency = True
    simulate = False
    saveFilesFramework = True
    saveFilesSim = saveFilesFramework
    strategyCodeOriginal = __name__
    strategyCode = strategyCodeOriginal #used by order tags and debug
    isEquity = False
    customFillModel = 1
    customSlippageModel = 1
    customFeeModel = 0
    customBuyingPowerModel = 0
    #Resolution
    resolutionMinutes   = 60
    resolutionMinutes_2 = 24*60
    maxWarmUpPeriod   = 500
    maxWarmUpPeriod_2 = 85
    barPeriod   =  timedelta(minutes=resolutionMinutes)
    barPeriod_2 =  timedelta(minutes=resolutionMinutes_2)
    if isEquity:
        warmupcalendardays = max(round(7/5*maxWarmUpPeriod/(7*(60/resolutionMinutes))), round(7/5*maxWarmUpPeriod_2/(7*(60/resolutionMinutes_2))))
    else:
        warmupcalendardays = max(round(7/5*maxWarmUpPeriod/(24*(60/resolutionMinutes))), round(7/5*maxWarmUpPeriod_2/(24*(60/resolutionMinutes_2))))
    #Switches
    plotIndicators = False
    #Risk Management
    strategyAllocation = 0.1 #Install can OverWrite it!!!
    maxAbsExposure = 4.0
    maxLongExposure = 4.0 
    maxNetLongExposure = 4.0
    maxShortExposure = -4.0
    maxNetShortExposure = -4.0
    maxSymbolAbsExposure = 0.50
    minSymbolAbsExposure = 0.02
    riskperLongTrade  = 0.50/100 
    riskperShortTrade = 0.50/100 
    maxLongVaR  = 5.0/100 
    maxShortVaR = 5.0/100 
    maxTotalVaR = 0.100
    mainVaRr = None #it is set by the first instance
    #Entry
    enableLong  = False
    enableShort = False
    liquidateLong  = False
    liquidateShort = False
    enableFlipLong  = True #Checked in self.algo.myPositionManager.EnterPosition_2
    enableFlipShort = True
    closeOnTriggerLong  = True #works if Flip is didabled
    closeOnTriggerShort = True 
    entryLimitOrderLong  = True  #If False: enter the position with Market Order
    entryLimitOrderShort = True
    limitEntryATRLong  = 0.01
    limitEntryATRShort = 0.01
    entryTimeInForceLong = timedelta(minutes=5*60)
    entryTimeInForceShort = timedelta(minutes=5*60)
    #Orders
    useTragetsLong  = True #TWS Sync is not ready yet to handle useTragets for Foreign Symbols
    useTragetsShort = True
    stopPlacerLong  = 0    #0)dch0 1)dch1 2)dch2) 3)dch3 4)minStopATR 5)bars_rw[0] 6)bars_rw[0-1] 7)dc01 8)dch02 else: dch1
    stopPlacerShort = 0
    targetPlacerLong  = 1    #1)max(dch2,minPayoff) 2)max(dch3,minPayoff)  else: minPayoff 
    targetPlacerShort = 1    
    minPayOffLong  = 1.75
    minPayOffShort = 1.75
    scratchTradeLong  = True 
    scratchTradeShort = True 
    stopTrailerLong  = 0 #0) no Trail 1)dhc2 2)dch3 3)dch1 4)dch0
    stopTrailerShort = 0 
    targetTrailerLong  = 0 #0) no Trail 1)dhc2 
    targetTrailerShort = 0 
    stopATRLong  = 0.5
    stopATRShort = 0.5
    minEntryStopATRLong  = 2.0    
    minEntryStopATRShort = 2.0    
    #Strategy Stats
    _totalTrades = 0
    _totalTradesLong = 0
    _totalTradesShort = 0
    _totalFlipTrades = 0
    _totalEntryRejections = 0
    _totalEntries =  0
    _totalFills = 0
    _totalEntryFills = 0
    _totalATRpct = 0
    _lastUpdated = datetime(year = 1968, month = 6, day = 25)
  
    #PiData. Dont use EURCAD before 2008 and GBPJPY before 2004, otherwise start with 2003
    #PiFx = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD", "AUDJPY", "CHFJPY", "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "GBPCHF", "GBPJPY"]
    myTickers = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD", "AUDJPY", "CHFJPY", "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "GBPCHF", "GBPJPY"]
    #myTickers = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD", "AUDJPY", "CHFJPY", "EURAUD", "EURCHF", "EURGBP", "EURJPY", "GBPCHF"]
    
    #My Selection
    #myTickers = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD", "USDCNH", "EURJPY", "EURSEK", "EURNOK", "USDMXN", "USDZAR", "USDSEK", "USDNOK", "EURHUF", "USDHUF"] #17 Symbols
    #myTickers = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD", "USDCNH", "EURJPY"] #9 Symbols
    #myTickers = ["EURUSD", "AUDUSD", "USDJPY", "USDZAR", "EURCHF", "EURCAD", "EURNOK", "EURCNH"] #8 Symbols
    #myTickers = ["EURUSD", "AUDUSD", "USDJPY", "EURCAD", "EURNOK", "EURCNH"] #6 Symbols
    #myTickers = ["EURUSD", "AUDUSD", "EURJPY", "EURCAD"] #4 Symbols
    myTickers = ["USDJPY"]

    #Simulation Signals [direction, disableBars, Enabled]
    simDict = {
       "L_Str_": [1,8,True], "S_Str_": [-1,8,True], 
       "L_Rej_": [1,8,True], "S_Rej_": [-1,8,True],
       "DB_": [1,8,True], "DT_": [-1,8,True], 
       "TB_": [1,8,False], "TT_": [-1,8,False],
       "IHS_": [1,8,False], "HS_": [-1,8,False]}
    
    exitSignalDict = {
       "L_Str_": 0, 
       "S_Str_": 0, 
       "L_Rej_": 0,
       "S_Rej_": 0}
    
    #AI ----
    loadAI = True
    aiDict = {}
    
    aiDict["L_Str_DCHs"] = {
        "enabled": False,
        "signalRegex": '^(?=.*L_Str)(?=.*L_DCH_s).*$',
        "direction": 1,
        "Type" : "LGB",
        "firstTradeHour": 0,
        "lastTradeHour": 24,
        "riskMultiple": 1.00,
        "modelURL": "https://www.dropbox.com/s/mw9pg986o59e2c8/LGB_FX15m_L_Str%26DCH_s_3MM_3MM_FeatSel_2003_Model_20200307-18_14_booster.txt?dl=1",
        "model": None,
        "featureCount": 0,
        "hiddenCount": 0,
        "outFeed": "",
        "usePCA": False,
        "pcaURL": '-',
        "pca": None,
        "rawFeatures": "rawFeatures1",
        "threshold": 0.0,
        "features": None,
        "featureFilter": "Feat5_4$|Feat13_2$|Feat13_1$|Feat11_1$|Feat13_0$|Feat11_2$|Feat5_0$|Feat13_3$|Feat17_0$|Feat15_0$|Feat14_0$|Feat15_11$|Feat8_0$|Feat6_4$|Feat6_3$|Feat18_11$|Feat7_0$|Feat11_0$|Feat18_1$|Feat5_2$|Feat8_4$|Feat0_0$|Feat9_0$|Feat11_3$|Feat9_4$|Feat14_1$|Feat15_10$|Feat10_4$|Feat14_2$|Feat13_4$|Feat16_18$|Feat10_0$|Feat6_1$|Feat15_4$|Feat7_3$|Feat15_12$|Feat16_21$|Feat16_1$|Feat16_15$|Feat18_3$|Feat16_17$|Feat16_0$|Feat18_9$|Feat15_5$|Feat0_1$|Feat14_3$|Feat7_1$|Feat10_1$|Feat1_2$|Feat10_2$",
        "customColumnFilter": [],
        "dfFilterPassed": True,
        "signal": False}
        
    aiDict['L_Str_1'] = {
        "enabled": False,
        "signalRegex": "L_Str",
        "direction": 1,
        "Type" : "LGB",
        "firstTradeHour": 0,
        "lastTradeHour": 24,
        "riskMultiple": 1.00,
        "modelURL": "https://www.dropbox.com/s/kw88w6r08cpvnp4/LGB_FX15m_L_Str_3MM_FeatSel_2003_Model_20200303-19_24_booster.txt?dl=1",
        "model": None,
        "featureCount": 0,
        "hiddenCount": 0,
        "outFeed": "",
        "usePCA": False,
        "pcaURL": '-',
        "pca": None,
        "rawFeatures": "rawFeatures1",
        "threshold" : 0.0,
        "features": None,
        "featureFilter": "Feat13_1$|Feat5_4$|Feat13_2$|Feat13_3$|Feat13_0$|Feat5_0$|Feat17_0$|Feat15_0$|Feat8_0$|Feat14_0$|Feat7_0$|Feat8_4$|Feat13_4$|Feat18_1$|Feat6_1$|Feat16_18$|Feat6_4$|Feat16_0$|Feat18_11$|Feat0_0$|Feat11_2$|Feat16_17$|Feat16_16$|Feat9_4$|Feat10_4$|Feat11_1$|Feat16_15$|Feat15_4$|Feat18_10$|Feat15_5$|Feat18_5$|Feat10_2$|Feat7_4$|Feat0_2$|Feat5_2$|Feat18_9$|Feat13_5$|Feat15_16$|Feat14_2$|Feat16_1$|Feat12_2$|Feat1_2$|Feat15_6$|Feat14_1$|Feat9_0$|Feat15_11$|Feat15_15$|Feat11_0$|Feat3_2$|Feat4_1$",
        "customColumnFilter": [],
        "dfFilterPassed": True,
        "signal": False}
    
    #Long Rej
    aiDict['L_Rej_1'] = {
        "enabled": False,
        "signalRegex": "L_Rej",
        "Type" : "LGB",
        "riskMultiple": 1.00,
        "firstTradeHour": 0,
        "lastTradeHour": 24,
        "direction": 1,
        "modelURL": "https://www.dropbox.com/s/hxc3utyilghmulw/LGB_FX15m_L_Rej_3MM_FeatSel_2003_Model_20200306-20_59_booster.txt?dl=1",
        "model": None,
        "featureCount": 127,
        "hiddenCount": 127*1,
        "outFeed": "_h3",
        "usePCA": False,
        "pcaURL": '-',
        "pca": None,
        "rawFeatures": "rawFeatures1",
        "threshold": None,
        "features": None,
        "featureFilter": "Feat13_0$|Feat14_0$|Feat5_4$|Feat15_0$|Feat15_12$|Feat11_4$|Feat5_0$|Feat0_1$|Feat15_2$|Feat10_4$|Feat15_3$|Feat15_13$|Feat18_6$|Feat13_2$|Feat16_3$|Feat15_11$|Feat6_2$|Feat14_4$|Feat16_2$|Feat6_4$|Feat13_1$|Feat11_2$|Feat7_3$|Feat9_1$|Feat9_2$|Feat8_4$|Feat15_1$|Feat16_6$|Feat18_5$|Feat1_1$|Feat9_4$|Feat16_4$|Feat13_5$|Feat15_14$|Feat15_4$|Feat13_3$|Feat18_3$|Feat15_6$|Feat16_7$|Feat15_16$|Feat16_5$|Feat7_4$|Feat17_3$|Feat3_1$|Feat15_15$|Feat16_10$|Feat18_7$|Feat11_1$|Feat11_3$|Feat15_7$",
        "customColumnFilter": [],
        "dfFilterPassed": True,
        "signal": False}
    
    #Short Str
    aiDict['S_Str_1'] = {
        "enabled": False,
        "signalRegex": "S_Str",
        "direction": -1,
        "Type" : "LGB",
        "riskMultiple": 1.00,
        "firstTradeHour": 0,
        "lastTradeHour": 24,
        "modelURL": "https://www.dropbox.com/s/p31319w0r2kid57/LGB_FX15m_S_Str_3MM_FeatSel_2003_Model_20200229-18_11_booster.txt?dl=1",
        "model": None,
        "featureCount": 0,
        "hiddenCount": 0,
        "outFeed": "_h2",
        "usePCA": False,
        "pcaURL": '-',
        "pca": None,
        "rawFeatures": "rawFeatures1",
        "threshold": 0.00,
        "features": None,
        "featureFilter": "Feat13_1$|Feat5_4$|Feat13_2$|Feat13_3$|Feat13_0$|Feat5_0$|Feat17_0$|Feat15_0$|Feat8_0$|Feat14_0$|Feat7_0$|Feat8_4$|Feat13_4$|Feat18_1$|Feat6_1$|Feat16_18$|Feat6_4$|Feat16_0$|Feat18_11$|Feat0_0$|Feat11_2$|Feat16_17$|Feat16_16$|Feat9_4$|Feat10_4$|Feat11_1$|Feat16_15$|Feat15_4$|Feat18_10$|Feat15_5$|Feat18_5$|Feat10_2$|Feat7_4$|Feat0_2$|Feat5_2$|Feat18_9$|Feat13_5$|Feat15_16$|Feat14_2$|Feat16_1$|Feat12_2$|Feat1_2$|Feat15_6$|Feat14_1$|Feat9_0$|Feat15_11$|Feat15_15$|Feat11_0$|Feat3_2$|Feat4_1$",
        "customColumnFilter": [],
        "dfFilterPassed": True,
        "signal": False}
    
    def __init__(self, caller, symbol, var):
        self.CL = self.__class__
        self.algo = caller
        self.symbol = symbol
        self.security = self.algo.Securities[symbol]
        self.var = var
        '''Consolidator(s)'''
        self.consolidator = TradeBarConsolidator(self.CL.barPeriod) if symbol.SecurityType == SecurityType.Equity else QuoteBarConsolidator(self.CL.barPeriod)
        self.consolidator.DataConsolidated += self.OnDataConsolidated
        self.algo.SubscriptionManager.AddConsolidator(symbol, self.consolidator)
        #Higher Timeframe
        self.consolidator_2 = TradeBarConsolidator(self.CL.barPeriod_2) if symbol.SecurityType == SecurityType.Equity else QuoteBarConsolidator(self.CL.barPeriod_2)
        self.consolidator_2.DataConsolidated += self.OnDataConsolidated_2
        self.algo.SubscriptionManager.AddConsolidator(symbol, self.consolidator_2)
        
        '''Symbol Data''' 
        self.bars_rw = RollingWindow[IBaseDataBar](100)
        self.barTimeDifference = timedelta(0)
        self.posEnabled = True
        #self.entryEnabled = False
        self.blockOrderCheck = False
        self.fillReleaseTime = self.algo.Time - timedelta(hours=20)
        self.fromTWS = False
        self.stateUpdateList = []
        '''Indicators'''
        self.atr1 = AverageTrueRange(24)
        self.algo.RegisterIndicator(self.symbol, self.atr1, self.consolidator)
        self.atr2 = AverageTrueRange(72)
        self.algo.RegisterIndicator(self.symbol, self.atr2, self.consolidator)
        
        self.sma1 = SimpleMovingAverage (72)
        self.algo.RegisterIndicator(self.symbol, self.sma1, self.consolidator)
        self.sma2 = SimpleMovingAverage (144)
        self.algo.RegisterIndicator(self.symbol, self.sma2, self.consolidator)
        self.sma3 = SimpleMovingAverage (336)
        self.algo.RegisterIndicator(self.symbol, self.sma3, self.consolidator)

        self.dch_s = DonchianChannel(4)
        self.algo.RegisterIndicator(self.symbol, self.dch_s, self.consolidator)
        self.dch0 = DonchianChannel(4)
        self.algo.RegisterIndicator(self.symbol, self.dch0, self.consolidator)
        self.dch01 = DonchianChannel(12)
        self.algo.RegisterIndicator(self.symbol, self.dch01, self.consolidator)
        self.dch02 = DonchianChannel(24)
        self.algo.RegisterIndicator(self.symbol, self.dch02, self.consolidator)
        self.dch1 = DonchianChannel(48)
        self.algo.RegisterIndicator(self.symbol, self.dch1, self.consolidator)
        self.dch2 = DonchianChannel(96)
        self.algo.RegisterIndicator(self.symbol, self.dch2, self.consolidator)
        self.dch3 = DonchianChannel(288)
        self.algo.RegisterIndicator(self.symbol, self.dch3, self.consolidator)
        self.dch4 = DonchianChannel(480)
        self.algo.RegisterIndicator(self.symbol, self.dch4, self.consolidator)
        
        self.rsi1 = RelativeStrengthIndex(48)
        self.algo.RegisterIndicator(self.symbol, self.rsi1, self.consolidator)
        self.rsi2 = RelativeStrengthIndex(120)
        self.algo.RegisterIndicator(self.symbol, self.rsi2, self.consolidator)
        
        self.myRelativePrice = MyRelativePrice(self, self.algo, self.symbol, "RelPrice", 500, self.atr1)
        self.algo.RegisterIndicator(self.symbol, self.myRelativePrice, self.consolidator)
        
        self.zz = MyZigZag(self, self.algo, self.symbol, name='zz', period=480, atr=self.atr1, lookback=10, thresholdType=2, threshold=10)
        self.algo.RegisterIndicator(self.symbol, self.zz, self.consolidator)
        
        self.vol = MyVolatility(self, self.algo, self.symbol, name='vol', period=480, atr=self.atr1)
        self.algo.RegisterIndicator(self.symbol, self.vol, self.consolidator)
        
        #self.priceNormaliser = MyPriceNormaliser(self, self.algo, self.symbol, "Normaliser", 100)
        #self.algo.RegisterIndicator(self.symbol, self.priceNormaliser, self.consolidator)

        '''Indicators Higher Timeframe'''
        self.atr1_2 = AverageTrueRange(5)
        self.algo.RegisterIndicator(self.symbol, self.atr1_2, self.consolidator_2)
        self.sma1_2 = SimpleMovingAverage (20)
        self.algo.RegisterIndicator(self.symbol, self.sma1_2, self.consolidator_2)
        self.sma2_2 = SimpleMovingAverage (50)
        self.algo.RegisterIndicator(self.symbol, self.sma2_2, self.consolidator_2)
        self.dch1_2 = DonchianChannel(30)
        self.algo.RegisterIndicator(self.symbol, self.dch1_2, self.consolidator_2)
        self.dch2_2 = DonchianChannel(50)
        self.algo.RegisterIndicator(self.symbol, self.dch2_2, self.consolidator_2)
        
        self.rsi1_2 = RelativeStrengthIndex(10)
        self.algo.RegisterIndicator(self.symbol, self.rsi1_2, self.consolidator_2)
        self.rsi2_2 = RelativeStrengthIndex(20)
        self.algo.RegisterIndicator(self.symbol, self.rsi2_2, self.consolidator_2)
        
        self.myRelativePrice_2 = MyRelativePrice(self, self.algo, self.symbol, "RelPrice_2", 80, self.atr1_2)
        self.algo.RegisterIndicator(self.symbol, self.myRelativePrice_2, self.consolidator_2)
        
        self.zz_2 = MyZigZag(self, self.algo, self.symbol, name='zz_2', period=80, atr=self.atr1_2, lookback=6, thresholdType=2, threshold=10)
        self.algo.RegisterIndicator(self.symbol, self.zz_2, self.consolidator_2)
        
        self.vol_2 = MyVolatility(self, self.algo, self.symbol, name='vol_2', period=80, atr=self.atr1_2)
        self.algo.RegisterIndicator(self.symbol, self.vol_2, self.consolidator_2)
        
        '''Symbol State and Features'''
        self.state_sma1 = MyMAState (self, self.sma1, self.sma2)
        self.state_sma2 = MyMAState (self, self.sma2, self.sma3)
        self.state_sma3 = MyMAState (self, self.sma3, self.sma1_2)
        self.state_sma1_2 = MyMAState (self, self.sma1_2, self.sma2_2)
        self.state_sma2_2 = MyMAState (self, self.sma2_2)

        self.state_dch_s = MyDCHState (self, self.dch_s, self.dch2, name='s')
        self.state_dch0 = MyDCHState (self, self.dch0, self.dch1, name='0')
        self.state_dch1 = MyDCHState (self, self.dch1, self.dch2, name='1')
        self.state_dch2 = MyDCHState (self, self.dch2, self.dch3, name='2')
        self.state_dch3 = MyDCHState (self, self.dch3, self.dch4, name='3')
        self.state_dch4 = MyDCHState (self, self.dch4, self.dch1_2, name='4')
        self.state_dch1_2 = MyDCHState (self, self.dch1_2, self.dch2_2, name='1_2')
        self.state_dch2_2 = MyDCHState (self, self.dch2_2, name='2_2')

        '''Signals'''
        self.barStrength1 = MyBarStrength(self, self.algo, self.symbol, name='barStrength1', period=10, atr=self.atr1, lookbackLong=2, lookbackShort=2, \
                priceActionMinATRLong=1.5, priceActionMaxATRLong=2.5, priceActionMinATRShort=1.5, priceActionMaxATRShort=2.5, referenceTypeLong='Close', referenceTypeShort='Close')
        self.algo.RegisterIndicator(self.symbol, self.barStrength1, self.consolidator)
        
        self.barRejection1 = MyBarRejection(self, self.algo, self.symbol, name='barRejection1', period=10, atr=self.atr1, lookbackLong=3, lookbackShort=3, \
               rejectionPriceTravelLong=1.50, rejectionPriceTravelShort=1.50, rejectionPriceRangeLong=0.35, rejectionPriceRangeShort=0.35, referenceTypeLong='Close', referenceTypeShort='Close')
        self.algo.RegisterIndicator(self.symbol, self.barRejection1, self.consolidator)
        
        '''Signals and Events string'''
        self.signals = ''
        self.events = ''

        '''Disable Bars'''
        self.longDisabledBars = 0
        self.shortDisabledBars = 0
        self.longSimDisabledBars = 0
        self.shortSimDisabledBars = 0

        '''Set up AI models'''
        self.signalDisabledBars = {}
        for aiKey, aiObj in self.CL.aiDict.items():
            if self.CL.loadAI and aiObj["enabled"] and aiObj["model"]==None:
                #Loading PCA Preprocessor if set
                if aiObj["usePCA"]: 
                    aiObj["pca"] = hp3.MyModelLoader.LoadModelPickled(self, aiObj["pcaURL"])
                    self.algo.Debug(f' PCA ({aiKey}) LOADED from url: {aiObj["pcaURL"]}')
                #Loading MODEL
                if aiObj["Type"] == "PT":
                    aiObj["model"] = NN_1(aiObj["featureCount"], hiddenSize=aiObj["hiddenCount"], outFeed=aiObj["outFeed"], softmaxout=False, outputs=2).to('cpu')
                    aiObj["model"] = hp3.MyModelLoader.LoadModelTorch(self, aiObj["modelURL"], existingmodel=aiObj["model"])
                    self.algo.Debug(f' Torch MODEL ({aiKey}/{self.CL.strategyCode}) LOADED from url: {aiObj["modelURL"]}')
                elif aiObj["Type"] == "SK":
                    aiObj["model"] = hp3.MyModelLoader.LoadModelPickled(self, aiObj["modelURL"])
                    self.algo.Debug(f' SKLearn MODEL ({aiKey}/{self.CL.strategyCode}) LOADED from url: {aiObj["modelURL"]}')
                elif aiObj["Type"] == "LGB":
                    #aiObj["model"] = hp3.MyModelLoader.LoadModelLGBtxt(self, aiObj["modelURL"])
                    aiObj["model"] = hp3.MyModelLoader.LoadModelLGB(self, aiObj["modelURL"])
                    self.algo.Debug(f' LGB MODEL ({aiKey}/{self.CL.strategyCode}) LOADED from url: {aiObj["modelURL"]}')
            self.signalDisabledBars[aiKey] = 0
        
        #Initialize signalDisabledBarsSim 
        self.signalDisabledBarsSim = {}
        for simKey, simObj in self.CL.simDict.items():
            self.signalDisabledBarsSim[simKey] = 0

        #SET FILES TO BE SAVED AT THE END OF SIMULATION
        if self.CL.simulate:
            if self.CL.saveFilesFramework:        
                #Framework specific filelist to be saved
                MySIMPosition.saveFiles.append([self.algo.myPositionManager.CL.file, '_pm'])
                MySIMPosition.saveFiles.append([self.algo.myPositionManagerB.CL.file, '_pmB'])
                MySIMPosition.saveFiles.append([self.algo.myVaR.CL.file, '_var'])
                MySIMPosition.saveFiles.append([self.algo.myHelpers.CL.file, '_hp'])
            if self.CL.saveFilesSim: 
                #Simulation Specific Files    
                MySIMPosition.saveFiles.append([self.algo.__class__.file, '_main'])
                MySIMPosition.saveFiles.append([self.CL.file, f'_{self.CL.strategyCodeOriginal}'])
                MySIMPosition.saveFiles.append([MySIMPosition.file, '_sim'])
                MySIMPosition.saveFiles.append([MyRelativePrice.file, '_ta1'])
                MySIMPosition.saveFiles.append([MyBarStrength.file, '_taB1'])
 
    '''
    On Consolidated =======================
    '''
    def OnDataConsolidated(self, sender, bar):
        #self.algo.Debug(str(self.algo.Time) + " " + str(bar))
        if False and self.symbol.Value == "XOM" and self.algo.StartDate < self.algo.Time: self.algo.MyDebug(" OnDataConsolidated {} Open:{} High:{} Low:{} Close:{}".format(str(self.symbol), str(bar.Open), str(bar.High), str(bar.Low), str(bar.Close)))
        self.barTimeDifference = self.algo.Time - bar.EndTime

        '''Update
        '''
        self.bars_rw.Add(bar)
        for i in self.stateUpdateList:
            i.Update(self, bar)

        if not self.CL.enabled or self.algo.IsWarmingUp or not self.posEnabled and not self.IsReady() or not self.WasJustUpdated(self.algo.Time):
            self.signals = ''
            self.events = ''
            return

        '''VOLATILITY CUTOFF
        '''
        volRef = self.algo.mySymbolDict[self.algo.benchmarkSymbol].vol
        #volRef = self.vol
        volatility = volRef.atrVolatility[0]
        self.CL._totalATRpct = volatility/100 #Dimension is inherited from eq2_2
        volatilityNorm = volRef.atrNormVolatility[0]
        volatilityChange = volRef.atrVolatility[0]-volRef.atrVolatility[10]
        
        #This is not used
        volatilityLimitLong = 1.0 
        volatilityLimitShort = 1.15
        
        #CUTOFF
        volatilityCutOffLong = 1.2 
        volatilityCutOffShort = 1.0
        if False and volatility > volatilityCutOffLong and self.algo.Portfolio[self.symbol].Quantity > 0:
            self.algo.myPositionManagerB.LiquidatePosition(self.symbol, "VolCutOff", "VolCutOff")
        if False and volatility < volatilityCutOffShort and self.algo.Portfolio[self.symbol].Quantity < 0:
            self.algo.myPositionManagerB.LiquidatePosition(self.symbol, "VolCutOff", "VolCutOff")
       
        '''TRADE SIGNALS
        '''
        loadFeatures1, loadFeatures2 = False, False
        for aiKey, aiObj in self.CL.aiDict.items():
            signalRegex = aiObj["signalRegex"] if "signalRegex" in aiObj.keys() else aiKey
            aiObj["signal"] = aiObj["enabled"] and re.search(signalRegex, self.signals) and aiObj["firstTradeHour"] <= self.algo.Time.hour and self.algo.Time.hour <= aiObj["lastTradeHour"]
            if aiObj["signal"]:
                #self.algo.MyDebug(f' Signal:{aiKey} {self.symbol}')
                self.algo.signalsTotal+=1
                if aiObj["rawFeatures"]=="rawFeatures1": loadFeatures1 = True
                if aiObj["rawFeatures"]=="rawFeatures2": loadFeatures2 = True
        
        '''SIMULATION SIGNALS
        '''
        longTriggerSim, shortTriggerSim = False, False
        #simObj : [direction, disableBars, enabled]
        for simKey, simObj in self.CL.simDict.items():
            self.signalDisabledBarsSim[simKey] = max(0, self.signalDisabledBarsSim[simKey]-1)
            if self.CL.simulate and simObj[2] and self.signalDisabledBarsSim[simKey]==0 and re.search(simKey, self.signals):
                self.signalDisabledBarsSim[simKey] = simObj[1]
                if   simObj[0]== 1:  longTriggerSim = True
                elif simObj[0]==-1: shortTriggerSim = True

        '''FEATURES
        '''
        if loadFeatures1 or (longTriggerSim or shortTriggerSim):
            self.rawFeatures1 = [ self.state_sma1.FeatureExtractor(Type=11), self.state_sma2.FeatureExtractor(Type=11), self.state_sma3.FeatureExtractor(Type=11), self.state_sma1_2.FeatureExtractor(Type=11), self.state_sma2_2.FeatureExtractor(Type=11), \
                            self.state_dch0.FeatureExtractor(Type=6), self.state_dch1.FeatureExtractor(Type=6), self.state_dch2.FeatureExtractor(Type=6), self.state_dch3.FeatureExtractor(Type=6), self.state_dch4.FeatureExtractor(Type=6), self.state_dch1_2.FeatureExtractor(Type=6), \
                            self.myRelativePrice.FeatureExtractor(Type=1, normalizationType=1, lookbacklist=[8,24,48,72,144], featureMask=[0,0,1,0]), self.myRelativePrice_2.FeatureExtractor(Type=1, normalizationType=1, lookbacklist=[10,20,30], featureMask=[0,0,1,0]), \
                            self.vol.FeatureExtractor(Type=51, lookbacklist=[24,48,72,144]), self.vol_2.FeatureExtractor(Type=51, lookbacklist=[10,20,30], avgPeriod=70), \
                            self.zz.FeatureExtractor(listLen=20, Type=11), self.zz.FeatureExtractor(listLen=20, Type=21), self.zz_2.FeatureExtractor(listLen=6, Type=11), self.zz_2.FeatureExtractor(listLen=6, Type=21), \
                            [self.rsi1.Current.Value, self.rsi2.Current.Value, self.rsi1_2.Current.Value, self.rsi2_2.Current.Value] ]
        
        if loadFeatures2 or (longTriggerSim or shortTriggerSim):    
            self.rawFeatures2 = []
        
        for aiObj in self.CL.aiDict.values():
            if aiObj["enabled"] and aiObj["signal"]: 
                featureFilter = aiObj["featureFilter"]
                myFeatures = self.algo.myHelpers.UnpackFeatures(getattr(self, aiObj["rawFeatures"]),  featureType=1, featureRegex=featureFilter, reshapeTuple=None)
                customColumnFilters = aiObj["customColumnFilter"]
                aiObj["dfFilterPassed"] = self.algo.myHelpers.FeatureCustomColumnFilter(myFeatures, customColumnFilters=customColumnFilters) if len(customColumnFilters)!=0 else True
                myFeatures = myFeatures.values
                if aiObj["usePCA"]:
                    myFeatures = aiObj["pca"].transform(myFeatures)
                if aiObj["Type"] == "PT":
                    myFeatures = torch.from_numpy(myFeatures.reshape(-1, aiObj["featureCount"])).float().to('cpu')
                aiObj["features"] = myFeatures

        '''SIGNAL FILTERING
        '''
        longTrigger, shortTrigger = False, False
        longRiskMultiple, shortRiskMultiple = 1.00, 1.00
        for aiKey, aiObj in self.CL.aiDict.items():
            #This is for Stat5 signal compatibility
            self.signalDisabledBars[aiKey] = max(self.signalDisabledBars[aiKey]-1,0)
            #Long
            if self.CL.loadAI and not (longTrigger or shortTrigger) and aiObj["enabled"] and aiObj["signal"] and aiObj["dfFilterPassed"] and self.signalDisabledBars[aiKey]==0:
                #MODEL INFERENCE
                if aiObj["Type"] == "PT":
                    aiObj["model"].eval()
                    trigger = aiObj["model"].Predict(aiObj["features"])
                elif aiObj["Type"] == "SK" or aiObj["Type"] == "LGB":
                    if "threshold" in aiObj.keys() and aiObj["threshold"]!=None:
                        trigger = np.where(aiObj["model"].predict(aiObj["features"])>aiObj["threshold"], 1, 0)
                    else:
                        trigger = aiObj["model"].predict(aiObj["features"])
                #Setting Trade Trigger Flags
                if trigger and aiObj["direction"]==1:
                    longTrigger = True
                    longRiskMultiple = min(1.00,max(aiObj["riskMultiple"],0.00))
                    self.CL.strategyCode = self.CL.strategyCode + "|" + aiKey
                elif trigger and aiObj["direction"]==-1:
                    shortTrigger = True
                    shortRiskMultiple = min(1.00,max(aiObj["riskMultiple"],0.00))
                    self.CL.strategyCode = self.CL.strategyCode + "|" + aiKey
            if aiObj["signal"] and self.signalDisabledBars[aiKey]==0: self.signalDisabledBars[aiKey] = 8

        '''POSITION ENTRY/FLIP (Flip Enabled is checked in EnterPosition_2)
        '''
        self.longDisabledBars = max(self.longDisabledBars-1,0)
        self.shortDisabledBars = max(self.shortDisabledBars-1,0)
        #---LONG POSITION
        if self.posEnabled and self.CL.enableLong and self.longDisabledBars==0 and longTrigger:
            self.algo.myPositionManager.EnterPosition_2(self.symbol, 1, myRiskMultiple=longRiskMultiple)
            if False: self.algo.MyDebug(f' Long Trade: {self.symbol}')
            self.longDisabledBars=3
        #---SHORT POSITION
        elif self.posEnabled and self.CL.enableShort and self.shortDisabledBars==0 and shortTrigger:
            self.algo.myPositionManager.EnterPosition_2(self.symbol, -1, myRiskMultiple=shortRiskMultiple)
            if False: self.algo.MyDebug(f' Short Trade: {self.symbol}')
            self.shortDisabledBars=3

        '''EXIT SIGNALS AND EXIT
        '''
        for exitSignal, exitDirection in self.CL.exitSignalDict.items():
            position = self.algo.Portfolio[self.symbol].Quantity
            if position!=0 and re.search(exitSignal, self.signals):
                if exitDirection == +1 and position < 0 and not longTrigger:
                    #Exit Short
                    self.algo.myPositionManagerB.LiquidatePosition(self.symbol, "L_ExitSignal", "L_ExitSignal")
                    if False: self.algo.MyDebug(f' Exit from Short: {self.symbol}, signal:{exitSignal}')
                elif exitDirection == -1 and position > 0 and not shortTrigger:
                    #Exit Long
                    self.algo.myPositionManagerB.LiquidatePosition(self.symbol, "S_ExitSignal", "S_ExitSignal")
                    if False: self.algo.MyDebug(f' Exit from Long: {self.symbol}, signal:{exitSignal}')
                    
        '''SIMULATION CALL
        '''
        debugSim  = False
        if longTriggerSim or shortTriggerSim: 
            myFeatures = self.rawFeatures1
        self.longSimDisabledBars=max(self.longSimDisabledBars-1,0)
        self.shortSimDisabledBars=max(self.shortSimDisabledBars-1,0)
        simTradeTypes=[0,2,3,4,7,8,10] #[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] 
        simMinMaxTypes=[0,1] #[0,1,2,3]
        lastEntryDate = datetime(year = 3019, month = 10, day = 7)
        if self.CL.simulate and self.posEnabled and longTriggerSim and (self.algo.Time<self.algo.simEndDate or self.algo.simEndDate==None):
            if debugSim: self.algo.MyDebug(f' {self.symbol} Sim call Long: {self.signals}')
            sim = MySIMPosition(self, direction=1, timestamp=self.algo.Time.strftime("%Y%m%d %H:%M"), signal=self.signals, features=myFeatures, simTradeTypes=simTradeTypes, simMinMaxTypes=simMinMaxTypes)
            #use 0 if signalspecific
            self.longSimDisabledBars=0
        if self.CL.simulate and self.posEnabled and shortTriggerSim and (self.algo.Time<self.algo.simEndDate or self.algo.simEndDate==None):
            if debugSim: self.algo.MyDebug(f' {self.symbol} Sim call Short: {self.signals}')
            sim = MySIMPosition(self, direction=-1, timestamp=self.algo.Time.strftime("%Y%m%d %H:%M"), signal=self.signals, features=myFeatures, simTradeTypes=simTradeTypes, simMinMaxTypes=simMinMaxTypes) 
            #use 0 if signalspecific
            self.shortSimDisabledBars=0
        
        '''FEATURE DEBUG
        '''
        debugTicker = 'EURUSD'
        debugSignal = 'L_Str'
        if False and self.symbol == self.algo.Securities[debugTicker].Symbol and re.search(debugSignal, self.signals):
            if True or ( datetime(2019, 1, 1, 12, 00) < self.algo.Time and self.algo.Time <= datetime(2019, 10, 11, 13, 00) ):
                df_debug = self.algo.myHelpers.UnpackFeatures(self.rawFeatures1,  featureType=1, featureRegex='Feat', reshapeTuple=None)
                self.PrintFeatures(df_debug, negativeOnly=False)
                pass
        
        '''RESET strategyCode, Signals and Events FOR THE NEXT BAR
        '''
        self.CL.strategyCode = self.CL.strategyCodeOriginal 
        #if False and re.search('ABCD_|DCH_', self.signals): self.algo.MyDebug(f' {self.symbol} self.signals before reset: {self.signals}')
        #if False and self.zz.patterns.doubleTop: self.algo.MyDebug(f' {self.symbol} self.signals before reset: {self.signals}')
        self.signals = ''
        self.events = ''
        return
    
    '''
    On Consolidated HIGHER TIMEFRAME =======================
    '''
    def OnDataConsolidated_2(self, sender, bar):
        pass
        
    '''
    UPDATE STATUS =======================
    '''
    def IsReady(self):
        indiacatorsReady = self.dch2_2.IsReady
        if False and indiacatorsReady:
            self.algo.MyDebug(" Consolidation Is Ready " + str(self.bars_rw.IsReady) + str(self.dch2.IsReady) + str(self.WasJustUpdated(self.algo.Time)))
        return indiacatorsReady
    
    def WasJustUpdated(self, currentTime):
        return self.bars_rw.Count > 0 and (currentTime - self.barTimeDifference - self.bars_rw[0].EndTime) < timedelta(seconds=10) \
                and (currentTime - self.barTimeDifference - self.bars_rw[0].EndTime).total_seconds() > timedelta(seconds=-10).total_seconds()
    
    '''
    OTHER =======================
    '''
    def PrintFeatures(self, df, negativeOnly=False):
        df = df.filter(regex = 'Feat')
        for col in df:
            item = df.loc[df.index[0], col]
            if not negativeOnly or (negativeOnly and item<0):
                self.algo.MyDebug(f' {col}: {item}')
            
    def TestPredict(self, model, featureNo):
        device = 'cpu'
        for i in range (10):
            np.random.seed(i)
            x_sample = np.random.rand(1, featureNo)
            x_sample = torch.from_numpy(x_sample.reshape(-1, featureNo)).float().to(device)
            y_hat = model.Predict2(x_sample)
            self.algo.MyDebug(f'{y_hat}')
 
'''
TORCH MODEL(S)
'''
class NN_1(nn.Module):
   def __init__(self, features, hiddenSize=None, outFeed='_h3', outputs=2, dropoutRateIn=0.0, dropoutRate=0.0, dropoutRate2=None, softmaxout=False, bn_momentum=0.1):
       super(NN_1, self).__init__()
       self.outputs = outputs
       self.features = features
       self.softmaxout = softmaxout
       if hiddenSize!=None:
           self.hiddenSize = hiddenSize
       else:
           self.hiddenSize = round(1.0*features)
       #https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/ , https://blog.paperspace.com/busting-the-myths-about-batch-normalization/
       #https://pytorch.org/docs/stable/nn.html?highlight=nn%20batchnorm1d#torch.nn.BatchNorm1d
       self.bn0 = nn.BatchNorm1d(num_features=features, momentum=bn_momentum)
       self.layer_In = torch.nn.Linear(features, self.hiddenSize)
       self.bn01 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
       self.layer_h1 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.bn1 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
       self.layer_h2 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.bn2 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
       self.layer_h3 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.bn3 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
       self.layer_h4 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.bn4 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
       self.layer_h5 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.bn5 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
       self.layer_h6 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.bn6 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
       self.layer_h7 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.bn7 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
      #  self.layer_h8 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
      #  self.bn8 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
       self.layer_Out = torch.nn.Linear(self.hiddenSize, self.outputs)
       #https://towardsdatascience.com/simplified-math-behind-dropout-in-deep-learning-6d50f3f47275
       self.L_out = {}
       self.outFeed = outFeed 
       #https://towardsdatascience.com/simplified-math-behind-dropout-in-deep-learning-6d50f3f47275
       self.dropoutRateIn = dropoutRateIn
       self.dropoutRate = dropoutRate
       self.dropoutRate2 = self.dropoutRate if dropoutRate2==None else dropoutRate2
       #https://mlfromscratch.com/activation-functions-explained/ 
       self.MyActivation = [F.relu, F.leaky_relu_, F.selu][2]
       self.useOutBn = True
       if self.useOutBn: self.bn_out = nn.BatchNorm1d(num_features=self.outputs, momentum=bn_momentum)
                     
   def forward(self, x):
       self.L_out['_x'] =  F.dropout(self.bn0(x), p=self.dropoutRateIn, training=self.training)
       #https://datascience.st  ackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks

       self.L_out['_in'] =  F.dropout(self.bn01(self.MyActivation(self.layer_In(self.L_out['_x']))),  p=self.dropoutRate, training=self.training)
       self.L_out['_h1'] =  F.dropout(self.bn1(self.MyActivation(self.layer_h1(self.L_out['_in']))),  p=self.dropoutRate2, training=self.training)
       self.L_out['_h2'] =  F.dropout(self.bn2(self.MyActivation(self.layer_h2(self.L_out['_h1']))),  p=self.dropoutRate2, training=self.training)
       self.L_out['_h3'] =  F.dropout(self.bn3(self.MyActivation(self.layer_h3(self.L_out['_h2']))),  p=self.dropoutRate2, training=self.training)
       self.L_out['_h4'] =  F.dropout(self.bn4(self.MyActivation(self.layer_h4(self.L_out['_h3']))),  p=self.dropoutRate2, training=self.training)
       #self.L_out['_h5'] =  F.dropout(self.bn5(self.MyActivation(self.layer_h5(self.L_out['_h4']))),  p=self.dropoutRate2, training=self.training)
       #self.L_out['_h6'] =  F.dropout(self.bn6(self.MyActivation(self.layer_h6(self.L_out['_h5']))),  p=self.dropoutRate2, training=self.training)
       #self.L_out['_h7'] =  F.dropout(self.bn7(self.MyActivation(self.layer_h7(self.L_out['_h6']))),  p=self.dropoutRate2, training=self.training)
       #self.L_out['_h8'] =  F.dropout(self.bn8(self.MyActivation(self.layer_h8(self.L_out['_h7']))),  p=self.dropoutRate2, training=self.training)
       if self.useOutBn:
         self.L_out['_out'] = self.bn_out(self.MyActivation(self.layer_Out(self.L_out[self.outFeed])))
       else:
         self.L_out['_out'] = self.MyActivation(self.layer_Out(self.L_out[self.outFeed]))
       #https://towardsdatascience.com/complete-guide-of-activation-functions-34076e95d044
       #https://medium.com/@himanshuxd/activation-functions-sigmoid-relu-leaky-relu-and-softmax-basics-for-neural-networks-and-deep-8d9c70eed91e
       if self.softmaxout:
           self.L_out['_out_sm'] = F.softmax(self.L_out['_out'].float(), dim=1)
           return self.L_out['_out_sm']
       else:
           return self.L_out['_out']
   
   def Predict(self, x):
       self.eval()
       prediction = self.forward(x)
       #prediction = prediction.data.numpy()
       prediction = np.argmax(prediction.data.numpy(), axis=1) #armax returns a tuple
       if len(prediction)==1:
           return prediction[0]
       return prediction
   
   def Predict2(self, x):
       self.eval()
       prediction = self.forward(x).data.numpy()
       return prediction