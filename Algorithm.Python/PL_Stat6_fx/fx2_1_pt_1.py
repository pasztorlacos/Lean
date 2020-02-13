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

class Fx2_1_pt_1():
    file = __file__
    '''
    Strategy Implementation
    '''
    '''STRATEGY SETTINGS'''
    enabled = True
    manageOrderConsistency = True
    simulate = False
    saveFilesFramework = False
    saveFilesSim = False
    strategyCodeOriginal = "fx2_1_pt_1"
    strategyCode = strategyCodeOriginal #used by order tags and debug
    isEquity = False
    customFillModel = 1
    customSlippageModel = 1
    customFeeModel = 0
    customBuyingPowerModel = 0
    #Resolution
    resolutionMinutes = 15
    maxWarmUpPeriod = 710
    barPeriod =  timedelta(minutes=resolutionMinutes) #TimeSpan.FromMinutes(resolutionMinutes)
    warmupcalendardays = round(7/5*maxWarmUpPeriod/(7*(60/resolutionMinutes))) if isEquity else round(7/5*maxWarmUpPeriod/(24*(60/resolutionMinutes)))
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
    enableFlipLong  = True
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
    stopPlacerLong  = 7    #0)dch0 1)dch1 2)dch2) 3)dch3 4)minStopATR 5)bars_rw[0] 6)bars_rw[0-1] 7)dc01 8)dch02 else: dch1
    stopPlacerShort = 7
    targetPlacerLong  = 3    #1)max(dch2,minPayoff) 2)max(dch3,minPayoff)  else: minPayoff 
    targetPlacerShort = 3    
    minPayOffLong  = 2.50
    minPayOffShort = 2.50
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
    
    #PiData FX
    myTickers = ["AUDJPY", "AUDUSD", "CHFJPY", "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURUSD", "GBPCHF", "GBPJPY", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
    myTickers = ["EURUSD", "GBPUSD", "EURGBP"]
    myTickers = ["EURUSD"]
    
    #My Selection
    #myTickers = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD", "USDCNH", "EURJPY", "EURSEK", "EURNOK", "USDMXN", "USDZAR", "USDSEK", "USDNOK", "EURHUF", "USDHUF"]
    #myTickers = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD", "USDCNH", "EURJPY"]
    #myTickers = ["EURUSD"]
    
    #Simulation Signals [direction, disableBars, disableCount]
    simDict = {"L_Str": [1,8,0], "L_Rej": [1,8,0], "S_Str": [-1,8,0], "S_Rej": [-1,8,0]}
    
    #AI ----
    loadAI = False
    aiDict = {}
    
    aiDict['L_Str'] = {
        "enabled": True,
        "folder": "20200205-20_35",
        "firstTradeHour": 0,
        "lastTradeHour": 14,
        "riskMultiple": 0.75,
        "direction": 1,
        "modelURL": "https://www.dropbox.com/s/e12iysgae1979wy/NNpt_L_2000_FeatAll_Str_3MM_c_Model_20200205-20_35.txt?dl=1",
        "model": None,
        "featureCount": 77,
        "hiddenCount": 77*1,
        "outFeed": "_h3",
        "usePCA": False,
        "pcaURL": '-',
        "pca": None,
        "rawFeatures": "rawFeatures2",
        "features": None,
        "featureFilter": "Feat",
        "customColumnFilter": [],
        "dfFilterPassed": True,
        "signal": False}
    
    #Long Rej
    aiDict['L_Rej'] = {
        "enabled": True,
        "folder": "20200206-09_41",
        "riskMultiple": 1.00,
        "firstTradeHour": 0,
        "lastTradeHour": 14,
        "direction": 1,
        "modelURL": "https://www.dropbox.com/s/9wrwi8rsgzb5odm/NNpt_L_2000_FeatAll_Rej_3MM_c_Model_20200206-09_41.txt?dl=1",
        "model": None,
        "featureCount": 77,
        "hiddenCount": 77*1,
        "outFeed": "_h1",
        "usePCA": False,
        "pcaURL": '-',
        "pca": None,
        "rawFeatures": "rawFeatures2",
        "features": None,
        "featureFilter": "Feat",
        "customColumnFilter": [],
        "dfFilterPassed": True,
        "signal": False}
    
    #Short Str
    aiDict['S_Str'] = {
        "enabled": True,
        "folder": "20200203-17_17",
        "riskMultiple": 1.00,
        "firstTradeHour": 0,
        "lastTradeHour": 14,
        "direction": -1,
        "modelURL": "https://www.dropbox.com/s/fgshv6a0j0805jz/NNpt_S_2000_FeatAll_Str_3MM_c_Model_20200203-17_17.txt?dl=1",
        "model": None,
        "featureCount": 77,
        "hiddenCount": 77*1,
        "outFeed": "_h2",
        "usePCA": False,
        "pcaURL": '-',
        "pca": None,
        "rawFeatures": "rawFeatures2",
        "features": None,
        "featureFilter": "Feat",
        "customColumnFilter": [],
        "dfFilterPassed": True,
        "signal": False}
    
    #Short Rej
    aiDict['S_Rej'] = {
        "enabled": True,
        "folder": "20200203-09_14",
        "riskMultiple": 1.00,
        "firstTradeHour": 0,
        "lastTradeHour": 14,
        "direction": -1,
        "modelURL": "https://www.dropbox.com/s/ljbdggpaf3ac4gk/NNpt_S_2000_FeatAll_Rej_5MM_c_Model_20200203-09_14.txt?dl=1",
        "model": None,
        "featureCount": 77,
        "hiddenCount": 77*1,
        "outFeed": "_h2",
        "usePCA": False,
        "pcaURL": '-',
        "pca": None,
        "rawFeatures": "rawFeatures2",
        "features": None,
        "featureFilter": "Feat",
        "customColumnFilter": [],
        "dfFilterPassed": True,
        "signal": False}
    
    def __init__(self, caller, symbol, var):
        self.CL = self.__class__
        self.algo = caller
        self.symbol = symbol
        self.security = self.algo.Securities[symbol]
        self.var = var
        '''Consolidator'''
        self.consolidator = TradeBarConsolidator(self.CL.barPeriod) if symbol.SecurityType == SecurityType.Equity else QuoteBarConsolidator(self.CL.barPeriod)
        #self.consolidator = TradeBarConsolidator(1) if symbol.SecurityType == SecurityType.Equity else QuoteBarConsolidator(1)
        self.consolidator.DataConsolidated += self.OnDataConsolidated
        self.algo.SubscriptionManager.AddConsolidator(symbol, self.consolidator)
        '''Symbol Data''' 
        self.bars_rw = RollingWindow[IBaseDataBar](100)
        self.barTimeDifference = timedelta(0)
        self.posEnabled = True
        self.entryEnabled = False
        self.blockOrderCheck = False
        self.fillReleaseTime = self.algo.Time - timedelta(hours=20)
        self.fromTWS = False
        self.stateUpdateList = []
        '''Indicators'''
        self.sma1 = SimpleMovingAverage (70)
        self.algo.RegisterIndicator(self.symbol, self.sma1, self.consolidator)
        self.sma2 = SimpleMovingAverage (140)
        self.algo.RegisterIndicator(self.symbol, self.sma2, self.consolidator)
        self.sma3 = SimpleMovingAverage (280)
        self.algo.RegisterIndicator(self.symbol, self.sma3, self.consolidator)
        self.sma4 = SimpleMovingAverage (420)
        self.algo.RegisterIndicator(self.symbol, self.sma4, self.consolidator)
        self.atr1 = AverageTrueRange(15)
        self.algo.RegisterIndicator(self.symbol, self.atr1, self.consolidator)
        self.atr2 = AverageTrueRange(70)
        self.algo.RegisterIndicator(self.symbol, self.atr2, self.consolidator)
        
        self.dch_s = DonchianChannel(7)
        self.algo.RegisterIndicator(self.symbol, self.dch_s, self.consolidator)
        self.dch0 = DonchianChannel(3)
        self.algo.RegisterIndicator(self.symbol, self.dch0, self.consolidator)
        self.dch01 = DonchianChannel(7)
        self.algo.RegisterIndicator(self.symbol, self.dch01, self.consolidator)
        self.dch02 = DonchianChannel(20)
        self.algo.RegisterIndicator(self.symbol, self.dch02, self.consolidator)
        self.dch1 = DonchianChannel(35)
        self.algo.RegisterIndicator(self.symbol, self.dch1, self.consolidator)
        self.dch2 = DonchianChannel(70)
        self.algo.RegisterIndicator(self.symbol, self.dch2, self.consolidator)
        self.dch3 = DonchianChannel(280)
        self.algo.RegisterIndicator(self.symbol, self.dch3, self.consolidator)
        self.dch4 = DonchianChannel(420)
        self.algo.RegisterIndicator(self.symbol, self.dch4, self.consolidator)
        
        self.rsi1 = RelativeStrengthIndex(35)
        self.algo.RegisterIndicator(self.symbol, self.rsi1, self.consolidator)
        
        '''Indicators B'''
        self.smaB1 = SimpleMovingAverage (70)
        self.algo.RegisterIndicator(self.symbol, self.sma1, Resolution.Daily)
        
        '''Symbol State and Features'''
        self.longSignalDisabledBars = 0
        self.shortSignalDisabledBars = 0
        self.longDisabledBars = 0
        self.shortDisabledBars = 0
        self.longSimDisabledBars = 0
        self.shortSimDisabledBars = 0

        self.state_sma1 = MyMAState (self, self.sma1, self.sma2)
        self.state_sma2 = MyMAState (self, self.sma2, self.sma3)
        self.state_sma3 = MyMAState (self, self.sma3, self.sma4)
        self.state_sma4 = MyMAState (self, self.sma4)

        self.state_dch_s = MyDCHState (self, self.dch_s, self.dch2)
        self.state_dch0 = MyDCHState (self, self.dch0, self.dch1)
        self.state_dch1 = MyDCHState (self, self.dch1, self.dch2)
        self.state_dch2 = MyDCHState (self, self.dch2, self.dch3)
        self.state_dch3 = MyDCHState (self, self.dch3, self.dch4)
        self.state_dch4 = MyDCHState (self, self.dch4)

        self.myRelativePrice = MyRelativePrice(self, self.algo, self.symbol, "RelPrice", 100, self.atr1)
        self.algo.RegisterIndicator(self.symbol, self.myRelativePrice, self.consolidator)
        
        self.zz1 = MyZigZag(self, self.algo, self.symbol, name='zz1', period=100, atr=self.atr1)
        self.algo.RegisterIndicator(self.symbol, self.zz1, self.consolidator)
        
        self.vol = MyVolatility(self, self.algo, self.symbol, name='vol_', period=110, atr=self.atr1)
        self.algo.RegisterIndicator(self.symbol, self.vol, self.consolidator)

        #self.priceNormaliser = MyPriceNormaliser(self, self.algo, self.symbol, "Normaliser", 100)
        #self.algo.RegisterIndicator(self.symbol, self.priceNormaliser, self.consolidator)

        self.barStrength1 = MyBarStrength(self, self.algo, self.symbol, name='barStrength1', period=10, atr=self.atr1, lookbackLong=2, lookbackShort=2, \
                priceActionMinATRLong=1.5, priceActionMaxATRLong=2.5, priceActionMinATRShort=1.25, priceActionMaxATRShort=2.25, referenceTypeLong='Close', referenceTypeShort='Close')
        self.algo.RegisterIndicator(self.symbol, self.barStrength1, self.consolidator)
        
        self.barRejection1 = MyBarRejection(self, self.algo, self.symbol, name='barRejection1', period=10, atr=self.atr1, lookbackLong=3, lookbackShort=3, \
               rejectionPriceTravelLong=2.0, rejectionPriceTravelShort=2.0, rejectionPriceRangeLong=1.0, rejectionPriceRangeShort=1.0, referenceTypeLong='Close', referenceTypeShort='Close')
        self.algo.RegisterIndicator(self.symbol, self.barRejection1, self.consolidator)
        
        '''Signals and Events string'''
        self.signals = ''
        self.events = ''
        
        '''Set up AI models'''
        self.signalDisabledBars = {}
        for aiKey, aiObj in self.CL.aiDict.items():
            if self.CL.loadAI and aiObj["enabled"] and aiObj["model"]==None:
                if aiObj["usePCA"]: 
                    aiObj["pca"] = hp3.MyModelLoader.LoadModelPickled(self, aiObj["pcaURL"])
                    self.algo.Debug(f' PCA ({aiKey}) LOADED from url: {aiObj["pcaURL"]}')
                aiObj["model"] = NN_1(aiObj["featureCount"], hiddenSize=aiObj["hiddenCount"], outFeed=aiObj["outFeed"], softmaxout=False, outputs=2).to('cpu')
                aiObj["model"] = hp3.MyModelLoader.LoadModelTorch(self, aiObj["modelURL"], existingmodel=aiObj["model"])
                self.algo.Debug(f' Torch MODEL ({aiKey}) LOADED from url: {aiObj["modelURL"]}')
            self.signalDisabledBars[aiKey] = 0

        #SET FILES TO BE SAVED AT THE END OF SIMULATION
        if self.CL.simulate:
            if self.CL.saveFilesFramework:        
                #Framework specific filelist to be saved
                MySIMPosition.saveFiles.append([self.algo.myPositionManager.CL.file, 'pmFile'])
                MySIMPosition.saveFiles.append([self.algo.myPositionManagerB.CL.file, 'pmBFile'])
                MySIMPosition.saveFiles.append([self.algo.myVaR.CL.file, 'varFile'])
                MySIMPosition.saveFiles.append([self.algo.myHelpers.CL.file, 'hpFile'])
            if self.CL.saveFilesSim: 
                #Simulation Specific Files    
                MySIMPosition.saveFiles.append([self.algo.__class__.file, 'algoFile'])
                MySIMPosition.saveFiles.append([self.CL.file, 'strategyFile'])
                MySIMPosition.saveFiles.append([MySIMPosition.file, 'simFile'])
                MySIMPosition.saveFiles.append([MyRelativePrice.file, 'ta1File'])
                MySIMPosition.saveFiles.append([MyBarStrength.file, 'taB1File'])
 
    '''
    On Consolidated
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

        if not self.CL.enabled or self.algo.IsWarmingUp: return

        if not self.IsReady() or not self.WasJustUpdated(self.algo.Time) or not self.posEnabled: 
            return

        '''VOLATILITY CUTOFF
        '''
        volRef = self.algo.mySymbolDict[self.algo.benchmarkSymbol].vol
        #volRef = self.vol
        volatility = volRef.atrVolatility[0]
        #Dimension is inherited from eq2_2
        self.CL._totalATRpct = volatility/100
        volatilityNorm = volRef.atrNormVolatility[0]
        volatilityChange = volRef.atrVolatility[0]-volRef.atrVolatility[10]
        
        volatilityLimitLong = 1.0 
        volatilityLimitShort = 1.15
        
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
            aiObj["signal"] = aiObj["enabled"] and re.search('(.+)'+str(aiKey)+'(.+)', self.signals) and aiObj["firstTradeHour"] <= self.algo.Time.hour and self.algo.Time.hour <= aiObj["lastTradeHour"]
            if aiObj["signal"]:
                if aiObj["rawFeatures"]=="rawFeatures1": loadFeatures1 = True
                if aiObj["rawFeatures"]=="rawFeatures2": loadFeatures2 = True
        
        '''FEATURES
        '''
        if loadFeatures1:
            self.rawFeatures1 = [self.state_sma1.FeatureExtractor(Type=1), self.state_sma2.FeatureExtractor(Type=1), self.state_sma3.FeatureExtractor(Type=1), \
                            self.state_dch0.FeatureExtractor(Type=6), self.state_dch1.FeatureExtractor(Type=6), self.state_dch2.FeatureExtractor(Type=6), self.state_dch3.FeatureExtractor(Type=6), \
                            self.myRelativePrice.FeatureExtractor(Type=1, normalizationType=1, lookbacklist=[3,7,14,35,70], featureMask=[0,0,1,0]), \
                            self.vol.FeatureExtractor(Type=51, lookbacklist=[15,35,70,100]), \
                            self.zz1.FeatureExtractor(listLen=10, Type=11), self.zz1.FeatureExtractor(listLen=10, Type=21)]
        
        if loadFeatures2:    
            self.rawFeatures2 = [self.state_sma1.FeatureExtractor(Type=1), self.state_sma2.FeatureExtractor(Type=1), self.state_sma3.FeatureExtractor(Type=1), \
                            self.state_dch0.FeatureExtractor(Type=6), self.state_dch1.FeatureExtractor(Type=6), self.state_dch2.FeatureExtractor(Type=6), self.state_dch3.FeatureExtractor(Type=6), \
                            self.myRelativePrice.FeatureExtractor(Type=1, normalizationType=1, lookbacklist=[3,7,14,35,70], featureMask=[0,0,1,1]), \
                            self.vol.FeatureExtractor(Type=5, lookbacklist=[15,35,70,100]), \
                            self.zz1.FeatureExtractor(listLen=10, Type=11), self.zz1.FeatureExtractor(listLen=10, Type=21)]
        
        for aiObj in self.CL.aiDict.values():
            if aiObj["enabled"] and aiObj["signal"]: 
                featureFilter = aiObj["featureFilter"]
                myFeatures = self.algo.myHelpers.UnpackFeatures(getattr(self, aiObj["rawFeatures"]),  featureType=1, featureRegex=featureFilter, reshapeTuple=None)
                customColumnFilters = aiObj["customColumnFilter"]
                aiObj["dfFilterPassed"] = self.algo.myHelpers.FeatureCustomColumnFilter(myFeatures, customColumnFilters=customColumnFilters) if len(customColumnFilters)!=0 else True
                myFeatures = myFeatures.values
                if aiObj["usePCA"]:
                    myFeatures = aiObj["pca"].transform(myFeatures)
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
            if not (longTrigger or shortTrigger) and aiObj["enabled"] and aiObj["signal"] and aiObj["dfFilterPassed"] and self.signalDisabledBars[aiKey]==0:
                aiObj["model"].eval()
                trigger = aiObj["model"].Predict(aiObj["features"])
                if trigger and aiObj["direction"]==1:
                    longTrigger = True
                    longRiskMultiple = min(1.00,max(aiObj["riskMultiple"],0.00))
                    self.CL.strategyCode = self.CL.strategyCode + "|" + aiKey
                elif trigger and aiObj["direction"]==-1:
                    shortTrigger = True
                    shortRiskMultiple = min(1.00,max(aiObj["riskMultiple"],0.00))
                    self.CL.strategyCode = self.CL.strategyCode + "|" + aiKey
            if aiObj["signal"] and self.signalDisabledBars[aiKey]==0: self.signalDisabledBars[aiKey] = 8

        #Feature Debug
        if False: #and self.symbol == self.algo.Securities['APRN'].Symbol
            self.algo.MyDebug(f'Signal: {self.symbol} Feat8_2:{round(self.vol.FeatureExtractor(Type=5, lookbacklist=[15,35,70,100])[2],3)}')
            if True: # and self.algo.Time > datetime(2019, 2, 6, 12, 00) and self.algo.Time <= datetime(2019, 2, 6, 13, 00):
                self.PrintFeatures(df_short)
                pass
            self.algo.MyDebug(f'Predict2: {self.CL.modelShort.Predict2(myFeaturesShort)}')
            #self.algo.MyDebug(f' Short prediction: {shortTrigger}')
      
        '''POSITION ENTRY/FLIP/CLOSE
        '''
        self.longDisabledBars = max(self.longDisabledBars-1,0)
        self.shortDisabledBars = max(self.shortDisabledBars-1,0)
        #---LONG POSITION
        if self.posEnabled and self.CL.enableLong and self.longDisabledBars==0 and longTrigger:
            self.algo.myPositionManager.EnterPosition_2(self.symbol, 1, myRiskMultiple=longRiskMultiple)
            self.longDisabledBars=3
        #---SHORT POSITION
        elif self.posEnabled and self.CL.enableShort and self.shortDisabledBars==0 and shortTrigger:
            self.algo.myPositionManager.EnterPosition_2(self.symbol, -1, myRiskMultiple=shortRiskMultiple)
            if False: self.algo.MyDebug(f' Short Trade: {self.symbol}')
            self.shortDisabledBars=3

        '''SIMULATION CALL
        '''
        longTriggerSim, shortgTriggerSim = False, False
        #simObj : [direction, disableBars, disableCount]
        for simKey, simObj in self.CL.simDict.items():
            simObj[2] = max(0, simObj[2])
            if re.search('(.+)'+str(simKey)+'(.+)', self.signals):
                if simObj[0]==1 and simObj[2]==0:
                    simObj[2]=simObj[1]
                    longTriggerSim = True
                elif simObj[0]==-1 and simObj[2]==0:
                    simObj[2]=simObj[1]
                    shorTriggerSim = True
        
        self.longSimDisabledBars=max(self.longSimDisabledBars-1,0)
        self.shortSimDisabledBars=max(self.shortSimDisabledBars-1,0)
        simTradeTypes=[0,2,3,4,7,8] #[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] 
        simMinMaxTypes=[0,1] #[0,1,2,3]
        lastEntryDate = datetime(year = 3019, month = 10, day = 7)
        if self.CL.simulate and longTriggerSim:
            sim = MySIMPosition(self, direction=1, timestamp=self.algo.Time.strftime("%Y%m%d %H:%M"), signal=self.signals, features=myFeatures, simTradeTypes=simTradeTypes, simMinMaxTypes=simMinMaxTypes)
            #use 0 if signalspecific
            self.longSimDisabledBars=0
        if self.CL.simulate and shorTriggerSim:
            sim = MySIMPosition(self, direction=-1, timestamp=self.algo.Time.strftime("%Y%m%d %H:%M"), signal=self.signals, features=myFeatures, simTradeTypes=simTradeTypes, simMinMaxTypes=simMinMaxTypes) 
            #use 0 if signalspecific
            self.shortSimDisabledBars=0

        '''Reset strategyCode, Signals and Evens for the next bar
        '''
        self.CL.strategyCode = self.CL.strategyCodeOriginal 
        self.signals = ''
        self.events = ''
        return
        
    '''
    Update Status
    '''
    def IsReady(self):
        indiacatorsReady = self.bars_rw.IsReady and self.dch4.IsReady
        if False and indiacatorsReady:
            self.algo.MyDebug(" Consolidation Is Ready " + str(self.bars_rw.IsReady) + str(self.dch2.IsReady) + str(self.WasJustUpdated(self.algo.Time)))
        return indiacatorsReady
    
    def WasJustUpdated(self, currentTime):
        return self.bars_rw.Count > 0 and (currentTime - self.barTimeDifference - self.bars_rw[0].EndTime) < timedelta(seconds=10) \
                and (currentTime - self.barTimeDifference - self.bars_rw[0].EndTime).total_seconds() > timedelta(seconds=-10).total_seconds()
    
    def PrintFeatures(self, df):
        df = df.filter(regex = 'Feat')
        for col in df:
            self.algo.MyDebug(f' {col}: {df.loc[df.index[0], col]}')
            
    def TestPredict(self, model, features):
        device = 'cpu'
        for i in range (10):
            np.random.seed(i)
            x_sample = np.random.rand(1, features)
            x_sample = torch.from_numpy(x_sample.reshape(-1,features)).float().to(device)
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
      #  self.layer_h5 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
      #  self.bn5 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
      #  self.layer_h6 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
      #  self.bn6 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
      #  self.layer_h7 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
      #  self.bn7 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
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