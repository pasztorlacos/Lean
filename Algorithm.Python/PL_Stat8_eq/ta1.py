### <summary>
### Technical Analysis 
### 
### </summary>
from QuantConnect.Orders import *
from QuantConnect.Orders.Fills import *
from QuantConnect.Orders.Fees import *
from QuantConnect.Data.Market import TradeBar

#import math
#import random
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from math import log, ceil, floor

class MyRelativePrice():
    '''
    Relative Price Indicator. self.Value = self.relativeClose[0] 
    simple Bar: self.myBars[] and Benchmark Close: self.benchmarkClose[] deque 
    Performance Method returns a list: performance, relativePerformance, performanceATR, relativePerformanceATR'''
    file = __file__
    
    def __init__(self, caller, algo, symbol, name, period, atr, benchmarkTicker=None):
        self.caller = caller
        self.algo = algo
        self.symbol = symbol
        self.name = name
        self.benchmarkTicker = benchmarkTicker
        self.benchmarkSymbol = None
        self.atr = atr
        self.isEquity = self.caller.CL.isEquity
        self.Value = 0
        self.IsReady = False
        self.myBars = deque(maxlen=period)
        self.relativeClose = deque(maxlen=period)
        self.benchmarkClose = deque(maxlen=period)
    
    # Update method is mandatory
    def Update(self, newbar):
        #This is canot be set up during _init_ as benchmark might not be available that time
        if self.benchmarkSymbol == None:
            if self.benchmarkTicker == None:
                #if NO Ticker use algo.benchmarkSymbol 
                self.benchmarkSymbol=self.algo.benchmarkSymbol
            else:
                #Get Symbol og the given Ticker
                self.benchmarkSymbol=self.algo.Securities[self.benchmarkTicker].Symbol
        
        benchmarkPrice = self.algo.Securities[self.benchmarkSymbol].Price if self.algo.Securities[self.benchmarkSymbol].Price!=0 else 1 
        self.myBars.appendleft(newbar)
        self.relativeClose.appendleft(newbar.Close/benchmarkPrice)
        self.benchmarkClose.appendleft(benchmarkPrice)
        self.IsReady = len(self.myBars) == self.myBars.maxlen
        #Relative Price ZZ: self.myBars.Updated += self._zzName.IndicatorUpdate must be used
        self.Value = self.relativeClose[0]
        return self.IsReady
    
    def Performance (self, lookback, normalizationType=0):
        '''
        Returns a list: performance, relativePerformance, performanceATR, relativePerformanceATR'''
        if self.IsReady and self.myBars[lookback].Close!=0  and self.benchmarkClose[lookback]!=0 and self.benchmarkClose[0]!=0:
            atrpct = self.atr.Current.Value/self.myBars[lookback].Close
            performance = self.myBars[0].Close/self.myBars[lookback].Close
            performanceATR = (performance-1)/atrpct if atrpct!=0 else 0.00
            performanceBenchmark = self.benchmarkClose[0]/self.benchmarkClose[lookback]
            relativePerformance = performance/performanceBenchmark
            relativePerformanceATR = (relativePerformance-1)/atrpct if atrpct!=0 else 0.00
        else:
            performance = 1.00
            relativePerformance = 1.00
            performanceATR = 0.00
            relativePerformanceATR = 0.00
        
        #Type=0: No Normalization
        if normalizationType==0:
            return [performance, relativePerformance, performanceATR, relativePerformanceATR]
        elif normalizationType==1:
            bias=0.5
            atrNorm=40
            perfMultiplier=2 
            performance = (performance-1)*perfMultiplier+bias
            relativePerformance = (relativePerformance-1)*perfMultiplier+bias
            performanceATR = performanceATR/atrNorm+bias
            relativePerformanceATR = relativePerformanceATR/atrNorm+bias
        return [performance, relativePerformance, performanceATR, relativePerformanceATR]
    
    #FEATURE EXTRACTOR
    def FeatureExtractor(self, Type=1, normalizationType=1, lookbacklist=[3,7,14,35,70], featureMask=[1,1,1,1]):
        '''
        Type==1: single list
        Type==2: list of lists each self.performance()'''
        features = []
        #!!! deleted=+1 bug fixed but it does not effect ed2_1_pt_1A and ed2_1_pt_1B as after the first deletion  deleted+=1==deleted=+1 amd there is no further deletion after the seconf one where deleted val is incorrect.
        if Type==1:
            for lookback in lookbacklist:
                performance = self.Performance(lookback, normalizationType)
                deleted=0
                #This is horribly complicated, so to be refactored to templist.append(performance(i) if mask 1) and then features.extend(templist)
                for i in range(len(featureMask)):
                    if featureMask[i]==0: 
                        del performance[i-deleted]
                        deleted+=1
                features.extend(performance)
        elif Type==2:
            for lookback in lookbacklist:
                performance = self.Performance(lookback, normalizationType)
                deleted=0
                for i in range(len(featureMask)):
                    if featureMask[i]==0: 
                        del performance[i-deleted]
                        deleted+=1
                features.append(performance)
        return features
'''
Pices Normalizer Indicator (Nominal an Reltive Price storage and Normalised Prices for CNN Antoencoder)
'''
class MyPriceNormaliser():
#Normaliser(symbol and benchmark): (Price-min*(1-margin))/(max*(1+margin)-min*(1-margin))
#Relative Normaliser: (1+normPrice)/(1+normBenchmark)-expectedValue=0.50

    def __init__(self, caller, algo, symbol, name, period, benchmarkSymbol=None):
        self.caller = caller
        self.algo = algo
        self.symbol = symbol
        self.name = name
        self.isEquity = self.caller.CL.isEquity
        self.Value = 0
        self.IsReady = False
        self.myPrices = deque(maxlen=period)
        #if new High/Low recalculate the entire deque
        self.recalculate = True
        self.priceMin, self.priceMax, self.volumeMin, self.volumeMax, self.benchmarkMin, self.benchmarkMax = 0, 0, 0, 0, 0, 0
        self.benchmarkSymbol = benchmarkSymbol

    # Update method is mandatory
    def Update(self, newbar):
        bar = TradeBar()
        # if not hasattr(self.algo, 'benchmarkSymbol'):
        #     return
        recalculate = False
        length =  len(self.myPrices) 
        
        newPriceValues = MyPriceStorage(self)
        if self.isEquity:
            bar = newbar
        else: 
            bar.Open = newbar.Open
            bar.High = newbar.High
            bar.Low = newbar.Low
            bar.Close = newbar.Close
            bar.Volume = 0
        newPriceValues.bar = bar
            
        if self.benchmarkSymbol==None:
            newPriceValues.benchmarkPrice = self.algo.Portfolio.Securities[self.algo.benchmarkSymbol].Price
        else:
            newPriceValues.benchmarkPrice = self.algo.Portfolio.Securities[self.benchmarkSymbol].Price
        if length==0:
            oldestPriceBar = newPriceValues.bar
            oldestbenchmarkPrice= newPriceValues.benchmarkPrice
        else:
            oldestPriceBar = self.myPrices[length-1].bar
            oldestbenchmarkPrice = self.myPrices[length-1].benchmarkPrice
        self.myPrices.appendleft(newPriceValues)
        
        #Check for retired and new highs/lows (try to avoid min/max i.e. iterating through myPrices)
        if oldestPriceBar.Low == self.priceMin: 
            self.priceMin = min(map(lambda x: x.bar.Low, self.myPrices))
            recalculate = True
        elif bar.Low < self.priceMin or length==0:
            self.priceMin = bar.Low
            recalculate = True
            
        if oldestPriceBar.High == self.priceMax:
            self.priceMax = max(map(lambda x: x.bar.High, self.myPrices))
            recalculate = True
        elif bar.High > self.priceMax or length==0:
            self.priceMax = bar.High
            recalculate = True
        
        if oldestPriceBar.Volume == self.volumeMin:
            self.volumeMin = min(map(lambda x: x.bar.Volume, self.myPrices))
            recalculate = True
        elif bar.Volume < self.volumeMin or length==0:
            self.volumeMin = bar.Volume
            recalculate = True
        
        if oldestPriceBar.Volume == self.volumeMax:
            self.volumeMax = max(map(lambda x: x.bar.Volume, self.myPrices))
            recalculate = True
        elif bar.Volume > self.volumeMax or length==0:
            self.volumeMax = bar.Volume
            recalculate = True
        
        if oldestbenchmarkPrice == self.benchmarkMin:
            self.benchmarkMin = min(map(lambda x: x.benchmarkPrice, self.myPrices))
            recalculate = True
        elif newPriceValues.benchmarkPrice < self.benchmarkMin or length==0:
            self.benchmarkMin = newPriceValues.benchmarkPrice
            recalculate = True
        
        if oldestbenchmarkPrice == self.benchmarkMax:
            self.benchmarkMax = max(map(lambda x: x.benchmarkPrice, self.myPrices))
            recalculate = True
        elif newPriceValues.benchmarkPrice > self.benchmarkMax or length==0:
            self.benchmarkMax = newPriceValues.benchmarkPrice
            recalculate = True

        #Calculate the latest item only
        item = self.myPrices[0]
        self.NormaliseItem(item)
        
        #Recalculate the entire deque (too slow)
        #but do it when warmed up anyway or if it is set so
        if not self.IsReady and length == self.myPrices.maxlen or (self.recalculate and recalculate):
            for item in self.myPrices:
                self.NormaliseItem(item)
        
        self.Value = bar.Close
        self.IsReady = length == self.myPrices.maxlen
        return self.IsReady
    
    def NormaliseItem(self, item):
        margin = 0.01
        expectedValue = 0.50
        
        priceDiv = (self.priceMax*(1+margin)-self.priceMin*(1-margin))
        if priceDiv!=0:
            item.normBar.Open  = (item.bar.Open-self.priceMin*(1-margin))/priceDiv
            item.normBar.High  = (item.bar.High-self.priceMin*(1-margin))/priceDiv
            item.normBar.Low   = (item.bar.Low-self.priceMin*(1-margin))/priceDiv
            item.normBar.Close = (item.bar.Close-self.priceMin*(1-margin))/priceDiv
        
        if (self.volumeMax*(1+margin)-self.volumeMin*(1-margin))!=0:
            item.normBar.Volume= (item.bar.Volume-self.volumeMin*(1-margin))/(self.volumeMax*(1+margin)-self.volumeMin*(1-margin))
        
        benchmarkDiv=(self.benchmarkMax*(1+margin)-self.benchmarkMin*(1-margin))
        if benchmarkDiv!=0:
            item.normBenchmarkPrice =(item.benchmarkPrice-self.benchmarkMin*(1-margin))/benchmarkDiv
        
        if (1+item.normBenchmarkPrice)!=0:
            item.relnormPrice  = (1+item.normBar.Close)/(1+item.normBenchmarkPrice)-expectedValue
        return

    def FeatureExtractor(self, onlyClose=False, icludeVolume=False, includeRelative=False, periods=10000, multiList=True):
        feturelist = []
        list_O = []
        list_H = []
        list_L = []
        list_C = []
        list_V = []
        list_Relative = []
        #list_Benchmark = []

        myrange = min(len(self.myPrices), periods)
        for i in range(myrange):
            item = self.myPrices[i].normBar
            list_O.append(item.Open)
            list_H.append(item.High)
            list_L.append(item.Low)
            list_C.append(item.Close)
            list_V.append(item.Volume)
            list_Relative.append(self.myPrices[i].relnormPrice)
            #list_Benchmark.append(self.myPrices[i].normBenchmarkPrice)
        
        if multiList:
            ohlcFeature=[]
            if not onlyClose: ohlcFeature.extend(list_O)
            if not onlyClose: ohlcFeature.extend(list_H)
            if not onlyClose: ohlcFeature.extend(list_L)
            ohlcFeature.extend(list_C)
            feturelist.append(ohlcFeature)
        else:
            if not onlyClose: feturelist.extend(list_O)
            if not onlyClose: feturelist.extend(list_H)
            if not onlyClose: feturelist.extend(list_L)
            feturelist.extend(list_C)
        if icludeVolume: 
            if multiList:
                feturelist.append(list_V)
            else: 
                feturelist.extend(list_V)
        if includeRelative: 
            if multiList:
                feturelist.append(list_Relative)
            else:
                feturelist.extend(list_Relative)
        return feturelist
    
    def PriceDebug(self, periods=10000):
        close = []
        normClose = []
        relnorm = []
        benchmark = []
        normbenchmark = []
        myrange= min(len(self.myPrices), periods)
        self.algo.MyDebug("  len(myPrices) "+ str(len(self.myPrices))) 
        for i in range(myrange):
            item = self.myPrices[i]
            close.append(item.bar.Close)
            normClose.append(item.normBar.Close)
            relnorm.append(item.relnormPrice)
            benchmark.append(item.benchmarkPrice)
            normbenchmark.append(item.normBenchmarkPrice)
        self.algo.MyDebug("  Symbol "+ str(self.symbol) + "  close:" + str(close))
        self.algo.MyDebug("  Symbol "+ str(self.symbol) + "  normClose:" + str(normClose))
        self.algo.MyDebug("  Symbol "+ str(self.symbol) + "  relnormClose:" + str(relnorm))
        self.algo.MyDebug("  Symbol "+ str(self.symbol) + "  benchmark:" + str(benchmark))
        self.algo.MyDebug("  Symbol "+ str(self.symbol) + "  normbenchmark:" + str(normbenchmark))
        return
    
class MyPriceStorage():
    def __init__(self, myPrice):
        self.myPriceObj = myPrice
        self.bar = TradeBar()
        self.normBar = TradeBar()
        self.relnormPrice = 0
        self.benchmarkPrice = 0
        self.normBenchmarkPrice = 0
'''
ZigZag
'''
class MyZigZag():
    def __init__(self, caller, algo, symbol, name, period, atr, lookback=10, thresholdType=1, threshold=0.05):
        self.caller = caller
        self.algo = algo
        self.symbol = symbol
        self.name = name
        self.roundDigits = 2
        self.period = period
        self.lookback = lookback
        self.thresholdType = thresholdType
        self.threshold = threshold
        self.currentThreshold = 0
        self.priceUnit = 0
        self.atr = atr  
        self.Time = datetime.min
        self.Value = 0
        self.IsReady = False

        self.lastLow = None
        self.lastHigh = None
        self.shortTrendCount = 0 # (-)n_th short trend
        self.longTrendCount = 0 # (+)n_th long trend
        self.trendDir = None
        self.trendChange = False
        self.bars = deque(maxlen=period)
        self.zzLow = deque(maxlen=period)
        self.shortTrendCount = deque(maxlen=period)
        self.zzHigh = deque(maxlen=period)
        self.longTrendCount = deque(maxlen=period)
        self.zigzag = deque(maxlen=period)
        self.barCount = 0
        
        self.zzPoints = []
        self.patterns = MyPatterns(self)
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return "Zig Zag Name:{}, Symbol:{}, IsReady:{}, Time:{}, Value:{}".format(self.Name, self.symbol, self.IsReady, self.Time, self.Value)


    def IndicatorUpdate(self, caller, updated):
        '''This is used when ZZ is applied on an indictor. 
        In this case self._IndicatorName.Updated += self._zzName.IndicatorUpdate must be used rather self.RegisterIndicator()'''
        bar = TradeBar()
        bar.Open = updated.Value
        bar.High = updated.Value
        bar.Low = updated.Value
        bar.Close = updated.Value
        bar.EndTime = updated.EndTime
        self.Update(bar)
    
    def ManualUpdate(self, value):
        '''Use this either 
        1) calling directly from OnDataConsolidated()
        2) or adding this method to self.stateUpdateList'''
        bar = TradeBar()
        bar.Open = value
        bar.High = value
        bar.Low = value
        bar.Close = value
        bar.EndTime = self.algo.Time
        self.Update(bar)
        return
    
    def Update(self, input):
        '''Update method is mandatory!'''
        bar = input
        self.bars.appendleft(bar)
        self.barCount = len(self.bars)
        self.Time = bar.EndTime
        self.IsReady = self.barCount == self.bars.maxlen 
        
        #update currentThreshold value
        if self.thresholdType==1:
            self.currentThreshold = self.bars[0].Close * self.threshold
            self.priceUnit = self.bars[0].Close * self.threshold/10
        elif self.thresholdType==2:
            self.currentThreshold = self.atr.Current.Value * self.threshold
            self.priceUnit = self.atr.Current.Value
        
        #Not enough bars yet to calculte  
        if self.barCount < self.lookback:
            self.zzLow.appendleft(0)
            self.shortTrendCount.appendleft(0)
            self.zzHigh.appendleft(0)
            self.longTrendCount.appendleft(0)
            self.zigzag.appendleft(0)
            return self.IsReady
        
        low, thisisLow , retraceLow = self.localLow()
        high, thisisHigh, retraceHigh = self.localHigh()
        self.trendChange = False
        
        #Disable retrace if it is a new high in the same bar. New shortZZ in short trend if new low.
        shortZZ = (thisisLow and self.trendDir!=-1) or (thisisLow and self.trendDir==-1 and low<=self.lastLow) or (retraceLow and not thisisHigh)
        if shortZZ:
            self.lastLow = low
            if self.trendDir == 1:
                self.trendChange = True
            self.trendDir = -1
        #Disable long if it is a shortZZ to prevent two zz poin in the same bar. New longZZ in long trend if new high.
        longZZ = ((thisisHigh and self.trendDir!=1) or (thisisHigh and self.trendDir==1 and high>=self.lastHigh) or retraceHigh) and not shortZZ
        if longZZ:
            self.lastHigh = high
            if self.trendDir == -1:
                self.trendChange = True
            self.trendDir = 1
        
        #Update ZZ Low
        #if thisisLow or retraceLow:
        if shortZZ:
            self.zzLow.appendleft(low)
        else: self.zzLow.appendleft(0)
        #Update Short TrendCount
        shortTrendCount = self.shortTrendCount[0] #This is still the previous bar value
        if self.trendChange and self.trendDir == -1:
            #New Trend
            shortTrendCount-=1 #this is the (-)n_th shortTrend
        self.shortTrendCount.appendleft(shortTrendCount) 

        #Update ZZ High
        #if thisisHigh or retraceHigh:
        if longZZ:
            self.zzHigh.appendleft(high)
        else: self.zzHigh.appendleft(0)
        #Update Long TrendCount
        longTrendCount = self.longTrendCount[0] #This is still the previous bar value
        if self.trendChange and self.trendDir == 1:
            #New Trend
            longTrendCount+=1 #this is the (+)n_th shortTrend
        self.longTrendCount.appendleft(longTrendCount) 
        
        #Update ZZ
        if self.zzLow[0]!=0:
            self.zigzag.appendleft(self.zzLow[0])
        else:
            self.zigzag.appendleft(self.zzHigh[0])
        self.Value = self.zigzag[0]
        
        #Erase previous values if any in this Short Trend
        i=1
        while i < len(self.bars)-1 and self.zzLow[0] !=0 and self.shortTrendCount[i] == shortTrendCount:
            if self.zzLow[i] == self.zigzag[i]:
                self.zigzag[i] = 0 
            self.zzLow[i] = 0
            i+=1        
        
        #Erase previous values if any in this Long Trend
        i=1
        while i < len(self.bars)-1 and self.zzHigh[0] != 0 and self.longTrendCount[i] == longTrendCount:
            if self.zzHigh[i] == self.zigzag[i]:
                self.zigzag[i] = 0            
            self.zzHigh[i] = 0
            i+=1              
        
        if False and self.IsReady :self.algo.MyDebug("Symbol:{}, Low:{}/{}/{}/{}/{}, High:{}/{}/{}/{}/{}, ZZ:{}/{}".format( \
                str(self.symbol), str(low),str(thisisLow),str(retraceLow),str(round(self.zzLow[0],self.roundDigits)),str(self.shortTrendCount[0]), \
                str(high),str(thisisHigh),str(retraceHigh),str(round(self.zzHigh[0],self.roundDigits)),str(self.longTrendCount[0]), \
                str(round(self.zigzag[0],self.roundDigits)),str(self.trendDir)))

        #Update zzPoints List [0] is the latest
        del self.zzPoints[:]
        for i in range(0, len(self.zigzag)):
            if self.zigzag[i]!=0:
                self.zzPoints.append(MyZZPoint(self,i)) #zzPoints[0] is the latest ZZ Point
                #self.zzPoints.insert(0,MyZZPoint(self,i))
       
        #Update Price Patterns
        self.patterns.Update()
        
        return self.IsReady
        
    #Local Low or Retracement (used in ZZ calculation)
    def localLow(self):
        thisisLow = False
        retrace = False
        low = self.bars[0].Low
        for i in range(1, self.lookback):
            if self.bars[i].Low < low: low = self.bars[i].Low
        thisisLow = self.bars[0].Low == low
        
        #trendDir is not updated yet so it is the previous value
        if self.lastHigh != None and self.trendDir == 1 and (self.lastHigh - self.bars[0].Low)>self.currentThreshold:
            retrace = True
            low = self.bars[0].Low 
        return low, thisisLow, retrace
    
    #Local High or Retracement (used in ZZ calculation)
    def localHigh(self):
        thisisHigh = False
        retrace = False
        high = self.bars[0].High
        for i in range(1, self.lookback):
            if self.bars[i].High > high: high = self.bars[i].High
        thisisHigh = self.bars[0].High == high
        
        #trendDir is not updated yet so it is the previous value
        if self.lastLow != None and self.trendDir == -1 and (self.bars[0].High - self.lastLow)>self.currentThreshold:
            retrace = True
            high = self.bars[0].High
        return high, thisisHigh, retrace 
    
    #FEATURE EXTRACTOR
    def FeatureExtractor(self, listLen=10, Type=1):
        atrNormaliser = 20*2
        bias=0.5
        features=[]
        
        #ATR distance of the zz point from current price 
        if Type==1:
            #zzPoints[0] is the latest
            for i in range (0,min(listLen,len(self.zzPoints))):
                #Distance from last Close in atr /atrNormaliser+bias
                #This is better for sklearn types
                zzNormalised=(self.zzPoints[i].value-self.bars[0].Close)/self.atr.Current.Value/atrNormaliser+bias
                features.append(zzNormalised)
                lastZZNormalised=zzNormalised
            i+=1
            #if not enough ZZ populate the rest with the latest
            if i<listLen:
                for j in range(i,listLen):
                    features.append(lastZZNormalised)
        
        #ATR distance of the zz point from current price but lows and highs are split to make the NN inputs less volatile. First half ZZLows Seconf half ZZHighs.
        elif Type==11:
            listLen_2 = floor(listLen/2)
            i_L, i_H = 0, 0
            zz_L, zz_H = [], []
            lastZZNormalised_L, lastZZNormalised_H = 0.50, 0.50
            #zzPoints[0] is the latest
            for i in range (0,min(listLen,len(self.zzPoints))):
                zzNormalised=(self.zzPoints[i].value-self.bars[0].Close)/self.atr.Current.Value/atrNormaliser+bias
                #if LowZZ points
                if self.zzPoints[i].trendCount<0:
                    zz_L.append(zzNormalised)
                    lastZZNormalised_L = zzNormalised
                    i_L+=1
                #if HighZZ points
                if self.zzPoints[i].trendCount>0:
                    zz_H.append(zzNormalised)
                    lastZZNormalised_H = zzNormalised
                    i_H+=1
            i_L+=1
            i_H+=1
            #if not enough ZZ populate the rest with the latest
            if i_L<listLen_2+1:
                for j in range(i_L,listLen_2+1):
                    zz_L.append(lastZZNormalised_L)
            if i_H<listLen_2+1:
                for j in range(i_H,listLen_2+1):
                    zz_H.append(lastZZNormalised_H)
            #Merge the LowZZ and HighZZ lists
            features = zz_L
            features.extend(zz_H)
        
        #zz point position in MIN MAX intervall
        elif Type==2:
            margin=0.1
            minZZ=self.bars[0].Close
            maxZZ=self.bars[0].Close
            #zzPoints[0] is the latest
            for i in range (0,min(listLen,len(self.zzPoints))):
                if self.zzPoints[i].value < minZZ: minZZ=self.zzPoints[i].value
                if self.zzPoints[i].value > maxZZ: maxZZ=self.zzPoints[i].value
            for i in range (0,min(listLen,len(self.zzPoints))):
                #ZZPoint position in the (maxZZ*(1+margin)-minZZ*(1-margin)) range
                #This is better for NN
                if (maxZZ-minZZ)!=0:
                    zzNormalised=(self.zzPoints[i].value-minZZ*(1-margin))/(maxZZ*(1+margin)-minZZ*(1-margin))
                else: 
                    zzNormalised = 0.5 #self.zzPoints[i].value
                features.append(zzNormalised)
                lastZZNormalised=zzNormalised
            i+=1
            #if not enough ZZ populate the rest with the latest
            if i<listLen:
                for j in range(i,listLen):
                    features.append(lastZZNormalised)
            if True:
                #Add Price position in the range
                priceFeature = (self.bars[0].Close-minZZ*(1-margin))/(maxZZ*(1+margin)-minZZ*(1-margin))
                features.append(priceFeature)
        
        #zz point position in MIN MAX intervall but lows and highs are split to make the NN inputs less volatile. First half ZZLows Seconf half ZZHighs.
        elif Type==21:
            margin=0.1
            minZZ=self.bars[0].Close
            maxZZ=self.bars[0].Close
            listLen_2 = floor(listLen/2)
            i_L, i_H = 0, 0
            zz_L, zz_H = [], []
            lastZZNormalised_L, lastZZNormalised_H = 0.50, 0.50
            if len(self.zzPoints)!=0:
               firstLow = self.zzPoints[0].trendCount < 0 
            else:
               firstLow = True
               #self.algo.MyDebug(f' ZZ({self.name}) is empty when calulating firstLow. Symbol:{self.symbol}')
            
            #zzPoints[0] is the latest
            for i in range (0,min(listLen,len(self.zzPoints))):
                if self.zzPoints[i].value < minZZ: minZZ=self.zzPoints[i].value
                if self.zzPoints[i].value > maxZZ: maxZZ=self.zzPoints[i].value
            for i in range (0,min(listLen,len(self.zzPoints))):
                #ZZPoint position in the (maxZZ*(1+margin)-minZZ*(1-margin)) range
                #This is better for NN
                if (maxZZ-minZZ)!=0:
                    zzNormalised=(self.zzPoints[i].value-minZZ*(1-margin))/(maxZZ*(1+margin)-minZZ*(1-margin))
                else: 
                    zzNormalised = 0.5 #self.zzPoints[i].value
                #if LowZZ points
                if self.zzPoints[i].trendCount<0:
                    zz_L.append(zzNormalised)
                    lastZZNormalised_L = zzNormalised
                    i_L+=1
                #if HighZZ points
                if self.zzPoints[i].trendCount>0:
                    zz_H.append(zzNormalised)
                    lastZZNormalised_H = zzNormalised
                    i_H+=1
            i_L+=1
            i_H+=1
            #if not enough ZZ populate the rest with the latest
            if i_L<listLen_2+1:
                for j in range(i_L,listLen_2+1):
                    zz_L.append(lastZZNormalised_L)
            if i_H<listLen_2+1:
                for j in range(i_H,listLen_2+1):
                    zz_H.append(lastZZNormalised_H)
            #Merge the LowZZ and HighZZ lists
            features = zz_L
            features.extend(zz_H)   

            #Add Price position in the range
            if True:
                priceFeature = (self.bars[0].Close-minZZ*(1-margin))/(maxZZ*(1+margin)-minZZ*(1-margin))
                features.append(priceFeature)       
            #Add Latest ZZ Type as this information is lost now (we don't know the order of lows and highs)
            if True:
                if firstLow: features.append(0.25) 
                else: features.append(0.75) 

        #LIST ZZ FEATURES
        if False:
            self.algo.MyDebug(f'\nZZ Feature Extractor Type:{Type}, Close:{self.bars[0].Close}, atrNormaliser:{atrNormaliser}, atr:{self.atr.Current.Value}, bias:{bias}')
            for i in range(len(features)):
                self.algo.MyDebug(f' Symbol:{self.symbol}, zzFeature[{i}]:{features[i]}')
        
        return features
    
    #DEBUG only
    def ListZZ(self, length=70):
        self.algo.Debug(f'\n Symbol:{str(self.symbol)}, >>>>> List of ZZ (ListZZ(self, length={str(length)}))')
        printItems = min(length, len(self.bars)-1)
        i=printItems
        while i >= 0:
            if self.zzLow[i]==self.zigzag[i] and self.zigzag[i]!=0:
                trendCount = self.shortTrendCount[i]
            elif self.zzHigh[i]==self.zigzag[i] and self.zigzag[i]!=0:
                trendCount = self.longTrendCount[i]
            else:
                trendCount = ""
            self.algo.Debug(" Symbol:{}, {} Date:{}, Close:{}, zzLow:{}/{}, zzHigh:{}/{}, ZZ:{}/{}".format( \
                str(self.symbol), str(self), str(self.bars[i].EndTime), str(round(self.bars[i].Close,self.roundDigits)), \
                str(round(self.zzLow[i],self.roundDigits)),str(self.shortTrendCount[i]), \
                str(round(self.zzHigh[i],self.roundDigits)),str(self.longTrendCount[i]), \
                str(round(self.zigzag[i],self.roundDigits)),str(trendCount)))
            i-=1
        return
    
    #DEBUG only
    def listZZPoints(self):
        self.algo.Debug(f'\n Symbol:{self.symbol}, >>>>> List of ZigZag Points')
        i=1
        for point in self.zzPoints[0:30]:
            #self.algo.Debug("Latest Date zzPoints[0]: " + str(self.zzPoints[0].endTime))
            self.algo.Debug(" Symbol:{}, {}/{}. Value:{}, EndTime:{}, Index:{}, TrendCount:{}".format( \
                str(self.symbol), str(self), str(i), \
                str(round(point.value,self.roundDigits)), str(point.endTime), str(point.index), str(point.trendCount),))
            i+=1
        return    

class MyZZPoint():
    def __init__(self, zz, index):
        self.zz = zz
        self.index = index
        self.endTime = zz.bars[index].EndTime
        self.value = zz.zigzag[index]
        if self.value == zz.zzLow[index]:
            self.trendCount = zz.shortTrendCount[index]
        else:
            self.trendCount = zz.longTrendCount[index]
        return

class MyPatterns():
    def __init__(self, zz):
        self.zz = zz
        self.debug = False
        self.roundDigits = 2
        self.isReady = False
        self.doubleTop = False
        self.doubleBottom = False
        self.tripleTop = False
        self.tripleBottom = False
        self.hs = False
        self.ihs = False
        self.abcdL = False
        self.abcdS = False
        
    def Update(self):
        priceUnitType = 2 #1 in Stat5
        if len(self.zz.zzPoints) <= 10:
            self.isReady = False
            if False and self.debug and not self.zz.algo.IsWarmingUp and not self.isReady and not self.zz.caller.CL.isEquity: self.zz.algo.MyDebug(f' Symbol:{self.zz.symbol} zz:{self.zz.name} MyPatterns is not ready. zz points:{len(self.zz.zzPoints)}')
            return False
        if False and self.debug and not self.zz.algo.IsWarmingUp and not self.zz.caller.CL.isEquity: self.zz.algo.MyDebug(f' Symbol:{self.zz.symbol} zz:{self.zz.name} zz points:{len(self.zz.zzPoints)}')
        bar=self.zz.bars[0]
        if priceUnitType==1:
            priceUnit = self.zz.priceUnit
        elif priceUnitType==2:
            #Force ATR even if zz thresholdType==1
            priceUnit = self.zz.atr.Current.Value
        A, B, C, D, E, F, G, H, I, J = self.zz.zzPoints[0:10]
        
        #Tops and Bottoms
        tolerance1 = priceUnit*1 #Same line
        tolerance2 = priceUnit*2 #Max Price move from extremum
        tolerance3 = priceUnit*0 #Min Price move from extremum
        channel1 = priceUnit*5 #Minimum Channel Size
        if False and self.debug: self.zz.algo.MyDebug(f' Symbol:{self.zz.symbol} zz:{self.zz.name} priceUnit:{priceUnit} A:{A.value} B:{B.value} C:{C.value}')
        
        #Double Top and Bottom
        self.doubleTop = False
        self.doubleBottom = False
        if abs(A.value-C.value)<tolerance1 and abs(A.value-B.value)>channel1  \
                and abs(A.value-bar.Close)>tolerance3 and abs(A.value-bar.Close)<=tolerance2:
            if A.trendCount>0 and A.value>B.value and A.value>bar.Close:
                self.doubleTop = True
                self.zz.caller.signals += f'DT_{self.zz.name}-'
                if False and self.debug: self.zz.algo.MyDebug("Symbol:{}, Double Top ZZ:{}, barClose:{}, zzValue:{}, zzEndTime:{}, Index:{}, TrendCount:{}".format( \
                    str(self.zz.symbol), str(A.zz), str(round(bar.Close,2)), \
                    str(round(A.value,self.roundDigits)), str(A.endTime), str(A.index), str(A.trendCount),))
            if A.trendCount<0 and A.value<B.value and A.value<bar.Close:
                self.doubleBottom = True
                self.zz.caller.signals += f'DB_{self.zz.name}-'
                if False and self.debug: self.zz.algo.MyDebug("Symbol:{}, Double Bottom ZZ:{}, barClose:{}, zzValue:{}, zzEndTime:{}, Index:{}, TrendCount:{}".format( \
                    str(self.zz.symbol), str(A.zz), str(round(bar.Close,2)), \
                    str(round(A.value,self.roundDigits)), str(A.endTime), str(A.index), str(A.trendCount),))
        #Triple Top and Bottom
        self.tripleTop = False
        self.tripleBottom = False
        if abs(A.value-C.value)<tolerance1 and abs(A.value-B.value)>channel1 and abs(E.value-C.value)<tolerance1 and abs(E.value-A.value)<tolerance1 \
                and abs(A.value-bar.Close)>tolerance3 and abs(A.value-bar.Close)<=tolerance2:
            if A.trendCount>0 and A.value>B.value and A.value>bar.Close:
                self.tripleTop = True
                self.zz.caller.signals += f'TT_{self.zz.name}-'
                if False and self.debug: self.zz.algo.MyDebug("Symbol:{}, triple Top ZZ:{}, barClose:{}, zzValue:{}, zzEndTime:{}, Index:{}, TrendCount:{}".format( \
                    str(self.zz.symbol), str(A.zz), str(round(bar.Close,2)), \
                    str(round(A.value,self.roundDigits)), str(A.endTime), str(A.index), str(A.trendCount),))
            if A.trendCount<0 and A.value<B.value and A.value<bar.Close:
                self.tripleBottom = True
                self.zz.caller.signals += f'TB_{self.zz.name}-'
                if False and self.debug: self.zz.algo.MyDebug("Symbol:{}, triple Bottom ZZ:{}, barClose:{}, zzValue:{}, zzEndTime:{}, Index:{}, TrendCount:{}".format( \
                    str(self.zz.symbol), str(A.zz), str(round(bar.Close,2)), \
                    str(round(A.value,self.roundDigits)), str(A.endTime), str(A.index), str(A.trendCount),))
        
        #Head and Shoulder and Inverse
        #NOT YET READY!!!!!
        tolerance1 = priceUnit*1.5 #Same line
        tolerance2 = priceUnit*2 #Max Price move from extremum
        tolerance3 = priceUnit*0 #Min Price move from extremum
        tolerance4 = priceUnit*4 #Min Price move of head from shoulders 
        self.hs = False
        self.ihs = False
        if abs(E.value-A.value)<tolerance1 and abs(D.value-B.value)<tolerance1 and abs(C.value-A.value)>tolerance4:
            if A.trendCount>0 and C.value>A.value and abs(A.value-bar.Close)<tolerance2 and abs(A.value-bar.Close)>tolerance3 and A.value>bar.Close:
                self.hs = True
                self.zz.caller.signals += f'HS_{self.zz.name}-'
                if False and self.debug: self.zz.algo.MyDebug("Symbol:{}, H&S ZZ:{}, barClose:{}, zzValue:{}, zzEndTime:{}, Index:{}, TrendCount:{}".format( \
                    str(self.zz.symbol), str(A.zz), str(round(bar.Close,2)), \
                    str(round(A.value,self.roundDigits)), str(A.endTime), str(A.index), str(A.trendCount),))
            if A.trendCount<0 and C.value<A.value and abs(A.value-bar.Close)<tolerance2 and abs(A.value-bar.Close)>tolerance3 and A.value<bar.Close:
                self.ihs = True
                self.zz.caller.signals += f'IHS_{self.zz.name}-'
                if False and self.debug: self.zz.algo.MyDebug("Symbol:{}, Inverse H&S ZZ:{}, barClose:{}, zzValue:{}, zzEndTime:{}, Index:{}, TrendCount:{}".format( \
                    str(self.zz.symbol), str(A.zz), str(round(bar.Close,2)), \
                    str(round(A.value,self.roundDigits)), str(A.endTime), str(A.index), str(A.trendCount),))        
 
        #ABCD
        #NOT YET READY!!!!!
        tolerance1 = priceUnit*10 #Minimum AB
        tolerance2 = 0.25 #MinRetracement
        tolerance3 = 0.5 #Max Retracement
        tolerance4 = tolerance1*2 #CD tolearance
        tolerance5 = priceUnit*1.2 #Max Price move from extremum
        tolerance6 = priceUnit*0.2 #Min Price move from extremum
        if abs(C.value-D.value)!=0:
            retracemet = abs(C.value-B.value)/abs(C.value-D.value)
        else: 
            retracemet = 0
        extensionDiff = abs(abs(C.value-D.value)-abs(C.value-D.value))
        #Long
        self.abcdL = False
        self.abcdS = False
        if abs(D.value-C.value)>tolerance1 and retracemet>=tolerance2 and retracemet<=tolerance3 and extensionDiff<tolerance4 \
                    and abs(A.value-bar.Close)>tolerance6 and abs(A.value-bar.Close)<tolerance5:
            if A.trendCount<0 and A.value<B.value and A.value-C.value>tolerance4:
                self.abcdL = True
                self.zz.caller.signals += f'L_ABCD_{self.zz.name}-'
                if True and self.debug: self.zz.algo.MyDebug("Symbol:{}, ABCD Long ZZ:{}, barClose:{}, zzValue:{}, zzEndTime:{}, Index:{}, TrendCount:{}".format( \
                    str(self.zz.symbol), str(A.zz), str(round(bar.Close,2)), \
                    str(round(A.value,self.roundDigits)), str(A.endTime), str(A.index), str(A.trendCount),))        
            if A.trendCount>0 and A.value>B.value and C.value-A.value>tolerance4:
                self.abcdS = True
                self.zz.caller.signals += f'S_ABCD_{self.zz.name}-'
                if True and self.debug: self.zz.algo.MyDebug("Symbol:{}, ABCD Short ZZ:{}, barClose:{}, zzValue:{}, zzEndTime:{}, Index:{}, TrendCount:{}".format( \
                    str(self.zz.symbol), str(A.zz), str(round(bar.Close,2)), \
                    str(round(A.value,self.roundDigits)), str(A.endTime), str(A.index), str(A.trendCount),))  
        
        self.isReady = True
        return True
        
#NOT YET READY!!!!!
class MySupportResistance():
    def __init__(self, zzList):
        self.zzList = zzList
        self.tolerance1 = 1.0
        self.minindex = 10 #the latest zz point(s) is to be ignored 
        self.lines = []
        self.strength = 0
        
    def Update(self):
        del self.lines[:]
        self.stength = 0
        for zz in self.zzList:
            lines=0
            if len(zz.zzPoints)>0:
                minindex = zz.lookback+1
                tolerance1 = zz.priceUnit*self.tolerance1
                price = zz.algo.Portfolio.Securities[zz.symbol].Price
                for point in zz.zzPoints:
                    if abs(price-point.value)<tolerance1 and point.index>=minindex:
                        lines+=1
                        self.strength+=1
            self.lines.append(lines)
        return
 
    def ListSR(self):
        i=0
        for zz in self.zzList:
            zz.algo.MyDebug("Symbol:{}, {}. ZigZag:{}, Relevant Lines:{} Misc:{}".format( \
                str(zz.symbol), str(i+1), str(zz), str(self.lines[i]), str(zz.priceUnit)))
            i+=1
        return

'''
Donchian Channel State
'''  
class MyDCHState():
    signalMarginATRLong =  0.1
    signalMarginATRShort =  0.1
    freezeSignalLong = 0
    freezeSignalShort = 0
    gabor_diffUpper=True
    
    def __init__(self, caller, baseDCH, referenceDCH=-1, name=None):
        self.debug = False 
        self.CL = self.__class__
        self.caller = caller
        self.baseDCH = baseDCH
        self.referenceDCH = referenceDCH
        self.caller.stateUpdateList.append(self)
        self.atr = 0
        self.name = name
        self.name_L = f'L_DCH_{name}-'
        self.name_S = f'S_DCH_{name}-'
        
        self.size = 0 #in atr
        self.sizeRef = 0 #in atr
        self.diffLower = 0 #in sizeRef
        self.diffUpper = 0 #in sizeRef
        self.priceFromUpper = 0 #in atr
        self.priceFromLower = 0 #in atr
        
        self.status = 0 #Last Event (DCH New High or Low direction)
        self.statusChange = 0
        self.statusAge = 0
        
        self.event = 0 #+/-1 DCH New High or Low
        self.eventMove = 0 
        self.lastEventAge = 0
        self.eventNum = 0
        self.eventMoveSum = 0
        self.eventMoveSumATR = 0
        
        self.signal = 0
        self.signalDisabledLong = 0
        self.signalDisabledShort = 0
        
    def Update(self, caller, bar):
        if False and self.caller.symbol.Value =="MCD" and self.caller.algo.Time > datetime(2019, 9, 1, 9, 00) and self.caller.algo.Time <= datetime(2019, 9, 3, 21, 00): 
            self.debug = True
        elif False:
            self.debug = False
            
        if False and self.debug: self.caller.algo.MyDebug("  MyDonchianChannelState Update Called {}".format(str(self.caller.symbol)))
        if (not self.baseDCH.IsReady) or (self.referenceDCH !=-1 and (not self.referenceDCH.IsReady)): 
            if self.debug: self.caller.algo.MyDebug("  MyDonchianChannelState Update Returned {}".format(str(self.caller.symbol)))
            return
        if self.caller.atr1.Current.Value !=0:
            self.atr = self.caller.atr1.Current.Value
        else:
            self.atr = 1
        
        self.size = (self.baseDCH.UpperBand.Current.Value - self.baseDCH.LowerBand.Current.Value)/self.atr
        if self.referenceDCH !=-1:
            self.sizeRef = (self.referenceDCH.UpperBand.Current.Value - self.referenceDCH.LowerBand.Current.Value)/self.atr
        else:
            self.sizeRef = 0
        
        if self.referenceDCH !=-1 and self.sizeRef != 0:
            #self.diffUpper = (self.referenceDCH.UpperBand.Current.Value - self.baseDCH.UpperBand.Current.Value)/self.atr/self.sizeRef
            if not self.CL.gabor_diffUpper:
                #Old version
                self.diffUpper = (self.referenceDCH.UpperBand.Current.Value - self.baseDCH.UpperBand.Current.Value)/self.atr/self.sizeRef
            else:
                self.diffUpper = (self.baseDCH.UpperBand.Current.Value - self.referenceDCH.LowerBand.Current.Value)/self.atr/self.sizeRef
            self.diffLower = (self.baseDCH.LowerBand.Current.Value - self.referenceDCH.LowerBand.Current.Value)/self.atr/self.sizeRef
        else:
            self.diffLower = 0
            self.diffUpper = 0
        self.priceFromUpper = (self.baseDCH.UpperBand.Current.Value - bar.Close)/self.atr
        self.priceFromLower = (bar.Close - self.baseDCH.LowerBand.Current.Value)/self.atr
            
        self.event = 0
        self.statusChange = 0
        if bar.High > self.baseDCH.UpperBand.Current.Value: 
            self.event = 1
            self.eventMove = bar.High - self.baseDCH.UpperBand.Current.Value
            if self.status == -1: 
                self.statusChange = 1
            self.status = 1
        if bar.Low < self.baseDCH.LowerBand.Current.Value: 
            self.event = -1
            self.eventMove = bar.Low - self.baseDCH.LowerBand.Current.Value
            if self.status == 1: 
                self.statusChange = -1
            self.status = -1
        
        if self.event !=0:
            self.lastEventAge = 0
        else:
            self.lastEventAge += 1
            
        if self.statusChange !=0:
            self.statusAge = 0
            self.eventNum = 1
            self.eventMoveSum = self.eventMove
        else:
            self.statusAge += 1
            if self.event !=0:
                self.eventNum += 1
                self.eventMoveSum += self.eventMove
        self.eventMoveSumATR = self.eventMoveSum/self.atr
        
        if bar.Close > (self.baseDCH.UpperBand.Current.Value + self.CL.signalMarginATRLong*self.atr) and self.signalDisabledLong==0:
            self.signal = 1
            self.caller.signals += self.name_L
            self.signalDisabledLong = self.CL.freezeSignalLong+1
        elif bar.Close < (self.baseDCH.LowerBand.Current.Value - self.CL.signalMarginATRShort*self.atr) and self.signalDisabledShort==0:
            self.signal = -1
            self.caller.signals += self.name_S
            self.signalDisabledShort = self.CL.freezeSignalShort+1
        else: self.signal = 0
        
        self.signalDisabledLong = max(self.signalDisabledLong - 1, 0)
        self.signalDisabledShort = max(self.signalDisabledShort - 1, 0)

        if self.debug: 
            self.caller.algo.MyDebug ("  " + str(self.caller.symbol))
            self.caller.algo.MyDebug ("  {} Low:{} DCH.Low:{} statusAge:{}".
                format(str(self.caller.symbol),
                        str(bar.Low),
                        str(self.baseDCH.LowerBand.Current.Value),
                        str(self.statusAge)))
        return
    
    #FEATURE EXTRACTOR
    def FeatureExtractor(self, Type=1):
        ageNormaliser = 50
        atrNormaliser = 20*2
        eventNormaliser = 4
        bias=0.5
        features=[]
        if Type==1:
            features=[self.size/atrNormaliser, self.sizeRef/atrNormaliser, self.diffLower, self.diffUpper, self.priceFromUpper/atrNormaliser+bias, self.priceFromLower/atrNormaliser+bias]
        elif Type==2:
            features=[self.eventMoveSumATR/atrNormaliser+bias]
        elif Type==3:
            features=[self.status/eventNormaliser+bias, self.statusChange/eventNormaliser+bias, self.statusAge/ageNormaliser]
        elif Type==4:
            features=[self.event/eventNormaliser+bias, self.eventNum/ageNormaliser, self.lastEventAge/ageNormaliser]
        elif Type==5:
            features=[self.signal/eventNormaliser+bias]
        elif Type==6:
            features=[self.size/atrNormaliser, self.diffLower, self.diffUpper, self.priceFromUpper/atrNormaliser+bias, self.priceFromLower/atrNormaliser+bias]
        return features

'''
Mooving Average (Simple or Exponential) State
'''  
class MyMAState():
    signalMarginATRLong =  0.5
    signalMarginATRShort =  0.5
    freezeSignalLong = 0
    freezeSignalShort = 0
    
    def __init__(self, caller, baseMA, referenceMA=None):
        self.debug = False 
        self.CL = self.__class__
        self.caller = caller
        self.baseMA = baseMA
        self.referenceMA = referenceMA
        self.caller.stateUpdateList.append(self)
        self.atr = 0
        self.name_L = f'L_MA_{baseMA.Period}_{referenceMA.Period}-' if referenceMA!=None else f'-L_MA_{baseMA.Period}_na-'
        self.name_S = f'S_MA_{baseMA.Period}_{referenceMA.Period}-' if referenceMA!=None else f'-S_MA_{baseMA.Period}_na-'
        
        self.diff = 0 #in atr
        self.breakMA = 0
        self.lastHigh = 0
        self.lastHighFromBreak = 0 #in atr
        self.lastHighFromMA = 0 #in atr
        self.lastLow = 0
        self.lastLowFromBreak = 0 #in atr
        self.lastLowFromMA = 0 #in atr
        self.priceFromMA = 0 #in atr
        self.priceFromBreak = 0 #in atr
        
        self.status = 0
        self.statusChange = 0
        self.statusAge = 0
        
        self.event = 0 #New Extremum since break
        self.lastEventAge = 0
        self.eventNum = 0 #No of new extremums since break (including the break)
        
        self.signal = 0 #MA break
        self.lastSignalAge = 0
        self.signalDisabledLong = 0
        self.signalDisabledShort = 0

    def Update(self, caller, bar):
        if False and self.caller.symbol.Value =="MCD" and self.caller.algo.Time > datetime(2019, 9, 1, 9, 00) and self.caller.algo.Time <= datetime(2019, 9, 3, 21, 00): 
            self.debug = True
        elif False:
            self.debug = False
        
        if False and self.debug: self.caller.algo.MyDebug("  MyMAState Update Called {}".format(str(self.caller.symbol)))
        if (not self.baseMA.IsReady) or (self.referenceMA!=None and (not self.referenceMA.IsReady)): 
            if self.debug: self.caller.algo.MyDebug("  MyMAState Update Returned {}".format(str(self.caller.symbol)))
            return
        if self.caller.atr1.Current.Value !=0:
            self.atr = self.caller.atr1.Current.Value
        else:
            self.atr = 1
        
        price = self.caller.algo.Portfolio.Securities[self.caller.symbol].Price
        #Initializing
        if self.breakMA == 0: self.breakMA = price 
        if self.lastHigh == 0: self.lastHigh = price
        if self.lastLow == 0: self.lastLow = price
        
        self.event = 0
        self.statusChange = 0
        if bar.Close > (self.baseMA.Current.Value + self.CL.signalMarginATRLong*self.atr): 
            #Break
            if self.status == -1: 
                self.statusChange = 1
                self.breakMA = self.baseMA.Current.Value
                self.lastHigh = self.baseMA.Current.Value
            self.status = 1
            #New High since Break
            if bar.High > self.lastHigh:
                self.event = 1
                self.lastHigh = bar.High

        if bar.Close < (self.baseMA.Current.Value - self.CL.signalMarginATRLong*self.atr): 
            #Break
            if self.status == 1: 
                self.statusChange = -1
                self.breakMA = self.baseMA.Current.Value
                self.lastLow = self.baseMA.Current.Value  
            self.status = -1
            #New Low since Break
            if bar.Low < self.lastLow:
                self.event = -1
                self.lastLow = bar.Low
        
        #Extremum positions
        self.lastHighFromBreak = (self.lastHigh-self.breakMA)/self.atr
        self.lastHighFromMA = (self.lastHigh-self.baseMA.Current.Value)/self.atr
        self.lastLowFromBreak = (self.lastLow-self.breakMA)/self.atr
        self.lastLowFromMA = (self.lastLow-self.baseMA.Current.Value)/self.atr
                
        if self.event !=0:
            self.lastEventAge = 0
        else:
            self.lastEventAge += 1
            
        if self.statusChange !=0:
            self.statusAge = 0
            if self.event !=0:
                self.eventNum = 1
        else:
            self.statusAge += 1
            if self.event !=0:
                self.eventNum += 1
        
        if self.referenceMA!=None:
            self.diff = (self.baseMA.Current.Value - self.referenceMA.Current.Value)/self.atr
        else:
            self.diff = 0
        self.priceFromMA = (bar.Close - self.baseMA.Current.Value)/self.atr
        self.priceFromBreak = (bar.Close - self.breakMA)/self.atr
        
        if self.statusChange == 1 and bar.Close > (self.baseMA.Current.Value + self.CL.signalMarginATRLong*self.atr) and self.signalDisabledLong==0:
            self.signal = 1
            self.caller.signals += self.name_L
            self.signalDisabledLong = self.CL.freezeSignalLong+1
            self.lastSignalAge=0
        elif self.statusChange == -1 and bar.Close < (self.baseMA.Current.Value - self.CL.signalMarginATRShort*self.atr) and self.signalDisabledShort==0:
            self.signal = -1
            self.caller.signals += self.name_S
            self.signalDisabledShort = self.CL.freezeSignalShort+1
            self.lastSignalAge=0
        else: self.signal = 0
        self.lastSignalAge+=1
        
        self.signalDisabledLong = max(self.signalDisabledLong - 1, 0)
        self.signalDisabledShort = max(self.signalDisabledShort - 1, 0)
        
        if self.debug: 
            self.caller.algo.MyDebug ("  " + str(self.caller.symbol))
            self.caller.algo.MyDebug ("  {} Pric:{} MA:{} statusAge:{}".
                format(str(self.caller.symbol),
                        str(bar.Close),
                        str(self.baseMA.Current.Value),
                        str(self.statusAge)))
        return
    
    #FEATURE EXTRACTOR
    def FeatureExtractor(self, Type=1):
        ageNormaliser = 50
        atrNormaliser = 20*2
        eventNormaliser = 4
        bias=0.5
        features=[]
        if Type==1:
            features=[self.diff/atrNormaliser+bias, self.priceFromMA/atrNormaliser+bias, self.priceFromBreak/2/atrNormaliser+bias]
        elif Type==11:
            if self.referenceMA!=None:
                features=[self.diff/atrNormaliser+bias, self.priceFromMA/atrNormaliser+bias, self.priceFromBreak/2/atrNormaliser+bias]
            else:
                features=[self.priceFromMA/atrNormaliser+bias, self.priceFromBreak/2/atrNormaliser+bias]
        elif Type==2:
            features=[self.lastHighFromBreak/2/atrNormaliser+bias*0, self.lastHighFromMA/atrNormaliser+bias*0, self.lastLowFromBreak/2/atrNormaliser+bias*2, self.lastLowFromMA/atrNormaliser+bias*2]
        elif Type==3:
            features=[self.status/eventNormaliser+bias, self.statusChange/eventNormaliser+bias, self.statusAge/ageNormaliser]
        elif Type==4:
            features=[self.event/eventNormaliser+bias, self.lastEventAge/ageNormaliser, self.eventNum/ageNormaliser]
        elif Type==5:
            features=[self.signal/eventNormaliser+bias, self.lastSignalAge/ageNormaliser]
        return features
'''
Volatility: 
'''  
class MyVolatility():
    def __init__(self, caller, algo, symbol, name, period, atr, benchmarkTicker=None, benchmarkVolAttr=None):
        self.caller = caller
        self.algo = algo
        self.symbol = symbol
        self.name = name
        self.atr = atr
        self.benchmarkTicker = benchmarkTicker
        self.benchmarkSymbol = None #During Initialize self.algo.Securities[self.benchmarkTicker].Symbol might not be available (initialized later)
        self.benchmarkVolAttr = benchmarkVolAttr
        self.benchmarkVol = None
        self.Time = datetime.min
        self.atrVolatility = deque(maxlen=period)
        self.atrNormVolatility = deque(maxlen=period)
        self.atrRelVolatility = deque(maxlen=period)
        self.atrNormRelVolatility = deque(maxlen=period)
        self.Value = 0.5
        self.IsReady = False      
        self.benchmark_atrVolatility = 0
        self.benchmark_atrNormVolatility = 0
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return "{0} -> IsReady: {1}. Time: {2}. Value: {3}".format(self.name, self.IsReady, self.Time, self.Value)
    
    # Update method is mandatory!
    def Update(self, bar):
        #First Update: Set up benchmarkSymbol and benchmarkVol. During init benchmarksymbol is not added yet, so it has to be done at the forst update. 
        if self.benchmarkSymbol==None:
            self.benchmarkSymbol = self.algo.benchmarkSymbol if self.benchmarkTicker==None else self.algo.Securities[self.benchmarkTicker].Symbol
        if self.benchmarkVol==None:
            if self.benchmarkVolAttr!=None and hasattr(self.algo.mySymbolDict[self.benchmarkSymbol], self.benchmarkVolAttr):
                self.benchmarkVol = getattr(self.algo.mySymbolDict[self.benchmarkSymbol], self.benchmarkVolAttr)
            else:
                self.benchmarkVol = self.algo.mySymbolDict[self.benchmarkSymbol].vol

        normMultiplier = 1
        self.Time = bar.EndTime
        if bar.Close!=0 and self.atr.Current.Value!=0: 
            self.Value = self.atr.Current.Value / bar.Close *100
        else: 
            self.Value = 0.5
        self.atrVolatility.appendleft(self.Value)
        if 1+self.Value*normMultiplier>0:
            self.atrNormVolatility.appendleft(log(1+self.Value*normMultiplier,10))
        else:
            self.atrNormVolatility.appendleft(0)
                
        if len(self.benchmarkVol.atrVolatility)!=0:
            #if first bar in warmup and benchmark update comes after the symbol (this would result in benchmark data 1bar delay)
            self.benchmark_atrVolatility = self.benchmarkVol.atrVolatility[0]
            self.benchmark_atrNormVolatility = self.benchmarkVol.atrNormVolatility[0]
        else:
            self.benchmark_atrVolatility = 0.5
            self.benchmark_atrNormVolatility = log(1.5,10)
        atrRelVolatility = self.atrVolatility[0]/self.benchmark_atrVolatility if self.benchmark_atrVolatility!=0 else 1
        self.atrRelVolatility.appendleft(atrRelVolatility)
        if 1+atrRelVolatility*normMultiplier>0:
            self.atrNormRelVolatility.appendleft(log(1+atrRelVolatility*normMultiplier,10))
        else:
             self.atrNormRelVolatility.appendleft(0)
        
        self.IsReady = len(self.atrVolatility) == self.atrVolatility.maxlen and len(self.atrNormVolatility) == self.atrNormVolatility.maxlen
        return self.IsReady
    
    def VolSlope(self, lookback, relative=False):
        if not self.IsReady: return (0,0,0,0,0)

        if not relative:
            period = min(lookback, len(self.atrVolatility))
            x = np.arange(period,0,-1) #This must be in reserve order as deque[0] is the latest==[lookback] 
            y = np.array(list(self.atrVolatility)[0:period])
        else:
            period = min(lookback, len(self.atrRelVolatility))
            x = np.arange(period,0,-1) #This must be in reserve order as deque[0] is the latest==[lookback] 
            y = np.array(list(self.atrRelVolatility)[0:period])

        try: 
            regresults = list(map(float, stats.linregress(x,y)))
        except:
            regresults = (0,0,0,0,0)
        #slope, intercept, r_value, p_value, std_err = regresults
        #self.algo.Debug(regresults[0]*100)
        return regresults
    
    def VolFromAverage(self, period, relative=False):
        if not relative:
            if not self.IsReady or len(self.atrVolatility)<=period: return 0
            avg = sum(list(self.atrVolatility)[0:period])/max(len(list(self.atrVolatility)[0:period]),1)
            if avg==0 or self.atrVolatility[0]<=0: return 0
            volFromAverage = log(self.atrVolatility[0]/avg,10)+0.5
        else:
            if not self.IsReady or len(self.atrRelVolatility)<=period: return 0
            avg = sum(list(self.atrRelVolatility)[0:period])/max(len(list(self.atrRelVolatility)[0:period]),1)
            if avg==0 or self.atrRelVolatility[0]<=0: return 0
            volFromAverage = log(self.atrRelVolatility[0]/avg,10)+0.5
        return volFromAverage
   
    def VolChange(self, period, relative=False):
        if not relative:
            if not self.IsReady or len(self.atrVolatility)<=period: return 0
            if self.atrVolatility[period]==0: return 0
            change = self.atrVolatility[0]/self.atrVolatility[period]
            if change>0:
                volChange=log(change,10)+0.5
            else:
                volChange=0
        else:
            if not self.IsReady or len(self.atrRelVolatility)<=period: return 0
            if self.atrRelVolatility[period]==0: return 0
            change = self.atrRelVolatility[0]/self.atrRelVolatility[period]
            if change>0:
                volChange=log(change,10)+0.5
            else:
                volChange=0
        return volChange
        
    #FEATURE EXTRACTOR
    def FeatureExtractor(self, Type=1, lookbacklist=[15,35,70,100], avgPeriod=100):
        if Type==1:
            features = list(self.atrVolatility)
        elif Type==2:
            features = list(self.atrNormVolatility)
        elif Type==3:
            features = [self.atrNormVolatility[0]]
        elif Type==4:
            features = [self.atrNormVolatility[0], self.atrNormVolatility[-1]]
        elif Type==5:
            features = []
            features.append(self.atrNormVolatility[0])
            features.append(self.VolFromAverage(avgPeriod))
            features.append(self.benchmarkVol.atrNormVolatility[0])
            features.append(self.benchmarkVol.VolFromAverage(avgPeriod))
            
            changes = []
            relativechanges = []
            changesBenchmark = []
            for lookback in lookbacklist:
                #Norm Volatility Change
                changes.append(self.VolChange(lookback))
                relativechanges.append(self.VolChange(lookback, relative=True))
                changesBenchmark.append(self.benchmarkVol.VolChange(lookback))
            features.extend(changes)
            features.extend(relativechanges)
            features.extend(changesBenchmark)
        elif Type==51:
            features = []
            features.append(self.atrNormVolatility[0])
            features.append(self.VolFromAverage(avgPeriod))
            
            changes = []
            for lookback in lookbacklist:
                #Norm Volatility Change
                changes.append(self.VolChange(lookback))
            features.extend(changes)
        return features