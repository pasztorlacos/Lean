### <summary>
### Technical Analysis  (part B.)
### 
### </summary>

from QuantConnect.Data.Market import TradeBar

#import math
#import random
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from math import log

class MyBarStrength():
    '''
    Price Action: Bar Strength over the lookback(=reference price period) period'''  
    file = __file__

    def __init__(self, caller, algo, symbol, name, period, atr, lookbackLong=2, lookbackShort=2, \
                    priceActionMinATRLong=1.0, priceActionMaxATRLong=2.0, priceActionMinATRShort=1.0, priceActionMaxATRShort=2.0, referenceTypeLong='Close', referenceTypeShort='Close'):
        self.caller = caller
        self.algo = algo
        self.symbol = symbol
        self.name = name
        self.period = period
        self.atr = atr
        self.lookbackLong = lookbackLong
        self.lookbackShort = lookbackShort
        self.Time = datetime.min
        self.priceActions = deque(maxlen=period)
        self.bars = deque(maxlen=max(lookbackLong,lookbackShort)+1)
        self.Value = 0
        self.IsReady = False
        
        self.priceActionMinATRLong = priceActionMinATRLong
        self.priceActionMaxATRLong = priceActionMaxATRLong
        self.priceActionMinATRShort = priceActionMinATRShort
        self.priceActionMaxATRShort = priceActionMaxATRShort
        self.referenceTypeLong = referenceTypeLong
        self.referenceTypeShort = referenceTypeShort

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return "{0} -> IsReady: {1}. Time: {2}. Value: {3}".format(self.name, self.IsReady, self.Time, self.Value)

    # Update method is mandatory!
    def Update(self, bar):
        self.bars.appendleft(bar)
        self.Time = bar.EndTime
        if self.IsReady:
            if self.referenceTypeLong=='Close':
                referenceLong = self.bars[self.lookbackLong].Close
            elif self.referenceTypeLong=='Open':
                referenceLong = self.bars[self.lookbackLong].Open
            elif self.referenceTypeLong=='High':
                referenceLong = self.bars[self.lookbackLong].High
            elif self.referenceTypeLong=='Low':
                referenceLong = self.bars[self.lookbackLong].Low

            if self.referenceTypeShort=='Close':
                referenceShort = self.bars[self.lookbackShort].Close
            elif self.referenceTypeShort=='Open':
                referenceShort = self.bars[self.lookbackShort].Open
            elif self.referenceTypeShort=='High':
                referenceShort = self.bars[self.lookbackShort].High
            elif self.referenceTypeShort=='Low':
                referenceShort = self.bars[self.lookbackShort].Low

        #Long
        if self.IsReady and bar.Close-referenceLong > self.priceActionMinATRLong*self.atr.Current.Value \
                and bar.Close-referenceLong < self.priceActionMaxATRLong*self.atr.Current.Value:
            self.Value = 1
        #Short
        elif self.IsReady and referenceShort-bar.Close > self.priceActionMinATRShort*self.atr.Current.Value \
                and referenceShort-bar.Close < self.priceActionMaxATRShort*self.atr.Current.Value:
            self.Value = -1
        else: self.Value = 0
        self.priceActions.appendleft(self.Value)

        self.IsReady = len(self.bars) == self.bars.maxlen
        return self.IsReady
    
    #FEATURE EXTRACTOR
    def FeatureExtractor(self, Type=1):
        eventNormaliser = 4
        bias=0.5
        if Type==1:
            features = list(self.priceActions)
        elif Type==2:
            features = list(self.priceActions)
            for i in range(0, len(features)):
                features[i] = features[i]/eventNormaliser+bias
        return features

'''
Price Action: Rejection over the lookback(=reference price period) period
'''  
class MyBarRejection():
    def __init__(self, caller, algo, symbol, name, period, atr, lookbackLong=3, lookbackShort=3, \
                    rejectionPriceTravelLong=2.0, rejectionPriceTravelShort=2.0, rejectionPriceRangeLong=1.0, rejectionPriceRangeShort=1.0, referenceTypeLong='Close', referenceTypeShort='Close'):
        self.caller = caller
        self.algo = algo
        self.symbol = symbol
        self.name = name
        self.period = period
        self.atr = atr
        self.lookbackLong = lookbackLong
        self.lookbackShort = lookbackShort
        self.Time = datetime.min
        self.priceActions = deque(maxlen=period)
        self.bars = deque(maxlen=max(lookbackLong,lookbackShort)+1)
        self.Value = 0
        self.IsReady = False
        
        self.rejectionPriceTravelLong = rejectionPriceTravelLong
        self.rejectionPriceTravelShort = rejectionPriceTravelShort
        self.rejectionPriceRangeLong = rejectionPriceRangeLong
        self.rejectionPriceRangeShort = rejectionPriceRangeShort
        self.referenceTypeLong = referenceTypeLong
        self.referenceTypeShort = referenceTypeShort

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return "{0} -> IsReady: {1}. Time: {2}. Value: {3}".format(self.name, self.IsReady, self.Time, self.Value)

    # Update method is mandatory!
    def Update(self, bar):
        self.bars.appendleft(bar)
        self.Time = bar.EndTime

        if self.IsReady:
            minLong = self.bars[0].Low
            maxLong = self.bars[0].High
            for i in range(0,self.lookbackLong):
                minLong = min(minLong, self.bars[i].Low)
                maxLong = max(maxLong, self.bars[i].High)
            minShort = self.bars[0].Low
            maxShort = self.bars[0].High
            for i in range(0,self.lookbackShort):
                minShort = min(minShort, self.bars[i].Low)
                maxShort = max(maxShort, self.bars[i].High)
        
            if self.referenceTypeLong=='Close':
                referenceLong = self.bars[self.lookbackLong].Close
            elif self.referenceTypeLong=='Open':
                referenceLong = self.bars[self.lookbackLong].Open
            elif self.referenceTypeLong=='High':
                referenceLong = self.bars[self.lookbackLong].High
            elif self.referenceTypeLong=='Low':
                referenceLong = self.bars[self.lookbackLong].Low

            if self.referenceTypeShort=='Close':
                referenceShort = self.bars[self.lookbackShort].Close
            elif self.referenceTypeShort=='Open':
                referenceShort = self.bars[self.lookbackShort].Open
            elif self.referenceTypeShort=='High':
                referenceShort = self.bars[self.lookbackShort].High
            elif self.referenceTypeShort=='Low':
                referenceShort = self.bars[self.lookbackShort].Low
                      
        #Long
        if self.IsReady and abs(bar.Close-referenceLong) < self.rejectionPriceRangeLong*self.atr.Current.Value \
                and bar.Close-minLong > self.rejectionPriceTravelLong*self.atr.Current.Value:
            self.Value = 1
        #Short
        elif self.IsReady and abs(bar.Close-referenceShort) < self.rejectionPriceRangeShort*self.atr.Current.Value \
                and maxShort-bar.Close > self.rejectionPriceTravelShort*self.atr.Current.Value:
            self.Value = -1
        else: self.Value = 0
        self.priceActions.appendleft(self.Value)

        self.IsReady = len(self.bars) == self.bars.maxlen
        return self.IsReady
    
    #FEATURE EXTRACTOR
    def FeatureExtractor(self, Type=1):
        eventNormaliser = 4
        bias=0.5
        if Type==1:
            features = list(self.priceActions)
        elif Type==2:
            features = list(self.priceActions)
            for i in range(0, len(features)):
                features[i] = features[i]/eventNormaliser+bias
        return features