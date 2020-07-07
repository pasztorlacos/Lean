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
from math import log, cos
import pickle
import codecs

class MyBarStrength():
    '''
    Price Action: Bar Strength over the lookback(=reference price period) period'''  
    file = __file__

    def __init__(self, caller, algo, symbol, name, period, atr, lookbackLong=2, lookbackShort=2, \
                    priceActionMinATRLong=1.0, priceActionMaxATRLong=2.0, priceActionMinATRShort=1.0, priceActionMaxATRShort=2.0, referenceTypeLong='Close', referenceTypeShort='Close'):
        self.debug = False
        self.caller = caller
        self.algo = algo
        self.symbol = symbol
        self.name = name
        self.name_L = f'L_Str_{lookbackLong}_{round(priceActionMinATRLong,2)}_{round(priceActionMaxATRLong,2)}-'
        self.name_S = f'S_Str_{lookbackShort}_{round(priceActionMinATRShort,2)}_{round(priceActionMaxATRShort,2)}-'
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
            self.caller.signals += self.name_L
            if self.debug and not self.algo.IsWarmingUp:
                self.algo.MyDebug(f'  Symbol:{self.symbol}, Signal:{self.name_L}')
        #Short
        elif self.IsReady and referenceShort-bar.Close > self.priceActionMinATRShort*self.atr.Current.Value \
                and referenceShort-bar.Close < self.priceActionMaxATRShort*self.atr.Current.Value:
            self.Value = -1
            self.caller.signals += self.name_S
            if self.debug and not self.algo.IsWarmingUp:
                self.algo.MyDebug(f'  Symbol:{self.symbol}, Signal:{self.name_S}')
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
        self.debug = False
        self.caller = caller
        self.algo = algo
        self.symbol = symbol
        self.name = name
        self.name_L = f'L_Rej_{lookbackLong}_{round(rejectionPriceTravelLong,2)}_{round(rejectionPriceRangeLong,2)}-'
        self.name_S = f'S_Rej_{lookbackShort}_{round(rejectionPriceTravelShort,2)}_{round(rejectionPriceRangeShort,2)}-'
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
            self.caller.signals += self.name_L
            if self.debug and not self.algo.IsWarmingUp:
                self.algo.MyDebug(f'  Symbol:{self.symbol}, Signal:{self.name_L}')
        #Short
        elif self.IsReady and abs(bar.Close-referenceShort) < self.rejectionPriceRangeShort*self.atr.Current.Value \
                and maxShort-bar.Close > self.rejectionPriceTravelShort*self.atr.Current.Value:
            self.Value = -1
            self.caller.signals += self.name_S
            if self.debug and not self.algo.IsWarmingUp:
                self.algo.MyDebug(f'  Symbol:{self.symbol}, Signal:{self.name_S}')
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
Price Action: Brooks Price Action
'''  
class MyBPA():
    def __init__(self, caller, algo, symbol, name, period, atr, lookbackLong=1, lookbackShort=1, \
                    minBarAtrLong=1.0, minBarAtrShort=1.0, referenceTypeLong='Close', referenceTypeShort='Close', gapATRLimitLong=1.0, gapATRLimitShort=1.0):
        self.debug = False
        self.caller = caller
        self.algo = algo
        self.symbol = symbol
        self.name = name
        self.name_L = f'L_BPA_{lookbackLong}_{referenceTypeLong}-'
        self.name_S = f'S_BPA_{lookbackShort}_{referenceTypeShort}-'
        self.period = period
        self.atr = atr
        self.lookbackLong = lookbackLong
        self.lookbackShort = lookbackShort
        self.gapATRLimitLong = gapATRLimitLong
        self.gapATRLimitShort = gapATRLimitShort
        self.Time = datetime.min
        self.priceActions = deque(maxlen=period)
        self.trend = deque(maxlen=period)
        self.bars = deque(maxlen=max(lookbackLong,lookbackShort)+1)
        self.Value = 0
        self.IsReady = False
        
        self.minBarAtrLong = minBarAtrLong
        self.minBarAtrShort = minBarAtrShort
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
            minLong = self.bars[1].Low
            maxLong = self.bars[1].High
            for i in range(1,self.lookbackLong+1):
                minLong = min(minLong, self.bars[i].Low)
                maxLong = max(maxLong, self.bars[i].High)
            minShort = self.bars[1].Low
            maxShort = self.bars[1].High
            for i in range(1,self.lookbackShort+1):
                minShort = min(minShort, self.bars[i].Low)
                maxShort = max(maxShort, self.bars[i].High)
        
            if self.referenceTypeLong=='Close':
                referenceLong = self.bars[0].Close
            elif self.referenceTypeLong=='Open':
                referenceLong = self.bars[0].Open
            elif self.referenceTypeLong=='High':
                referenceLong = self.bars[0].High
            elif self.referenceTypeLong=='Low':
                referenceLong = self.bars[0].Low

            if self.referenceTypeShort=='Close':
                referenceShort = self.bars[0].Close
            elif self.referenceTypeShort=='Open':
                referenceShort = self.bars[0].Open
            elif self.referenceTypeShort=='High':
                referenceShort = self.bars[0].High
            elif self.referenceTypeShort=='Low':
                referenceShort = self.bars[0].Low
        
        gapATRLong = max(0, bar.Open - self.bars[1].High)/self.atr.Current.Value if self.IsReady and self.atr.Current.Value!=0 else 0
        gapATRShort = max(0, self.bars[1].Low - bar.Open)/self.atr.Current.Value if self.IsReady and self.atr.Current.Value!=0 else 0 
        
        #Long
        if self.IsReady and self.trend[0]==-1 and referenceLong>maxLong and (bar.Close - bar.Low)>self.atr.Current.Value*self.minBarAtrLong and gapATRLong<=self.gapATRLimitLong:
            self.Value = 1
            self.caller.signals += self.name_L
            self.trend.appendleft(1)
            if self.debug and not self.algo.IsWarmingUp:
                self.algo.MyDebug(f'  Symbol:{self.symbol}, Signal:{self.name_L}')
        #Short
        elif self.IsReady and self.trend[0]==1 and referenceShort < minShort and (bar.High - bar.Close)>self.atr.Current.Value*self.minBarAtrShort and gapATRShort<=self.gapATRLimitShort:
            self.Value = -1
            self.caller.signals += self.name_S
            self.trend.appendleft(-1)
            if self.debug and not self.algo.IsWarmingUp:
                self.algo.MyDebug(f'  Symbol:{self.symbol}, Signal:{self.name_S}')
        else: 
            self.Value = 0
            self.trend.appendleft(self.trend[0]) if len(self.trend)>0 else self.trend.appendleft(1)
        self.priceActions.appendleft(self.Value)

        self.IsReady = len(self.bars) == self.bars.maxlen
        return self.IsReady
    
    #FEATURE EXTRACTOR
    def FeatureExtractor(self, Type=1):
        eventNormaliser = 4
        bias=0.5
        if Type==1:
            features = list(self.priceActions)
        return features

'''
GASF Indicator
'''  
class MyGASF():
    '''
    Gramian Angular Summation Field for Convolutional Neural Network fetures '''
    file = __file__
        
    def __init__(self, caller, algo, symbol, name, period, atr, benchmarkTicker=None):
        self.caller = caller
        self.algo = algo
        self.symbol = symbol
        self.name = name
        self.benchmarkTicker = benchmarkTicker
        self.benchmarkSymbol = None
        self.atr = atr
        self.isEquity = self.caller.CL.isEquity if hasattr(self.caller.CL, 'isEquity') else False
        self.Value = 0
        self.IsReady = False
        self.myBars = deque(maxlen=period)
        self.benchmarkClose = deque(maxlen=period)
        self.relativeClose = deque(maxlen=period)
        self.volatility = deque(maxlen=period)

    # Update method is mandatory
    def Update(self, newbar):
        #This cannot be set up during _init_ as benchmark might not be available at that time
        if self.benchmarkSymbol == None:
            if self.benchmarkTicker == None:
                #if None Ticker use algo.benchmarkSymbol 
                self.benchmarkSymbol=self.algo.benchmarkSymbol
            else:
                #Get Symbol of the given Ticker
                self.benchmarkSymbol=self.algo.Securities[self.benchmarkTicker].Symbol
        
        benchmarkPrice = self.algo.Securities[self.benchmarkSymbol].Price if self.algo.Securities[self.benchmarkSymbol].Price!=0 else 1 
        self.myBars.appendleft(newbar)
        self.benchmarkClose.appendleft(benchmarkPrice)
        self.relativeClose.appendleft(newbar.Close/benchmarkPrice)
        self.volatility.appendleft(self.atr.Current.Value/newbar.Close)

        self.IsReady = len(self.myBars) == self.myBars.maxlen
        self.Value = self.myBars[0].Close
        return self.IsReady
    
    # Gramian Angular Summation Field Transformation
    # assumes x as normalized multidimensional (channels are in rows) numpy either as it is or pickled
    #   pickle:     x = codecs.encode(pickle.dumps(x, protocol= pickle.HIGHEST_PROTOCOL), "base64").decode()
    #   unpickle:   x = pickle.loads(codecs.decode(x.encode(), "base64"))
    def GASF_Transform (self, x, unPickle=False, intDecode=None, unPickleOnly=False, positiveNormalize=True):
        if unPickle: x = pickle.loads(codecs.decode(x.encode(), "base64"))
        if intDecode!=None: x = self.IntCoding(x, intDecode, encode=False)
        if unPickleOnly: return x

        #Transformation Function
        def GASF(x_v):
            n = len(x_v)
            x_v = np.arccos(x_v)
            x_v_gasf = np.zeros(n*n, dtype=np.float64).reshape(n, n)
            for i in range(n):
                for j in range(n):
                    x_v_gasf[i,j] = cos(x_v[i] + x_v[j])/2+0.5 if positiveNormalize else cos(x_v[i] + x_v[j])
            return x_v_gasf
        
        if len(x.shape) == 1:
            #Single vector
            #add 1 extra dim as cnn needs channels even if it is 1
            x_t = np.expand_dims(GASF(x), axis=0)
        else:
            #n=x.shape[0] channel matrix
            x_t = np.zeros(x.shape[0]*x.shape[1]*x.shape[1], dtype=np.float64).reshape(x.shape[0], x.shape[1], x.shape[1])
            for i in range(x.shape[0]):
                x_t[i] = GASF(x[i])
        return x_t
    
    #Normalize Numpy
    def Norm(self, x):
        _max = np.max(x)
        _min = np.min(x)
        _delta = _max - _min if _max!=_min else 1
        return (x-_min)/_delta

    # INTEGER CODING
    # It assumes Numpy array normalised between [0.0, 1.0]
    # Encodes data between [-128, 128] etc..
    def IntCoding(self, x, encodeType=np.int16, decodeType=np.single, encode=True):
        intNorm = {
            np.int8:   128,
            np.int16:  32767,
            np.int32:  2147483647,
            np.uint8:  255,
            np.uint16: 65535,
            np.uint32: 4294967295}
        if encode:
            if encodeType==np.int8 or encodeType==np.int16 or encodeType==np.int32:
                x_enc = x * 2*intNorm[encodeType]-intNorm[encodeType]
            else:
                x_enc = np.round(x * intNorm[encodeType])
            x_enc = x_enc.astype(encodeType)
            return x_enc
        else:
            x = x.astype(decodeType)
            if encodeType==np.int8 or encodeType==np.int16 or encodeType==np.int32:
                x_dec = (x + intNorm[encodeType])/(2*intNorm[encodeType])
            else:
                x_dec = x /  intNorm[encodeType]
            x_dec = x_dec.astype(decodeType)
            return x_dec
        
    #FEATURE EXTRACTOR
    #   For Simulation Stats use useGASF=False, picleFeatures=True
    def FeatureExtractor(self, featureType="Close", useGAP=True, useGASF=True, picleFeatures=False, useFloat32=True, intCode=None, preProcessor=None):

        if featureType == "Close":
            #Normaized Close only 
            _C_n = np.array([bar.Close for bar in self.myBars])
            Features = self.Norm(_C_n)
        
        elif featureType == "OHLC":
            #OHLC
            _O = self.Norm(np.array([bar.Open for bar in self.myBars]))
            _H = self.Norm(np.array([bar.High for bar in self.myBars]))
            _L = self.Norm(np.array([bar.Low for bar in self.myBars]))
            _C = self.Norm(np.array([bar.Close for bar in self.myBars]))
            Features = np.vstack((_O, _H, _L, _C))
        
        elif featureType in ["CULBG", "ULBG", "_U", "_L", "_B", "_G"]:
            #Close (if "CULBG"), Upper shadow, Lower shadow, Body, Gap
            _C_n = np.array([bar.Close for bar in self.myBars])
            _C = self.Norm(_C_n)
            _U = self.Norm(np.array([bar.High-bar.Close if bar.Close>bar.Open else bar.High-bar.Open for bar in self.myBars]))
            _L = self.Norm(np.array([bar.Open-bar.Low if bar.Close>bar.Open else bar.Close-bar.Low for bar in self.myBars]))
            _B = self.Norm(np.array([bar.Close-bar.Open for bar in self.myBars]))
            _C_n_t1 = np.roll(_C_n, shift=-1, axis = 0) 
            #_C_n_t1[len(_C_n_t1)-1] = 0 #this is no good as creates a huge gap at [len-1]
            _C_n_t1[len(_C_n_t1)-1] = _C_n_t1[max(len(_C_n_t1)-2,0)] #this creates a 0 gap at [len-1] so no need for clear it
            _G = np.array([bar.Open for bar in self.myBars])-_C_n_t1
            #_G[len(_G)-1] = 0
            _G = self.Norm(_G)
            if featureType == "CULBG":
                Features = np.vstack((_C, _U, _L, _B, _G)) if useGAP else np.vstack((_C, _U, _L, _B))
            elif featureType == "ULBG": 
                Features = np.vstack((_U, _L, _B, _G)) if useGAP else np.vstack((_U, _L, _B))
            elif featureType == "_U":
                Features = _U
            elif featureType == "_L":
                Features = _L
            elif featureType == "_B":
                Features = _B
            elif featureType == "_G":
                Features = _G

        elif featureType == "ULRange":
            #Normalized Upper and Lower Range
            _UR = self.Norm(np.array([bar.High-bar.Close for bar in self.myBars]))
            _LR = self.Norm(np.array([bar.Close-bar.Low for bar in self.myBars]))
            Features = np.vstack((_UR, _LR))

        elif featureType == "ULRG":
            #Normalized Upper and Lower Range and Gap
            _UR = self.Norm(np.array([bar.High-bar.Close for bar in self.myBars]))
            _LR = self.Norm(np.array([bar.Close-bar.Low for bar in self.myBars]))
            _C_n = np.array([bar.Close for bar in self.myBars])
            _C_n_t1 = np.roll(_C_n, shift=-1, axis = 0) 
            _C_n_t1[len(_C_n_t1)-1] = _C_n_t1[max(len(_C_n_t1)-2,0)] #this creates a 0 gap at [len-1]
            _G = self.Norm(np.array([bar.Open for bar in self.myBars])-_C_n_t1)
            Features = np.vstack((_UR, _LR, _G))

        elif featureType == "RelativePrice":
            Features = self.Norm(np.array(self.relativeClose))
        
        elif featureType == "Volume" and hasattr(self.myBars[0], 'Volume'):
            Features = self.Norm(np.array([bar.Volume for bar in self.myBars]))
        
        elif featureType == "Volatility":
            Features = self.Norm(np.array(self.volatility))
        
        #in case of vectors (time series only) one extra dim as channel=1 added during gasf so output also is in (1, ch, n, n) format
        #   if no gasf is used only pickled, numpy time series as vectors or OHLC stacked to matrix is pickled
        if useGASF: Features = self.GASF_Transform(Features, positiveNormalize=True)
        if useFloat32: Features = Features.astype(np.single)
        if intCode!=None: Features = self.IntCoding(Features, encodeType=intCode)
        if preProcessor!=None: Features = preProcessor.PreProcess(Features)
        if picleFeatures: Features = codecs.encode(pickle.dumps(Features, protocol= pickle.HIGHEST_PROTOCOL), "base64").decode()
        
        return Features