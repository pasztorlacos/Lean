### <summary>
### Technical Analysis  (part B.)
### 
### </summary>
from QuantConnect.Data.Market import TradeBar

from collections import deque
import pickle
import codecs

from datetime import datetime, timedelta
import numpy as np
from math import log, cos, sqrt
#import random

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
        self.name_L = f'L_Str_{name}-'
        self.name_S = f'S_Str_{name}-'
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
            referenceLong = getattr(self.bars[self.lookbackLong], self.referenceTypeLong, self.bars[self.lookbackLong].Close)
            referenceShort = getattr(self.bars[self.lookbackShort], self.referenceTypeShort, self.bars[self.lookbackShort].Close)

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
        self.name_L = f'L_Rej_{name}-'
        self.name_S = f'S_Rej_{name}-'
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
            # minLong = self.bars[0].Low
            # maxLong = self.bars[0].High
            # for i in range(0,self.lookbackLong):
            #     minLong = min(minLong, self.bars[i].Low)
            #     maxLong = max(maxLong, self.bars[i].High)
            # minShort = self.bars[0].Low
            # maxShort = self.bars[0].High
            # for i in range(0,self.lookbackShort):
            #     minShort = min(minShort, self.bars[i].Low)
            #     maxShort = max(maxShort, self.bars[i].High)
            minLong  = min([x.Low  for x in self.bars][0:self.lookbackLong])
            maxLong  = max([x.High for x in self.bars][0:self.lookbackLong])
            
            minShort = min([x.Low  for x in self.bars][0:self.lookbackShort])
            maxShort = max([x.High for x in self.bars][0:self.lookbackShort])   
            
            referenceLong = getattr(self.bars[self.lookbackLong], self.referenceTypeLong, self.bars[self.lookbackLong].Close)
            referenceShort = getattr(self.bars[self.lookbackShort], self.referenceTypeShort, self.bars[self.lookbackShort].Close)
 
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
                    minBarAtrLong=1.0, minBarAtrShort=1.0, referenceTypeLong='Close', referenceTypeShort='Close', gapATRLimitLong=1.0, gapATRLimitShort=1.0, forceTrend=True, needBPAinTrend=True, atrTrendChange=None):
        self.debug = False
        self.caller = caller
        self.algo = algo
        self.symbol = symbol
        self.name = name
        self.name_L = f'L_BPA_{name}-'
        self.name_S = f'S_BPA_{name}-'
        self.period = period
        self.atr = atr
        self.lookbackLong = lookbackLong
        self.lookbackShort = lookbackShort
        self.gapATRLimitLong = gapATRLimitLong
        self.gapATRLimitShort = gapATRLimitShort
        self.forceTrend = forceTrend
        self.needBPAinTrend = needBPAinTrend
        self.atrTrendChange = atrTrendChange
        self.Time = datetime.min
        self.priceActions = deque(maxlen=period)
        self.trend = deque(maxlen=period)
        self.bars = deque(maxlen=max(lookbackLong,lookbackShort)+1)
        self.Value = 0
        self.IsReady = False
        self.trendChangePrice = 0
        
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
        if self.trendChangePrice==0: self.trendChangePrice = bar.Close
        
        if self.IsReady:
            # minLong = self.bars[1].Low
            # maxLong = self.bars[1].High
            # for i in range(1,self.lookbackLong+1):
            #     minLong = min(minLong, self.bars[i].Low)
            #     maxLong = max(maxLong, self.bars[i].High)
            # minShort = self.bars[1].Low
            # maxShort = self.bars[1].High
            # for i in range(1,self.lookbackShort+1):
            #     minShort = min(minShort, self.bars[i].Low)
            #     maxShort = max(maxShort, self.bars[i].High)
            minLong  = min([x.Low  for x in self.bars][1:self.lookbackLong+1])
            maxLong  = max([x.High for x in self.bars][1:self.lookbackLong+1])
            
            minShort = min([x.Low  for x in self.bars][1:self.lookbackShort+1])
            maxShort = max([x.High for x in self.bars][1:self.lookbackShort+1])           
            
            referenceLong = getattr(self.bars[0], self.referenceTypeLong, self.bars[0].Close)
            referenceShort = getattr(self.bars[0], self.referenceTypeShort, self.bars[0].Close)

            gapATRLong = max(0, bar.Open - self.bars[1].High)/self.atr.Current.Value if self.IsReady and self.atr.Current.Value!=0 else 0
            gapATRShort = max(0, self.bars[1].Low - bar.Open)/self.atr.Current.Value if self.IsReady and self.atr.Current.Value!=0 else 0 
        
        #Long BPA Signal
        if self.IsReady and (self.trend[0]==-1 or not self.forceTrend) and referenceLong>maxLong and (bar.Close - bar.Low)>self.atr.Current.Value*self.minBarAtrLong and gapATRLong<=self.gapATRLimitLong:
            self.Value = 1
            self.caller.signals += self.name_L
            self.trend.appendleft(1)
            self.trendChangePrice = bar.Close
            if self.debug and not self.algo.IsWarmingUp:
                self.algo.MyDebug(f'  Symbol:{self.symbol}, Signal:{self.name_L}')
        
        #Short BPA Signal
        elif self.IsReady and (self.trend[0]==1 or not self.forceTrend) and referenceShort<minShort and (bar.High - bar.Close)>self.atr.Current.Value*self.minBarAtrShort and gapATRShort<=self.gapATRLimitShort:
            self.Value = -1
            self.caller.signals += self.name_S
            self.trend.appendleft(-1)
            self.trendChangePrice = bar.Close
            if self.debug and not self.algo.IsWarmingUp:
                self.algo.MyDebug(f'  Symbol:{self.symbol}, Signal:{self.name_S}')
        
        #No BPA Signal
        else: 
            self.Value = 0
            if self.needBPAinTrend:
                #It needs Signal to change the trand so trand remains the same
                self.trend.appendleft(self.trend[0]) if len(self.trend)>0 else self.trend.appendleft(1)
            else:
                #It is enough to have just Higher/Lower price OR (if set) atrTrendChange to change the trand (we do not look at price action size when calculating the trend)
                refPriceUp = bar.High
                refPriceDown = bar.Low
                if self.IsReady and self.trend[0]==-1 and ((self.atrTrendChange==None and referenceLong>maxLong) or (self.atrTrendChange!=None and refPriceUp-self.trendChangePrice>self.atrTrendChange*self.atr.Current.Value)):
                    self.trend.appendleft(1)
                    self.trendChangePrice = refPriceUp
                elif self.IsReady and self.trend[0]==1 and ((self.atrTrendChange==None and referenceShort<minShort) or (self.atrTrendChange!=None and self.trendChangePrice-refPriceDown>self.atrTrendChange*self.atr.Current.Value)):
                    self.trend.appendleft(-1)
                    self.trendChangePrice = refPriceDown
                else:
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
    GASF:   Gramian Angular Summation Field for Convolutional Neural Network fetures
    DFT:    Discrete Fourier Transform
    HE:     Hurst Exponent. HE is a number between 0 and 1, with H < 0.5 indicating mean reversion, H > 0.5 indicating a trending time series and H = 0.5 indicating a random walk.
    
'''  
class MyGASF():
    file = __file__
        
    def __init__(self, caller, algo, symbol, name, period, atr, benchmarkTicker=None):
        self.caller = caller
        self.algo = algo
        self.symbol = symbol
        self.name = name
        self.benchmarkTicker = benchmarkTicker
        self.benchmarkSymbol = None
        self.atr = atr
        if self.caller!=None:
            self.isEquity = True if not hasattr(self.caller.CL, 'isEquity') else self.caller.CL.isEquity
        else:
            self.isEquity = True
        self.Value = 0
        self.IsReady = False
        self.myBars = deque(maxlen=period)
        self.benchmarkClose = deque(maxlen=period)
        self.relativeClose = deque(maxlen=period)
        self.volatility = deque(maxlen=period)
        self.myValues = deque(maxlen=period)

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
        self.myValues.appendleft(self.Value)
        return self.IsReady
    
    # Gramian Angular Summation Field Transformation
    # assumes x as normalized multidimensional (channels are in rows) numpy either as it is or pickled
    #   pickle:     x = codecs.encode(pickle.dumps(x, protocol= pickle.HIGHEST_PROTOCOL), "base64").decode()
    #   unpickle:   x = pickle.loads(codecs.decode(x.encode(), "base64"))
    def GASF_Transform (self, x, unPickle=False, intDecode=None, unPickleOnly=False, positiveNormalize=True):
        if unPickle: x = pickle.loads(codecs.decode(x.encode(), "base64"))
        if intDecode!=None: x = self.IntCoding(x, intDecode, encode=False)
        if unPickleOnly: 
            if len(x.shape) == 1:
                #add 1 extra dim as cnn needs channels even if it is 1
                x = np.expand_dims(x, axis=0)
            return x

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
        
    # FEATURE EXTRACTOR
    #   For Simulation Stats use useGASF=False, picleFeatures=True
    def FeatureExtractor(self, n=None, featureType="Close", useGASF=False, useFloat32=True, intCode=None, preProcessor=None, picleFeatures=False, returnList=False):
        # Slicing to be after Norm() to preserve spatial context 
        n = min(n, len(self.myBars)) if n!=None else len(self.myBars)
        
        if featureType == "Close":
            #Normaized Close only 
            Features = self.Norm(np.array([bar.Close for bar in self.myBars]))[0:n]
        
        elif featureType == "OHLC":
            _O = np.array([bar.Open for bar in self.myBars])
            _H = np.array([bar.High for bar in self.myBars])
            _L = np.array([bar.Low for bar in self.myBars])
            _C = np.array([bar.Close for bar in self.myBars])
            Features = self.Norm(np.vstack((_O, _H, _L, _C)))[:,0:n]
        
        elif featureType in ["ULBG", "ULB", "_U", "_L", "_B", "_G"]:
            #Close (if "CULBG"), Upper shadow, Lower shadow, Body, Gap
            _C = np.array([bar.Close for bar in self.myBars])
            _U = np.array([bar.High-bar.Close if bar.Close>bar.Open else bar.High-bar.Open for bar in self.myBars])
            _L = np.array([bar.Open-bar.Low if bar.Close>bar.Open else bar.Close-bar.Low for bar in self.myBars])
            _B = np.array([bar.Close-bar.Open for bar in self.myBars])
            _C_t1 = np.roll(_C, shift=-1, axis = 0) 
            _G = np.array([bar.Open for bar in self.myBars])-_C_t1
            _G[len(_G)-1] = 0   #this creates a 0 gap at [len-1] as we don't know what the gap was there

            if featureType == "ULBG": 
                Features = self.Norm(np.vstack((_U, _L, _B, _G)))[:,0:n]
            elif featureType == "ULB": 
                Features = self.Norm(np.vstack((_U, _L, _B)))[:,0:n]
            elif featureType == "_U":
                Features = self.Norm(_U)[0:n]
            elif featureType == "_L":
                Features = self.Norm(_L)[0:n]
            elif featureType == "_B":
                Features = self.Norm(_B)[0:n]
            elif featureType == "_G":
                Features = self.Norm(_G)[0:n]

        elif featureType == "ULRange":
            #Normalized Upper and Lower Range
            _UR = np.array([bar.High-bar.Close for bar in self.myBars])
            _LR = np.array([bar.Close-bar.Low for bar in self.myBars])
            Features = self.Norm(np.vstack((_UR, _LR)))[:,0:n]

        elif featureType == "ULRG":
            #Normalized Upper and Lower Range and Gap
            _UR = np.array([bar.High-bar.Close for bar in self.myBars])
            _LR = np.array([bar.Close-bar.Low for bar in self.myBars])
            _C = np.array([bar.Close for bar in self.myBars])
            _C_t1 = np.roll(_C, shift=-1, axis = 0) 
            _G = np.array([bar.Open for bar in self.myBars])-_C_t1
            _G[len(_G)-1] = 0   #this creates a 0 gap at [len-1] as we don't know what the gap was there
            Features = self.Norm(np.vstack((_UR, _LR, _G)))[:,0:n]

        elif featureType == "RelativePrice":
            Features = self.Norm(np.array(self.relativeClose))[0:n]
        
        elif featureType == "Volume" and hasattr(self.myBars[0], 'Volume'):
            Features = self.Norm(np.array([bar.Volume for bar in self.myBars]))[0:n]
        
        elif featureType == "Volatility":
            Features = self.Norm(np.array(self.volatility))[0:n]
        
        # in case of vectors (time series only) one extra dim as channel=1 added during gasf so output also is in (1, ch, n, n) format
        #   if no gasf is used only pickled, numpy time series as vectors or OHLC like stacked to matrix is pickled
        if useGASF: Features = self.GASF_Transform(Features, positiveNormalize=True)
        if useFloat32: Features = Features.astype(np.single)
        if intCode!=None: Features = self.IntCoding(Features, encodeType=intCode)
        if preProcessor!=None: 
            if len(Features.shape)==1:
                #add 1 extra dim as cnn needs channels even if it is 1 (in PreProcess this is handled anyway)
                Features = np.expand_dims(Features, axis=0)
            Features = preProcessor.PreProcess(Features)
        if picleFeatures: Features = codecs.encode(pickle.dumps(Features, protocol= pickle.HIGHEST_PROTOCOL), "base64").decode()
        if not picleFeatures and returnList: 
            return np.reshape(Features, -1).tolist()
        else:
            return Features

    # DFT: Discrete Fourier Transform FEATURE EXTRACTOR
    #   
    #Normalize Numpy for DFT
    def NormDFT(self, x, bias=0.0, distribution=False):
        if distribution:
            if sum(x)==0: return x
            return x/sum(x)
        else:
            _max = np.max(x)
            _min = np.min(x)
            _delta = _max - _min if _max!=_min else 1
        return (x-_min)/_delta+bias
    
    #Commpress Spectrum to n_compressed
    def CompressedSpectrum(self, x, n_compressed, useLog=True, distribution=True):
        n = x.size
        n_dft = n//2

        fft = np.fft.fft(x)
        fft_amp = np.abs(fft)[:n_dft]
        
        n_compressed = min(n_compressed, n_dft)
        windowSize = n_dft/n_compressed
        currentSpectumValue = 0.0
        currentWindowFill = 0
        spectrum = list()
        for i in  range (n_dft):
            #This is the last portion of the current window
            if currentWindowFill+1 == windowSize or i==n_dft-1:
                currentSpectumValue += fft_amp[i]
                spectrum.append(currentSpectumValue)
                currentSpectumValue = 0.0
                currentWindowFill = 0
            elif currentWindowFill+1 > windowSize:
                currentSpectumValue += fft_amp[i] * (windowSize-currentWindowFill)
                spectrum.append(currentSpectumValue)
                currentSpectumValue = fft_amp[i] * (1-(windowSize-currentWindowFill))
                currentWindowFill = 1-(windowSize-currentWindowFill)
            else:
                currentSpectumValue += fft_amp[i]
                currentWindowFill += 1
        spectrum = np.asarray(spectrum) 
        spectrum = self.NormDFT(np.log(1+spectrum), distribution=distribution) if useLog else self.NormDFT(spectrum, distribution=distribution)
        return spectrum 
    
    # DFT Sate: Signal-Filtered_Signal Normalized
    # Low energy spectral components are filtered out
    # threshold: min energy contribution
    # k is used if not None: above and k_th highest energy component
    def DFT_State(self, x, threshold=0.025, k=None):
        n = x.size
        n_dft = n//2
        
        fft = np.fft.fft(x)
        fft_amp = np.abs(fft)[:n_dft]
        fft_energy = sum(fft_amp)
        
        if k!=None: 
            kth_largest = np.abs(fft[np.argsort(np.abs(fft))[-k]])
            fft_mod = [component if np.abs(component)>=kth_largest else  0.+0.j for component in fft]
        else:
            fft_mod = [component if np.abs(component)/fft_energy>=threshold else  0.+0.j for component in fft]
        fft_mod_n = np.sum(np.where(np.abs(fft_mod)!=0, 1, 0))
        x_mod = np.fft.ifft(fft_mod).real
        x0_difference = x[0] -x_mod[0]
        # Normalizing the difference: min_diff=-1.00, max_diff=+1.00, delta=2.00 
        x0_difference = (x0_difference + 1)/2
    
        #fft_mod_amp = np.abs(fft_mod)[:n_dft]
        fft_mod_density = fft_mod_n/len(fft_mod)
        return [x0_difference, fft_mod_density] 

    def FeatureExtractor_DFT(self, inputType="Close", n=None, featureType='Compressed', n_compressed=5, useLog=True, distribution=True, threshold=0.02, k=None):
        # Slicing to be before Norm() as no nedd to preserve spatial context rather we want to minimize the DC component of the spectrum
        n = min(n, len(self.myBars)) if n!=None else len(self.myBars)
        if n_compressed==None: n_compressed = n
        
        if inputType == "Close":
            x = self.NormDFT(np.array([bar.Close for bar in self.myBars][0:n]), bias=-0.50, distribution=False)
        elif inputType == "RelativePrice":
            x = self.NormDFT(np.array(self.relativeClose[0:n]), bias=-0.50, distribution=False)
        elif inputType == "Volatility":
            x = self.NormDFT(np.array(self.volatility[0:n]), bias=-0.50, distribution=False)
        
        if featureType=='Compressed':
            #CompressedSpectrum return numpy
            Features = self.CompressedSpectrum(x, n_compressed, useLog=useLog, distribution=distribution).tolist()
        
        if featureType=='DFT_Sate':
            Features = self.DFT_State(x, threshold=threshold, k=k)
        return Features
        
    # HE: HURST EXPONENT FEATURE EXTRACTOR
    #
    # HE Calculator
    def HurstExponent(self, x, lag1, lag2):
        lags = range(lag1, lag2)
        tau = [sqrt(np.std(np.subtract(x[:-lag], x[lag:]))) for lag in lags]
        m = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = m[0]*2
        return hurst
    
    def FeatureExtractor_HE(self, hursts, inputType="Close", n=None):
        # hursts must be a list of tuples (lag1, lag2)
        # No need to normalize as we analyze returns
        n = min(n, len(self.myBars)) if n!=None else len(self.myBars)
        
        if inputType == "Close":
            x = np.array([bar.Close for bar in self.myBars][0:n])
        elif inputType == "RelativePrice":
            x = np.array(self.relativeClose[0:n])
        elif inputType == "Volatility":
            x = np.array(self.volatility[0:n])
        
        Features = list()
        for hurst in hursts:
            Features.append(self.HurstExponent(x, hurst[0], hurst[1]))
        return Features