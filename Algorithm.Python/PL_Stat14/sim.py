### <summary>
### Simulator for AI Statistics
### 
### </summary>
from QuantConnect.Orders import *
from QuantConnect.Orders.Fills import *
from QuantConnect.Orders.Fees import *
from QuantConnect.Orders import OrderStatus

import os
from sys import getsizeof as getsizeof
import shutil

from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
#import math
#import random

'''
Simulator to create data for machine learning.
Data is saved to a csv file.
It can be used only in local Lean environment 
'''  
class MySIMPosition():
    file = __file__
    debug = True
    modeQuantconnect = False
    debugFrequency = 1000
    rawClosedPositionsData = [] #list of completed positionData lists
    hasPosition=False
    useHeader=True
    statFolder="X:\\My Drive\\QCStats\\Stat14\\Stat-2005_2019-Eq-Sig_Sel\\"
    subStatName="S1" #.csv statFile filename starts with it
    saveFiles = [] #List of lists or tuples (filePath, fileName)
    simsOpened = 0
    simsClosed = 0
    
    @classmethod
    def SaveData(cls, algo):
        #Don't do anything if no sim position to save
        if len(cls.rawClosedPositionsData)<2:
             algo.MyDebug("    No Simulation Data to be saved!" )
             return

        if cls.modeQuantconnect:
            #In QC debug one item
            for item in range(0, len(cls.rawClosedPositionsData)):
                algo.MyDebug("  Sim Row "+str(item) +":"+ str(cls.rawClosedPositionsData[item]))
            algo.MyDebug("   Sims Opened:" + str(cls.simsOpened) + " Sims Closed:" +str(cls.simsClosed))
            return
        
        
        if not os.path.exists(cls.statFolder):
            os.makedirs(cls.statFolder)
              
        #Save Files
        timeStr = datetime.now().strftime("%Y%m%d_%H_%M")
        for savefile in cls.saveFiles:
            shutil.copyfile(os.path.abspath(savefile[0]), cls.statFolder + timeStr + '_' + savefile[1] +'.py')
        cls.saveFiles = []
            
        #Save rawClosedPositionsData to csv file
        statFile = cls.statFolder + cls.subStatName + "_"+ timeStr + ".csv"
        if os.path.exists(statFile):
            os.remove(statFile)
        df = pd.DataFrame(cls.rawClosedPositionsData[1:], columns=cls.rawClosedPositionsData[0])
        algo.MyDebug(f'    Saving data to {statFile}\n    {getsizeof(df)/10**6} MB')
        df = df.sort_values(by=["TimeStamp"])
        df.insert(loc=3, column='TimeConverted', value=datetime.now())
        for i in range(0, df.shape[0]):
            df.loc[df.index[i], 'TimeConverted'] = cls.GetBacktestTime(df.loc[df.index[i], 'TimeStamp'])
        df.to_csv(statFile, sep=',', index=False, header=True)
        return
    
    @classmethod
    def GetBacktestTime(cls, timeStamp):
        year=int(timeStamp[0:4])
        month=int(timeStamp[4:6])
        day=int(timeStamp[6:8])
        hour=int(timeStamp[9:11])
        min=int(timeStamp[12:14])
        btTime = datetime(year, month, day, hour, min)
        return btTime

    def __init__(self, symbolStrat, direction, timestamp, signal, features, \
                    simTradeTypes = ({'s':5, 't':35, 'mpo':1.5, 'sc':True, 'st':None, 'msd':3.0}), \
                    simMinMaxTypes = ({'n':5}),  \
                    MinMaxNormMethod = ["atr", "pct"][1], \
                    openUntil = datetime(2168, 6, 25, 15, 55), \
                    dataConsolidated = None):
        self.CL = self.__class__
        self.symbolStrat = symbolStrat
        self.algo = self.symbolStrat.algo
        self.direction = direction
        self.openUntil = openUntil
        self.entryPrice = self.algo.Portfolio.Securities[self.symbolStrat.symbol].Price
        self.positionData = []
        self.positionData.append(str(self.symbolStrat.symbol))
        self.positionData.append(direction)
        self.positionData.append(timestamp)
        self.positionData.append(signal)
        #self.positionData.extend(features)
        self.openTrades = []
        self.openPriceMinMaxes = []
        self.dataConsolidated = self.symbolStrat.consolidator.DataConsolidated if dataConsolidated==None else dataConsolidated
        self.dataConsolidated += self.Update
        self.rawDataHeader = []
        self.isClosed = False
        self.CL.simsOpened +=1

        #if self.CL.simsOpened % 10 !=0: self.algo.MyDebug(f' Sims Opened: {self.CL.simsOpened}')

        #If this is the first instance Initialize rawDataHeader
        if not self.CL.hasPosition: 
            self.rawDataHeader.extend(("Symbol","Direction","TimeStamp","Signal"))
            #Fatures Header
            for i in range(0, len(features)):
                if isinstance(features[i], list):
                    #features[i] is a list
                    for j in range(0, len(features[i])):
                        self.rawDataHeader.append("Feat"+str(i)+'_'+str(j))
                else:
                    #features[i] is a single number 
                    self.rawDataHeader.append("Feat"+str(i))

        #Fill in Features
        for i in range(0, len(features)):
            if isinstance(features[i], list):
                #features[i] is a list, extend unpacks the list
                self.positionData.extend(features[i])
            else:
                #features[i] is a single number
                self.positionData.append(features[i])

        '''Simulated Trades
        '''
        tradeNo = 0
        singleTradeOutput = [0, 0, 0, 0] #__PPct, _PAtr, _PayOff, _Bars
        if False: simTradeTypes=[] #Disable simTradeTypes
        for simTrade in simTradeTypes:
            stopPlacer =        simTrade['s'] 
            targetPlacer =      simTrade['t']       # Use None for payOff only. Target = max/min(minimum Payoff if any,targetPlacer)
            minPayOff =         simTrade['mpo']     # Use None to have no minPayOff
            scratchTrade =      simTrade['sc'] 
            stopTrailer =       simTrade['st']     # None if no trail
            minStopDistance =   simTrade['msd']*self.symbolStrat.atr1.Current.Value if 'msd' in simTrade else  3.0*self.symbolStrat.atr1.Current.Value
            targetPrice = None
            
            #If the first sim: Initialize rawDataHeader
            if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr", str(tradeNo)+"_PayOff", str(tradeNo)+"_Bars"))
            
            if self.direction==1:
                stopPriceTarget = min([bar.Low for bar in self.symbolStrat.bars_rw][0:stopPlacer])
                stopPrice = min(stopPriceTarget, self.entryPrice - minStopDistance)
                mintargetPrice = self.entryPrice + minPayOff * (self.entryPrice - stopPrice) if minPayOff!=None else self.entryPrice
                targetPriceTarget = max([bar.High for bar in self.symbolStrat.bars_rw][0:targetPlacer]) if targetPlacer!=None else self.entryPrice
                targetPrice = max(mintargetPrice, targetPriceTarget) if minPayOff!=None or targetPlacer!=None else None
                newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, stopPrice, targetPrice, scratchTrade, stopTrailer)
            if self.direction==-1:
                stopPriceTarget = max([bar.High for bar in self.symbolStrat.bars_rw][0:stopPlacer])
                stopPrice = max(stopPriceTarget, self.entryPrice + minStopDistance)
                mintargetPrice = self.entryPrice + minPayOff * (self.entryPrice - stopPrice) if minPayOff!=None else self.entryPrice
                targetPriceTarget = min([bar.Low for bar in self.symbolStrat.bars_rw][0:targetPlacer]) if targetPlacer!=None else self.entryPrice
                targetPrice = min(mintargetPrice, targetPriceTarget) if minPayOff!=None or targetPlacer!=None else None
                newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, stopPrice, targetPrice, scratchTrade, stopTrailer)
            self.positionData.extend(singleTradeOutput)
            tradeNo+=1
                  
        '''Simulated Period MinMaxes and Price change
        '''
        minMaxNo = 0
        singleMinMaxOutput = [0, 0, 0]
        if False: simMinMaxTypes = []   #Disable simMinMaxTypes
        for simMinMax in simMinMaxTypes:
            mmPeriod = simMinMax['n']
            
            #If the first sim: Initialize rawDataHeader
            if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(minMaxNo)+"_P", str(minMaxNo)+"_Min", str(minMaxNo)+"_Max"))
            
            newPriceMinMax = MyPriceMinMax(self, len(self.positionData), entryPrice=self.entryPrice, period=mmPeriod, normMethod=MinMaxNormMethod)
            self.positionData.extend(singleMinMaxOutput)
            minMaxNo+=1
 
        #If the first sim: Initialize rawClosedPositionsData (csv) Header with rawDataHeader
        if not self.CL.hasPosition and self.CL.useHeader: self.CL.rawClosedPositionsData.append(self.rawDataHeader)
        self.CL.hasPosition = True
        return
    
    def ClosePosition(self):
        self.CL.rawClosedPositionsData.append(self.positionData)
        self.CL.simsClosed +=1
        self.isClosed = True
        #Unsubscribes for DataConsolidated
        self.dataConsolidated -= self.Update

        if self.CL.simsClosed % self.CL.debugFrequency == 0: self.algo.MyDebug(f' ------- Total Sims Closed: {self.CL.simsClosed}, Running Sims:{self.CL.simsOpened-self.CL.simsClosed}')
        return
    
    def Update(self, caller, bar):
        #If this was the last(both lists are empty) and it is not closed yet: close Positions
        if not self.isClosed and len(self.openTrades)==0 and len(self.openPriceMinMaxes)==0:
            self.ClosePosition()
        return

'''
Simulated Trade
'''  
class MySIMTrade():
    def __init__(self, position, index, entryPrice, stopPrice, targetPrice=None, scratchTrade=False, stopTrailer=None):
        self.position = position
        self.index = index  #self.positionData index where the trade results to be put 
        self.direction = self.position.direction
        self.entryPrice = entryPrice
        self.stopPrice = stopPrice
        self.targetPrice = targetPrice
        self.scratchTrade =  scratchTrade
        self.stopTrailer = stopTrailer
        self.stopOrder = MySIMOrder(self, OrderType.StopMarket, stopPrice)
        if targetPrice!=None:
            self.targetOrder = MySIMOrder(self, OrderType.Limit, targetPrice)
            self.targetOrder.ocoOrderPair=self.stopOrder
            self.stopOrder.ocoOrderPair=self.targetOrder
        else:
            self.targetOrder = None
        self.atr = 0
        self.price = 0
        self.risk = self.direction*(self.entryPrice-self.stopPrice)
        self.closePrice = None
        self.profitPct = 0
        self.profitAtr = 0
        self.payoff = 0
        self.isClosed = False
        
        self.position.openTrades.append(self)
        #Subscribes for DataConsolidated
        self.position.dataConsolidated += self.Update
        self.openBars = 0
        return

    def Update(self, caller, bar):
        if self.isClosed:
            return
        self.openBars+=1 
        self.price = bar.Close
        self.atr = self.position.symbolStrat.atr1.Current.Value
        #Check Stop Order
        if self.stopOrder.satus == OrderStatus.Submitted:
            self.stopOrder.Update(bar)
        #Check Target Order
        if self.targetOrder!=None and self.targetOrder.satus == OrderStatus.Submitted:
            self.targetOrder.Update(bar)    
        
        #Close if DayTrade and still open (stop is still submitted): cancel orders and close trade
        if self.position.openUntil <= self.position.algo.Time and self.stopOrder.satus == OrderStatus.Submitted:
            slippage = self.stopOrder.Slippage()
            closePrice = bar.Close - slippage*self.direction
            self.stopOrder.satus = OrderStatus.Canceled
            if self.targetOrder!=None: self.targetOrder.satus = OrderStatus.Canceled
            self.CloseTrade(closePrice)
        return
    
    def CloseTrade(self, closePrice):
        self.closePrice = closePrice
        profit = self.direction*(self.closePrice-self.entryPrice)-self.Commission()
        if self.entryPrice!= 0: self.profitPct = profit/self.entryPrice
        if self.atr!= 0: self.profitAtr = profit/self.atr
        if self.risk!= 0: self.payoff = profit/self.risk
        #log Trade Results
        self.position.positionData[self.index] = self.profitPct
        self.position.positionData[self.index+1] = self.profitAtr
        self.position.positionData[self.index+2] = self.payoff
        self.position.positionData[self.index+3] = self.openBars

        self.position.openTrades.remove(self)
        self.isClosed = True
        #Unsubscribes for DataConsolidated
        self.position.dataConsolidated -= self.Update
        return
    
    #COMMISSION MODEL
    def Commission (self):
        commissionRatioEQ = 0.0005
        commissionRatioFX = 0.00004/1.1000
        if self.position.symbolStrat.isEquity:
            commission = 2*commissionRatioEQ * self.price
        else:
            commission = 2*commissionRatioFX * self.price
        return commission

'''
Simulated Order
'''  
class MySIMOrder():
    def __init__(self, trade, orderType, orderPrice):
        self.trade = trade
        self.orderType = orderType
        self.orderPrice = orderPrice
        self.satus = OrderStatus.Submitted
        self.fillPrice = None
        self.ocoOrderPair = None
        return

    def Update(self, bar):
        #Check Fill if it is a Stop Order and the order is still submitted
        if self.orderType == OrderType.StopMarket and self.satus == OrderStatus.Submitted:
            if self.trade.direction == 1:
                if bar.Low <= self.orderPrice:
                    self.satus = OrderStatus.Filled
                    if self.ocoOrderPair != None: self.ocoOrderPair.satus = OrderStatus.Canceled
                    self.fillPrice = min(bar.Open, self.orderPrice)-self.Slippage() #use min/max for gap down/up
                    self.trade.CloseTrade(self.fillPrice)
            else:
                if bar.High >= self.orderPrice:
                    self.satus = OrderStatus.Filled
                    if self.ocoOrderPair != None: self.ocoOrderPair.satus = OrderStatus.Canceled
                    self.fillPrice = max(bar.Open, self.orderPrice)+self.Slippage() #use min/max for gap down/up
                    self.trade.CloseTrade(self.fillPrice)
            
        #Check Fill if it is a Target Order and the order is still submitted
        if self.orderType == OrderType.Limit and self.satus == OrderStatus.Submitted:
            if self.trade.direction == 1:
                if bar.High >= self.orderPrice:
                    self.satus = OrderStatus.Filled
                    self.ocoOrderPair.satus = OrderStatus.Canceled
                    self.fillPrice = max(bar.Open,  self.orderPrice) #use max/min for gap up/down
                    self.trade.CloseTrade(self.fillPrice)
            else:
                if bar.Low <= self.orderPrice:
                    self.satus = OrderStatus.Filled
                    self.ocoOrderPair.satus = OrderStatus.Canceled
                    self.fillPrice = min(bar.Open, self.orderPrice) #use max/min for gap up/down
                    self.trade.CloseTrade(self.fillPrice)
        
        #if there is no fill Trail Stop
        if self.orderType == OrderType.StopMarket and self.satus == OrderStatus.Submitted:
            self.TrailStop(bar)
        return
    
    #STOP TRAILER
    def TrailStop(self, bar):
        if self.orderType!=OrderType.StopMarket:
            return
        
        scratchMargin = 0.20*self.trade.atr
        #scratchTrade
        if self.trade.scratchTrade and self.satus == OrderStatus.Submitted:
            #long Trade
            if self.trade.direction==1 and self.orderPrice<self.trade.entryPrice and (bar.Close-self.trade.entryPrice)>(self.trade.entryPrice-self.orderPrice+scratchMargin):
                self.orderPrice = self.trade.entryPrice+scratchMargin
                return
            #Short Trade
            if self.trade.direction==-1 and self.orderPrice>self.trade.entryPrice and (self.trade.entryPrice-bar.Close)>(self.orderPrice-self.trade.entryPrice+scratchMargin):
                self.orderPrice = self.trade.entryPrice-scratchMargin
                return
        
        #Trail rw[n]
        if self.trade.stopTrailer!=None and self.satus == OrderStatus.Submitted:
            #Long Trade if scratched
            if self.trade.direction == 1 and self.orderPrice >= self.trade.entryPrice:
                orderPriceTarget = min([bar.Low for bar in self.trade.position.symbolStrat.bars_rw][0:self.trade.stopTrailer])
                self.orderPrice = max(self.orderPrice, orderPriceTarget)
                return
            #Short Trade if scratched
            if self.trade.direction == -1 and self.orderPrice <= self.trade.entryPrice:
                orderPriceTarget = max([bar.High for bar in self.trade.position.symbolStrat.bars_rw][0:self.trade.stopTrailer])
                self.orderPrice = min(self.orderPrice, orderPriceTarget)
                return
        return

    #SLIPPAGE MODEL    
    def Slippage (self):
        slippageRatioEq = 0.01 
        slippageRatioFX = 0.01
        if self.trade.position.symbolStrat.isEquity:
            slippage = slippageRatioEq * self.trade.atr
        else:
            slippage = slippageRatioFX * self.trade.atr
        return slippage

'''
Price Min Max on the given period
'''  
class MyPriceMinMax():
    def __init__(self, position, index, entryPrice, period, normMethod):
        self.position = position
        self.index = index #self.positionData index where the trade results to be put 
        self.entryPrice = entryPrice
        self.priceMin = entryPrice
        self.priceMax = entryPrice
        self.period = period
        self.currentPeriod = 1 #as it incremented at the end of Update
        self.atr =  self.position.symbolStrat.atr1.Current.Value
        self.normMethod = normMethod
        self.normalizer = 5 if self.normMethod=="atr" else 0.01
        self.isClosed= False
        
        self.position.openPriceMinMaxes.append(self)
        self.position.dataConsolidated += self.Update
        return
    
    def Update(self, caller, bar):
        if self.isClosed or self.currentPeriod > self.period:
            return
        #Update Min Max
        self.priceMin = min(self.priceMin, bar.Low)
        self.priceMax = max(self.priceMax, bar.High)
        #Close it if this is the period
        if self.currentPeriod == self.period:
            if self.normMethod=="atr" and self.atr!=0:
                norm = self.normalizer * self.atr
            elif self.normMethod=="pct":
                norm = self.normalizer * bar.Close
            else:
                norm = 100
            self.position.positionData[self.index  ] = (bar.Close-self.entryPrice)/norm
            self.position.positionData[self.index+1] = (self.priceMin-self.entryPrice)/norm
            self.position.positionData[self.index+2] = (self.priceMax-self.entryPrice)/norm
            
            self.position.openPriceMinMaxes.remove(self)
            self.isClosed = True
            self.position.dataConsolidated -= self.Update
        self.currentPeriod+=1
        return