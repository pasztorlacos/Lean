### <summary>
### Simulator for AI Statistics
### 
### </summary>
from QuantConnect.Orders import *
from QuantConnect.Orders.Fills import *
from QuantConnect.Orders.Fees import *
from QuantConnect.Orders import OrderStatus

#import math
#import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import os

import shutil

'''
Simulator
'''  
class MySIMPosition():
    file = __file__
    debug = True
    debugFrequency = 1000
    rawClosedPositionsData = [] #list of completed positionData lists
    useSingleFeatureList=False
    hasPosition=False
    modeQuantconnect = False
    useHeader=True
    subStatName="\\2015"
    statName = "Stat6_fx"+subStatName
    #statFolder="C:\\Github\\Stats\\"+statName+"\\"
    statFolder="X:\\My Drive\\QuantConnect\\Stats\\"+statName+"\\"
    saveFiles = [] #List of lists or tuples (filePath, fileName)
    simsOpened = 0
    simsClosed = 0
    
    @classmethod
    def SaveData(cls, algo):
        debug = False
        #Don't do anything if no sim position to save
        if len(cls.rawClosedPositionsData)<2:
             algo.MyDebug("    No Simulation Data to be saved!" )
             return

        if cls.modeQuantconnect and debug:
            #In QC debug one item
            for item in range(0, len(cls.rawClosedPositionsData)):
                algo.MyDebug("  Sim Row "+str(item) +":"+ str(cls.rawClosedPositionsData[item]))
            algo.MyDebug("   Sims Opened:" + str(cls.simsOpened) + " Sims Closed:" +str(cls.simsClosed))
            return
        
        
        if not os.path.exists(cls.statFolder):
            os.makedirs(cls.statFolder)
              
        #Save Files
        for savefile in cls.saveFiles:
            shutil.copyfile(os.path.abspath(savefile[0]), cls.statFolder + datetime.now().strftime("%Y%m%d_%H_%M") + '_' + savefile[1] +'.py')
            
        #Save rawClosedPositionsData to csv file
        statFile = cls.statFolder + datetime.now().strftime("%Y%m%d_%H_%M") + ".csv"
        if os.path.exists(statFile):
            os.remove(statFile)
        df = pd.DataFrame(cls.rawClosedPositionsData[1:], columns=cls.rawClosedPositionsData[0])
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


    def __init__(self, symbolStrat, direction, timestamp, signal, features, simTradeTypes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], simMinMaxTypes=[0,1,2,3]):
        self.CL = self.__class__
        self.symbolStrat = symbolStrat
        self.algo = self.symbolStrat.algo
        self.direction = direction
        self.entryPrice = self.algo.Portfolio.Securities[self.symbolStrat.symbol].Price
        self.positionData = []
        self.positionData.append(str(self.symbolStrat.symbol))
        self.positionData.append(direction)
        self.positionData.append(timestamp)
        self.positionData.append(signal)
        #self.positionData.extend(features)
        self.openTrades = []
        self.openPriceMinMaxes = []
        self.symbolStrat.consolidator.DataConsolidated += self.Update
        self.rawDataHeader = []
        self.isClosed = False
        self.CL.simsOpened +=1

        #if self.CL.simsOpened % 10 !=0: self.algo.MyDebug(f' Sims Opened: {self.CL.simsOpened}')

        #If this is the first instance Initialize rawDataHeader
        if not self.CL.hasPosition: 
            self.rawDataHeader.extend(("Symbol","Direction","TimeStamp","Signal"))
            if isinstance(features[0], list) and not self.CL.useSingleFeatureList:
                #if features is a list of lists
                for i in range(0, len(features)):
                    for j in range(0, len(features[i])):
                        self.rawDataHeader.append("Feat"+str(i)+'_'+str(j))
            else:
                #if features is a single list or useSingleFeatureList
                for i in range(len(features)):
                    self.rawDataHeader.append("Feat"+str(i))

        #Fill in Features
        if isinstance(features[0], list) and not self.CL.useSingleFeatureList:
            #if features is a list of lists
            for i in range(0, len(features)):
                self.positionData.extend(features[i])
        else:
            #if features is a single list
            self.positionData.extend(features)
            
        '''Simulated Trades
        '''
        #simTradeTypes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        #simTradeTypes = [0,3]
        tradeNo = 0
        singleTradeOutput = [0, 0, 0, 0] #__PPct, _PAtr, _PayOff, _Bars
        basePayOff = 1.5
        if True:
            #Trade_0 dch0 (3)
            if 0 in simTradeTypes:
                payOff=basePayOff #Use 0 if no targetPrice
                scratchTrade=True
                stopTrailer=None #None if no trail
                targetPrice=None #This is overwritten if payOff!=0
                minStopDistance = 2*self.symbolStrat.atr1.Current.Value
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr", str(tradeNo)+"_PayOff", str(tradeNo)+"_Bars"))
                if self.direction==1:
                    stopPrice=min(self.symbolStrat.dch0.LowerBand.Current.Value, self.entryPrice-minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                if self.direction==-1:
                    stopPrice=max(self.symbolStrat.dch0.UpperBand.Current.Value, self.entryPrice+minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                self.positionData.extend(singleTradeOutput)
                tradeNo+=1
            #Trade_1 dch0 (3)
            if 1 in simTradeTypes:
                payOff=2.0 #Use 0 if no targetPrice
                scratchTrade=True
                stopTrailer=None #None if no trail
                targetPrice=None #This is overwritten if payOff!=0
                minStopDistance = 2*self.symbolStrat.atr1.Current.Value
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr", str(tradeNo)+"_PayOff", str(tradeNo)+"_Bars"))
                if self.direction==1:
                    stopPrice=min(self.symbolStrat.dch0.LowerBand.Current.Value, self.entryPrice-minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                if self.direction==-1:
                    stopPrice=max(self.symbolStrat.dch0.UpperBand.Current.Value, self.entryPrice+minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                self.positionData.extend(singleTradeOutput)
                tradeNo+=1
            #Trade_2 dch0 (3)
            if 2 in simTradeTypes:
                payOff=3.0 #Use 0 if no targetPrice
                scratchTrade=True
                stopTrailer=None #None if no trail
                targetPrice=None #This is overwritten if payOff!=0
                minStopDistance = 2*self.symbolStrat.atr1.Current.Value
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr", str(tradeNo)+"_PayOff", str(tradeNo)+"_Bars"))
                if self.direction==1:
                    stopPrice=min(self.symbolStrat.dch0.LowerBand.Current.Value, self.entryPrice-minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                if self.direction==-1:
                    stopPrice=max(self.symbolStrat.dch0.UpperBand.Current.Value, self.entryPrice+minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                self.positionData.extend(singleTradeOutput)
                tradeNo+=1
            #Trade_3 dch01 (7)
            if 3 in simTradeTypes:
                payOff=basePayOff #Use 0 if no targetPrice
                scratchTrade=True
                stopTrailer=None #None if no trail
                targetPrice=None #This is overwritten if payOff!=0
                minStopDistance = 2*self.symbolStrat.atr1.Current.Value
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr", str(tradeNo)+"_PayOff", str(tradeNo)+"_Bars"))
                if self.direction==1:
                    stopPrice=min(self.symbolStrat.dch01.LowerBand.Current.Value, self.entryPrice-minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                if self.direction==-1:
                    stopPrice=max(self.symbolStrat.dch01.UpperBand.Current.Value, self.entryPrice+minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                self.positionData.extend(singleTradeOutput)
                tradeNo+=1
            #Trade_4 dch01 (7)
            if 4 in simTradeTypes:
                payOff=2.0 #Use 0 if no targetPrice
                scratchTrade=True
                stopTrailer=None #None if no trail
                targetPrice=None #This is overwritten if payOff!=0
                minStopDistance = 2*self.symbolStrat.atr1.Current.Value
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr", str(tradeNo)+"_PayOff", str(tradeNo)+"_Bars"))
                if self.direction==1:
                    stopPrice=min(self.symbolStrat.dch01.LowerBand.Current.Value, self.entryPrice-minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                if self.direction==-1:
                    stopPrice=max(self.symbolStrat.dch01.UpperBand.Current.Value, self.entryPrice+minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                self.positionData.extend(singleTradeOutput)
                tradeNo+=1
            #Trade_5 dch01 (7)
            if 5 in simTradeTypes:
                payOff=3.0 #Use 0 if no targetPrice
                scratchTrade=True
                stopTrailer=None #None if no trail
                targetPrice=None #This is overwritten if payOff!=0
                minStopDistance = 2*self.symbolStrat.atr1.Current.Value
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr", str(tradeNo)+"_PayOff", str(tradeNo)+"_Bars"))
                if self.direction==1:
                    stopPrice=min(self.symbolStrat.dch01.LowerBand.Current.Value, self.entryPrice-minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                if self.direction==-1:
                    stopPrice=max(self.symbolStrat.dch01.UpperBand.Current.Value, self.entryPrice+minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                self.positionData.extend(singleTradeOutput)
                tradeNo+=1
            #Trade_6 dch01 (7)
            if 6 in simTradeTypes:
                payOff=4.0 #Use 0 if no targetPrice
                scratchTrade=True
                stopTrailer=None #None if no trail
                targetPrice=None #This is overwritten if payOff!=0
                minStopDistance = 2*self.symbolStrat.atr1.Current.Value
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr", str(tradeNo)+"_PayOff", str(tradeNo)+"_Bars"))
                if self.direction==1:
                    stopPrice=min(self.symbolStrat.dch01.LowerBand.Current.Value, self.entryPrice-minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                if self.direction==-1:
                    stopPrice=max(self.symbolStrat.dch01.UpperBand.Current.Value, self.entryPrice+minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                self.positionData.extend(singleTradeOutput)
                tradeNo+=1
            #Trade_7 dch02 (20)
            if 7 in simTradeTypes:
                payOff=basePayOff  #Use 0 if no targetPrice
                scratchTrade=True
                stopTrailer=None #None if no trail
                targetPrice=None #This is overwritten if payOff!=0
                minStopDistance = 2*self.symbolStrat.atr1.Current.Value
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr", str(tradeNo)+"_PayOff", str(tradeNo)+"_Bars"))
                if self.direction==1:
                    stopPrice=min(self.symbolStrat.dch02.LowerBand.Current.Value, self.entryPrice-minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                if self.direction==-1:
                    stopPrice=max(self.symbolStrat.dch02.UpperBand.Current.Value, self.entryPrice+minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                self.positionData.extend(singleTradeOutput)
                tradeNo+=1       
            #Trade_8 dch02 (20)
            if 8 in simTradeTypes:
                payOff=2.0 #Use 0 if no targetPrice
                scratchTrade=True
                stopTrailer=None #None if no trail
                targetPrice=None #This is overwritten if payOff!=0
                minStopDistance = 2*self.symbolStrat.atr1.Current.Value
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr", str(tradeNo)+"_PayOff", str(tradeNo)+"_Bars"))
                if self.direction==1:
                    stopPrice=min(self.symbolStrat.dch02.LowerBand.Current.Value, self.entryPrice-minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                if self.direction==-1:
                    stopPrice=max(self.symbolStrat.dch02.UpperBand.Current.Value, self.entryPrice+minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                self.positionData.extend(singleTradeOutput)
                tradeNo+=1   
            #Trade_9 dch02 (20)
            if 9 in simTradeTypes:
                payOff=3.0 #Use 0 if no targetPrice
                scratchTrade=True
                stopTrailer=None #None if no trail
                targetPrice=None #This is overwritten if payOff!=0
                minStopDistance = 2*self.symbolStrat.atr1.Current.Value
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr",str(tradeNo)+"_PayOff", str(tradeNo)+"_Bars"))
                if self.direction==1:
                    stopPrice=min(self.symbolStrat.dch02.LowerBand.Current.Value, self.entryPrice-minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                if self.direction==-1:
                    stopPrice=max(self.symbolStrat.dch02.UpperBand.Current.Value, self.entryPrice+minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                self.positionData.extend(singleTradeOutput)
                tradeNo+=1   
            #Trade_10 dch1 (35)
            if 10 in simTradeTypes:
                payOff=basePayOff #Use 0 if no targetPrice
                scratchTrade=True
                stopTrailer=None #None if no trail
                targetPrice=None #This is overwritten if payOff!=0
                minStopDistance = 2*self.symbolStrat.atr1.Current.Value
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr", str(tradeNo)+"_PayOff", str(tradeNo)+"_Bars"))
                if self.direction==1:
                    stopPrice=min(self.symbolStrat.dch1.LowerBand.Current.Value, self.entryPrice-minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                if self.direction==-1:
                    stopPrice=max(self.symbolStrat.dch1.UpperBand.Current.Value, self.entryPrice+minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                self.positionData.extend(singleTradeOutput)
                tradeNo+=1  
            #Trade_11 dch1 (35)
            if 11 in simTradeTypes:
                payOff=2.0 #Use 0 if no targetPrice
                scratchTrade=True
                stopTrailer=None #None if no trail
                targetPrice=None #This is overwritten if payOff!=0
                minStopDistance = 2*self.symbolStrat.atr1.Current.Value
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr", str(tradeNo)+"_PayOff", str(tradeNo)+"_Bars"))
                if self.direction==1:
                    stopPrice=min(self.symbolStrat.dch1.LowerBand.Current.Value, self.entryPrice-minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                if self.direction==-1:
                    stopPrice=max(self.symbolStrat.dch1.UpperBand.Current.Value, self.entryPrice+minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                self.positionData.extend(singleTradeOutput)
                tradeNo+=1     
            #Trade_12 dch1 (35)
            if 12 in simTradeTypes:
                payOff=3.0 #Use 0 if no targetPrice
                scratchTrade=True
                stopTrailer=None #None if no trail
                targetPrice=None #This is overwritten if payOff!=0
                minStopDistance = 2*self.symbolStrat.atr1.Current.Value
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr", str(tradeNo)+"_PayOff", str(tradeNo)+"_Bars"))
                if self.direction==1:
                    stopPrice=min(self.symbolStrat.dch1.LowerBand.Current.Value, self.entryPrice-minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                if self.direction==-1:
                    stopPrice=max(self.symbolStrat.dch1.UpperBand.Current.Value, self.entryPrice+minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                self.positionData.extend(singleTradeOutput)
                tradeNo+=1
            #Trade_13 dch2 (70)
            if 13 in simTradeTypes:
                payOff=basePayOff #Use 0 if no targetPrice
                scratchTrade=True
                stopTrailer=None #None if no trail
                targetPrice=None #This is overwritten if payOff!=0
                minStopDistance = 2*self.symbolStrat.atr1.Current.Value
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr", str(tradeNo)+"_PayOff", str(tradeNo)+"_Bars"))
                if self.direction==1:
                    stopPrice=min(self.symbolStrat.dch2.LowerBand.Current.Value, self.entryPrice-minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                if self.direction==-1:
                    stopPrice=max(self.symbolStrat.dch2.UpperBand.Current.Value, self.entryPrice+minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                self.positionData.extend(singleTradeOutput)
                tradeNo+=1   
            #Trade_14 dch2 (70)
            if 14 in simTradeTypes:
                payOff=2.0 #Use 0 if no targetPrice
                scratchTrade=True
                stopTrailer=None #None if no trail
                targetPrice=None #This is overwritten if payOff!=0
                minStopDistance = 2*self.symbolStrat.atr1.Current.Value
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr", str(tradeNo)+"_PayOff"))
                if self.direction==1:
                    stopPrice=min(self.symbolStrat.dch2.LowerBand.Current.Value, self.entryPrice-minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                if self.direction==-1:
                    stopPrice=max(self.symbolStrat.dch2.UpperBand.Current.Value, self.entryPrice+minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                self.positionData.extend(singleTradeOutput)
                tradeNo+=1   
            #Trade_15 dch2 (70)
            if 15 in simTradeTypes:
                payOff=3.0 #Use 0 if no targetPrice
                scratchTrade=True
                stopTrailer=None #None if no trail
                targetPrice=None #This is overwritten if payOff!=0
                minStopDistance = 2*self.symbolStrat.atr1.Current.Value
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(tradeNo)+"_PPct", str(tradeNo)+"_PAtr", str(tradeNo)+"_PayOff"))
                if self.direction==1:
                    stopPrice=min(self.symbolStrat.dch2.LowerBand.Current.Value, self.entryPrice-minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                if self.direction==-1:
                    stopPrice=max(self.symbolStrat.dch2.UpperBand.Current.Value, self.entryPrice+minStopDistance)
                    if payOff!=0: targetPrice=self.entryPrice+payOff*(self.entryPrice-stopPrice)
                    newTrade = MySIMTrade(self, len(self.positionData), self.entryPrice, \
                            stopPrice, targetPrice, scratchTrade, stopTrailer)
                self.positionData.extend(singleTradeOutput)
                tradeNo+=1 
                
        '''Simulated Period MinMaxes and Price change
        '''
        #simMinMaxTypes=[0,1,2,3]
        #simMinMaxTypes = [0,1]
        minMaxNo = 0
        singleMinMaxOutput = [0, 0, 0]
        if True:
            #MinMax_1
            if 0 in simMinMaxTypes:
                mmPeriod=7
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(minMaxNo)+"_P", str(minMaxNo)+"_Min", str(minMaxNo)+"_Max"))
                newPriceMinMax = MyPriceMinMax(self, len(self.positionData), entryPrice=self.entryPrice, period=mmPeriod)
                self.positionData.extend(singleMinMaxOutput)
                minMaxNo+=1
            #Minmax_2
            if 1 in simMinMaxTypes:
                mmPeriod=15
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(minMaxNo)+"_P", str(minMaxNo)+"_Min", str(minMaxNo)+"_Max"))
                newPriceMinMax = MyPriceMinMax(self, len(self.positionData), entryPrice=self.entryPrice, period=mmPeriod)
                self.positionData.extend(singleMinMaxOutput)
                minMaxNo+=1
            #Minmax_3
            if 2 in simMinMaxTypes:
                mmPeriod=35
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(minMaxNo)+"_P", str(minMaxNo)+"_Min", str(minMaxNo)+"_Max"))
                newPriceMinMax = MyPriceMinMax(self, len(self.positionData), entryPrice=self.entryPrice, period=mmPeriod)
                self.positionData.extend(singleMinMaxOutput)
                minMaxNo+=1        
            #Minmax_4
            if 3 in simMinMaxTypes:
                mmPeriod=70
                if not self.CL.hasPosition and self.CL.useHeader: self.rawDataHeader.extend((str(minMaxNo)+"_P", str(minMaxNo)+"_Min", str(minMaxNo)+"_Max"))
                newPriceMinMax = MyPriceMinMax(self, len(self.positionData), entryPrice=self.entryPrice, period=mmPeriod)
                self.positionData.extend(singleMinMaxOutput)
                minMaxNo+=1     

        #Initialize rawClosedPositionsData (csv source) Header with rawDataHeader
        if not self.CL.hasPosition and self.CL.useHeader: self.CL.rawClosedPositionsData.append(self.rawDataHeader)
        self.CL.hasPosition = True
        return
    
    def ClosePosition(self):
        self.CL.rawClosedPositionsData.append(self.positionData)
        self.CL.simsClosed +=1
        self.isClosed = True
        
        if self.CL.simsClosed % self.CL.debugFrequency == 0: self.algo.MyDebug(f' ------- Total Sims Closed: {self.CL.simsClosed}, Running Sims:{self.CL.simsOpened-self.CL.simsClosed}')
        return
    
    def Update(self, caller, bar):
        #Subscribes for symbolStrat.consolidator.DataConsolidated
        #If this was the last(both lists are empty) and it is not closed yet: close Positions
        if not self.isClosed and len(self.openTrades)==0 and len(self.openPriceMinMaxes)==0:
            self.ClosePosition()
        return

'''
Simulated Trade
'''  
class MySIMTrade():
    def __init__(self, position, index, entryPrice, stopPrice, targetPrice=None,  scratchTrade=False, stopTrailer=None):
        self.position = position
        self.index = index #self.positionData index where the trade results to be put 
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
            self.targetOrder=None
        self.atr = 0
        self.price = 0
        self.risk = self.direction*(self.entryPrice-self.stopPrice)
        self.closePrice = None
        self.profitPct=0
        self.profitAtr=0
        self.payoff=0
        self.isClosed = False
        
        self.position.openTrades.append(self)
        self.position.symbolStrat.consolidator.DataConsolidated += self.Update
        self.openBars = 0
        return

    def Update(self, caller, bar):
        #Subscribes for symbolStrat.consolidator.DataConsolidated
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
        return
    
    def CloseTrade(self, closePrice):
        self.closePrice = closePrice
        profit = self.direction*(self.closePrice-self.entryPrice)-self.Commission()
        if self.entryPrice != 0: self.profitPct = profit/self.entryPrice
        if self.atr != 0: self.profitAtr = profit/self.atr
        if self.risk != 0: self.payoff = profit/self.risk
        #log Trade Results
        self.position.positionData[self.index] = self.profitPct
        self.position.positionData[self.index+1] = self.profitAtr
        self.position.positionData[self.index+2] = self.payoff
        self.position.positionData[self.index+3] = self.openBars

        self.position.openTrades.remove(self)
        self.isClosed = True
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
        scratchMargin = self.trade.atr*0.2
        #scratchTrade
        if self.trade.scratchTrade and self.orderType == OrderType.StopMarket and self.satus == OrderStatus.Submitted:
            #long Trade
            if self.trade.direction==1 and self.orderPrice<self.trade.entryPrice and (bar.Close-self.trade.entryPrice)>(self.trade.entryPrice-self.orderPrice+scratchMargin):
                self.orderPrice = self.trade.entryPrice+scratchMargin
                return
            #Short Trade
            if self.trade.direction==-1 and self.orderPrice>self.trade.entryPrice and (self.trade.entryPrice-bar.Close)>(self.orderPrice-self.trade.entryPrice+scratchMargin):
                self.orderPrice = self.trade.entryPrice-scratchMargin
                return
        
        #Trail 1)dch1
        if self.trade.stopTrailer==1 and self.orderType == OrderType.StopMarket and self.satus == OrderStatus.Submitted:
            #long Trade
            if self.trade.direction==1 and self.orderPrice>=self.trade.entryPrice:
                self.orderPrice = max(self.orderPrice, min(bar.Low, self.trade.position.symbolStrat.dch1.LowerBand.Current.Value))
                return
            #Short Trade
            if self.trade.direction==-1 and self.orderPrice<=self.trade.entryPrice:
                self.orderPrice = min(self.orderPrice, max(bar.High, self.trade.position.symbolStrat.dch1.UpperBand.Current.Value))
                return
        return

    #SLIPPAGE MODEL    
    def Slippage (self):
        slippageRatioEq = 0.01 
        slippageRatioFX = 0.01
        if self.trade.position.symbolStrat.isEquity:
            slippage = self.trade.atr*slippageRatioEq
        else:
            slippage = self.trade.atr*slippageRatioFX
        return slippage

'''
Price Min Max on the given period
'''  
class MyPriceMinMax():
    def __init__(self, position, index, entryPrice, period):
        self.position = position
        self.index = index #self.positionData index where the trade results to be put 
        self.entryPrice = entryPrice
        self.priceMin = entryPrice
        self.priceMax = entryPrice
        self.period = period
        self.currentPeriod = 1
        self.atr =  self.position.symbolStrat.atr1.Current.Value
        self.normalizer=5
        self.isClosed= False
        
        self.position.openPriceMinMaxes.append(self)
        self.position.symbolStrat.consolidator.DataConsolidated += self.Update
        return
    
    def Update(self, caller, bar):
        if self.isClosed or self.currentPeriod > self.period:
            return
        #Update Min Max
        self.priceMin = min(self.priceMin, bar.Low)
        self.priceMax = max(self.priceMax, bar.High)
        #Close it if this is the period
        if self.currentPeriod == self.period:
            if self.atr!=0:
                self.position.positionData[self.index  ] = (bar.Close-self.entryPrice)/self.atr/self.normalizer
                self.position.positionData[self.index+1] = (self.priceMin-self.entryPrice)/self.atr/self.normalizer
                self.position.positionData[self.index+2] = (self.priceMax-self.entryPrice)/self.atr/self.normalizer
            self.position.openPriceMinMaxes.remove(self)
            self.isClosed = True
        self.currentPeriod+=1
        return