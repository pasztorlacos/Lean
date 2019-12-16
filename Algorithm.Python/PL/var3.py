### <summary>
### VaR Statistics
### Charting
### 
### </summary>
from System.Drawing import Color
from datetime import timedelta
from QuantConnect import SecurityType, Chart, Series, SeriesType, ScatterMarkerSymbol
from QuantConnect.Orders import OrderDirection, OrderStatus

#--------------MyVaR CLASS for EXPOSURE
class MyVaR:
    '''
    Value at Risk and Exposure Calcualtions and Charting
    '''
    file = __file__
    #Portfolio Risk Limits
    buyingPowerUtilisation = 0.9
    manageTWSSymbols = True
    consistencyCheckSec = timedelta(seconds=30.123456789)
    '''Portfolio Risk Limits
    '''
    #Total VaR Limit
    maxTotalVaR = 0.1
    #Total Exposure Ratio Limits
    maxAbsExposure = 4.0
    maxLongExposure = 4.0 
    maxNetLongExposure = 4.0
    maxShortExposure = -4.0
    maxNetShortExposure = -4.0
    #Equity Exposure Ratio Limits
    maxAbsExposureEQ = 2.0
    maxLongExposureEQ = 2.0 
    maxNetLongExposureEQ = 2.0
    maxShortExposureEQ = -2.0
    maxNetShortExposureEQ = -2.0
    #Algo Liquidation Levels
    exposureBreachLimit = 1.2
    varBreachLimit = 1.2
    '''Portfolio Risk Values
    '''
    #Total Exposure Ratios
    _absExposure = 0
    _longExposure = 0
    _longPositions =0
    _shortExposure = 0
    _shortPositions = 0
    _netExposure = 0
    #Equity Exposure Ratios
    _absExposureEQ = 0
    _longExposureEQ = 0
    _longPositionsEQ =0
    _shortExposureEQ = 0
    _shortPositionsEQ = 0
    _netExposureEQ = 0
    #Total VaR Ratios
    _activeStops = 0
    _activeLimits = 0
    _openTotalVaR = 0
    _lastEntryOpenTotalVaR = 0
    _openLongVaR = 0
    _openShortVaR = 0
    '''Swiches
    '''
    _locked = False

    '''
    INITIALIZE: instantiated for each strategy
    '''
    def __init__(self, caller, strategy):
        self.CL = self.__class__
        self.algo = caller
        self.strategy = strategy
        self.icnludeinTotalVaR = True
        self.debug = False #self.algo.debug
        '''Startegy Exposure Ratios (per Allocated Capital)
        '''
        self.absExposure = 0
        self.longExposure = 0
        self.longPositions =0
        self.shortExposure = 0
        self.shortPositions = 0
        self.netExposure = 0
        #Actual Exposure Limits for the Startegy 
        self.longExposureLimit = 0
        self.shortExposureLimit = 0
        '''Startegy VaR Ratios (per Allocated Capital)
        '''
        self.activeStops = 0
        self.activeLimits = 0
        self.openTotalVaR = 0 
        self.openLongVaR = 0
        self.openShortVaR = 0
        #Actual VaR Limits for the Startegy 
        self.longVaRLimit = 0
        self.shortVaRLimit = 0
        
    '''
    UPDATE Stategy and Portfolio(Class Level)
    '''
    def Update (self):
        #Before TWS Sync Key exeption if foreign symbol exists
        if (not self.CL._locked) and (not self.algo.IsWarmingUp) and self.algo.twsSynced:
            if not self.algo.LiveMode: self.algo.myPositionManagerB.AllOrdersConsistency()
            self.CL._locked = True
            self.UpdateExposure()
            self.UpdateRiskLimits()
            self.UpdateVaR()
            self.CheckGlobalBreach()
            self.algo.myStats.Update()
            if not self.algo.IsWarmingUp:
                self.algo.myCharts.Update()
            self.CL._locked = False
        return
    '''
    EXPOSURE RATIOS UPDATE
    UPDATES EVERY MyVaR INSTANCE
    '''
    def UpdateExposure (self):
        #EXPOSURE RATIO Calculation
        #Portfolio
        self.CL._absExposure = 0
        self.CL._longExposure = 0
        self.CL._longPositions =0
        self.CL._shortExposure = 0
        self.CL._shortPositions = 0
        self.CL._netExposure = 0
        self.CL._absExposureEQ = 0
        self.CL._longExposureEQ = 0
        self.CL._longPositionsEQ =0
        self.CL._shortExposureEQ = 0
        self.CL._shortPositionsEQ = 0
        self.CL._netExposureEQ = 0
        #Strategies
        for var in self.algo.myVaRList:
            var.absExposure = 0
            var.longExposure = 0
            var.longPositions =0
            var.shortExposure = 0
            var.shortPositions = 0
            var.netExposure = 0

        portfolioValue = self.algo.Portfolio.TotalPortfolioValue
        '''POSITIONS EXPOSURE RATIOS'''
        for pos in self.algo.Portfolio:
            symbol = pos.Key
            price = self.algo.Portfolio.Securities[symbol].Price
            conversionRate = self.algo.Portfolio.Securities[symbol].QuoteCurrency.ConversionRate
            quantity = self.algo.Portfolio[symbol].Quantity
            var = self.algo.mySymbolDict[symbol].var
            strategy = self.algo.mySymbolDict[symbol].CL
            strategyCode = self.algo.mySymbolDict[symbol].CL.strategyCode
            strategyAllocation = strategy.strategyAllocation
            strategyMultiple  = 1000000 if strategyAllocation == 0 else 1/strategyAllocation
            isEquity = symbol.SecurityType == SecurityType.Equity
            
            #THIS IS CODE DUPLICATION TO BE REFACTORED!
            #Market Order Exposure of Entry and Liquidation(should offset position if liquidated after market hours)
            _SM = 0
            for tempTicket in self.algo.openMarketOrders:
                if tempTicket.Symbol == symbol and (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == 6 or tempTicket.Status == OrderStatus.New):
                    _SM += tempTicket.Quantity - tempTicket.QuantityFilled 
            if portfolioValue != 0:
                positionExposure = (quantity + _SM)* price * conversionRate / portfolioValue
            else: positionExposure = 0
            if quantity> 0:
                #Portfolio
                self.CL._absExposure += abs(positionExposure)
                self.CL._longExposure += positionExposure
                self.CL._longPositions += 1
                if isEquity:
                    self.CL._absExposureEQ += abs(positionExposure)
                    self.CL._longExposureEQ += positionExposure
                    self.CL._longPositionsEQ += 1
                #Strategy
                    var.absExposure += abs(positionExposure) * strategyMultiple
                    var.longExposure += positionExposure * strategyMultiple
                    var.longPositions += 1
            if quantity < 0 :
                #Portfolio
                self.CL._absExposure += abs(positionExposure)
                self.CL._shortExposure += positionExposure
                self.CL._shortPositions += 1 
                if isEquity:
                    self.CL._absExposureEQ += abs(positionExposure)
                    self.CL._shortExposureEQ += positionExposure
                    self.CL._shortPositionsEQ += 1 
                #Strategy
                var.absExposure += abs(positionExposure) * strategyMultiple
                var.shortExposure += positionExposure * strategyMultiple
                var.shortPositions += 1 
            #Debug
            if self.debug and positionExposure!=0: 
                self.algo.MyDebug("     Position Exposure Symbol: {}({}), Exposure:{}, TotalLongExposure:{}, TotalShortExposure:{}"
                    .format(str(symbol),
                            str(strategyCode),
                            str(round(positionExposure,3)),
                            str(round(self.CL._longExposure,3)),
                            str(round(self.CL._shortExposure,3))))
        
        '''PENDING LIMIT ENTRY POTENTIAL EXPOSURE RATIOS'''
        for symbol in self.algo.mySymbolDict:
            price = self.algo.Securities[symbol].Price
            conversionRate = self.algo.Portfolio.Securities[symbol].QuoteCurrency.ConversionRate
            var = self.algo.mySymbolDict[symbol].var
            strategy = self.algo.mySymbolDict[symbol].CL
            strategyCode = self.algo.mySymbolDict[symbol].CL.strategyCode
            strategyAllocation = strategy.strategyAllocation
            strategyMultiple  = 1000000 if strategyAllocation == 0 else 1/strategyAllocation
            isEquity = symbol.SecurityType == SecurityType.Equity
            
            #THIS IS CODE DUPLICATION TO BE REFACTORED!
            _SS, _SL, _SM = 0, 0, 0
            pendingEntryExposure = 0
            #Stop Orders
            for tempTicket in self.algo.openStopMarketOrders:
                if tempTicket.Symbol == symbol and (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == 6 or tempTicket.Status == OrderStatus.New):
                    _SS += tempTicket.Quantity - tempTicket.QuantityFilled
            #Limit Orders
            for tempTicket in self.algo.openLimitOrders:
                if tempTicket.Symbol == symbol and (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == 6 or tempTicket.Status == OrderStatus.New):
                    _SL += tempTicket.Quantity - tempTicket.QuantityFilled
            #Market Orders (just for condition as taken into account above)
            for tempTicket in self.algo.openMarketOrders:
                if tempTicket.Symbol == symbol and (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == 6 or tempTicket.Status == OrderStatus.New):
                    _SM += tempTicket.Quantity - tempTicket.QuantityFilled
            #Works if orders are consistent, otherwise it would overestimate (conservative)
            if portfolioValue !=0:
                useTargets =  strategy.useTragetsLong if _SS < 0 else strategy.useTragetsShort
                #If there is no market order (Liquidation or Entry)
                if _SM == 0:
                    pendingEntryExposure =  (_SL - _SS * useTargets) * price * conversionRate / portfolioValue
                else:
                    pendingEntryExposure = 0
                if pendingEntryExposure > 0:
                    #Portfolio
                    self.CL._absExposure += abs(pendingEntryExposure)
                    self.CL._longExposure += pendingEntryExposure
                    self.CL._longPositions += 0                    
                    if isEquity:
                        self.CL._absExposureEQ += abs(pendingEntryExposure)
                        self.CL._longExposureEQ += pendingEntryExposure
                        self.CL._longPositionsEQ += 0                          
                    #Strategy
                    var.absExposure += abs(pendingEntryExposure) * strategyMultiple
                    var.longExposure += pendingEntryExposure * strategyMultiple
                    var.longPositions += 0
                if pendingEntryExposure < 0:
                    #Portfolio
                    self.CL._absExposure += abs(pendingEntryExposure)
                    self.CL._shortExposure += pendingEntryExposure
                    self.CL._shortPositions += 0 
                    if isEquity:
                        self.CL._absExposureEQ += abs(pendingEntryExposure)
                        self.CL._shortExposureEQ += pendingEntryExposure
                        self.CL._shortPositionsEQ += 0 
                    #Strategy 
                    var.absExposure += abs(pendingEntryExposure) * strategyMultiple
                    var.shortExposure += pendingEntryExposure * strategyMultiple
                    var.shortPositions += 0             
                #Debug
                if self.debug and pendingEntryExposure!=0: 
                    self.algo.MyDebug("     Pending Entry Exposure Symbol: {}({}), Pending Exposure:{}, TotalLongExposure:{}, TotalShortExposure:{}"
                        .format(str(symbol),
                                str(strategyCode),
                                str(round(pendingEntryExposure,3)),
                                str(round(self.CL._longExposure,3)),
                                str(round(self.CL._shortExposure,3))))
        #Net Exposure
        self.CL._netExposure = self.CL._longExposure + self.CL._shortExposure
        self.CL._netExposureEQ = self.CL._longExposureEQ + self.CL._shortExposureEQ
        for var in self.algo.myVaRList:
            var.netExposure = var.longExposure + var.shortExposure
        return
    '''
    VaR RATIOS UPDATE
    UPDATES EVERY MyVaR INSTANCE
    '''
    def UpdateVaR (self):
        #Portfolio
        self.CL._activeStops = 0
        self.CL._activeLimits = 0
        self.CL._openTotalVaR = 0 
        self.CL._openLongVaR = 0
        self.CL._openShortVaR = 0
        #Srategies
        for var in self.algo.myVaRList:
            var.activeStops = 0
            var.activeLimits = 0
            var.openTotalVaR = 0 
            var.openLongVaR = 0
            var.openShortVaR = 0
        
        #Stops and VaR 
        #This includes Positions and Pending Positions as it relies only on Stop Orders
        #Pending Positions are overstated as current price is the basis not the entryprice. This is conservative.
        for tempTicket in self.algo.openStopMarketOrders:
            symbol = tempTicket.Symbol
            portfolioValue = self.algo.Portfolio.TotalPortfolioValue
            conversionRate =  self.algo.Portfolio.Securities[symbol].QuoteCurrency.ConversionRate
            strategy = self.algo.mySymbolDict[symbol].CL
            strategyAllocation = strategy.strategyAllocation
            strategyMultiple  = 1000000 if strategyAllocation == 0 else 1/strategyAllocation
            var = self.algo.mySymbolDict[symbol].var
            
            if portfolioValue !=0 and (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == 6 or tempTicket.Status == OrderStatus.New):
                self.CL._activeStops += 1
                var.activeStops += 1
                tempStopPrice = self.algo.Transactions.GetOrderById(tempTicket.OrderId).StopPrice
                if tempTicket.Quantity < 0: 
                    #Long
                    longVaR = (self.algo.Portfolio[symbol].Price - tempStopPrice) * conversionRate * -1*tempTicket.Quantity/portfolioValue
                    #Portfolio
                    self.CL._openLongVaR += longVaR
                    #Strategy
                    var.openLongVaR += longVaR * strategyMultiple
                else:
                    #Short
                    shortVaR = (tempStopPrice - self.algo.Portfolio[symbol].Price) * conversionRate * tempTicket.Quantity/portfolioValue
                    #Portfolio
                    self.CL._openShortVaR += shortVaR
                    #Strategy              
                    var.openShortVaR += shortVaR * strategyMultiple
            #Total VaR
            #Portfolio
            self.CL._openTotalVaR = self.CL._openLongVaR + self.CL._openShortVaR
            #Strategy
            var.openTotalVaR = var.openLongVaR + var.openShortVaR
        #Targets
        for tempTicket in self.algo.openLimitOrders:
            var = self.algo.mySymbolDict[tempTicket.Symbol].var
            if (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == 6):
                self.CL._activeLimits += 1
                var.activeLimits += 1
        return
    '''
    CHECK FOR GLOBAL LIMITS AND DISABLE ALGO IF BREACHED
    '''
    def CheckGlobalBreach(self):
        varBreachLimit = self.CL.varBreachLimit
        exposureBreachLimit = self.CL.exposureBreachLimit
        
        if self.CL._lastEntryOpenTotalVaR > self.CL.maxTotalVaR * varBreachLimit \
        or self.CL._absExposure > self.CL.maxAbsExposure * exposureBreachLimit \
        or self.CL._longExposure > self.CL.maxLongExposure * exposureBreachLimit \
        or self.CL._shortExposure < self.CL.maxShortExposure * exposureBreachLimit \
        or self.CL._netExposure > self.CL.maxNetLongExposure * exposureBreachLimit or self.CL._netExposure < self.CL.maxNetShortExposure * exposureBreachLimit \
        or self.CL._absExposureEQ > self.CL.maxAbsExposureEQ * exposureBreachLimit \
        or self.CL._longExposureEQ > self.CL.maxLongExposureEQ * exposureBreachLimit \
        or self.CL._shortExposureEQ < self.CL.maxShortExposureEQ * exposureBreachLimit \
        or self.CL._netExposureEQ  > self.CL.maxNetLongExposureEQ * exposureBreachLimit or self.CL._netExposureEQ < self.CL.maxNetShortExposureEQ * exposureBreachLimit:
            self.algo.MyDebug(" TOTAL RISK LIMIT BREACH: _TVaR:{}, _AbsExp:{}, _LExp:{}, _SExp:{}, _NExp:{}, _AbsExpEQ:{}, _LExpEQ:{}, _SExpEQ:{}, _NExpEQ:{}"
                        .format(str(round(self.CL._openTotalVaR,4)),
                                str(round(self.CL._absExposure,4)),
                                str(round(self.CL._longExposure,4)),
                                str(round(self.CL._shortExposure,4)),
                                str(round(self.CL._netExposure,4)),
                                str(round(self.CL._absExposureEQ,4)),
                                str(round(self.CL._longExposureEQ,4)),
                                str(round(self.CL._shortExposureEQ,4)),
                                str(round(self.CL._netExposureEQ,4))))
            if self.algo.enabled: self.algo.myPositionManagerB.LiquidateAlgo()
        return
    '''
    RISK LIMITS UPDATE
    UPDATES EVERY MyVaR INSTANCE
    '''
    def UpdateRiskLimits (self):
    #Portfolio EXPOSURE RATIO Limit
        totalLongLimit = max(min(self.CL.maxAbsExposure - self.CL._absExposure, \
                                    self.CL.maxLongExposure - self.CL._longExposure, \
                                    self.CL.maxNetLongExposure - self.CL._netExposure),0)
        totalshortLimit = min(max(-1*(self.CL.maxAbsExposure - self.CL._absExposure), \
                                    self.CL.maxShortExposure - self.CL._shortExposure, \
                                    self.CL.maxNetShortExposure - self.CL._netExposure),0)
        
        totalLongLimitE = max(min(totalLongLimit,
                                    self.CL.maxAbsExposureEQ - self.CL._absExposureEQ, \
                                    self.CL.maxLongExposureEQ - self.CL._longExposureEQ, \
                                    self.CL.maxNetLongExposureEQ - self.CL._netExposureEQ),0)
        totalshortLimitE = min(max(totalshortLimit,
                                    -1*(self.CL.maxAbsExposureEQ - self.CL._absExposureEQ), \
                                    self.CL.maxShortExposureEQ - self.CL._shortExposureEQ, \
                                    self.CL.maxNetShortExposureEQ - self.CL._netExposureEQ),0)
        #Portfolio VaR RATIO Limit
        totalVaRLimit = max(0, self.CL.maxTotalVaR-self.CL._openTotalVaR)
        
        #Strategies
        #Strategy Ratios are on Strategy level so they are applied to: self.algo.Portfolio.TotalPortfolioValue * sd.CL.
        #therefore Portfolio level limits must be adjusted by strategyMultiple
        for var in self.algo.myVaRList:
            strategyAllocation = var.strategy.strategyAllocation
            strategyMultiple  = 0 if strategyAllocation == 0 else 1/strategyAllocation
            if var.strategy.isEquity:
                mytotalLongLimit = totalLongLimitE * strategyMultiple
                mytotalshortLimit = totalshortLimitE * strategyMultiple
            else: 
                mytotalLongLimit = totalLongLimit * strategyMultiple
                mytotalshortLimit = totalshortLimit * strategyMultiple
            
            var.longExposureLimit = max(min(mytotalLongLimit, \
                                            var.strategy.maxSymbolAbsExposure, \
                                            var.strategy.maxLongExposure - var.longExposure, \
                                            var.strategy.maxNetLongExposure - var.netExposure, \
                                            var.strategy.maxAbsExposure - var.absExposure),0)
            var.shortExposureLimit = min(max(mytotalshortLimit, \
                                            -1*var.strategy.maxSymbolAbsExposure, \
                                            var.strategy.maxShortExposure - var.shortExposure, \
                                            var.strategy.maxNetShortExposure - var.netExposure, \
                                            -1*(var.strategy.maxAbsExposure - var.absExposure)),0)
            #VaR Limit Calculations
            var.longVaRLimit = max(0, min(totalVaRLimit*strategyMultiple, var.strategy.maxLongVaR-var.openLongVaR, var.strategy.riskperLongTrade))
            var.shortVaRLimit = max(0, min(totalVaRLimit*strategyMultiple, var.strategy.maxShortVaR-var.openShortVaR, var.strategy.riskperShortTrade))
    '''
    ORDER LIST    Lists Stop, Limit and Market Orders, only Debug and Log Functionality
    '''
    def OrderList(self):
        self.algo.MyDebug(" ---- ACTIVE ORDERS:")
        orderCount=0
        for x in self.algo.openStopMarketOrders :
            if (x.Status == OrderStatus.Submitted or x.Status == 6):
                orderCount += 1
                self.algo.MyDebug("      {}. STOP Symbol:{}({}), ID:{}, BrokerID:{}, Quantity:{}, StopPrice:{}, Staus:{}, Position Quantity:{}"
                   .format(str(orderCount),
                           str(x.Symbol),
                           str(self.algo.mySymbolDict[x.Symbol].CL.strategyCode),
                           str(x.OrderId),
                           str(self.algo.Transactions.GetOrderById(x.OrderId).BrokerId),
                           str(x.Quantity),
                           str(self.algo.Transactions.GetOrderById(x.OrderId).StopPrice),
                           str(x.Status),
                           str(round(self.algo.Portfolio[x.Symbol].Quantity))))    
        for x in self.algo.openLimitOrders :
            if (x.Status == OrderStatus.Submitted or x.Status == 6):
                orderCount += 1
                self.algo.MyDebug("      {}. LIMIT Symbol:{}({}), ID:{}, BrokerID:{}, Quantity:{}, LimitPrice:{}, Staus:{}, Position Quantity:{}"
                   .format(str(orderCount),
                           str(x.Symbol),
                           str(self.algo.mySymbolDict[x.Symbol].CL.strategyCode),
                           str(x.OrderId),
                           str(self.algo.Transactions.GetOrderById(x.OrderId).BrokerId),
                           str(x.Quantity),
                           str(self.algo.Transactions.GetOrderById(x.OrderId).LimitPrice),
                           str(x.Status),
                           str(round(self.algo.Portfolio[x.Symbol].Quantity))))    
        for x in self.algo.openMarketOrders :
            if (x.Status == OrderStatus.Submitted or x.Status == 6):
                orderCount += 1
                self.algo.MyDebug("      {}. MARKET Symbol:{}({}), ID:{}, BrokerID:{}, Quantity:{}, Staus:{}, Position Quantity:{}"
                   .format(str(orderCount),
                           str(x.Symbol),
                           str(self.algo.mySymbolDict[x.Symbol].CL.strategyCode),
                           str(x.OrderId),
                           str(self.algo.Transactions.GetOrderById(x.OrderId).BrokerId),
                           str(x.Quantity),
                           str(x.Status),
                           str(round(self.algo.Portfolio[x.Symbol].Quantity))))    
        if orderCount == 0:
            self.algo.MyDebug("       NO Active Orders")
        return
    
    '''
    PORTFOLIO LIST    Lists Portfolio Items, only Debug and Log Functionality
    '''
    def PortfolioList(self, positiononly):
        i = 1 
        if self.algo.LiveMode:
            self.algo.MyDebug(" ---- PORTFOLIO ITEMS:")
            self.algo.MyDebug("       Positions Only: " + str(positiononly))
            self.algo.MyDebug("       Portfolio Items:" + str(self.algo.Portfolio.Count) + "  Securities:" + str(self.algo.Securities.Count)+ "  mySymbols:" + str(len(self.algo.mySymbolDict))) 
            self.algo.MyDebug("       TotalHoldingsValue/TotalPortfolioValue:" +str(round(self.algo.Portfolio.TotalHoldingsValue)) + "/" + str(round(self.algo.Portfolio.TotalPortfolioValue)))
            self.algo.MyDebug("       Margin Remaining (Long/Short) for " + str(self.algo.benchmarkSymbol) + ": "+ str(round(self.algo.Portfolio.GetMarginRemaining(self.algo.benchmarkSymbol,OrderDirection.Buy)))+ "/" + str(round(self.algo.Portfolio.GetMarginRemaining(self.algo.benchmarkSymbol,OrderDirection.Sell))))
            self.algo.MyDebug("       Buying Power (Long/Short) for " + str(self.algo.benchmarkSymbol) + ": "+ str(round(self.algo.Portfolio.GetBuyingPower(self.algo.benchmarkSymbol,OrderDirection.Buy)))+ "/" + str(round(self.algo.Portfolio.GetBuyingPower(self.algo.benchmarkSymbol,OrderDirection.Sell))))
            self.algo.MyDebug("       Conversion Rate for " + str(self.algo.benchmarkSymbol) + ": "+ str(round(self.algo.Portfolio.Securities[self.algo.benchmarkSymbol].QuoteCurrency.ConversionRate,5)))
            
            for x in self.algo.Portfolio:
                symbol = x.Key
                if self.algo.Portfolio[symbol].Quantity !=0 or not positiononly:
                    totalTickets = 0
                    totalStops = 0
                    quantityStops = 0
                    totalLimits = 0
                    quantityLimits = 0
                    totalMarkets = 0
                    quantityMarkets =0
                    for tempTicket in self.algo.Transactions.GetOrderTickets():
                        totalTickets +=1
                        if (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == 6) or False:    
                            if tempTicket.Symbol == symbol and tempTicket.OrderType == OrderType.StopMarket:
                                totalStops += 1
                                quantityStops += tempTicket.Quantity
                            elif tempTicket.Symbol == symbol and tempTicket.OrderType == OrderType.Limit:
                                totalLimits += 1
                                quantityLimits += tempTicket.Quantity
                            elif tempTicket.Symbol == symbol and (tempTicket.OrderType == OrderType.Market or tempTicket.OrderType == OrderType.MarketOnOpen):
                                totalMarkets += 1
                                quantityMarkets += tempTicket.Quantity
                    self.algo.MyDebug("      " \
                                    + str(round(i/1))
                                    + ". Symbol:" + str(x.Key)
                                    + "(" + str (self.algo.mySymbolDict[x.Key].CL.strategyCode) + ")"
                                    + ", Quantity:" + str(self.algo.Portfolio[x.Key].Quantity)
                                    + ", Stops:" + str(round(quantityStops))
                                    + "/" + str(totalStops)
                                    + ", Limits:" + str(round(quantityLimits))
                                    + "/" + str(totalLimits)
                                    + ", Markets:" + str(round(quantityMarkets))
                                    + "/" + str(totalMarkets))  
                i += 1
        return
'''
CHARTS
'''
class MyCharts:
    '''
    Exposure and Indicator Chart
    '''
    #Exposure Chart
    plotChart1 = True
    #Indicator Chart
    plotChart2 = True
    #VaR Chart
    plotChart3 = True
    #Trades
    plotChart4 = True
    
    def __init__(self, caller, symbol, backtestUpdateHours=23):
        self.CL = self.__class__
        self.algo = caller
        if self.algo.LiveMode:
            self.updateTimeFrequency = 1 * timedelta(minutes=1)
        else:
            self.updateTimeFrequency = backtestUpdateHours * timedelta(hours=1)
        self.nextUpdateTime = self.algo.Time - timedelta(hours=20)
        self.chartSymbol = symbol
        self.myChartTitle1 = "Exposure"
        self.myChartTitle2 = "Indicators"
        self.myChartTitle3 = "VaR"
        self.myChartTitle4 = "Trades"
        
        if self.algo.LiveMode:
            self.CL.plotChart1 = True
            self.CL.plotChart2 = False
            self.CL.plotChart3 = True
            self.CL.plotChart4 = True
            
        myPlot1 = Chart(self.myChartTitle1)
        myPlot1.AddSeries(Series('Abs Exposure', SeriesType.Line, '$', Color.Brown))
        myPlot1.AddSeries(Series('Net Exposure', SeriesType.Line, '$', Color.Blue))
        myPlot1.AddSeries(Series('Short Exposure', SeriesType.Line, '$', Color.Red))
        myPlot1.AddSeries(Series('Long Exposure', SeriesType.Line, '$', Color.Green))
        myPlot2 = Chart(self.myChartTitle2)
        myPlot2.AddSeries(Series('Price', SeriesType.Line, 0))
        myPlot2.AddSeries(Series('Srategy ATRpct', SeriesType.Line, '$', Color.Brown))
        myPlot2.AddSeries(Series('Benchmark ATRpct', SeriesType.Line, '$', Color.Blue))
        myPlot2.AddSeries(Series('emaFast', SeriesType.Line, '$', Color.Red))
        myPlot2.AddSeries(Series('emaSlow', SeriesType.Line, '$', Color.Yellow))
        myPlot2.AddSeries(Series('dchUp', SeriesType.Line, '$', Color.Blue))
        myPlot2.AddSeries(Series('dchLow', SeriesType.Line, '$', Color.Blue))
        myPlot2.AddSeries(Series('dchMid', SeriesType.Line, '$', Color.Blue))
        myPlot2.AddSeries(Series('sma', SeriesType.Line, '$', Color.Green))
        myPlot2.AddSeries(Series('Buy', SeriesType.Scatter, '$', Color.Green, ScatterMarkerSymbol.Triangle))
        myPlot2.AddSeries(Series('Sell', SeriesType.Scatter, '$', Color.Red, ScatterMarkerSymbol.TriangleDown))
        myPlot3 = Chart(self.myChartTitle3)
        myPlot3.AddSeries(Series('Total VaR(%)', SeriesType.Line, '$', Color.Blue))
        myPlot3.AddSeries(Series('Long VaR(%)', SeriesType.Line, '$', Color.Green))
        myPlot3.AddSeries(Series('Short VaR(%)', SeriesType.Line, '$', Color.Red))
        myPlot3.AddSeries(Series('Last Enty VaR(%)', SeriesType.Line, '$', Color.Purple))
        myPlot4 = Chart(self.myChartTitle4)
        myPlot4.AddSeries(Series('Trades', SeriesType.Line, '$', Color.Green))
        myPlot4.AddSeries(Series('Rejected Trades', SeriesType.Line, '$', Color.Red))
        myPlot4.AddSeries(Series('Entries', SeriesType.Line, '$', Color.Purple))
        myPlot4.AddSeries(Series('Entry Fills', SeriesType.Line, '$', Color.Blue))
        if self.CL.plotChart1: self.algo.AddChart(myPlot1)
        if self.CL.plotChart2: self.algo.AddChart(myPlot2)
        if self.CL.plotChart3: self.algo.AddChart(myPlot3)
        if self.CL.plotChart4: self.algo.AddChart(myPlot4)
        
        return
    
    '''Charts Update
    '''
    def Update (self, forced=False):
        #self.algo.MyDebug("Chart Update1")
        if self.nextUpdateTime > self.algo.Time and not forced:
            return
        self.nextUpdateTime = self.algo.Time + self.updateTimeFrequency
        #self.algo.MyDebug("Chart Update2")
        totalTrades, totalFlipTrades, totalEntryRejections, totalEntries, totalEntryFills = 0, 0, 0, 0, 0
        for strat in self.algo.myStrategyClassList:
            totalTrades += strat._totalTrades
            totalFlipTrades += strat._totalFlipTrades
            totalEntryRejections += strat._totalEntryRejections
            totalEntries += strat._totalEntries
            totalEntryFills += strat._totalEntryFills

        varCL = self.algo.myVaR.CL

        if self.CL.plotChart1: self.algo.Plot(self.myChartTitle1, "Abs Exposure", varCL._netExposure)
        if self.CL.plotChart1: self.algo.Plot(self.myChartTitle1, "Net Exposure", varCL._netExposure)
        if self.CL.plotChart1: self.algo.Plot(self.myChartTitle1, "Long Exposure", varCL._longExposure)
        if self.CL.plotChart1: self.algo.Plot(self.myChartTitle1, "Short Exposure", varCL._shortExposure)
        if self.CL.plotChart3: self.algo.Plot(self.myChartTitle3, "Total VaR(%)", 100*varCL._openTotalVaR)
        if self.CL.plotChart3: self.algo.Plot(self.myChartTitle3, "Long VaR(%)", 100*varCL._openLongVaR)
        if self.CL.plotChart3: self.algo.Plot(self.myChartTitle3, "Short VaR(%)", 100*varCL._openShortVaR)
        if self.CL.plotChart3: self.algo.Plot(self.myChartTitle3, "Last Enty VaR(%)", 100*varCL._lastEntryOpenTotalVaR)
        if self.CL.plotChart4: self.algo.Plot(self.myChartTitle4, "Trades", totalTrades)
        if False and self.CL.plotChart4: self.algo.Plot(self.myChartTitle4, "FlipTrades", totalFlipTrades)
        if self.CL.plotChart4: self.algo.Plot(self.myChartTitle4, "Rejected Trades", totalEntryRejections)
        if True and self.CL.plotChart4: self.algo.Plot(self.myChartTitle4, "Entries", totalEntries)
        if self.CL.plotChart4: self.algo.Plot(self.myChartTitle4, "Entry Fills", totalEntryFills)
        
        self.algo.SetRuntimeStatistic("AbsE: ", str(round(varCL._absExposure,2))+"/"+str(varCL._longPositions+varCL._shortPositions))
        self.algo.SetRuntimeStatistic("NetE: ", str(round(varCL._netExposure,2))+"/"+str(varCL._longPositions+varCL._shortPositions))
        self.algo.SetRuntimeStatistic("LongE: ", str(round(varCL._longExposure,2))+"/"+str(varCL._longPositions))
        self.algo.SetRuntimeStatistic("ShortE: ", str(round(varCL._shortExposure,2))+"/"+str(varCL._shortPositions))

        #Update Indicator Chart
        #if self.CL.plotChart2: self.algo.Plot(self.myChartTitle2, "emaFast", self.algo.mySymbolDict[self.chartSymbol].emaFast.Current.Value)
        #if self.CL.plotChart2: self.algo.Plot(self.myChartTitle2, "emaSlow", self.algo.mySymbolDict[self.chartSymbol].emaSlow.Current.Value)
        #if self.CL.plotChart2: self.algo.Plot(self.myChartTitle2, "dchMid", self.algo.mySymbolDict[self.chartSymbol].dch2.Current.Value)
        #if self.CL.plotChart2: self.algo.Plot(self.myChartTitle2, "dchUp", self.algo.mySymbolDict[self.chartSymbol].dch3.UpperBand.Current.Value)
        #if self.CL.plotChart2: self.algo.Plot(self.myChartTitle2, "dchLow", self.algo.mySymbolDict[self.chartSymbol].dch3.LowerBand.Current.Value)
        #if self.CL.plotChart2: self.algo.Plot(self.myChartTitle2, "Price", self.algo.Securities[self.algo.chartSymbol].Price)
        if self.CL.plotChart2: self.algo.Plot(self.myChartTitle2, "Srategy ATRpct", self.algo.myStrategyClassList[0]._totalATRpct*100)
        #if self.CL.plotChart2: self.algo.Plot(self.myChartTitle2, "Benchmark ATRpct", self.algo.myStrategyClassList[1]._totalATRpct*100)
        #if self.CL.plotChart2: self.algo.Plot(self.myChartTitle2, "Srategy ATRpct", self.algo.mySymbolDict[self.algo.benchmarkSymbol].rsi1.Current.Value)
'''
RUNTIME STATISTICS
'''
class MyStats:
    def __init__(self, caller):
        self.algo = caller
        if self.algo.LiveMode:
            self.updateTimeFrequency = 1 * timedelta(seconds=10)
        else:
            self.updateTimeFrequency = 1 * timedelta(hours=23)
        self.nextUpdateTime = self.algo.Time - timedelta(hours=20)
        
    def Update (self, forced = False):
        if self.nextUpdateTime > self.algo.Time and not forced:
            return
        self.nextUpdateTime = self.algo.Time + self.updateTimeFrequency
        
        varCL = self.algo.myVaR.CL
        
        self.algo.SetRuntimeStatistic("Stop/Limit: ", str(varCL._activeStops)+"/"+str(varCL._activeLimits))    
        self.algo.SetRuntimeStatistic("VaR(%):", str(round(varCL._openTotalVaR*100,1)))
        self.algo.SetRuntimeStatistic("LVaR(%):", str(round(varCL._openLongVaR*100,1)))
        self.algo.SetRuntimeStatistic("SVaR(%):", str(round(varCL._openShortVaR*100,1)))
        #self.algo.SetRuntimeStatistic("Securities: ", str(self.algo.Securities.Count)) 
        #self.algo.SetRuntimeStatistic("SymbDic: ", str(len(self.algo.mySymbolDict)))
    
    def PrintStrategyTradeStats(self):
        for strategy in self.algo.myStrategyClassList:
            self.algo.MyDebug("\n  TRADE STATISTICS:{} Strategy {}: Trades:{}, LongTrades:{}, ShortTrades:{}, FlipTrades:{}, EntryRejections:{}, Entries:{}, EntryFills:{}"
                   .format(str('\n' + str(' '*20)),
                           str(strategy.strategyCode),
                           str(strategy._totalTrades),
                           str(strategy._totalTradesLong),
                           str(strategy._totalTradesShort),
                           str(strategy._totalFlipTrades),
                           str(strategy._totalEntryRejections),
                           str(strategy._totalEntries),
                           str(strategy._totalEntryFills))) 
        return