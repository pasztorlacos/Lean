### <summary>
### Position Manager
###
### </summary
import math
from QuantConnect import SecurityType
from QuantConnect.Orders import UpdateOrderFields, OrderDirection, OrderStatus, TimeInForce

#--------------MyPositionManager CLASS 
class MyPositionManager:
    '''
    Enter and Exit Positions 
    Manage orders and
    Order Consistency
    '''
    file = __file__
    pendingFlipPositions = []

    def __init__(self, caller):
        self.CL = self.__class__
        self.algo = caller
        self.checkBuyingPower = False #It doesn't work anymore: https://github.com/QuantConnect/Lean/commit/a697d7380e343d6ab710047e8f940d2e43ef9713
        #Quick Fix for IB FX max order size
        self.maxIBForexQuantity = 4000000
        self.debug = self.algo.debug
        if self.algo.LiveMode: self.debug = True
        return
    
    '''
    SYNCCRONIZE WITH TWS
    '''
    def TWS_Sync (self):
        '''
        Sync TWS orders'''
        totalTickets = 0
        totalStopsAdded = 0
        totalLimitsAdded = 0
        totalMarketsAdded = 0
        totalOrdersAdded = 0
        if self.debug: self.algo.MyDebug(" ---- ORDER SYNC WITH TWS:" )
        for tempTicket in self.algo.Transactions.GetOrderTickets():
            totalTickets +=1
            #Add symbol if it isn't in mySymbolDict. No new positions is allowed, hence we did not intend to trade it.
            if tempTicket.Symbol not in self.algo.mySymbolDict:
                #Subscribe to Data
                if tempTicket.Symbol.SecurityType == SecurityType.Equity:
                    self.algo.AddEquity(tempTicket.Symbol.Value, self.algo.mainResolution)
                elif tempTicket.Symbol.SecurityType == SecurityType.Forex: 
                    self.algo.AddForex(self.algo.Securities[tempTicket.Symbol].Symbol.Value, self.algo.mainResolution)
                #Add to mySymbolDict
                self.algo.myHelpers.addSymbolDict(tempTicket.Symbol, self.algo.myStrategyClassList[0], self.algo.foreignVaR)
                self.algo.mySymbolDict[tempTicket.Symbol].posEnabled = False
                self.algo.mySymbolDict[tempTicket.Symbol].fromTWS = True
                if self.algo.LiveMode or self.debug: self.algo.MyDebug(" ORDER SYMBOL ADDED ID:{}, Symbol:{}, Type:{}, Quantity:{}, Status:{}"
                    .format(str(tempTicket.OrderId),
                            str(tempTicket.Symbol),
                            str(tempTicket.OrderType),
                            str(tempTicket.Quantity),
                            str(tempTicket.Status)))  
            #self.algo.Transactions.GetOrderById(orderID)
            #tempTicket.Symbol in self.algo.mySymbolDict and
            if (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == OrderStatus.UpdateSubmitted or tempTicket.Status == 6) or False:    
                if tempTicket.OrderType == OrderType.StopMarket:
                    self.algo.openStopMarketOrders.append(tempTicket)
                    totalStopsAdded += 1
                elif tempTicket.OrderType == OrderType.Limit:
                    self.algo.openLimitOrders.append(tempTicket)
                    totalLimitsAdded += 1
                elif tempTicket.OrderType == OrderType.Market or tempTicket.OrderType == OrderType.MarketOnOpen:
                    self.algo.openMarketOrders.append(tempTicket)
                    totalMarketsAdded += 1

        if self.debug: self.algo.MyDebug("      Orders Added:" + str(totalTickets)+ "     Stops:" + str(totalStopsAdded) \
                + " Limits:" + str(totalLimitsAdded) + " Market:" + str(totalMarketsAdded))
        totalOrdersAdded = totalMarketsAdded + totalLimitsAdded
        self.algo.twsSynced = True
        return totalOrdersAdded
    
    '''
    ENTER POSITION
        Custom Settings Handles: 
        customPositionSettings={'riskperTrade': ,'stopPlacer': ,'targetPlacer': ,'minPayOff': , 'stopATR': ,'minEntryStopATR': ,'entryLimitOrder': ,'limitEntryATR': ,'entryPrice': , 'stopPrice': , 'targetPrice': }
        stopPlacer must be a Tuple (Type, arg2, arg2...): ("minStop"), ("dch",'dch_attributeName'), ("rw",n)
        targetPlacer must be a Tuple (Type, arg2, arg2...): (None,0), ("minPayOff"), ("dch",'dch_attributeName'), ("rw",n)
        Trailing and Scratch ccannot be set as it is not available after entry!
        
    '''
    def EnterPosition(self, symbol, myDirection, myReverse=False, myRiskMultiple=1.00, customPositionSettings={}):
        debugEntry = self.debug
        positionMaxQuantity = 0
        positionTargetQuantity = 0
        positionTargetExposure = 0
        positionQuantity = 0
        buyingPower = 0
        marginRemaining = 0
        orderQuantity = 0
        reducedAbsExposure = 0
        entryPrice, minStopPrice, currentLow, stopPrice, targetPrice , targetRisk, entryRisk = 0, 0 , 0, 0, 0, 0, 0
        strategyCode = str(self.algo.mySymbolDict[symbol].CL.strategyCode)
        
        self.algo.myVaR.Update()
        
        #Exit if ALGO IS DISABLED (after VaR Limit Breach Liquidation)
        if not self.algo.enabled:
            if debugEntry or self.algo.LiveMode: self.algo.MyDebug("     ENTRY IS REJECTED (0)! Symbol: {}({}) Algo is Disabled"
                .format(str(symbol),
                        str(strategyCode)))            
            return
        
        #Exit if SYMBOL is NOT TRADABLE or STRADEGY IS DISABLED
        if not self.algo.mySymbolDict[symbol].posEnabled or not self.algo.mySymbolDict[symbol].CL.enabled :
            if debugEntry or self.algo.LiveMode: self.algo.MyDebug("     ENTRY IS REJECTED (1)! Symbol: {}({}) Position is Disabled: Position Enabled:{}, Entry Enabled:{}, Strategy Enabled:{}"
                .format(str(symbol),
                        str(strategyCode),
                        str(self.algo.mySymbolDict[symbol].posEnabled),
                        str(self.algo.mySymbolDict[symbol].entryEnabled),
                        str(self.algo.mySymbolDict[symbol].CL.enabled)))            
            return
        
        #Exit if DIRECTION is NOT ENABLED or LIQUIDATED
        directionLiquidation = False
        if myDirection == 1:
            if hasattr(self.algo.mySymbolDict[symbol].CL, 'liquidateLong'):
                directionLiquidation = self.algo.mySymbolDict[symbol].CL.liquidateLong
            directionEnabled = self.algo.mySymbolDict[symbol].CL.enableLong
        elif myDirection == -1:
            if hasattr(self.algo.mySymbolDict[symbol].CL, 'liquidateShort'):
                directionLiquidation = self.algo.mySymbolDict[symbol].CL.liquidateShort
            directionEnabled = self.algo.mySymbolDict[symbol].CL.enableShort
        if not directionEnabled or directionLiquidation:
            if debugEntry or self.algo.LiveMode: self.algo.MyDebug("     ENTRY IS REJECTED (1.1)! Symbol: {}({}) Direction({}) is Disabled or Liquidated: Direction Enabled:{}, Direction Liquidated:{}"
                .format(str(symbol),
                        str(strategyCode),
                        str(myDirection),
                        str(directionEnabled),
                        str(directionLiquidation)))            
            return            
        
        #It is a NEW TRADE
        self.algo.mySymbolDict[symbol].CL._totalTrades += 1
        if myDirection==1:
            self.algo.mySymbolDict[symbol].CL._totalTradesLong += 1
        elif myDirection==-1:
            self.algo.mySymbolDict[symbol].CL._totalTradesShort += 1
        
        #Check position and pending Entry
        _SP = self.algo.Portfolio[symbol].Quantity #self.algo.myPositionManagerB.SumPosition(symbol)[1]
        _SM = self.algo.myPositionManagerB.SumOrder(symbol, self.algo.openMarketOrders)[1]
        _ASS = self.algo.myPositionManagerB.SumOrder(symbol, self.algo.openStopMarketOrders)[2]
        _ASL = self.algo.myPositionManagerB.SumOrder(symbol, self.algo.openLimitOrders)[2]
 
        #Reject if already in Position (and not in Liquidation) or Pending Position
        #self.algo.mySymbolDict[symbol].entryEnabled is set to False before entry and released in Consistency if _CP == 0 and _CS == 0 and _CL == 0 and _CM == 0 
        if not self.algo.mySymbolDict[symbol].entryEnabled or _SP+_SM != 0 or _ASS != 0 or _ASL != 0:
            self.algo.mySymbolDict[symbol].CL._totalEntryRejections += 1
            if debugEntry or self.algo.LiveMode: self.algo.MyDebug("     ENTRY IS REJECTED (2)! Symbol: {}({}) already has (Pending)Position: _SP:{}, _SM:{}, _ASS:{}, _ASL:{}, entryEnabled:{}"
                .format(str(symbol),
                        str(strategyCode),
                        str(_SP),
                        str(_SM),
                        str(_ASS),
                        str(_ASL),
                        str(self.algo.mySymbolDict[symbol].entryEnabled)))            
            return
        
        #Reject if Blocked for furter fills so Position in change
        if self.algo.Time < self.algo.mySymbolDict[symbol].fillReleaseTime:
            self.algo.mySymbolDict[symbol].CL._totalEntryRejections += 1
            timeDifference = max(timedelta(0),self.algo.mySymbolDict[symbol].fillReleaseTime - self.algo.Time)
            if debugEntry or self.algo.LiveMode: self.algo.MyDebug("     ENTRY IS REJECTED (3)! Symbol: {}({}) is Blocked for Order Fill: Block Release timeDifference:{}"
                .format(str(symbol),
                        str(strategyCode),
                        str(timeDifference)))
            return
        
        conversionRate = self.algo.Portfolio.Securities[symbol].QuoteCurrency.ConversionRate
        priceRoundingDigits = round(-1*math.log(self.algo.Securities[symbol].SymbolProperties.MinimumPriceVariation,10))
        sd = self.algo.mySymbolDict[symbol]
        #THIS IS ON STARTEGY LEVEL ALLOCATED EQUITY!
        TotalPortfolioValue = self.algo.Portfolio.TotalPortfolioValue * sd.CL.strategyAllocation
        
        #Update VaR numbers so limits are up to date
        sd.var.Update()
        
        if not myReverse:
            direction = myDirection
        else:
            direction = -1*myDirection

        '''LONG POSITION'''
        '''Position Sizing and Exposure Limit'''
        if direction == 1:
            riskMultiple = min(1.00,max(myRiskMultiple,0.00))
            riskperTradeTarget = sd.CL.riskperLongTrade     if 'riskperTrade' not in customPositionSettings else customPositionSettings['riskperTrade']
            stopPlacerType = sd.CL.stopPlacerLong           if 'stopPlacer' not in customPositionSettings else customPositionSettings['stopPlacer']
            targetPlacerType = sd.CL.targetPlacerLong       if 'targetPlacer' not in customPositionSettings else customPositionSettings['targetPlacer']
            useTargets = getattr(self.algo.mySymbolDict[symbol].CL, "useTragetsLong", targetPlacerType[0]!=None)
            riskperTrade = min(sd.var.longVaRLimit, riskperTradeTarget*riskMultiple)
            if debugEntry and riskperTrade < riskperTradeTarget*riskMultiple: 
                self.algo.MyDebug("     REDUCED RISK per TRADE from:{}%, to:{}% due to Startegy VaR LIMIT. Symbol: {}({}), TLongLimit:{}%, TLimit:{}%, StrOpenLVaR:{}%, StrOpenTVaR:{}%"
                    .format(str(round(riskperTradeTarget*riskMultiple*100,2)),
                            str(round(riskperTrade*100,2)),
                            str(symbol),
                            str(strategyCode),
                            str(round(sd.CL.maxLongVaR*100,2)),
                            str(round(sd.CL.maxTotalVaR*100,2)),
                            str(round(sd.var.openLongVaR*100,2)),
                            str(round(sd.var.openTotalVaR*100,2))))           
            
            if not myReverse:
                entryPrice, stopPrice, targetPrice = self.StopTargetPlacer(symbol, direction, customPositionSettings=customPositionSettings)
                #Overwrite if Custom Order Prices are given
                if 'entryPrice'  in customPositionSettings: entryPrice = round(min(customPositionSettings['entryPrice'], self.algo.Securities[symbol].Price), priceRoundingDigits)
                if 'stopPrice'   in customPositionSettings: stopPrice = customPositionSettings['stopPrice']
                if 'targetPrice' in customPositionSettings: targetPrice = customPositionSettings['targetPrice']
            else:
                entryPrice, targetPrice, stopPrice = self.StopTargetPlacer(symbol, -1*direction, customPositionSettings=customPositionSettings)
                entryPrice = round(min(entryPrice, self.algo.Securities[symbol].Price), priceRoundingDigits)
                #Overwrite if Custom Order Prices are given
                if 'entryPrice'  in customPositionSettings: entryPrice = round(min(customPositionSettings['entryPrice'], self.algo.Securities[symbol].Price), priceRoundingDigits)
                if 'targetPrice' in customPositionSettings: stopPrice = customPositionSettings['targetPrice']
                if 'stopPrice'   in customPositionSettings: targetPrice = customPositionSettings['stopPrice']
            if debugEntry: self.algo.MyDebug(f' entryPrice:{entryPrice}, stopPrice:{stopPrice}, targetPrice:{targetPrice}') 
                
            if entryPrice != 0: 
                positionMaxQuantity =  (sd.var.longExposureLimit * TotalPortfolioValue) / (entryPrice * conversionRate)
            if (entryPrice-stopPrice) != 0:
                positionTargetQuantity = round((riskperTrade * TotalPortfolioValue)/((entryPrice-stopPrice)*conversionRate))
                if TotalPortfolioValue !=0: 
                    positionTargetExposure = positionTargetQuantity*entryPrice*conversionRate/TotalPortfolioValue
                #Exposure Limit Constraint
                positionQuantity = round(max(0,min(positionMaxQuantity, positionTargetQuantity)))
            else:
                positionQuantity = 0
            #if TotalPortfolioValue != 0: targetRisk = (entryPrice-stopPrice)*conversionRate*positionTargetQuantity/TotalPortfolioValue
            if debugEntry and positionQuantity < positionTargetQuantity: 
                self.algo.MyDebug("     REDUCED POS. SIZE from:{}, to:{} due to Str. EXP. LIMIT Symbol: {}({}), TargetE:{}, LimitE:{}, longE:{}, shortE:{}, netE:{}, absE:{}, Longs:{}"
                    .format(str(positionTargetQuantity),
                            str(positionQuantity),
                            str(symbol),
                            str(strategyCode),
                            str(round(positionTargetExposure,2)),
                            str(round(sd.var.longExposureLimit,3)),
                            str(round(sd.var.longExposure,3)),
                            str(round(sd.var.shortExposure,3)),
                            str(round(sd.var.netExposure,3)),
                            str(round(sd.var.absExposure,3)),
                            str(sd.var.longPositions)))

        '''SHORT POSITION'''
        '''Position Sizing and Exposure Limit'''
        if direction == -1:
            riskMultiple = min(1.00,max(myRiskMultiple,0.00))
            riskperTradeTarget = sd.CL.riskperShortTrade    if 'riskperTrade' not in customPositionSettings else customPositionSettings['riskperTrade']
            stopPlacerType = sd.CL.stopPlacerShort          if 'stopPlacer' not in customPositionSettings else customPositionSettings['stopPlacer']
            targetPlacerType = sd.CL.targetPlacerShort      if 'targetPlacer' not in customPositionSettings else customPositionSettings['targetPlacer']
            useTargets = getattr(self.algo.mySymbolDict[symbol].CL, "useTragetsShort", targetPlacerType[0]!=None)
            
            
            riskperTrade = min(sd.var.shortVaRLimit, riskperTradeTarget*riskMultiple)
            if debugEntry and riskperTrade < riskperTradeTarget*riskMultiple: 
                self.algo.MyDebug("     REDUCED RISK per TRADE from:{}%, to:{}% due to Str. VaR LIMIT. Symbol: {}({}), TShortLimit:{}%, TLimit:{}%, StrOpenSVaR:{}%, StrOpenTVaR:{}%"
                    .format(str(round(riskperTradeTarget*riskMultiple*100,2)),
                            str(round(riskperTrade*100,2)),
                            str(symbol),
                            str(strategyCode),
                            str(round(sd.CL.maxShortVaR*100,2)),
                            str(round(sd.CL.maxTotalVaR*100,2)),
                            str(round(sd.var.openShortVaR*100,2)),
                            str(round(sd.var.openTotalVaR*100,2))))  
            
            if not myReverse:
                entryPrice, stopPrice, targetPrice = self.StopTargetPlacer(symbol, direction, customPositionSettings=customPositionSettings)
                #Overwrite if Custom Order Prices are given
                if 'entryPrice'  in customPositionSettings: entryPrice = round(max(customPositionSettings['entryPrice'], self.algo.Securities[symbol].Price), priceRoundingDigits)
                if 'stopPrice'   in customPositionSettings: stopPrice = customPositionSettings['stopPrice']
                if 'targetPrice' in customPositionSettings: targetPrice = customPositionSettings['targetPrice']
            else:
                entryPrice, targetPrice, stopPrice = self.StopTargetPlacer(symbol, -1*direction, customPositionSettings=customPositionSettings)
                entryPrice = round(max(entryPrice, self.algo.Securities[symbol].Price), priceRoundingDigits)
                #Overwrite if Custom Order Prices are given
                if 'entryPrice'  in customPositionSettings: entryPrice = round(max(customPositionSettings['entryPrice'], self.algo.Securities[symbol].Price), priceRoundingDigits)
                if 'targetPrice' in customPositionSettings: stopPrice = customPositionSettings['targetPrice']
                if 'stopPrice'   in customPositionSettings: targetPrice = customPositionSettings['stopPrice']
            if debugEntry: self.algo.MyDebug(f' entryPrice:{entryPrice}, stopPrice:{stopPrice}, targetPrice:{targetPrice}')    
            
            if entryPrice != 0: 
                positionMaxQuantity =  (sd.var.shortExposureLimit * TotalPortfolioValue) / (entryPrice * conversionRate)
            if (stopPrice-entryPrice) !=0:
                positionTargetQuantity = round((riskperTrade * TotalPortfolioValue)/((entryPrice-stopPrice)*conversionRate))
                if TotalPortfolioValue !=0: 
                    positionTargetExposure = positionTargetQuantity*entryPrice*conversionRate/TotalPortfolioValue
                #Exposure Limit Constraint
                positionQuantity = round(min(0,max(positionMaxQuantity, positionTargetQuantity))) #entryPrice-stopPrice as it is short
            else:
                positionQuantity = 0
            #if TotalPortfolioValue != 0: targetRisk = (entryPrice-stopPrice)*conversionRate*positionTargetQuantity/TotalPortfolioValue
            if debugEntry and positionQuantity > positionTargetQuantity: 
                self.algo.MyDebug("     REDUCED POS. SIZE from:{}, to:{} due to Str. EXP. LIMIT. Symbol: {}({}), TargetE:{}, LimitE:{}, LE:{}, SE:{}, NE:{}, AE:{}, Shorts:{}"
                    .format(str(positionTargetQuantity),
                            str(positionQuantity),
                            str(symbol),
                            str(strategyCode),
                            str(round(positionTargetExposure,2)),
                            str(round(sd.var.shortExposureLimit,3)),
                            str(round(sd.var.longExposure,3)),
                            str(round(sd.var.shortExposure,3)),
                            str(round(sd.var.netExposure,3)),
                            str(round(sd.var.absExposure,3)),
                            str(sd.var.shortPositions)))

        '''Checking for Buying Power Constraint'''
        #!!!! Quick Fix for IB max Forex Quantity, Slicing to be implemented
        #self.algo.MyDebug("   LBP: " +str(self.algo.Portfolio.GetBuyingPower(symbol, OrderDirection.Buy)) + " SBP:" +str(self.algo.Portfolio.GetBuyingPower(symbol, OrderDirection.Sell)))
        if symbol.SecurityType == SecurityType.Forex:
            buyingPowerMultiple = 5 
        elif symbol.SecurityType == SecurityType.Equity:
            buyingPowerMultiple = 2 
        else: buyingPowerMultiple = 1         
        
        if self.checkBuyingPower and entryPrice !=0:
            if positionQuantity > 0:
                marginRemaining = self.algo.Portfolio.GetMarginRemaining(symbol, OrderDirection.Buy)
                buyingPower = buyingPowerMultiple * sd.var.CL.buyingPowerUtilisation * self.algo.Portfolio.GetBuyingPower(symbol, OrderDirection.Buy)
                #Need Buying Power for the Target Order as well
                if useTargets: buyingPower = buyingPower/2
                orderQuantity = round(min(positionQuantity, buyingPower/(entryPrice*conversionRate), self.maxIBForexQuantity))
            elif positionQuantity < 0:
                marginRemaining = self.algo.Portfolio.GetMarginRemaining(symbol, OrderDirection.Sell)
                buyingPower = buyingPowerMultiple * sd.var.CL.buyingPowerUtilisation * self.algo.Portfolio.GetBuyingPower(symbol, OrderDirection.Sell)
                #Need Buying Power for the Target Order as well
                if useTargets: buyingPower = buyingPower/2
                orderQuantity = round(max(positionQuantity, -1* buyingPower/(entryPrice*conversionRate), -1*self.maxIBForexQuantity))
        elif entryPrice !=0: orderQuantity = positionQuantity
        if debugEntry and orderQuantity != positionQuantity: 
            self.algo.MyDebug("     REDUCED POS. SIZE from:{}, to:{} due to BUYING POWER LIMIT. Symbol: {}({}), Orig Exp:{}, My BPower:{}, Halved:{}, MRemaining:{}"
                    .format(str(positionQuantity),
                            str(orderQuantity),
                            str(symbol),
                            str(strategyCode),
                            str(round(positionQuantity*entryPrice)),
                            str(round(buyingPower)),
                            str(useTargets),
                            str(round(marginRemaining))))
        
        #Round to Forex nearest lot size
        if symbol.SecurityType == SecurityType.Forex:
            orderQuantity = round(orderQuantity/1000)*1000

        #Check Total Reductions
        if TotalPortfolioValue != 0:
            reducedAbsExposure = abs(orderQuantity*entryPrice*conversionRate/TotalPortfolioValue)
        
        '''If Total Reduction is too big reject entry'''
        if reducedAbsExposure < sd.CL.minSymbolAbsExposure:
            orderQuantity = 0
            sd.CL._totalEntryRejections += 1
            if debugEntry: self.algo.MyDebug("     ENTRY IS REJECTED! Symbol: {}({})   Reduced Pos. Abs. Exposure is too Small:{}, Limit is:{}"
                    .format(str(symbol),
                            str(strategyCode),
                            str(round(reducedAbsExposure,3)),
                            str(round(sd.CL.minSymbolAbsExposure,3))))
            return

        #Submit pending position to mySymbolDict and Order to Broker. Existing position and Enabled is double checked
        if orderQuantity !=0 and self.algo.Portfolio[symbol].Quantity == 0 and self.algo.mySymbolDict[symbol].entryEnabled:
            #self.SetHoldings(symbol, self.holdingSize)
            
            #Update Last Entry VaR as Total VaR increases with positive running profit if no stops are trailed meanwhile
            if TotalPortfolioValue != 0 and sd.CL.strategyAllocation != 0:
                newPositionPortfolioRisk = abs(orderQuantity*(entryPrice-stopPrice))*conversionRate/TotalPortfolioValue
                sd.var.CL._lastEntryOpenTotalVaR = sd.var.CL._openTotalVaR + newPositionPortfolioRisk
            
            '''SUBMITTING ENTRY ORDERS
            '''
            #Block further Entry
            self.algo.mySymbolDict[symbol].entryEnabled = False
            self.algo.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilCanceled
            enterOrderId = -1
            #First Submit Stop and Target Orders so tickets would be available upon entry fill
            #Submit Stop Order
            orderTag = "S_[{}]({}): {}".format(str(strategyCode), str(stopPlacerType[0])+'-'+str(stopPlacerType[1]), str(stopPrice))
            stopTicket = self.algo.StopMarketOrder(symbol, -1*orderQuantity, stopPrice, orderTag)
            if stopTicket is not None: self.algo.openStopMarketOrders.append(stopTicket)                
            #Submit Target Order
            if useTargets:
                orderTag = "T_[{}]({}): {}".format(str(strategyCode),  str(targetPlacerType[0])+'-'+str(targetPlacerType[1]), str(targetPrice))
                targetTicket = self.algo.LimitOrder(symbol, -1*orderQuantity, targetPrice, orderTag)
                if targetTicket is not None: self.algo.openLimitOrders.append(targetTicket)
            #Submit Position Entry Order
            entryLimitOrder = self.algo.mySymbolDict[symbol].CL.entryLimitOrderLong if myDirection == 1 else self.algo.mySymbolDict[symbol].CL.entryLimitOrderShort
            entryTimeInForce = self.algo.mySymbolDict[symbol].CL.entryTimeInForceLong if myDirection == 1 else self.algo.mySymbolDict[symbol].CL.entryTimeInForceShort
            if entryLimitOrder:
                #Set TimeInForce
                self.algo.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilDate(self.algo.Time + entryTimeInForce)
                #Submit Limit Order
                orderTag = "LE_[{}]({}/{}): {}".format(str(strategyCode), str(stopPlacerType[0])+'-'+str(stopPlacerType[1]), str(targetPlacerType[0])+'-'+str(targetPlacerType[1]), str(entryPrice))
                positionTicket = self.algo.LimitOrder(symbol, orderQuantity, entryPrice, orderTag)
                enterOrderId = positionTicket.OrderId
                if positionTicket is not None: self.algo.openLimitOrders.append(positionTicket)
                #Set TimeInForce Back 
                self.algo.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilCanceled
            else:
                #Set TimeInForce
                self.algo.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilDate(self.algo.Time + entryTimeInForce)
                #Submit Market Order
                orderTag = "ME_[{}]({}/{}): {}".format(str(strategyCode), str(stopPlacerType[0])+'-'+str(stopPlacerType[1]), str(targetPlacerType[0])+'-'+str(targetPlacerType[1]), str(entryPrice))
                positionTicket = self.algo.MarketOrder(symbol, orderQuantity, False, orderTag)
                enterOrderId = positionTicket.OrderId
                if positionTicket is not None: self.algo.openMarketOrders.append(positionTicket)
                #Set TimeInForce Back 
                self.algo.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilCanceled
            if TotalPortfolioValue !=0: entryRisk = abs((entryPrice-stopPrice)*conversionRate*orderQuantity/TotalPortfolioValue)
            if debugEntry: self.algo.MyDebug(" > POSITION ENTRY for {} [{}]: S/T:{}/{}, Risk:{}, Quantity:{}, EntryPrice:{}, Stop:{}, Target:{}, OrderId:{}, FxRate:{}"
                    .format(str(symbol),
                            str(strategyCode),
                            str(stopPlacerType[0])+'-'+str(stopPlacerType[1]),
                            str(targetPlacerType[0])+'-'+str(targetPlacerType[1]),
                            str(round(entryRisk,4)),
                            str(orderQuantity),
                            str(entryPrice),
                            str(stopPrice),
                            str(targetPrice),
                            str(enterOrderId),
                            str(round(conversionRate,4))))
            #This is a New Entry
            sd.CL._totalEntries += 1
            #Check for rejection so stop and limit tickets to be cancelled!!!!!
            
            '''Updating Exposure and VaR numbers'''
            sd.var.Update()
            
            #Update Chart
            if orderQuantity > 0 and self.algo.myCharts.plotChart2  and symbol == self.algo.chartSymbol:
                self.algo.Plot(self.algo.myCharts.myChartTitle2, "Buy", self.algo.Securities[self.algo.chartSymbol].Price)
            elif orderQuantity < 0 and self.algo.myCharts.plotChart2  and symbol == self.algo.chartSymbol:
                self.algo.Plot(self.algo.myCharts.myChartTitle2, "Sell", self.algo.Securities[self.algo.chartSymbol].Price)
        return
            
    '''
    STOP AND TARGET PLACER
    '''
    def StopTargetPlacer(self, symbol, direction, customPositionSettings):
        sd = self.algo.mySymbolDict[symbol]
        if direction == 1:
            stopPlacerType =    sd.CL.stopPlacerLong        if 'stopPlacer' not in customPositionSettings       else customPositionSettings['stopPlacer']
            targetPlacerType =  sd.CL.targetPlacerLong      if 'targetPlacer' not in customPositionSettings     else customPositionSettings['targetPlacer']
            minPayOff =         sd.CL.minPayOffLong         if 'minPayOff' not in customPositionSettings        else customPositionSettings['minPayOff']
            stopATR =           sd.CL.stopATRLong           if 'stopATR' not in customPositionSettings          else customPositionSettings['stopATR']
            minEntryStopATR =   sd.CL.minEntryStopATRLong   if 'minEntryStopATR' not in customPositionSettings  else customPositionSettings['minEntryStopATR']
            entryLimitOrder =   sd.CL.entryLimitOrderLong   if 'entryLimitOrder' not in customPositionSettings  else customPositionSettings['entryLimitOrder']
            limitEntryATR =     sd.CL.limitEntryATRLong     if 'limitEntryATR' not in customPositionSettings    else customPositionSettings['limitEntryATR']
        else:
            stopPlacerType =    sd.CL.stopPlacerShort       if 'stopPlacer' not in customPositionSettings       else customPositionSettings['stopPlacer']
            targetPlacerType =  sd.CL.targetPlacerShort     if 'targetPlacer' not in customPositionSettings     else customPositionSettings['targetPlacer']
            minPayOff =         sd.CL.minPayOffShort        if 'minPayOff' not in customPositionSettings        else customPositionSettings['minPayOff']
            stopATR =           sd.CL.stopATRShort          if 'stopATR' not in customPositionSettings          else customPositionSettings['stopATR']
            minEntryStopATR =   sd.CL.minEntryStopATRShort  if 'minEntryStopATR' not in customPositionSettings  else customPositionSettings['minEntryStopATR']
            entryLimitOrder =   sd.CL.entryLimitOrderShort  if 'entryLimitOrder' not in customPositionSettings  else customPositionSettings['entryLimitOrder']
            limitEntryATR =     sd.CL.limitEntryATRShort    if 'limitEntryATR' not in customPositionSettings    else customPositionSettings['limitEntryATR']
            
        entryPrice, stopPrice, targetPrice, minStopPrice, currentLow, currentHigh = 0, 0, 0, 0, 0, 0
        priceRoundingDigits = round(-1*math.log(self.algo.Securities[symbol].SymbolProperties.MinimumPriceVariation,10))

        if direction == 1:
            if entryLimitOrder: entryPrice = round(self.algo.Securities[symbol].Price - limitEntryATR * sd.atr1.Current.Value, priceRoundingDigits)
            else: entryPrice = self.algo.Securities[symbol].Price
            minStopPrice = round(entryPrice - minEntryStopATR * sd.atr1.Current.Value , priceRoundingDigits)
            currentLow = sd.bars_rw[0].Low  #as DCH does not take into account last Bar
        if direction == -1:
            if entryLimitOrder: entryPrice = round(self.algo.Securities[symbol].Price + limitEntryATR * sd.atr1.Current.Value, priceRoundingDigits)
            else: entryPrice = self.algo.Securities[symbol].Price
            minStopPrice = round(entryPrice + minEntryStopATR * sd.atr1.Current.Value, priceRoundingDigits)
            currentHigh = sd.bars_rw[0].High    #as DCH does not take into account last Bar
        
        '''
        STOP PLACER ---------------
        '''
        if stopPlacerType[0] == "minStop":
            ''' Stop: minEntryStopATR
            '''
            if direction == 1:
                stopPrice = minStopPrice
            if direction == -1:
                stopPrice = minStopPrice
        elif stopPlacerType[0] == "dch":
            '''Stop: min(minEntryStopATR, sd.dch),
            '''
            if direction == 1:
                stopPrice = round(min(minStopPrice ,min(currentLow, getattr(sd, stopPlacerType[1]).LowerBand.Current.Value) - stopATR * sd.atr1.Current.Value), priceRoundingDigits)
            if direction == -1:
                stopPrice = round(max (minStopPrice, max(currentHigh, getattr(sd, stopPlacerType[1]).UpperBand.Current.Value) + stopATR * sd.atr1.Current.Value), priceRoundingDigits)
        elif stopPlacerType[0] == "rw":
            ''' Stop: min(minEntryStopATR, rw(n)),  (use [0:n] to include the latest bar unlike dch)
            '''
            if direction == 1:
                stopPrice = round(min(minStopPrice ,min(currentLow, min([bar.Low for bar in sd.bars_rw][0:stopPlacerType[1]])) - stopATR * sd.atr1.Current.Value), priceRoundingDigits)
            if direction == -1:
                stopPrice = round(max (minStopPrice, max(currentHigh, max([bar.High for bar in sd.bars_rw][0:stopPlacerType[1]])) + stopATR * sd.atr1.Current.Value), priceRoundingDigits)
        else:
            ''' Base Case Stop: min(minEntryStopATR, rw(10)),  (use [0:n] to include the latest bar unlike dch)
            '''
            if direction == 1:
                stopPrice = round(min(minStopPrice ,min(currentLow, min([bar.Low for bar in sd.bars_rw][0:10])) - stopATR * sd.atr1.Current.Value), priceRoundingDigits)
            if direction == -1:
                stopPrice = round(max (minStopPrice, max(currentHigh, max([bar.High for bar in sd.bars_rw][0:10])) + stopATR * sd.atr1.Current.Value), priceRoundingDigits)

        '''
        TARGET PLACER ---------------
        '''
        if targetPlacerType[0]=="minPayOff" or targetPlacerType[0]==None:
            ''' minPayOff
            '''
            if direction == 1:
                targetPrice = round(entryPrice+minPayOff*(entryPrice-stopPrice), priceRoundingDigits)
            if direction == -1:
                targetPrice = round(entryPrice-minPayOff*(stopPrice-entryPrice), priceRoundingDigits)
        elif targetPlacerType[0] == "dch":
            '''Stop: min(minEntryStopATR, sd.dch),
            '''
            if direction == 1:
                targetPrice = round(max(entryPrice+minPayOff*(entryPrice-stopPrice), getattr(sd, targetPlacerType[1]).UpperBand.Current.Value - 0.01 * sd.atr1.Current.Value), priceRoundingDigits)
            if direction == -1:
                targetPrice = round(min(entryPrice-minPayOff*(stopPrice-entryPrice), getattr(sd, targetPlacerType[1]).LowerBand.Current.Value + 0.01 * sd.atr1.Current.Value), priceRoundingDigits)
        elif targetPlacerType[0] == "rw":
            ''' max/min(dch(targetPlacerType), minPayOff),  (use [0:n] to include the latest bar unlike dch)
            '''
            if direction == 1:
                targetPrice = round(max(entryPrice+minPayOff*(entryPrice-stopPrice), max([bar.High for bar in sd.bars_rw][0:targetPlacerType[1]]) - 0.01 * sd.atr1.Current.Value), priceRoundingDigits)
            if direction == -1:
                targetPrice = round(min(entryPrice-minPayOff*(stopPrice-entryPrice), min([bar.Low for bar in sd.bars_rw][0:targetPlacerType[1]]) + 0.01 * sd.atr1.Current.Value), priceRoundingDigits)
        else:
            ''' minPayOff
            '''
            if direction == 1:
                targetPrice = round(entryPrice+minPayOff*(entryPrice-stopPrice), priceRoundingDigits)
            if direction == -1:
                targetPrice = round(entryPrice-minPayOff*(stopPrice-entryPrice), priceRoundingDigits)

        return entryPrice, stopPrice, targetPrice
    
    '''
    STOP TRAILER
    stopTrailer must be a Tuple (Type, arg2, arg2...): (None,0), ("dch",'dch_attributeName'), ("rw",n)
    '''
    def TrailStops (self):
        debugTrail = self.debug
        minTrailStopVarianceATR = 0.25
        
        #List of Positions
        positionKeys = [x.Key for x in self.algo.Portfolio]
        if not positionKeys: return
        updateOrderFields = UpdateOrderFields() 
        
        for i in positionKeys:
            atrMargin = 1
            currentPrice = self.algo.Portfolio[i].Price
            averagePrice = self.algo.Portfolio[i].AveragePrice
            priceRoundingDigits = round(-1*math.log(self.algo.Securities[i].SymbolProperties.MinimumPriceVariation,10))
            sd = self.algo.mySymbolDict[i]
            wasJustUpdated = sd.WasJustUpdated(self.algo.Time)
            minTrailStopVariance = minTrailStopVarianceATR * sd.atr1.Current.Value
            if self.algo.Portfolio[i].Quantity > 0:
                scratchTrade = sd.CL.scratchTradeLong
                stopTrailer = sd.CL.stopTrailerLong
            else:
                scratchTrade = sd.CL.scratchTradeShort
                stopTrailer = sd.CL.stopTrailerShort
            
            '''SCRATCH TRADE
            '''
            if scratchTrade and wasJustUpdated:
                trailedStopPrice = 0
                #Long Positions
                for y in self.algo.openStopMarketOrders:
                    if self.algo.Portfolio[i].Symbol == y.Symbol and (y.Status == OrderStatus.Submitted or y.Status == OrderStatus.UpdateSubmitted or y.Status == 6):
                        tempStopOrder = self.algo.Transactions.GetOrderById(y.OrderId)
                        currentStopPrice = tempStopOrder.StopPrice
                        #LONG POSITION
                        if self.algo.Portfolio[i].Quantity > 0 and y.Quantity < 0:
                            if currentPrice - (averagePrice + minTrailStopVariance) > averagePrice - currentStopPrice:
                                trailedStopPrice = round(averagePrice + minTrailStopVariance, priceRoundingDigits)
                            if trailedStopPrice < currentStopPrice + minTrailStopVariance: trailedStopPrice = 0
                        #SHORT POSITION
                        elif self.algo.Portfolio[i].Quantity < 0 and y.Quantity > 0:
                            if (averagePrice - minTrailStopVariance) - currentPrice > currentStopPrice - averagePrice:
                                trailedStopPrice = round(averagePrice - minTrailStopVariance, priceRoundingDigits)
                            if trailedStopPrice > currentStopPrice - minTrailStopVariance: trailedStopPrice = 0
                        if trailedStopPrice != 0:
                            updateOrderFields.StopPrice = trailedStopPrice
                            updateOrderFields.Tag = tempStopOrder.Tag + "-S_sc: {}".format(str(trailedStopPrice))
                            #updateOrderFields.QuantityFilled = 0
                            y.Update(updateOrderFields)
                            if debugTrail: self.algo.MyDebug(" > STOP SCRATCHED for Symbol: {}({}), OrderID:{}, Status:{}, Quantity:{}, Stop Price:{}"
                                    .format(str(y.Symbol),
                                        str(sd.CL.strategyCode),
                                        str(y.OrderId),
                                        str(y.Status),
                                        str(y.Quantity),
                                        str(trailedStopPrice)))
                                                    
            if stopTrailer[0] == None and wasJustUpdated:
                '''0) NO STOPTRAIL
                '''
                pass
            if stopTrailer[0] == 'dch' and wasJustUpdated:
                '''1) If in Profit: dch -/+ atrMargin
                '''
                trailedStopPrice = 0
                scratchable = False
                #Long Positions
                for y in self.algo.openStopMarketOrders:
                    if self.algo.Portfolio[i].Symbol == y.Symbol and (y.Status == OrderStatus.Submitted or y.Status == OrderStatus.UpdateSubmitted or y.Status == 6):
                        tempStopOrder = self.algo.Transactions.GetOrderById(y.OrderId)
                        currentStopPrice = tempStopOrder.StopPrice
                        #LONG POSITION
                        if self.algo.Portfolio[i].Quantity > 0 and y.Quantity < 0:
                            referencePrice = getattr(sd, stopTrailer[1]).LowerBand.Current.Value
                            scratchable = currentPrice - (averagePrice + minTrailStopVariance) > averagePrice - currentStopPrice
                            trailedStopPrice = round(referencePrice - atrMargin * sd.atr1.Current.Value, priceRoundingDigits)
                            if trailedStopPrice < currentStopPrice + minTrailStopVariance: trailedStopPrice = 0
                        #SHORT POSITION
                        elif self.algo.Portfolio[i].Quantity < 0 and y.Quantity > 0:
                            referencePrice = getattr(sd, stopTrailer[1]).UpperBand.Current.Value
                            scratchable = (averagePrice - minTrailStopVariance) - currentPrice > currentStopPrice - averagePrice
                            trailedStopPrice = round(referencePrice + atrMargin * sd.atr1.Current.Value, priceRoundingDigits) 
                            if trailedStopPrice > currentStopPrice - minTrailStopVariance: trailedStopPrice = 0
                        if scratchable and trailedStopPrice != 0:
                            updateOrderFields.StopPrice = trailedStopPrice
                            updateOrderFields.Tag = tempStopOrder.Tag + f"-S_t({stopTrailer[0]}/{stopTrailer[1]}): {trailedStopPrice}"
                            #updateOrderFields.QuantityFilled = 0
                            y.Update(updateOrderFields)
                            if debugTrail: self.algo.MyDebug(" > STOP TRAILED({}) for Symbol: {}({}), OrderID:{}, Status:{}, Quantity:{}, Stop Price:{}"
                                    .format(str(stopTrailer[0]),
                                        str(y.Symbol),
                                        str(sd.CL.strategyCode),
                                        str(y.OrderId),
                                        str(y.Status),
                                        str(y.Quantity),
                                        str(trailedStopPrice)))
            if stopTrailer[0] == 'rw' and wasJustUpdated:
                '''1) If in Profit: min/max(rw(n)) -/+ atrMargin
                '''
                trailedStopPrice = 0
                scratchable = False
                #Long Positions
                for y in self.algo.openStopMarketOrders:
                    if self.algo.Portfolio[i].Symbol == y.Symbol and (y.Status == OrderStatus.Submitted or y.Status == OrderStatus.UpdateSubmitted or y.Status == 6):
                        tempStopOrder = self.algo.Transactions.GetOrderById(y.OrderId)
                        currentStopPrice = tempStopOrder.StopPrice
                        #LONG POSITION
                        if self.algo.Portfolio[i].Quantity > 0 and y.Quantity < 0:
                            referencePrice = min([bar.Low for bar in sd.bars_rw][0:stopTrailer[1]])
                            scratchable = currentPrice - (averagePrice + minTrailStopVariance) > averagePrice - currentStopPrice
                            trailedStopPrice = round(referencePrice - atrMargin * sd.atr1.Current.Value, priceRoundingDigits)
                            if trailedStopPrice < currentStopPrice + minTrailStopVariance: trailedStopPrice = 0
                        #SHORT POSITION
                        elif self.algo.Portfolio[i].Quantity < 0 and y.Quantity > 0:
                            referencePrice = max([bar.High for bar in sd.bars_rw][0:stopTrailer[1]])
                            scratchable = (averagePrice - minTrailStopVariance) - currentPrice > currentStopPrice - averagePrice
                            trailedStopPrice = round(referencePrice + atrMargin * sd.atr1.Current.Value, priceRoundingDigits) 
                            if trailedStopPrice > currentStopPrice - minTrailStopVariance: trailedStopPrice = 0
                        if scratchable and trailedStopPrice != 0:
                            updateOrderFields.StopPrice = trailedStopPrice
                            updateOrderFields.Tag = tempStopOrder.Tag + f"-S_t({stopTrailer[0]}/{stopTrailer[1]}): {trailedStopPrice}"
                            #updateOrderFields.QuantityFilled = 0
                            y.Update(updateOrderFields)
                            if debugTrail: self.algo.MyDebug(" > STOP TRAILED({}) for Symbol: {}({}), OrderID:{}, Status:{}, Quantity:{}, Stop Price:{}"
                                    .format(str(stopTrailer[0]),
                                        str(y.Symbol),
                                        str(sd.CL.strategyCode),
                                        str(y.OrderId),
                                        str(y.Status),
                                        str(y.Quantity),
                                        str(trailedStopPrice)))
    
        return
    '''
    TRAIL TARGET
    targetTrailer must be a Tuple (Type, arg2, arg2...): (None,0), ("dch",'dch_attributeName'), ("rw",n)
    '''
    def TrailTargets (self):
        debugTrail = self.debug
        minTrailLimitVarianceATR = 0.25
        
        #List of Positions
        positionKeys = [x.Key for x in self.algo.Portfolio]
        if not positionKeys: return
        updateOrderFields = UpdateOrderFields() 
        
        for i in positionKeys:
            atrMargin = 1
            currentPrice = self.algo.Portfolio[i].Price
            averagePrice = self.algo.Portfolio[i].AveragePrice
            priceRoundingDigits = round(-1*math.log(self.algo.Securities[i].SymbolProperties.MinimumPriceVariation,10))
            sd = self.algo.mySymbolDict[i]
            wasJustUpdated = sd.WasJustUpdated(self.algo.Time)
            minTrailLimitVariance = minTrailLimitVarianceATR * sd.atr1.Current.Value
            if self.algo.Portfolio[i].Quantity > 0:
                targetTrailer = sd.CL.targetTrailerLong
            else:
                targetTrailer = sd.CL.targetTrailerShort

            if targetTrailer[0] == None:
                ''' NO TARGETTRAIL
                '''
                pass
            
            if targetTrailer[0] == 'dch' and wasJustUpdated:
                ''' dch -/+ atrMargin
                '''
                trailedLimitPrice = 0
                #Long Positions
                for y in self.algo.openLimitOrders:
                    if self.algo.Portfolio[i].Symbol == y.Symbol and (y.Status == OrderStatus.Submitted or y.Status == OrderStatus.UpdateSubmitted or y.Status == 6):
                        tempLimitOrder = self.algo.Transactions.GetOrderById(y.OrderId)
                        currentLimitPrice = tempLimitOrder.LimitPrice
                        #LONG POSITION
                        if self.algo.Portfolio[i].Quantity > 0 and y.Quantity < 0:
                            referencePrice = getattr(sd, targetTrailer[1]).UpperBand.Current.Value
                            trailedLimitPrice = round(referencePrice - atrMargin * sd.atr1.Current.Value, priceRoundingDigits)
                            if trailedLimitPrice > currentLimitPrice + minTrailLimitVariance: trailedLimitPrice = 0
                        #SHORT POSITION
                        elif self.algo.Portfolio[i].Quantity < 0 and y.Quantity > 0:
                            referencePrice = getattr(sd, targetTrailer[1]).LowerBand.Current.Value
                            trailedLimitPrice = round(referencePrice + atrMargin * sd.atr1.Current.Value, priceRoundingDigits) 
                            if trailedLimitPrice < currentLimitPrice - minTrailLimitVariance: trailedLimitPrice = 0
                        if trailedLimitPrice != 0:
                            updateOrderFields.LimitPrice = trailedLimitPrice
                            updateOrderFields.Tag = tempLimitOrder.Tag + "-L_t({}): {}".format(str(targetTrailer[0])+'-'+tr(targetTrailer[1]), str(trailedLimitPrice))
                            #updateOrderFields.QuantityFilled = 0
                            y.Update(updateOrderFields)
                            if debugTrail: self.algo.MyDebug(" > TARGET TRAILED({}) for Symbol: {}({}), OrderID:{}, Status:{}, Quantity:{}, Limit Price:{}"
                                    .format(str(targetTrailer[0]),
                                        str(y.Symbol),
                                        str(sd.CL.strategyCode),
                                        str(y.OrderId),
                                        str(y.Status),
                                        str(y.Quantity),
                                        str(trailedLimitPrice)))
                                        
            if targetTrailer[0] == 'rw' and wasJustUpdated:
                ''' rw(n) -/+ atrMargin
                '''
                trailedLimitPrice = 0
                #Long Positions
                for y in self.algo.openLimitOrders:
                    if self.algo.Portfolio[i].Symbol == y.Symbol and (y.Status == OrderStatus.Submitted or y.Status == OrderStatus.UpdateSubmitted or y.Status == 6):
                        tempLimitOrder = self.algo.Transactions.GetOrderById(y.OrderId)
                        currentLimitPrice = tempLimitOrder.LimitPrice
                        #LONG POSITION
                        if self.algo.Portfolio[i].Quantity > 0 and y.Quantity < 0:
                            referencePrice =  max([bar.High for bar in sd.bars_rw][0:targetTrailer[1]])
                            trailedLimitPrice = round(referencePrice - atrMargin * sd.atr1.Current.Value, priceRoundingDigits)
                            if trailedLimitPrice > currentLimitPrice + minTrailLimitVariance: trailedLimitPrice = 0
                        #SHORT POSITION
                        elif self.algo.Portfolio[i].Quantity < 0 and y.Quantity > 0:
                            referencePrice = min([bar.Low for bar in sd.bars_rw][0:targetTrailer[1]])
                            trailedLimitPrice = round(referencePrice + atrMargin * sd.atr1.Current.Value, priceRoundingDigits) 
                            if trailedLimitPrice < currentLimitPrice - minTrailLimitVariance: trailedLimitPrice = 0
                        if trailedLimitPrice != 0:
                            updateOrderFields.LimitPrice = trailedLimitPrice
                            updateOrderFields.Tag = tempLimitOrder.Tag + "-L_t({}): {}".format(str(targetTrailer[0])+'-'+tr(targetTrailer[1]), str(trailedLimitPrice))
                            #updateOrderFields.QuantityFilled = 0
                            y.Update(updateOrderFields)
                            if debugTrail: self.algo.MyDebug(" > TARGET TRAILED({}) for Symbol: {}({}), OrderID:{}, Status:{}, Quantity:{}, Limit Price:{}"
                                    .format(str(targetTrailer[0]),
                                        str(y.Symbol),
                                        str(sd.CL.strategyCode),
                                        str(y.OrderId),
                                        str(y.Status),
                                        str(y.Quantity),
                                        str(trailedLimitPrice)))     
        return
    
    '''
    FLIP POSITION (Flip or Close on Opposite Trigger)
    '''
    def EnterPosition_2(self, symbol, myDirection, myRiskMultiple=1.00, customPositionSettings={}):
        self.CheckPendingEntry()
        pendingEntries = self.CalculatePendingEntry(symbol)
        
        #If everything is closed and no Pending Entry, Enter Position
        if pendingEntries['_SP']+pendingEntries['_SSubmitted']==0 and pendingEntries['_PFlip']==0:
            self.EnterPosition(symbol, myDirection, myReverse=False, myRiskMultiple=myRiskMultiple, customPositionSettings=customPositionSettings)
            return
        
        enableFlip = self.algo.mySymbolDict[symbol].CL.enableFlipLong if myDirection==1 else self.algo.mySymbolDict[symbol].CL.enableFlipShort
        closeOnTrigger = self.algo.mySymbolDict[symbol].CL.closeOnTriggerLong if myDirection==1 else self.algo.mySymbolDict[symbol].CL.closeOnTriggerShort
        
        #If there is existing open or submitted position with opposite direction and Flip Enabled
        if (pendingEntries['_SP']+pendingEntries['_SSubmitted'])*myDirection < 0 and enableFlip:
            self.algo.mySymbolDict[symbol].CL._totalFlipTrades += 1
            self.FlipPosition(symbol, myDirection, myRiskMultiple=myRiskMultiple, customPositionSettings=customPositionSettings)
            return
        
        #If there is existing open or submitted position with opposite direction  and closeOnTriggerLong Enabled
        if (pendingEntries['_SP']+pendingEntries['_SSubmitted'])*myDirection < 0 and closeOnTrigger:
            self.algo.myPositionManagerB.LiquidatePosition(symbol, "OppositeTrigger", "OppositeTrigger")
            return
        return
        
    def FlipPosition(self, symbol, myDirection, myRiskMultiple=1.00, customPositionSettings={}):
        debugFlip = self.debug
        
        #Remove eny existing pending position for the symbol
        #This could only be opposite as this is the condition of calling FlipPosition
        self.RemovePendingFlipPosition(symbol)
        
        #Liquidate position or submitted order if any 
        self.algo.myPositionManagerB.LiquidatePosition(symbol, "FlipPosition", "FlipPosition")
        
        #Submit pending position that is entered by CheckPendingEntry once _SP+_SM == 0 and _ASS == 0 and _ASL == 0:
        entryTimeInForce = self.algo.mySymbolDict[symbol].CL.entryTimeInForceLong if myDirection == 1 else self.algo.mySymbolDict[symbol].CL.entryTimeInForceShort
        expiryDate = self.algo.Time + entryTimeInForce #+ timedelta(days=5*60)
        #self.CL.pendingFlipPositions.append((symbol, myDirection, expiryDate, myEntry, myStop, myTarget, myRiskMultiple, myStopPlacer, myTargetPlacer, myMinPayOff))
        self.CL.pendingFlipPositions.append({'symbol':symbol, 'direction':myDirection, 'expiryDate':expiryDate, 'riskMultiple':myRiskMultiple, 'customPositionSettings':customPositionSettings})
        if debugFlip or self.algo.LiveMode: self.algo.MyDebug(" > POSITION FLIPPED: {}({}), Direction:{}"
                        .format(str(symbol), str(self.algo.mySymbolDict[symbol].CL.strategyCode), str(myDirection)))
        #Check if pending position already submittable 
        self.CheckPendingEntry()
        return
    
    def RemovePendingFlipPosition(self, symbol):
        i=0
        for pos in self.CL.pendingFlipPositions:
            if pos['symbol'] == symbol:
                self.CL.pendingFlipPositions.remove(pos)
                i+=1
        return i
    
    def CheckPendingEntry(self):
        debugFlipCheck = self.debug
         
        #Remove Expired Enries
        for pos in self.CL.pendingFlipPositions:
            if pos['expiryDate'] < self.algo.Time:
                self.CL.pendingFlipPositions.remove(pos)
        
        #Exit If No Pending Entry is left
        if len(self.CL.pendingFlipPositions) == 0:
            return
        
        #Submint Entry if no orders and positions
        for pos in self.CL.pendingFlipPositions:
            symbol = pos['symbol']
            direction = pos['direction']
            #Run consistency
            self.algo.myPositionManagerB.SymbolOrderConsistency(symbol)
            #Check position and pending Entry
            _SP = self.algo.Portfolio[symbol].Quantity
            _SM = self.algo.myPositionManagerB.SumOrder(symbol, self.algo.openMarketOrders)[1]
            _ASS = self.algo.myPositionManagerB.SumOrder(symbol, self.algo.openStopMarketOrders)[2]
            _ASL = self.algo.myPositionManagerB.SumOrder(symbol, self.algo.openLimitOrders)[2]
            if _SP+_SM == 0 and _ASS == 0 and _ASL == 0 and self.algo.mySymbolDict[symbol].entryEnabled:
                if debugFlipCheck or self.algo.LiveMode: self.algo.MyDebug("  FLIP IS SUBMITTED: {}({}), Direction:{}"
                        .format(str(symbol), str(self.algo.mySymbolDict[symbol].CL.strategyCode), str(direction)))
                #Submit Entry (Note that entryTimeInForceLong would be reset by EnterPosition)
                self.EnterPosition(symbol, direction, myReverse=False, myRiskMultiple=pos['riskMultiple'], customPositionSettings=pos['customPositionSettings'])
                #Remove Pendig Entry
                self.CL.pendingFlipPositions.remove(pos)
        return
    
    def CalculatePendingEntry(self, symbol):
        pendingEntries = {
            '_SSubmitted': 0.0,
            '_PFlip': 0.0,
            '_PE': 0.0,
            '_SP': 0.0,
            '_SM': 0.0,
            '_SS': 0.0,
            '_SL': 0.0,
            '_ASL': 0.0
        }
        
        pendingEntries['_SP'] = self.algo.Portfolio[symbol].Quantity
        pendingEntries['_SM'] = self.algo.myPositionManagerB.SumOrder(symbol, self.algo.openMarketOrders)[1]
        pendingEntries['_SS'] = self.algo.myPositionManagerB.SumOrder(symbol, self.algo.openStopMarketOrders)[1]
        pendingEntries['_SL'] = self.algo.myPositionManagerB.SumOrder(symbol, self.algo.openLimitOrders)[1]
        pendingEntries['_ASL'] = self.algo.myPositionManagerB.SumOrder(symbol, self.algo.openLimitOrders)[2]

        # IF customPositionSettings OVERRRDES DEFAULT SETTINGS _UT and _SSubmitted ARE WRONG VALUES!!!!!
        if pendingEntries['_SS'] < 0:
            _UT = getattr(self.algo.mySymbolDict[symbol].CL, "useTragetsLong", self.algo.mySymbolDict[symbol].CL.targetPlacerLong[0]!=None)
        else:
            _UT = getattr(self.algo.mySymbolDict[symbol].CL, "useTragetsShort", self.algo.mySymbolDict[symbol].CL.targetPlacerShort[0]!=None)
        #Submitted orders that meant to change the position (not stop or target). Limit and Market entry
        pendingEntries['_SSubmitted'] = pendingEntries['_SL'] + pendingEntries['_SM'] - _UT * pendingEntries['_SS']

        for pos in self.CL.pendingFlipPositions:
            if pos['symbol'] == symbol:
                #direction of Pending Entry
                pendingEntries['_PFlip']  = pos['direction']
            else:
                pendingEntries['_PFlip']  = 0
        #Pending Entry that is already submitted or pending as waiting fro the opposit position to close
        pendingEntries['_PE'] = pendingEntries['_SSubmitted'] + pendingEntries['_PFlip']
        return pendingEntries