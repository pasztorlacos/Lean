### <summary>
### Position Manager (part B.)
###
### </summary>
#import math
from datetime import datetime, timedelta
from QuantConnect import SecurityType
from QuantConnect.Orders import UpdateOrderFields, OrderDirection, OrderStatus, TimeInForce

#--------------MyPositionManager CLASS 
class MyPositionManagerB:
    '''
    Enter and Exit Positions 
    Manage orders and
    Order Consistency (continued due to 64k Limit)
    '''
    file = __file__
    def __init__(self, caller):
        self.CL = self.__class__
        self.algo = caller
        self.debug = self.algo.debug
        if self.algo.LiveMode: self.debug = True
        return
    
    '''
    ORDER EVENT HANDLER
    '''
    def MyOrderEventHandler(self, OrderEvent):
        debugPartialFill = False
        orderID = OrderEvent.OrderId
        fetched = self.algo.Transactions.GetOrderById(orderID)
        ticket = self.algo.Transactions.GetOrderTicket(orderID)
        symbol = fetched.Symbol
        
        '''EXIT IF Symbol not in mySymbolDict'''
        if symbol not in self.algo.mySymbolDict: return
        
        #Cancel or CancelPending
        if OrderEvent.Status == OrderStatus.Canceled or OrderEvent.Status == OrderStatus.CancelPending:
            self.SymbolOrderConsistency(symbol)
        #Deal with Filled Orders
        if OrderEvent.FillQuantity == 0:
            return

        partialFill = True
        fillMessage = "PARTIAL FILL"
        if ticket.Quantity == ticket.QuantityFilled:
            partialFill = False
            fillMessage = "FILL"
        orderType = ""
        if fetched.Type == 1: 
            orderType = "Limit"
        elif fetched.Type == 2:
            orderType = "Stop"
        else: 
            orderType = "Market"
 
        if self.algo.debugOrderFill and (debugPartialFill or not partialFill): self.algo.MyDebug(" > {} Symbol: {}({}), OrderID:{}, Type:{}/{}, Status:{}, TQuantity:{}, TQFilled:{}, EventFillQ:{}, FillPrice:{}, Direction:{}, Holding:{}"
                    .format(fillMessage,
                            str(symbol),
                            str(self.algo.mySymbolDict[symbol].CL.strategyCode),
                            str(orderID),
                            str(fetched.Type),
                            str(orderType),
                            str(fetched.Status),
                            str(ticket.Quantity),
                            str(ticket.QuantityFilled),
                            str(OrderEvent.FillQuantity),
                            str(round(OrderEvent.FillPrice , 2)),
                            str(OrderEvent.Direction),
                            str(self.algo.Portfolio[fetched.Symbol].Quantity)))
        #Don't call at every partiall fill, only when the order is fully filled as the fill process cannot be synchronously handled
        #   Issue1: while reading order values of a position one was read (stop) can already change while reading the other (target)
        #   Issue2: OnOrderEvent is invoked before the new order ticket is assigned to a variable (first bolg fill)
        #            => cannot use the order ticket at this stage. Orders starts to fill partially with the first bloc.
        #If Partiall Fill 
        #or Gap Up/Down and Entry and Stop triggreded at the same time: wait until the position get synced with the broker 
        if self.algo.LiveMode and partialFill: 
            #Block Consistency for consistencyCheckSec
            self.algo.mySymbolDict[symbol].fillReleaseTime = self.algo.Time + self.algo.myVaR.CL.consistencyCheckSec
        elif self.algo.LiveMode and not partialFill and (fetched.Type == 1 or fetched.Type == 2):
            #Release Consistency Block
            self.algo.mySymbolDict[symbol].fillReleaseTime = self.algo.Time - timedelta(seconds=5)
            self.SymbolOrderConsistency(symbol)
        if not partialFill: self.algo.mySymbolDict[symbol].CL._totalFills += 1
        if not partialFill and self.algo.Portfolio[symbol].Quantity != 0: self.algo.mySymbolDict[symbol].CL._totalEntryFills += 1
        self.algo.myVaR.Update()
        self.algo.myCharts.Update(True)
        self.algo.myStats.Update(True)
        return
    
    '''
    POSITION SUMMARY for symbol
    '''
    def SumPosition (self, symbol):
        _C, _S, _AS = 0, 0, 0
        for x in self.algo.Portfolio:
            if x.Key == symbol and self.algo.Portfolio[x.Key].Quantity !=0:
                _C += 1
                _S += self.algo.Portfolio[x.Key].Quantity
                _AS += abs(self.algo.Portfolio[x.Key].Quantity)
    
        return _C, _S, _AS
    '''
    ORDER SUMMARY for symbol
    '''
    def SumOrder (self, symbol, orderlist):
        _C, _S, _AS = 0, 0, 0
        _lastTicket = None
        for tempTicket in orderlist:
            if tempTicket.Symbol == symbol and (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == OrderStatus.UpdateSubmitted or tempTicket.Status == 6 or tempTicket.Status == OrderStatus.New):
                _C += 1
                _S += tempTicket.Quantity - tempTicket.QuantityFilled
                _AS += abs(tempTicket.Quantity - tempTicket.QuantityFilled)
                _marketTicket = tempTicket
        if _lastTicket != None: lastTicketID = _lastTicket.OrderId
        else: lastTicketID = -1
        return _C, _S, _AS, _lastTicket, lastTicketID
        
    '''
    LIQUIDATE ALGO
    '''
    def LiquidateAlgo(self):
        self.algo.MyDebug(" ------------ ALGO LIQUIDATION! ----------") 
        for x in self.algo.Portfolio:
            if self.algo.Portfolio[x.Key].Quantity != 0:
                self.LiquidatePosition(x.Key, "AlgoLiq", " --- Algo Liquidation")
        self.algo.enabled = False
   
    '''
    LIQUIDATE STRATEGY
    '''
    def LiquidateStrategy(self, strategy, direction):
        self.algo.MyDebug(" ------------ STRATEGY LIQUIDATION! Strategy: " + str(strategy.strategyCode) + "Dir: " + str(direction)) 
        for x in self.algo.Portfolio:
            if self.algo.mySymbolDict[orderSymbol].CL == strategy and self.algo.Portfolio[x.Key].Quantity != 0:
                if direction == 0:
                    self.LiquidatePosition(x.Key, "StrLi", " --- Strategy Liquidation! Strategy: " +str(str(strategy.strategyCode)) + "Dir: " + str(direction))
                elif direction == 1 and self.algo.Portfolio[x.Key].Quantity > 0:
                    self.LiquidatePosition(x.Key, "StrLi", " --- Strategy Liquidation! Strategy: " +str(str(strategy.strategyCode)) + "Dir: " + str(direction))
                elif direction == -1 and self.algo.Portfolio[x.Key].Quantity < 0:
                    self.LiquidatePosition(x.Key, "StrLi", " --- Strategy Liquidation! Strategy: " +str(str(strategy.strategyCode)) + "Dir: " + str(direction))
    
    '''
    LIQUIDATE POSITION
    '''
    def LiquidatePosition(self, orderSymbol, orderTag, message):
        debugLiquidation = self.debug
        strategyCode = str(self.algo.mySymbolDict[orderSymbol].CL.strategyCode)
        
        quantityLimit = 1
        if orderSymbol.SecurityType == SecurityType.Forex:
            quantityLimit = 1000
        
        #Stops: Count, SumQuantity, SumAbsQuantity
        _CS, _SS, _ASS= 0, 0, 0
        #Limits: Count, SumQuantity, SumAbsQuantity
        _CL, _SL, _ASL= 0, 0, 0
        #Markets: Count, SumQuantity, SumAbsQuantity
        _CM, _SM, _ASM= 0, 0, 0        
        #Positions: Count, SumQuantity, SumAbsQuantity
        _CP, _SP, _ASP= 0, 0, 0 
        
        #Block consistency
        self.algo.mySymbolDict[orderSymbol].blockOrderCheck = True
        
        #THIS IS CODE DUPLICATION TO BE REFACTORED!
        #Positions
        for x in self.algo.Portfolio:
            if x.Key == orderSymbol and self.algo.Portfolio[x.Key].Quantity !=0:
                _CP += 1
                _SP += self.algo.Portfolio[x.Key].Quantity
                _ASP += abs(self.algo.Portfolio[x.Key].Quantity)
        
        #Market Orders
        for tempTicket in self.algo.openMarketOrders:
            if tempTicket.Symbol == orderSymbol and (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == OrderStatus.UpdateSubmitted or tempTicket.Status == 6 or tempTicket.Status == OrderStatus.New):
                _CM += 1
                _SM += tempTicket.Quantity - tempTicket.QuantityFilled
                _ASM += abs(tempTicket.Quantity - tempTicket.QuantityFilled)
                _marketTicket = tempTicket
        #Stops
        for tempTicket in self.algo.openStopMarketOrders:
            if tempTicket.Symbol == orderSymbol and (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == OrderStatus.UpdateSubmitted or tempTicket.Status == 6):
                _CS += 1
                _SS += tempTicket.Quantity - tempTicket.QuantityFilled
                _ASS += abs(tempTicket.Quantity - tempTicket.QuantityFilled)
        #Limits
        for tempTicket in self.algo.openLimitOrders:
            if tempTicket.Symbol == orderSymbol and (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == OrderStatus.UpdateSubmitted or tempTicket.Status == 6):
                _CL += 1
                _SL += tempTicket.Quantity - tempTicket.QuantityFilled
                _ASL += abs(tempTicket.Quantity - tempTicket.QuantityFilled)      
        
        #No orders to cancel and quantity is too small
        if _ASL==0 and _ASS==0 and abs(self.algo.Portfolio[orderSymbol].Quantity + _SM)<quantityLimit:
            return

        if debugLiquidation: self.algo.MyDebug(" > LIQUIDATION STARTED: " + str(orderSymbol) + "(" + str(strategyCode) + ") --- " +str(message))
        updateOrderFields = UpdateOrderFields()
        #Cancelling Stops
        for tempTicket in self.algo.openStopMarketOrders:
            if tempTicket.Symbol == orderSymbol and (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == OrderStatus.UpdateSubmitted or tempTicket.Status == 6) and tempTicket.Status != OrderStatus.CancelPending:
                updateOrderFields.Tag = tempTicket.Tag + "_Liq " + " ---_SP:"+str(_SP)+" _SS:"+str(_SS)+" _SL:"+str(_SL)+" _SM:"+str(_SM)+"_"+str(orderTag)+"_"+str(self.algo.Time)
                tempTicket.Update(updateOrderFields)
                response = tempTicket.Cancel()
                if self.debug or debugLiquidation: 
                    if response.IsSuccess: self.algo.MyDebug(" > {} CANCEL STOP for Symbol: {}({}), OrderID:{}, Status:{}, Quantity:{}, StopPrice:{}, _SP:{}, _SS:{}, _SL:{}, _SM:{}"
                        .format(str(message),
                                str(tempTicket.Symbol),
                                str(strategyCode),
                                str(tempTicket.OrderId),
                                str(tempTicket.Status),
                                str(tempTicket.Quantity),
                                str(self.algo.Transactions.GetOrderById(tempTicket.OrderId).StopPrice),
                                str(_SP),
                                str(_SS),
                                str(_SL),
                                str(_SM)))
                    else: self.algo.MyDebug(" > FAILED TO CANCEL STOP for Symbol: " + str(orderSymbol) + " _SP:" + str (_SP) + " _SL:" + str (_SL))
        
        #Cancelling Limits
        for tempTicket in self.algo.openLimitOrders:
            if tempTicket.Symbol == orderSymbol and (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == OrderStatus.UpdateSubmitted or tempTicket.Status == 6) and tempTicket.Status != OrderStatus.CancelPending:
                updateOrderFields.Tag = tempTicket.Tag + "_Liq " + " ---_SP:"+str(_SP)+" _SS:"+str(_SS)+" _SL:"+str(_SL)+" _SM:"+str(_SM)+"_"+str(orderTag)+"_"+str(self.algo.Time)
                tempTicket.Update(updateOrderFields)
                response = tempTicket.Cancel()            
                if self.debug or debugLiquidation: 
                    if response.IsSuccess: self.algo.MyDebug(" > {} CANCEL LIMIT for Symbol: {}({}), OrderID:{}, Status:{}, Quantity:{}, LimitPrice:{}, _SP:{}, _SS:{}, _SL:{}, _SM:{}"
                        .format(str(message),
                                str(tempTicket.Symbol),
                                str(strategyCode),
                                str(tempTicket.OrderId),
                                str(tempTicket.Status),
                                str(tempTicket.Quantity),
                                str(self.algo.Transactions.GetOrderById(tempTicket.OrderId).LimitPrice),
                                str(_SP),
                                str(_SS),
                                str(_SL),
                                str(_SM)))
                    else: self.algo.MyDebug(" > FAILED TO CANCEL LIMIT for Symbol:" + str(orderSymbol) + " _SP:" + str (_SP) + " _SL:" + str (_SL))

        #Exiting Position if not in Liquidation
        positionQuantity = self.algo.Portfolio[orderSymbol].Quantity
        if abs(positionQuantity + _SM)>=quantityLimit:
            #Use Market Order so no repetative Liquidations as liquidation order gets into consistency check
            entryPrice = self.algo.Securities[orderSymbol].Price
            orderTag2 = "Liq: " + str(entryPrice) +" ---_SP:"+str(_SP)+" _SS:"+str(_SS)+" _SL:"+str(_SL)+"_SM:"+str(_SM)+"_"+str(orderTag)+"_"+str(self.algo.Time)
            #without -_SM it can get into an infinite liquidation loop 
            orderQuantity = -1*positionQuantity - _SM
            positionTicket = self.algo.MarketOrder(orderSymbol, orderQuantity,  False, orderTag2)
            enterOrderId = positionTicket.OrderId
            if debugLiquidation: self.algo.MyDebug(" > LIQUIDATION Symbol: {}({}), Liquidated Quantity:{}, _SP:{}, _SS:{}, _SL:{}, _SM:{}, >> {}"
                        .format(str(orderSymbol),
                                str(strategyCode),
                                str(self.algo.Portfolio[orderSymbol].Quantity),
                                str(_SP),
                                str(_SS),
                                str(_SL),
                                str(_SM),
                                str(message)))

            self.algo.openMarketOrders.append(positionTicket)
            #Block Consistency to wait for Market order fill
            self.algo.mySymbolDict[orderSymbol].fillReleaseTime = self.algo.Time + 5*self.algo.myVaR.CL.consistencyCheckSec
        if debugLiquidation: self.algo.MyDebug(" > LIQUIDATION FINISHED: " + str(orderSymbol) + "(" + str(strategyCode) + ") --- " +str(message))
        
        #Release consistency block
        self.algo.mySymbolDict[orderSymbol].blockOrderCheck = False
        return
    
    '''
    ALL SYMBOLS CONSISTENCY
    '''
    def AllOrdersConsistency(self):
        #Call consistency check for symbols
        for x in self.algo.Securities:
            #ConvertionRate Currency (EURUSD) and like would be in mySymbolDict so it would throw KeyError exception  
            if self.algo.Securities[x.Key].Symbol in self.algo.mySymbolDict:
                self.SymbolOrderConsistency(self.algo.Securities[x.Key].Symbol)
        #PENDING ENTRIES: NO THIS CAUSES RECURSIVE LOOP as CheckPendingEntry->EnterPosition->VaR->AllOrdersConsistency->CheckPendingEntry
        #self.algo.myPositionManager.CheckPendingEntry()
        return
    
    '''
    SYMBOL CONSISTENCY
    '''
    def SymbolOrderConsistency(self, orderSymbol):
        debugConsistency= False
        adjustResubmit = False
        
        '''EXIT IF Symbol not in mySymbolDict'''
        if orderSymbol not in self.algo.mySymbolDict: return
        
        strategyCode = str(self.algo.mySymbolDict[orderSymbol].CL.strategyCode)
        
        '''EXIT IF WarmingUp or Wait for full IB Sync'''
        if self.algo.IsWarmingUp or self.algo.Time < self.algo.consistencyStartUpReleaseTime: 
            return
       
        '''EXIT IF Consistency Check for Strategy is disabled'''
        if hasattr(self.algo.mySymbolDict[orderSymbol].CL, 'manageOrderConsistency'):
            if not self.algo.mySymbolDict[orderSymbol].CL.manageOrderConsistency:
                return
        
        '''EXIT IF consistency check is blocked for symbol'''
        timeDifference = max(timedelta(0),self.algo.mySymbolDict[orderSymbol].fillReleaseTime - self.algo.Time)
        twsBlock = self.algo.mySymbolDict[orderSymbol].fromTWS and not self.algo.myVaR.CL.manageTWSSymbols
        if self.algo.mySymbolDict[orderSymbol].blockOrderCheck or self.algo.mySymbolDict[orderSymbol].fillReleaseTime > self.algo.Time or twsBlock:
            if False and self.debug: self.algo.MyDebug("     CONSISTENCY CHECK WAS BLOCKED for Symbol: {}({}), blockOrderCheck:{}, timeDifference:{}, TWSBlock:{}".format(
                str(orderSymbol),
                str(strategyCode),
                str(self.algo.mySymbolDict[orderSymbol].blockOrderCheck),
                str(timeDifference),
                str(twsBlock)))
            return
        #Set Block to avoid concurent checks and actions
        self.algo.mySymbolDict[orderSymbol].blockOrderCheck = True
        
        '''LIQUIDATE IF STARTAGEY liquidateLong or LiquidateShort'''
        if hasattr(self.algo.mySymbolDict[orderSymbol].CL, 'liquidateLong') and self.algo.Portfolio[orderSymbol].Quantity>0:
            if self.algo.mySymbolDict[orderSymbol].CL.liquidateLong:
                self.LiquidatePosition(orderSymbol, "liquidateLong", "liquidateLong")
        if hasattr(self.algo.mySymbolDict[orderSymbol].CL, 'liquidateShort') and self.algo.Portfolio[orderSymbol].Quantity<0:
            if self.algo.mySymbolDict[orderSymbol].CL.liquidateShort:
                self.LiquidatePosition(orderSymbol, "liquidateShort", "liquidateShort")               
        
        if False and orderSymbol.Value =="HD" and self.algo.Time > datetime(2019, 7, 22, 9, 0) and self.algo.Time <= datetime(2019, 7, 22, 13, 00): debugConsistency = True
        else: debugConsistency = False
        #Stops: Count, SumQuantity, SumAbsQuantity
        _CS, _SS, _ASS= 0, 0, 0
        #Limits: Count, SumQuantity, SumAbsQuantity
        _CL, _SL, _ASL= 0, 0, 0
        #Markets: Count, SumQuantity, SumAbsQuantity
        _CM, _SM, _ASM= 0, 0, 0        
        #Positions: Count, SumQuantity, SumAbsQuantity
        _CP, _SP, _ASP= 0, 0, 0             
        _UT = True #in case there is no stop
        _adjustStopTicket = None
        _adjustLimitTicket = None
        _marketTicket = None
        _SDIROK = False
        _LDIROK = False
        _0, _I, _I_A, _I_B, _I_C, _II, _II_A, _II_B, II_C = False, False, False, False, False, False, False, False, False
        
        if debugConsistency: self.algo.MyDebug(" SYMBOL: " +str(orderSymbol) + "(" + str(strategyCode) + ")   ----  CONSISTENCY CHECK STARTED")
        '''GET CURRENT STATUS'''
        #THIS IS CODE DUPLICATION TO BE REFACTORED!
        #Submitted and None(IB sync at Sturtup) orders that matter
        #Stops
        for tempTicket in self.algo.openStopMarketOrders:
            if tempTicket.Symbol == orderSymbol and (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == OrderStatus.UpdateSubmitted or tempTicket.Status == 6 or tempTicket.Status == OrderStatus.New):
                _CS += 1
                _SS += tempTicket.Quantity - tempTicket.QuantityFilled
                _ASS += abs(tempTicket.Quantity - tempTicket.QuantityFilled)
                #Change order condition imply that cannot be more than one
                _adjustStopTicket = tempTicket
        if _adjustStopTicket != None: lastTicketID = _adjustStopTicket.OrderId
        else: lastTicketID = -1
        if debugConsistency: self.algo.MyDebug(" > _CS:{}, _SS:{}, _ASS:{}, LastOrderID:{}".format(str(_CS), str(_SS),str(_ASS),str(lastTicketID)))
        #Limits
        for tempTicket in self.algo.openLimitOrders:
            if tempTicket.Symbol == orderSymbol and (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == OrderStatus.UpdateSubmitted or tempTicket.Status == 6 or tempTicket.Status == OrderStatus.New):
                _CL += 1
                _SL += tempTicket.Quantity - tempTicket.QuantityFilled
                _ASL += abs(tempTicket.Quantity - tempTicket.QuantityFilled)      
                #Change order condition imply that cannot be more than one
                _adjustLimitTicket = tempTicket
        if _adjustLimitTicket != None: lastTicketID = _adjustLimitTicket.OrderId
        else: lastTicketID = -1
        if debugConsistency: self.algo.MyDebug(" > _CL:{}, _SL:{}, _ASL:{}, LastOrderID:{}".format(str(_CL), str(_SL),str(_ASL),str(lastTicketID)))
        #Market Orders
        for tempTicket in self.algo.openMarketOrders:
            #It never would be ==Submitted
            if tempTicket.Symbol == orderSymbol and (tempTicket.Status == OrderStatus.Submitted or tempTicket.Status == OrderStatus.UpdateSubmitted or tempTicket.Status == 6 or tempTicket.Status == OrderStatus.New):
                _CM += 1
                _SM += tempTicket.Quantity - tempTicket.QuantityFilled
                _ASM += abs(tempTicket.Quantity - tempTicket.QuantityFilled)
                _marketTicket = tempTicket
        if _marketTicket != None: lastTicketID = _marketTicket.OrderId
        else: lastTicketID = -1
        if debugConsistency: self.algo.MyDebug(" > _CM:{}, _SM:{}, _ASM:{}, LastOrderID:{}".format(str(_CM), str(_SM),str(_ASM),str(lastTicketID)))
        #Positions
        for x in self.algo.Portfolio:
            if x.Key == orderSymbol and self.algo.Portfolio[x.Key].Quantity !=0:
                _CP += 1
                _SP += self.algo.Portfolio[x.Key].Quantity
                _ASP += abs(self.algo.Portfolio[x.Key].Quantity)
        if debugConsistency: self.algo.MyDebug(" > _CP:{}, _SP:{}, _ASP:{}".format(str(_CP), str(_SP),str(_ASP)))
        
        if _SS < 0:
            _UT = self.algo.mySymbolDict[orderSymbol].CL.useTragetsLong
        else:
            _UT = self.algo.mySymbolDict[orderSymbol].CL.useTragetsShort
        
        #If everything is empty nothing to do with orders. Release entry block and exit.
        if _CP == 0 and _CS == 0 and _CL == 0 and _CM == 0:
            self.algo.mySymbolDict[orderSymbol].entryEnabled = True
            #Release consistency block
            self.algo.mySymbolDict[orderSymbol].blockOrderCheck = False
            return
        
        '''CONSISTENCY CHECK'''
        #_0: IF CS=0 and CL=0 and SP+SM=0  Pending liquidation tipically premarket submission
        if _CS==0 and _CL==0 and (_SM + _SP)==0:
            _0 = True
            if debugConsistency: self.algo.MyDebug(" > _0:{}".format(str(_0)))
            
        #_SDIROK
        if (_SS<0 and _SP>0) or (_SS>0 and _SP<0) or (_SS==0 and _SP==0): _SDIROK = True
        if debugConsistency: self.algo.MyDebug(" > _SDIROK:{}".format(str(_SDIROK)))
        #_LDIROK
        if (_SL<0 and _SP>0) or (_SL>0 and _SP<0) or (_SL==0 and _SP==0): _LDIROK = True
        if debugConsistency: self.algo.MyDebug(" > _LDIROK:{}".format(str(_LDIROK)))
        #_I_A
        if _ASS+_UT*_ASS == _ASP+_ASL+_ASM: _I_A = True
        if debugConsistency: self.algo.MyDebug(" > _I_A:{}".format(str(_I_A)))
        #_I_B
        if _ASP!=0:
            _I_B = _SDIROK
        else: _I_B = True
        if debugConsistency: self.algo.MyDebug(" > _I_B:{}".format(str(_I_B)))
        #_I_C
        if _SL+_SP+_SM+((not _UT)*_SS) == 0: _I_C = True
        if debugConsistency: self.algo.MyDebug(" > _I_C:{}  _UT:{}".format(str(_I_C), str(_UT)))
        #_I
        if _I_A and _I_B  and _I_C: _I = True
        if debugConsistency: self.algo.MyDebug(" > _I:{}".format(str(_I)))
        
        #II_A
        if _CL-_UT == 0: _II_A = True
        #if debugConsistency: self.algo.MyDebug(" > _II_A:{}".format(str(_II_A)))
        #II_B
        if _UT:
            if _LDIROK and _CL==1: _II_B = True
        else:
            if _CL == 0 : _II_B = True
        #if debugConsistency: self.algo.MyDebug(" > _II_B:{}".format(str(_II_B)))
        #_II
        if _ASP!=0 and _II_A and _II_B and _CS==1 and _SDIROK: _II = True
        if debugConsistency: self.algo.MyDebug(" > _II:{}".format(str(_II)))
     
        '''ACTION'''
        #if 0. or I: DO NOTHING everything is ok
        if _0 or _I:
            if debugConsistency: self.algo.MyDebug(" > ACTION: 0. or I. Do nothing all orders are OK")
            #Release consistency block
            self.algo.mySymbolDict[orderSymbol].blockOrderCheck = False
            return
        
        '''if II: ADJUST STOP and LIMIT (if any) to P Size '''
        #Cancel and Submit New Orders if it is to be to changed so Quantity and FillQuantity field remain consistent
        #The conditions imply that there is no more that one S and no or one L order(s) to adjust so Tickets set previously are correct
        if _II: 
            updateOrderFields = UpdateOrderFields()
            if debugConsistency: self.algo.MyDebug(" > ACTION: II. Adjust S and L(if any) to P Size")
            '''ADJUST STOP'''
            if _adjustStopTicket != None:
                newOrderPrice = self.algo.Transactions.GetOrderById(_adjustStopTicket.OrderId).StopPrice
                newOrderQuantity = -_SP
                if adjustResubmit and _adjustStopTicket.Status != OrderStatus.CancelPending:
                    #Adjust Stop quantity to Position Quantity by cancelling and submitting new
                    updateOrderFields.Tag = _adjustStopTicket.Tag + "_ca"
                    _adjustStopTicket.Update(updateOrderFields)
                    _adjustStopTicket.Cancel()
                    orderTag = "SP_a: " + str(newOrderPrice)
                    newTicket = self.algo.StopMarketOrder(orderSymbol, newOrderQuantity, newOrderPrice, orderTag)
                    newOrderId = newTicket.OrderId
                    newOrderStatus = newTicket.Status
                    self.algo.openStopMarketOrders.append(newTicket)
                else:
                    updateOrderFields.Quantity = newOrderQuantity
                    updateOrderFields.QuantityFilled = 0
                    updateOrderFields.StopPrice = newOrderPrice
                    updateOrderFields.Tag = _adjustStopTicket.Tag + "_a"
                    _adjustStopTicket.Update(updateOrderFields)
                    newOrderId = _adjustStopTicket.OrderId
                    newOrderStatus = _adjustStopTicket.Status
                if self.debug: self.algo.MyDebug(" > ADJUST CONSISTENCY STOP Symbol: {}({}), OrderID:{}, Status:{}, New Quantity:{}, newOrderPrice:{}, _SP:{}, _SS:{}, _SL:{}, _SM:{}"
                   .format(str(orderSymbol),
                            str(strategyCode),
                            str(newOrderId),
                            str(newOrderStatus),
                            str(newOrderQuantity),
                            str(newOrderPrice),
                            str(_SP),
                            str(_SS),
                            str(_SL),
                            str(_SM)))                
            '''ADJUST LIMIT'''
            if _adjustLimitTicket != None:
                newOrderPrice = self.algo.Transactions.GetOrderById(_adjustLimitTicket.OrderId).LimitPrice
                newOrderQuantity = -_SP
                if adjustResubmit and _adjustLimitTicket.Status != OrderStatus.CancelPending:
                    #Adjust Limit quantity to Position Quantity by cancelling and submitting new
                    updateOrderFields.Tag = _adjustLimitTicket.Tag + "_ca"
                    _adjustLimitTicket.Update(updateOrderFields)
                    _adjustLimitTicket.Cancel()
                    orderTag = "LP_a: " + str(newOrderPrice)
                    newTicket = self.algo.LimitOrder(orderSymbol, newOrderQuantity, newOrderPrice, orderTag)
                    newOrderId = newTicket.OrderId
                    newOrderStatus = newTicket.Status
                    self.algo.openLimitOrders.append(newTicket)
                else:
                    updateOrderFields.Quantity = newOrderQuantity
                    updateOrderFields.QuantityFilled = 0
                    updateOrderFields.LimitPrice = newOrderPrice
                    updateOrderFields.Tag = _adjustLimitTicket.Tag + "_a"
                    _adjustLimitTicket.Update(updateOrderFields)
                    newOrderId = _adjustLimitTicket.OrderId
                    newOrderStatus = _adjustLimitTicket.Status
                if self.debug: self.algo.MyDebug(" > ADJUST CONSISTENCY LIMIT Symbol: {}({}), OrderID:{}, Status:{} New Quantity:{}, newOrderPrice:{}, _SP:{}, _SS:{}, _SL:{}, _SM:{}"
                   .format(str(orderSymbol),
                            str(strategyCode),
                            str(newOrderId),
                            str(newOrderStatus),
                            str(newOrderQuantity),
                            str(newOrderPrice),
                            str(_SP),
                            str(_SS),
                            str(_SL),
                            str(_SM)))
            #Release consistency block
            self.algo.mySymbolDict[orderSymbol].blockOrderCheck = False
            return
        
        '''not I. not II. CANCEL ALL ORDERS AND EXIT POSITION (if it gets here the conditions are true)'''
        if not(_0) and not(_I) and not(_II):
            if debugConsistency: self.algo.MyDebug(" > ACTION: not I. not II. Exit Position and Cancel All Orders")
            self.LiquidatePosition(orderSymbol, "CONSISTENCY", "CONSISTENCY")
            if debugConsistency: self.algo.MyDebug(" > CONSISTENCY LIQUIDATION FINISHED")
        #Release consistency block
        self.algo.mySymbolDict[orderSymbol].blockOrderCheck = False
        return
    
    '''REMOVE ORDERS
    '''
    def ClearOrderList(self):
        ordersRemoved = 0
        for tempTicket in self.algo.openStopMarketOrders:
            if tempTicket.Status != OrderStatus.Submitted and tempTicket.Status != 6 and tempTicket.Status != OrderStatus.New \
                            and tempTicket.Status != OrderStatus.PartiallyFilled and tempTicket.Status != OrderStatus.CancelPending \
                            and self.algo.Portfolio[tempTicket.Symbol].Quantity == 0:
                self.algo.openStopMarketOrders.remove(tempTicket)
                ordersRemoved += 1
        #Limit Orders
        for tempTicket in self.algo.openLimitOrders:
            if tempTicket.Status != OrderStatus.Submitted and tempTicket.Status != 6 and tempTicket.Status != OrderStatus.New \
                            and tempTicket.Status != OrderStatus.PartiallyFilled and tempTicket.Status != OrderStatus.CancelPending \
                            and self.algo.Portfolio[tempTicket.Symbol].Quantity == 0:
                self.algo.openLimitOrders.remove(tempTicket)
                ordersRemoved += 1
        #Market Orders
        for tempTicket in self.algo.openMarketOrders:
            if tempTicket.Status != OrderStatus.Submitted and tempTicket.Status != 6 and tempTicket.Status != OrderStatus.New \
                            and tempTicket.Status != OrderStatus.PartiallyFilled and tempTicket.Status != OrderStatus.CancelPending \
                            and self.algo.Portfolio[tempTicket.Symbol].Quantity == 0:
                self.algo.openMarketOrders.remove(tempTicket)
                ordersRemoved += 1
        if ordersRemoved > 0 and (self.algo.LiveMode or self.debug): self.algo.MyDebug(" Orders Removed: " + str(ordersRemoved))
        return