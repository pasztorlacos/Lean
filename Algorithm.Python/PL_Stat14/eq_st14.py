### <summary>
### 
### </summary>
from QuantConnect import SecurityType, Resolution
from QuantConnect.Indicators import RollingWindow, ExponentialMovingAverage, SimpleMovingAverage, IndicatorExtensions, AverageTrueRange, DonchianChannel, RelativeStrengthIndex, IndicatorDataPoint
from QuantConnect.Data.Consolidators import TradeBarConsolidator, QuoteBarConsolidator
from QuantConnect.Data.Market import IBaseDataBar, TradeBar

import re
from datetime import datetime, timedelta, date
from pandas import DataFrame
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ta import MyRelativePrice, MyZigZag, MyDCHState, MyMAState, MySupportResistance, MyPatterns, MyVolatility
from taB import MyBarStrength, MyBarRejection, MyBPA, MyGASF
from sim import MySIMPosition
import hp

class Eq_St14():
    file = __file__
    '''
    Strategy Implementation
    '''
    '''STRATEGY SETTINGS'''
    enabled = True
    manageOrderConsistency = True
    simulate = True
    saveFilesFramework = True
    saveFilesSim = saveFilesFramework
    strategyCodeOriginal = __name__     #This is used by ReadSettings as Signal tags strategyCode (self.CL.strategyCode = self.CL.strategyCode + "|" + aiKey)
    strategyCode = strategyCodeOriginal #used by order tags, debug
    isEquity = True
    customFillModel = 1
    customSlippageModel = 1
    customFeeModel = 0
    customBuyingPowerModel = 0
    #Resolution
    resolutionMinutes   = 60
    resolutionMinutes_2 = 24*60
    maxWarmUpPeriod   = 160
    maxWarmUpPeriod_2 = 160
    barPeriod   =  timedelta(minutes=resolutionMinutes)
    barPeriod_2 =  timedelta(minutes=resolutionMinutes_2)
    if isEquity:
        warmupcalendardays = max( round(7/5*maxWarmUpPeriod/(7*(60/min(resolutionMinutes,60*7) ))), round(7/5*maxWarmUpPeriod_2/(7*(60/min(resolutionMinutes_2,60*7) ))) )
        lastTradeHour = 15
    else:
        warmupcalendardays = max(round(7/5*maxWarmUpPeriod/(24*(60/resolutionMinutes))), round(7/5*maxWarmUpPeriod_2/(24*(60/resolutionMinutes_2))))
        lastTradeHour = 24
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
    riskperLongTrade  = 0.60/100 
    riskperShortTrade = 0.50/100 
    maxLongVaR  = 5.0/100 
    maxShortVaR = 5.0/100 
    maxTotalVaR = 10.0/100
    mainVaRr = None #it is set by the first instance
    #Entry
    enableLong  = False
    enableShort = False
    liquidateLong  = False
    liquidateShort = False
    enableFlipLong  = True #Checked in self.algo.myPositionManager.EnterPosition_2
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
    stopPlacerLong  =   [("minStop",), ("dch", 'dch01'), ("rw",4+1)][2]     # Tuple (Type, arg2, arg2...) type else:('rw',10): ("minStop",'na'). ("dch",'dch_attributeName'), ("rw",n)
    stopPlacerShort =   [("minStop",), ("dch", 'dch01'), ("rw",  3)][2] 
    targetPlacerLong  = [(None,'na'), ("minPayOff",'na'), ("dch", 'dch2'), ("rw",35+1)][3]   # Tuple (Type, arg2, arg2...) type else:: minPayOff. If targetPlacer[0]!=None then customPositionSettings['targetPlacer']==None must raise an exeption during installation otherwise _UT would become inconsistent
    targetPlacerShort = [(None,'na'), ("minPayOff",'na'), ("dch", 'dch2'), ("rw",  22)][3]   # TWS Sync is not ready yet to handle useTragets for Foreign Symbols   
    minPayOffLong  = 2.00
    minPayOffShort = 1.50
    scratchTradeLong  = True 
    scratchTradeShort = True 
    stopTrailerLong  =   [(None,'na'), ("dch", 'dch01'), ("rw",8) ][0]      # Tuple (Type, arg2, arg2...): (None,'na'), ("dch",'dch_attributeName'), ("rw",n)
    stopTrailerShort =   [(None,'na'), ("dch", 'dch01'), ("rw",8) ][0] 
    targetTrailerLong  = [(None,'na'), ("dch", 'dch2'),  ("rw",40)][0]      # Tuple (Type, arg2, arg2...): (None,'na'), ("dch",'dch_attributeName'), ("rw",n)
    targetTrailerShort = [(None,'na'), ("dch", 'dch2'),  ("rw",40)][0] 
    stopATRLong  = 0.5
    stopATRShort = 0.5
    minEntryStopATRLong  = 2.0    
    minEntryStopATRShort = 1.0    
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
    
    myTickers =["A", "AA"]
    #Lean noETFs SLICE_n (41)
    myTickers = ["A", "AA", "AABA", "AAL", "AAXN", "ABBV", "ACIA", "ADM", "ADT", "AIG", "AKAM", "AKS", "ALLY", "ALTR", "AMAT", "AMC", "AMCX", "AMD", "AMGN", "AMZN", "AN", "ANF", "ANTM", "AOBC", "APO", "APRN", "ARLO", "ATUS", "ATVI", "AUY", "AVGO", "AVTR", "AWK", "BABA", "BAC", "BAH", "BB", "BBBY", "BBY", "BIDU", "BJ"]    
    #SP100 (100)
    #myTickers = ["AAPL", "ABBV", "ABT", "ACN", "ADBE", "AGN", "AIG", "ALL", "AMGN", "AMZN", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CELG", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CSCO", "CVS", "CVX", "DD", "DHR", "DIS", "DUK", "EMR", "EXC", "F", "FB", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTN", "SBUX", "SLB", "SO", "SPG", "T", "TGT", "TXN", "UNH", "UNP", "UPS", "USB", "UTX", "V", "VZ", "WBA", "WFC", "WMT", "XOM"]
    #SP100 S_1
    #mySymbols = ["AAPL", "ABBV", "ABT", "ACN", "ADBE", "AGN", "AIG", "ALL", "AMGN", "AMZN", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CELG", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CSCO", "CVS", "CVX", "DD", "DHR"]
  
    #ES&NQ (181)
    #myTickers = ["AAPL", "ABBV", "ABT", "ACN", "ADBE", "AGN", "AIG", "ALL", "AMGN", "AMZN", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CELG", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CSCO", "CVS", "CVX", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FB", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTN", "SBUX", "SLB", "SO", "SPG", "T", "TGT", "TXN", "UNH", "UNP", "UPS", "USB", "UTX", "V", "VZ", "WBA", "WFC", "WMT", "XOM", "ATVI", "AMD", "ALXN", "ALGN", "AAL", "ADI", "AMAT", "ASML", "ADSK", "ADP", "BIDU", "BMRN", "AVGO", "CDNS", "CERN", "CHKP", "CTAS", "CTXS", "CTSH", "CSX", "CTRP", "DLTR", "EBAY", "EA", "EXPE", "FAST", "FISV", "FOX", "FOXA", "HAS", "HSIC", "IDXX", "ILMN", "INCY", "INTU", "ISRG", "JBHT", "JD", "KLAC", "LRCX", "LBTYA", "LBTYK", "LULU", "MAR", "MXIM", "MELI", "MCHP", "MU", "MNST", "MYL", "NTAP", "NTES", "NXPI", "ORLY", "PCAR", "PAYX", "REGN", "ROST", "SIRI", "SWKS", "SYMC", "SNPS", "TMUS", "TTWO", "TSLA", "ULTA", "UAL", "VRSN", "VRSK", "VRTX", "WDAY", "WDC", "WLTW", "WYNN", "XEL", "XLNX", "STX", "TSLA", "VRSK", "WYNN", "XLNX"]
    #ES&NQ S_1 (90)
    #myTickers = ["BK", "CL", "SPG", "PFE", "MSFT", "NEE", "AGN", "INTC", "NFLX", "HD", "PYPL", "USB", "MDLZ", "OXY", "MS", "C", "GD", "BRK.B", "ALL", "DUK", "ABT", "TGT", "DD", "QCOM", "BMY", "LLY", "CHTR", "WBA", "BA", "PM", "LOW", "GILD", "GOOG", "MMM", "COF", "SO", "JNJ", "DIS", "WMT", "CVX", "UNH", "GS", "CELG", "MCD", "BIIB", "CAT", "COP", "T", "AMZN", "ABBV", "HON", "RTN", "VZ", "PG", "SLB", "KMI", "IBM", "BLK", "ACN", "GE", "CVS", "UPS", "MDT", "MRK", "DHR", "BKNG", "NVDA", "V", "AXP", "XOM", "EXC", "JPM", "ORCL", "KHC", "KO", "AIG", "CSCO", "UNP", "TXN", "CMCSA", "AAPL", "LMT", "PEP", "UTX", "F", "SBUX", "COST", "WFC", "FDX", "AMGN"]
    #Ecluding DD dor Backtest: 2019-05-24T14:00:00Z Open order cancelled on symbol changed event 
    #myTickers = ["BK", "CL", "SPG", "PFE", "MSFT", "NEE", "AGN", "INTC", "NFLX", "HD", "PYPL", "USB", "MDLZ", "OXY", "MS", "C", "GD", "BRK.B", "ALL", "DUK", "ABT", "TGT", "QCOM", "BMY", "LLY", "CHTR", "WBA", "BA", "PM", "LOW", "GILD", "GOOG", "MMM", "COF", "SO", "JNJ", "DIS", "WMT", "CVX", "UNH", "GS", "CELG", "MCD", "BIIB", "CAT", "COP", "T", "AMZN", "ABBV", "HON", "RTN", "VZ", "PG", "SLB", "KMI", "IBM", "BLK", "ACN", "GE", "CVS", "UPS", "MDT", "MRK", "DHR", "BKNG", "NVDA", "V", "AXP", "XOM", "EXC", "JPM", "ORCL", "KHC", "KO", "AIG", "CSCO", "UNP", "TXN", "CMCSA", "AAPL", "LMT", "PEP", "UTX", "F", "SBUX", "COST", "WFC", "FDX", "AMGN"]
    #ES&NQ S_2 (90)
    #myTickers = ["NKE", "MET", "ADBE", "GM", "MA", "MO", "BAC", "EMR", "FB", "TSLA", "EXPE", "AAL", "CTAS", "FISV", "UAL", "BIDU", "CHKP", "WDC", "SIRI", "ULTA", "EA", "FAST", "MAR", "ASML", "ADSK", "PAYX", "ATVI", "NTES", "MXIM", "VRSN", "AMD", "FOXA", "TTWO", "CSX", "VRSK", "VRTX", "BMRN", "WYNN", "KLAC", "ALXN", "JBHT", "WYNN", "XLNX", "INCY", "CERN", "HSIC", "AMAT", "VRSK", "SWKS", "ALGN", "XLNX", "INTU", "LBTYK", "STX", "MNST", "WDAY", "REGN", "ROST", "IDXX", "ILMN", "MELI", "ADP", "AVGO", "PCAR", "FOX", "JD", "GOOGL", "MYL", "ISRG", "ADI", "LBTYA", "DLTR", "CTXS", "EBAY", "MCHP", "XEL", "NXPI", "CDNS", "SNPS", "ORLY", "HAS", "LRCX", "LULU", "WLTW", "MU", "TMUS", "NTAP", "CTSH", "SYMC", "TSLA", "CTRP"]
    
    #SP500-ES&NQ
    #myTickers = ["CRM", "TMO", "LIN", "AMT", "FIS", "CME", "CB", "BDX", "SYK", "TJX", "ANTM", "SPGI", "NOC", "D", "CCI", "ZTS", "BSX", "PNC", "CI", "PLD", "ECL", "ICE", "MMC", "DE", "APD", "KMB", "LHX", "EQIX", "WM", "NSC", "AON", "EW", "SCHW", "EL", "AEP", "ITW", "PGR", "EOG", "SHW", "BAX", "PSX", "DG", "PSA", "SRE", "TRV", "ROP", "HUM", "AFL", "WELL", "BBT", "YUM", "MCO", "SYY", "DAL", "STZ", "JCI", "ETN", "NEM", "PRU", "MPC", "HCA", "GIS", "VLO", "EQR", "TEL", "TWTR", "PEG", "WEC", "MSI", "SBAC", "AVB", "OKE", "IR", "ED", "WMB", "ZBH", "AZO", "HPQ", "VTR", "VFC", "TSN", "STI", "HLT", "BLL", "APH", "MCK", "TROW", "PPG", "DFS", "GPN", "ES", "TDG", "FLT", "LUV", "DLR", "EIX", "IQV", "DTE", "INFO", "O", "FE", "AWK", "A", "CTVA", "HSY", "TSS", "GLW", "APTV", "CMI", "ETR", "PPL", "HIG", "PH", "ADM", "ESS", "FTV", "PXD", "LYB", "SYF", "CMG", "CLX", "SWK", "MTB", "MKC", "MSCI", "RMD", "BXP", "CHD", "AME", "WY", "RSG", "STT", "FITB", "KR", "CNC", "NTRS", "AEE", "VMC", "HPE", "KEYS", "ROK", "CMS", "RCL", "EFX", "ANSS", "CCL", "AMP", "CINF", "TFX", "ARE", "OMC", "HCP", "DHI", "LH", "KEY", "AJG", "MTD", "COO", "CBRE", "HAL", "EVRG", "AMCR", "MLM", "HES", "K", "EXR", "CFG", "IP", "CPRT", "FANG", "BR", "CBS", "NUE", "DRI", "FRC", "MKTX", "BBY", "LEN", "WAT", "RF", "AKAM", "CXO", "MAA", "MGM", "CE", "HBAN", "CAG", "CNP", "KMX", "PFG", "XYL", "DGX", "WCG", "UDR", "DOV", "CBOE", "FCX", "HOLX", "GPC", "L", "ATO", "ABC", "KSU", "CAH", "TSCO", "LDOS", "IEX", "LNT", "EXPD", "GWW", "XRAY", "MAS", "ANET", "UHS", "DRE", "HST", "IT", "SJM", "NDAQ", "HRL", "CHRW", "FTNT", "FMC", "BHGE", "JKHY", "IFF", "NBL", "WAB", "NI", "CTL", "NCLH", "REG", "LNC", "PNW", "CF", "TXT", "VNO", "FTI", "ARNC", "LW", "ETFC", "JEC", "SIVB", "MRO", "AES", "RJF", "AAP", "AVY", "GRMN", "BF.B", "VAR", "RE", "FRT", "TAP", "NRG", "WU", "CMA", "PKG", "DISCK", "DVN", "TIF", "PKI", "IRM", "GL", "ALLE", "EMN", "VIAB", "DXC", "URI", "WRK", "PHM", "ABMD", "WHR", "HII", "QRVO", "CPB", "SNA", "APA", "LKQ", "JNPR", "DISH", "BEN", "IPG", "NOV", "FFIV", "KIM", "KSS", "AIV", "AIZ", "ZION", "ALK", "NLSN", "COG", "MHK", "FBHS", "HFC", "DVA", "SLG", "BWA", "FLIR", "AOS", "ALB", "RHI", "MOS", "NWL", "IVZ", "SEE", "TPR", "PRGO", "PVH", "XRX", "PNR", "PBCT", "ADS", "UNM", "FLS", "NWSA", "HOG", "HBI", "HRB", "PWR", "LEG", "ROL", "JEF", "RL", "M", "DISCA", "IPGP", "XEC", "HP", "CPRI", "AMG", "TRIP", "LB", "GPS", "UAA", "UA", "JWN", "MAC", "NKTR", "COTY", "NWS"]
    #SP500-ES&NQ S_1 (115)
    #myTickers = ["XRAY", "CPB", "L", "TDG", "O", "FBHS", "RHI", "MAC", "CMS", "ANTM", "EQR", "KSU", "BR", "TFX", "CHD", "BBY", "XYL", "SJM", "HRL", "KEYS", "FTNT", "DAL", "A", "GRMN", "BXP", "GPC", "ROP", "WAB", "LIN", "BDX", "AMT", "SHW", "TSN", "NBL", "HUM", "ETN", "WRK", "FCX", "ETFC", "ES", "AEE", "HFC", "AME", "EXR", "EMN", "NUE", "LW", "CB", "LNT", "PPG", "NTRS", "HCA", "ARE", "HES", "DHI", "LEN", "MCO", "MKC", "KMB", "ZION", "EW", "MPC", "ETR", "FRC", "MTD", "EFX", "CLX", "UNM", "RE", "CPRI", "TMO", "DISH", "DRE", "KEY", "DVN", "PSA", "WCG", "MOS", "UHS", "PKI", "FANG", "URI", "CF", "AWK", "EIX", "MGM", "ICE", "LEG", "KIM", "CTL", "QRVO", "ALLE", "DE", "ZBH", "ED", "APA", "CE", "GL", "CBS", "DVA", "GPN", "EL", "ALK", "MSCI", "TXT", "PPL", "RMD", "CMG", "JWN", "COTY", "PNR", "DG", "KSS", "BF.B", "CNC"]
    #SP500-ES&NQ S_2 (115)
    #myTickers = ["PKG", "SLG", "PFG", "AVY", "DTE", "FITB", "SYY", "MCK", "AFL", "AAP", "SNA", "HCP", "EXPD", "LNC", "CHRW", "OKE", "STT", "LUV", "IPGP", "TSS", "HOG", "VTR", "REG", "LYB", "VAR", "NCLH", "IP", "SBAC", "DXC", "TROW", "HP", "FLT", "HBAN", "PNW", "MLM", "NOV", "WU", "JKHY", "AIV", "GWW", "MTB", "HII", "CNP", "PVH", "AEP", "JEC", "PNC", "TRV", "MMC", "FMC", "RJF", "RCL", "HPQ", "HOLX", "IPG", "DGX", "ITW", "NDAQ", "KMX", "WHR", "PLD", "NWL", "TRIP", "NWSA", "ADS", "ROL", "APD", "COO", "NSC", "HIG", "ADM", "HBI", "IR", "EVRG", "MHK", "SRE", "JEF", "MAA", "CMI", "STI", "PBCT", "AON", "AJG", "LKQ", "STZ", "BSX", "EQIX", "ANSS", "TIF", "FIS", "SCHW", "CAG", "UA", "YUM", "ESS", "CXO", "BLL", "NEM", "SYF", "NKTR", "BEN", "SEE", "HAL", "PEG", "MRO", "AKAM", "VIAB", "RSG", "PRU", "TEL", "DLR", "CINF", "NI", "FRT", "VFC"]
    #SP500-ES&NQ S_3 (107)
    #myTickers = ["CRM", "MSI", "INFO", "HSY", "OMC", "VMC", "BAX", "TJX", "EOG", "AMP", "MAS", "LH", "BHGE", "FLS", "K", "IEX", "ATO", "CI", "PXD", "NLSN", "ALB", "COG", "DOW", "CAH", "WM", "WMB", "AIZ", "HRB", "SPGI", "ABMD", "JCI", "AOS", "TAP", "JNPR", "IVZ", "ZTS", "WY", "LB", "AES", "PGR", "PHM", "FE", "VLO", "FLIR", "SIVB", "CMA", "WELL", "DISCK", "RF", "CBOE", "ROK", "CPRT", "PH", "GIS", "PWR", "HPE", "GLW", "BBT", "IFF", "IRM", "CCI", "DOV", "HLT", "NRG", "NWS", "UAA", "HST", "TPR", "XEC", "GPS", "CFG", "D", "PRGO", "WAT", "DISCA", "SYK", "CCL", "M", "FTI", "AZO", "IT", "LHX", "ANET", "TWTR", "ABC", "XRX", "FFIV", "NOC", "MKTX", "AVB", "AMG", "ECL", "KR", "UDR", "CME", "DRI", "WEC", "PSX", "SWK", "TSCO", "ARNC", "DFS", "VNO", "FTV", "APH", "LDOS", "APTV", "BWA"]

    #Simulation Signals. Key is the Signal. Use simKey[0:3]=="ALL" to bypass signal filtering
    # "ALL_L":     {"direction":+1, "Enabled": True, "disableBars":8, "currentDisabledBars":0}
    # "S_BPA":     {"direction":-1, "Enabled": True, "disableBars":8, "currentDisabledBars":0}
    simALL=False
    simDict = {
       "ALL_L":     {"direction":+1,    "Enabled": simALL,      "disableBars":len(myTickers),   "currentDisabledBars":0},   "ALL_S":    {"direction":-1,  "Enabled": simALL,      "disableBars":len(myTickers),   "currentDisabledBars":0},
       "L_Str_":    {"direction":+1,    "Enabled": not simALL,  "disableBars":8,                "currentDisabledBars":0},   "S_Str_":   {"direction":-1,  "Enabled": not simALL,  "disableBars":8,                "currentDisabledBars":0}, 
       "L_Rej_":    {"direction":+1,    "Enabled": not simALL,  "disableBars":8,                "currentDisabledBars":0},   "S_Rej_":   {"direction":-1,  "Enabled": not simALL,  "disableBars":8,                "currentDisabledBars":0},
       "L_BPA_10":  {"direction":+1,    "Enabled": not simALL,  "disableBars":8,                "currentDisabledBars":0},   "S_BPA_10": {"direction":-1,  "Enabled": not simALL,  "disableBars":8,                "currentDisabledBars":0}, 
       "L_BPA_15":  {"direction":+1,    "Enabled": not simALL,  "disableBars":8,                "currentDisabledBars":0},   "S_BPA_15": {"direction":-1,  "Enabled": not simALL,  "disableBars":8,                "currentDisabledBars":0}, 
       "DB_":       {"direction":+1,    "Enabled": not simALL,  "disableBars":8,                "currentDisabledBars":0},   "DT_":      {"direction":-1,  "Enabled": not simALL,  "disableBars":8,                "currentDisabledBars":0}, 
       "TB_":       {"direction":+1,    "Enabled": not simALL,  "disableBars":8,                "currentDisabledBars":0},   "TT_":      {"direction":-1,  "Enabled": not simALL,  "disableBars":8,                "currentDisabledBars":0},
       "IHS_":      {"direction":+1,    "Enabled": not simALL,  "disableBars":8,                "currentDisabledBars":0},   "HS_":      {"direction":-1,  "Enabled": not simALL,  "disableBars":8,                "currentDisabledBars":0}}
    
    #Exitsignals: Key is the Signal item is the Direction (+1:Shorts Closed, -1:Longs Closed). 
    exitSignalDict = {
       "L_Str_": 0, 
       "S_Str_": 0, 
       "L_Rej_": 0,
       "S_Rej_": 0}
    
    #AI ----
    loadAI = True
    preprocDict = {}
    aiDict = {}

 #PREPROCESSORS   -------------------------------------------
    preprocDict ["CAE_1D_100-16_sm"] = {
        "enabled": True,
        "Type": ["PT"][0],
        "modelClass": "CNN_AE_1",
        "modelParams" : {"cnn_Dim": [1,2][0],
                       "inH": 1,
                       "inW": 100,
                       "inChannels": 1,
                       "cnnChannels1": 64,
                       "cnnChannels2": 128,
                       "outputs": 16,
                       "useConvTranspose": False,
                       "hiddenSize": [None, 64][0],
                       "hiddenSizeDecode": [None, 64][0],
                       "dropoutRate": 0.10},
        "modelURL": "https://www.dropbox.com/s/ram9c51d5m1cvqn/CAE_1D_100-16_sm_Model_20200711-16_41_c.txt?dl=1",
        "model": None}
    preprocDict ["CAE_1D_50-8_sm"] = {
        "enabled": True,
        "Type": ["PT"][0],
        "modelClass": "CNN_AE_1",
        "modelParams" : {"cnn_Dim": [1,2][0],
                       "inH": 1,
                       "inW": 50,
                       "inChannels": 1,
                       "cnnChannels1": 64,
                       "cnnChannels2": 128,
                       "outputs": 8,
                       "useConvTranspose": False,
                       "hiddenSize": [None, 64][0],
                       "hiddenSizeDecode": [None, 64][0],
                       "dropoutRate": 0.10},
        "modelURL": "https://www.dropbox.com/s/u80fqif5h3zfpyk/CAE_1D_50-8_sm_Model_20200711-18_14_c.txt?dl=1",
        "model": None}

#MODELS   -------------------------------------------   
    aiDict["L_BPA-A"] = {
        "enabled": False if enableLong else False,
        "signalRegex": "L_BPA_1",
        "direction": 1,
        "Type" : ["SK", "LGB", "PT", "PT_CNN"][2],
        "firstTradeHour": 0,
        "lastTradeHour": lastTradeHour,
        "riskMultiple": 0.50,
        'customPositionSettings': {},
        "stopPlacer": None,
        "targetPlacer": None,
        "minPayOff": None,
        "modelClass": "NN_2",
        "featureCount": 136,
        "modelParams" : {"features":    136,
                        "inputSize":    136,
                        "hiddenSize1":  272,
                        "outFeed":      '_h1',
                        "outputs":      2 , 
                        "softmaxout":   False},       
        "modelURL": "https://www.dropbox.com/s/oxkjlaymbeu5qm9/NNpt_L_BPA_2005_FeatAll_5MM_Model_20200722-14_2954_c.txt?dl=1",
        "model": None,
        "usePCA": False,
        "pcaURL": None,
        "pca": None,
        "rawFeatures": "rawFeatures1",
        "threshold": None,
        "features": None,
        "featureFilter": "Feat",
        "customColumnFilter": [],
        "dfFilterPassed": True,
        "signal": False}

    aiDict["L_BPA-B"] = {
        "enabled": False if enableLong else False,
        "signalRegex": "L_BPA_1",
        "direction": 1,
        "Type" : ["SK", "LGB", "PT", "PT_CNN"][2],
        "firstTradeHour": 0,
        "lastTradeHour": lastTradeHour,
        "riskMultiple": 0.50,
        'customPositionSettings': {},
        "stopPlacer": None,
        "targetPlacer": None,
        "minPayOff": None,
        "modelClass": "NN_2",
        "featureCount": 136,
        "modelParams" : {"features":    136,
                        "inputSize":    136,
                        "hiddenSize1":  136,
                        "outFeed":      '_h1',
                        "outputs":      2 , 
                        "softmaxout":   False},       
        "modelURL": "https://www.dropbox.com/s/hd06fjj9ok7062q/NNpt_L_BPA_2005_FeatAll_5MM_Model_20200721-23_3829_c.txt?dl=1",
        "model": None,
        "usePCA": False,
        "pcaURL": None,
        "pca": None,
        "rawFeatures": "rawFeatures1",
        "threshold": None,
        "features": None,
        "featureFilter": "Feat",
        "customColumnFilter": [],
        "dfFilterPassed": True,
        "signal": False}

    # {"targetPlacer": ("rw", 15)}
    aiDict["S_Rej-A"] = {
        "enabled": False if enableShort else False,
        "signalRegex": "S_Rej",
        "direction": -1,
        "Type" : ["SK", "LGB", "PT", "PT_CNN"][2],
        "firstTradeHour": 0,
        "lastTradeHour": lastTradeHour,
        "riskMultiple": 1.00,
        'customPositionSettings': {},
        "modelClass": "NN_2",
        "featureCount": 136,
        "modelParams" : {"features":    136,
                        "inputSize":    136,
                        "hiddenSize1":  272,
                        "outFeed":      '_h1',
                        "outputs":      2 , 
                        "softmaxout":   False,
                        "dropoutRate_in": 0.0,
                        "dropoutRate_h":  0.0,
                        "bn_momentum": 0.1},       
        "modelURL": "https://www.dropbox.com/s/bwh14h4p4isc3eb/NNpt_S_Rej_2005_FeatAll_5MM_Model_20200722-16_5447_c.txt?dl=1",
        "model": None,
        "usePCA": False,
        "pcaURL": None,
        "pca": None,
        "rawFeatures": "rawFeatures1",
        "threshold": None,
        "features": None,
        "featureFilter": "Feat",
        "customColumnFilter": [],
        "dfFilterPassed": True,
        "signal": False}

    # {"targetPlacer": ("rw", 15)}
    aiDict["S_Rej-B"] = {
        "enabled": False if enableShort else False,
        "signalRegex": "S_Rej",
        "direction": -1,
        "Type" : ["SK", "LGB", "PT", "PT_CNN"][2],
        "firstTradeHour": 0,
        "lastTradeHour": lastTradeHour,
        "riskMultiple": 1.00,
        'customPositionSettings': {"targetPlacer": ("rw", 15)},
        "modelClass": "NN_2",
        "featureCount": 136,
        "modelParams" : {"features":    136,
                        "inputSize":    136,
                        "hiddenSize1":  272,
                        "outFeed":      '_h1',
                        "outputs":      2 , 
                        "softmaxout":   False,
                        "dropoutRate_in": 0.0,
                        "dropoutRate_h":  0.0,
                        "bn_momentum": 0.1},       
        "modelURL": "https://www.dropbox.com/s/46vjzzqw8laideh/NNpt_S_Rej_2005_FeatAll_5MM_Model_20200722-21_5850_c.txt?dl=1",
        "model": None,
        "usePCA": False,
        "pcaURL": None,
        "pca": None,
        "rawFeatures": "rawFeatures1",
        "threshold": None,
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
        '''Consolidator(s)'''
        self.consolidator = TradeBarConsolidator(self.CL.barPeriod) if symbol.SecurityType == SecurityType.Equity else QuoteBarConsolidator(self.CL.barPeriod)
        self.consolidator.DataConsolidated += self.OnDataConsolidated
        self.algo.SubscriptionManager.AddConsolidator(symbol, self.consolidator)
        #Higher Timeframe
        self.consolidator_2 = TradeBarConsolidator(self.CL.barPeriod_2) if symbol.SecurityType == SecurityType.Equity else QuoteBarConsolidator(self.CL.barPeriod_2)
        self.consolidator_2.DataConsolidated += self.OnDataConsolidated_2
        self.algo.SubscriptionManager.AddConsolidator(symbol, self.consolidator_2)
        
        '''Symbol Data''' 
        self.bars_rw =      RollingWindow[IBaseDataBar](100)
        self.bars_rw_2 =    RollingWindow[IBaseDataBar](100)
        self.barTimeDifference = timedelta(0)
        self.posEnabled = True
        #self.entryEnabled = False
        self.blockOrderCheck = False
        self.fillReleaseTime = self.algo.Time - timedelta(hours=20)
        self.fromTWS = False
        self.stateUpdateList = []
        '''Indicators'''
        self.atr1 = AverageTrueRange(15)
        self.algo.RegisterIndicator(self.symbol, self.atr1, self.consolidator)
        self.atr2 = AverageTrueRange(50)
        self.algo.RegisterIndicator(self.symbol, self.atr2, self.consolidator)
        
        self.dch01 = DonchianChannel(4)
        self.algo.RegisterIndicator(self.symbol, self.dch01, self.consolidator)
        self.dch2 = DonchianChannel(35)
        self.algo.RegisterIndicator(self.symbol, self.dch2, self.consolidator)
        
        self.vol = MyVolatility(self, self.algo, self.symbol, name='vol', period=150, atr=self.atr1)
        self.algo.RegisterIndicator(self.symbol, self.vol, self.consolidator)
        
        self.gasf1 = MyGASF(self, self.algo, self.symbol, name='gasf1', period=150, atr=self.atr1, benchmarkTicker=None)
        self.algo.RegisterIndicator(self.symbol, self.gasf1, self.consolidator)

        '''Indicators Higher Timeframe'''
        self.atr1_2 = AverageTrueRange(5)
        self.algo.RegisterIndicator(self.symbol, self.atr1_2, self.consolidator_2)

        self.gasf1_2 = MyGASF(self, self.algo, self.symbol, name='gasf1_2', period=150, atr=self.atr1_2, benchmarkTicker=None)
        self.algo.RegisterIndicator(self.symbol, self.gasf1_2, self.consolidator_2)

        '''Reagy Check Indicator'''
        self.readyIndicator = self.gasf1_2
        
        '''Symbol State and Features'''
        #GASF indicator implements FeatureExtraction method

        '''Signals'''
        self.barStrength1 = MyBarStrength(self, self.algo, self.symbol, name='1', period=10, atr=self.atr1, lookbackLong=2, lookbackShort=2, \
                priceActionMinATRLong=1.5, priceActionMaxATRLong=2.5, priceActionMinATRShort=1.5, priceActionMaxATRShort=2.5, referenceTypeLong='Close', referenceTypeShort='Close')
        self.algo.RegisterIndicator(self.symbol, self.barStrength1, self.consolidator)
        
        self.barRejection1 = MyBarRejection(self, self.algo, self.symbol, name='1', period=10, atr=self.atr1, lookbackLong=3, lookbackShort=3, \
               rejectionPriceTravelLong=1.75, rejectionPriceTravelShort=1.75, rejectionPriceRangeLong=0.40, rejectionPriceRangeShort=0.40, referenceTypeLong='Close', referenceTypeShort='Close')
        self.algo.RegisterIndicator(self.symbol, self.barRejection1, self.consolidator)

        self.bpa1 = MyBPA(self, self.algo, self.symbol, name='10', period=10, atr=self.atr1, lookbackLong=1, lookbackShort=1, \
                    minBarAtrLong=1.0, minBarAtrShort=1.0, referenceTypeLong='Close', referenceTypeShort='Close', gapATRLimitLong=1.0, gapATRLimitShort=1.0, needBPAinTrend=False, atrTrendChange=3.0)
        self.algo.RegisterIndicator(self.symbol, self.bpa1, self.consolidator)

        self.bpa2 = MyBPA(self, self.algo, self.symbol, name='15', period=10, atr=self.atr1, lookbackLong=1, lookbackShort=1, \
                    minBarAtrLong=1.5, minBarAtrShort=1.5, referenceTypeLong='Close', referenceTypeShort='Close', gapATRLimitLong=1.0, gapATRLimitShort=1.0, needBPAinTrend=False, atrTrendChange=3.0)
        self.algo.RegisterIndicator(self.symbol, self.bpa2, self.consolidator)

        '''Signals and Events string Container'''
        self.signals = ''
        self.events = ''

        '''Disable Bars'''
        self.longDisabledBars = 0
        self.shortDisabledBars = 0
        self.longSimDisabledBars = 0
        self.shortSimDisabledBars = 0
        
        '''Set up Preprocessors'''
        for preprKey, preprObj in self.CL.preprocDict.items():
            if self.CL.loadAI and preprObj["enabled"] and preprObj["model"]==None:
                #Loading MODEL
                if preprObj["Type"] == "PT":
                    preprObj["model"] = globals()[preprObj["modelClass"]](**preprObj["modelParams"]).to('cpu')
                    preprObj["model"] = hp.MyModelLoader.LoadModelTorch(self, preprObj["modelURL"], existingmodel=preprObj["model"])
                    self.algo.Debug(f' PT(Prep) MODEL ({preprKey}/{self.CL.strategyCode}) LOADED from url: {preprObj["modelURL"]}')

        '''Set up AI models'''
        self.signalDisabledBars = {}
        for aiKey, aiObj in self.CL.aiDict.items():
            if self.CL.loadAI and aiObj["enabled"] and aiObj["model"]==None:
                #Checking customPositionSettings['targetPlacer'] Consistency
                defaultTargetPlecer = self.CL.targetPlacerLong if aiObj["direction"]==1 else self.CL.targetPlacerShort
                if (defaultTargetPlecer[0]==None and ('customPositionSettings' in aiObj and 'targetPlacer' in aiObj['customPositionSettings'] and aiObj['customPositionSettings']['targetPlacer'][0]!=None)) or \
                        (defaultTargetPlecer[0]!=None and ('customPositionSettings' in aiObj and 'targetPlacer' in aiObj['customPositionSettings'] and aiObj['customPositionSettings']['targetPlacer'][0]==None)):
                    raise Exception(f"AI customPositionSettings _UL Mismatch! defaultTargetPlecer: {defaultTargetPlecer[0]} and customPositionSettings['targetPlacer']: {aiObj['customPositionSettings']['targetPlacer'][0]}")    
                #Loading PCA Preprocessor if set
                if aiObj["usePCA"]: 
                    aiObj["pca"] = hp.MyModelLoader.LoadModelPickled(self, aiObj["pcaURL"])
                    self.algo.Debug(f' PCA ({aiKey}/{self.CL.strategyCode}) LOADED from url: {aiObj["pcaURL"]}')
                #Loading MODEL
                if aiObj["Type"] == "PT" or aiObj["Type"] == "PT_CNN":
                    aiObj["model"] = globals()[aiObj["modelClass"]](**aiObj["modelParams"]).to('cpu')
                    aiObj["model"] = hp.MyModelLoader.LoadModelTorch(self, aiObj["modelURL"], existingmodel=aiObj["model"])
                    self.algo.Debug(f' PT MODEL ({aiKey}/{self.CL.strategyCode}) LOADED from url: {aiObj["modelURL"]}')
                elif aiObj["Type"] == "SK":
                    aiObj["model"] = hp.MyModelLoader.LoadModelPickled(self, aiObj["modelURL"])
                    self.algo.Debug(f' SKL MODEL ({aiKey}/{self.CL.strategyCode}) LOADED from url: {aiObj["modelURL"]}')
                elif aiObj["Type"] == "LGB":
                    #aiObj["model"] = hp.MyModelLoader.LoadModelLGBtxt(self, aiObj["modelURL"])
                    aiObj["model"] = hp.MyModelLoader.LoadModelLGB(self, aiObj["modelURL"])
                    self.algo.Debug(f' LGB MODEL ({aiKey}/{self.CL.strategyCode}) LOADED from url: {aiObj["modelURL"]}')
            self.signalDisabledBars[aiKey] = 0
        
        #Initialize signalDisabledBars
        for simKey, simObj in self.CL.simDict.items():
            simObj["currentDisabledBars"] = random.randint(1, 20) #random.randint(1, len(self.CL.myTickers))

        #SET FILES TO BE SAVED AT THE END OF SIMULATION
        if self.CL.simulate and len(MySIMPosition.saveFiles)==0:
            if self.CL.saveFilesFramework:        
                #Framework specific filelist to be saved
                MySIMPosition.saveFiles.append([self.algo.myPositionManager.CL.file, '_pm'])
                MySIMPosition.saveFiles.append([self.algo.myPositionManagerB.CL.file, '_pmB'])
                MySIMPosition.saveFiles.append([self.algo.myVaR.CL.file, '_var'])
                MySIMPosition.saveFiles.append([self.algo.myHelpers.CL.file, '_hp'])
            if self.CL.saveFilesSim: 
                #Simulation Specific Files    
                MySIMPosition.saveFiles.append([self.algo.__class__.file, '_main'])
                MySIMPosition.saveFiles.append([self.CL.file, f'_{self.CL.strategyCodeOriginal}'])
                MySIMPosition.saveFiles.append([MySIMPosition.file, '_sim'])
                MySIMPosition.saveFiles.append([MyRelativePrice.file, '_ta'])
                MySIMPosition.saveFiles.append([MyBarStrength.file, '_taB'])
 
    '''
    On Consolidated =======================
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

        if not self.CL.enabled or self.algo.IsWarmingUp or not self.posEnabled or not self.IsReady() or not self.WasJustUpdated(self.algo.Time):
            self.signals = ''
            self.events = ''
            return

        '''VOLATILITY CUTOFF
        '''
        volRef = self.algo.mySymbolDict[self.algo.benchmarkSymbol].vol
        #volRef = self.vol
        volatility = volRef.atrVolatility[0]
        self.CL._totalATRpct = volatility/100 #Dimension is inherited from eq2_2
        volatilityNorm = volRef.atrNormVolatility[0]
        volatilityChange = volRef.atrVolatility[0]-volRef.atrVolatility[10]
        
        #This is not used
        volatilityLimitLong = 1.0 
        volatilityLimitShort = 1.15
        
        #CUTOFF
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
            signalRegex = aiObj["signalRegex"] if "signalRegex" in aiObj.keys() else aiKey
            #aiObj["signal"] = aiObj["enabled"] and (signalRegex=="ALL" or re.search(signalRegex, self.signals)) and aiObj["firstTradeHour"] <= self.algo.Time.hour and self.algo.Time.hour <= aiObj["lastTradeHour"]
            aiObj["signal"] = aiObj["enabled"] and (signalRegex=="ALL" or re.search(signalRegex, self.signals)) and aiObj["firstTradeHour"] <= bar.EndTime.hour and bar.EndTime.hour <= aiObj["lastTradeHour"]
            if aiObj["signal"]:
                #self.algo.MyDebug(f' Signal:{aiKey} {self.symbol}')
                #self.algo.signalsTotal+=1
                if aiObj["rawFeatures"]=="rawFeatures1": loadFeatures1 = True
                if aiObj["rawFeatures"]=="rawFeatures2": loadFeatures2 = True
        
        '''SIMULATION SIGNALS
        '''
        longTriggerSim, shortTriggerSim = False, False
        for simKey, simObj in self.CL.simDict.items():
            simObj["currentDisabledBars"] = max(0, simObj["currentDisabledBars"]-1)
            if self.CL.simulate and simObj["Enabled"] and simObj["currentDisabledBars"]==0 and (simKey[0:3]=="ALL" or re.search(simKey, self.signals)):
                simObj["currentDisabledBars"] = simObj["disableBars"]
                if   simObj["direction"]== 1:  longTriggerSim = True
                elif simObj["direction"]==-1: shortTriggerSim = True

        '''FEATURES
        '''
        #rawFeatures1: a)Simulation uses b)aiObj default Feature 
        gasfSim = False
        intCode = [None, np.uint8, np.uint16][2] 
        if False:
            preProcessedFeature1 = self.gasf1.FeatureExtractor(featureType="Close", useGASF=True, picleFeatures=False, preProcessor=self.CL.preprocDict["CNNAE_1"]["model"])
            self.algo.MyDebug(f'preProcessedFeature1: {preProcessedFeature1.shape}')
        if loadFeatures1 or (longTriggerSim or shortTriggerSim):
            self.rawFeatures1 = [self.gasf1.FeatureExtractor(n=100, featureType="Close", preProcessor=self.CL.preprocDict["CAE_1D_100-16_sm"]["model"], returnList=True), \
                                 self.gasf1.FeatureExtractor(n=8, featureType="ULBG", returnList=True), \
                                 self.gasf1.FeatureExtractor(n=50, featureType="RelativePrice", preProcessor=self.CL.preprocDict["CAE_1D_50-8_sm"]["model"], returnList=True), \
                                 self.gasf1.FeatureExtractor(n=50, featureType="Volatility", preProcessor=self.CL.preprocDict["CAE_1D_50-8_sm"]["model"], returnList=True), \
                                 self.gasf1.FeatureExtractor_HE(hursts=((1,20),(15,35),(30,50)), inputType="Close", n=None), \
                                 self.gasf1_2.FeatureExtractor(n=100, featureType="Close", preProcessor=self.CL.preprocDict["CAE_1D_100-16_sm"]["model"], returnList=True), \
                                 self.gasf1_2.FeatureExtractor(n=8, featureType="ULBG", returnList=True), \
                                 self.gasf1_2.FeatureExtractor(n=50, featureType="RelativePrice", preProcessor=self.CL.preprocDict["CAE_1D_50-8_sm"]["model"], returnList=True), \
                                 self.gasf1_2.FeatureExtractor(n=50, featureType="Volatility", preProcessor=self.CL.preprocDict["CAE_1D_50-8_sm"]["model"], returnList=True), \
                                 self.gasf1_2.FeatureExtractor(n=8, featureType="Volume", returnList=True), \
                                 self.gasf1_2.FeatureExtractor_HE(hursts=((1,20),(15,35),(30,50)), inputType="Close", n=None)]
        
        if loadFeatures2:
            self.rawFeatures2 = []
 
        for aiObj in self.CL.aiDict.values():
            if aiObj["enabled"] and aiObj["signal"]:
                if self.symbol == 'CI' and (True or self.algo.Time == datetime(year = 3019, month = 10, day = 7)):
                    hp.MyModelLoader.PrintFeatures(self, self.rawFeatures1, featureRegex='Feat', negativeOnly=False)
                myFeatures = self.algo.myHelpers.UnpackFeatures(getattr(self, aiObj["rawFeatures"], self.rawFeatures1),  featureType=aiObj["featureType"] if "featureType" in aiObj else 1, \
                                featureRegex=aiObj["featureFilter"] if "featureFilter" in aiObj else 'Feat', reshapeTuple=None)
                customColumnFilters = aiObj["customColumnFilter"]
                aiObj["dfFilterPassed"] = self.algo.myHelpers.FeatureCustomColumnFilter(myFeatures, customColumnFilters=customColumnFilters) if len(customColumnFilters)!=0 else True
                if aiObj["Type"] != "PT_CNN": myFeatures = myFeatures.values
                if aiObj["usePCA"]:
                    myFeatures = aiObj["pca"].transform(myFeatures)
                if aiObj["Type"] == "PT":
                    myFeatures = torch.from_numpy(myFeatures.reshape(-1, aiObj["featureCount"])).float().to('cpu')
                aiObj["features"] = myFeatures

        '''SIGNAL FILTERING AND TRADE TRIGGERS
        '''
        longTrigger, shortTrigger = False, False
        for aiKey, aiObj in self.CL.aiDict.items():
            #This is for Stat5 signal compatibility
            self.signalDisabledBars[aiKey] = max(self.signalDisabledBars[aiKey]-1,0)
            #Long
            if self.CL.loadAI and not (longTrigger or shortTrigger) and aiObj["enabled"] and aiObj["model"]!=None and aiObj["signal"] and aiObj["dfFilterPassed"] and self.signalDisabledBars[aiKey]==0:
                #MODEL INFERENCE
                if aiObj["Type"] == "PT" or aiObj["Type"] == "PT_CNN":
                    aiObj["model"].eval()
                    trigger = aiObj["model"].Predict(aiObj["features"])
                elif aiObj["Type"] == "SK" or aiObj["Type"] == "LGB":
                    if "threshold" in aiObj.keys() and aiObj["threshold"]!=None:
                        trigger = np.where(aiObj["model"].predict(aiObj["features"])>aiObj["threshold"], 1, 0)
                    else:
                        trigger = aiObj["model"].predict(aiObj["features"])
                #Setting Trade Trigger Flags
                if trigger and aiObj["direction"]==1:
                    if shortTrigger:
                        #Both Direction -> Undecided
                        longTrigger, shortTrigger = False, False
                    else:
                        longTrigger = True
                        riskMultiple = min(1.00, max(aiObj["riskMultiple"], 0.00)) if "riskMultiple" in aiObj else 1.00
                        customPositionSettings = aiObj["customPositionSettings"] if "customPositionSettings" in aiObj else {}
                        self.CL.strategyCode = self.CL.strategyCode + "|" + aiKey
                elif trigger and aiObj["direction"]==-1:
                    if longTrigger:
                        #Both Direction -> Undecided
                        longTrigger, shortTrigger = False, False
                    else:
                        shortTrigger = True
                        riskMultiple = min(1.00, max(aiObj["riskMultiple"], 0.00)) if "riskMultiple" in aiObj else 1.00
                        customPositionSettings = aiObj["customPositionSettings"] if "customPositionSettings" in aiObj else {}
                        self.CL.strategyCode = self.CL.strategyCode + "|" + aiKey
            if aiObj["signal"] and self.signalDisabledBars[aiKey]==0: self.signalDisabledBars[aiKey] = 8

        '''POSITION ENTRY/FLIP (Flip Enabled is checked in EnterPosition_2)
        '''
        self.longDisabledBars = max(self.longDisabledBars-1,0)
        self.shortDisabledBars = max(self.shortDisabledBars-1,0)
        #---LONG POSITION
        if self.posEnabled and self.CL.enableLong and self.longDisabledBars==0 and longTrigger:
            self.algo.myPositionManager.EnterPosition_2(self.symbol, 1, myRiskMultiple=riskMultiple, customPositionSettings=customPositionSettings)
            if False: self.algo.MyDebug(f' Long Trade: {self.symbol}')
            self.longDisabledBars=3
        #---SHORT POSITION
        elif self.posEnabled and self.CL.enableShort and self.shortDisabledBars==0 and shortTrigger:
            self.algo.myPositionManager.EnterPosition_2(self.symbol, -1, myRiskMultiple=riskMultiple, customPositionSettings=customPositionSettings)
            if False: self.algo.MyDebug(f' Short Trade: {self.symbol}')
            self.shortDisabledBars=3

        '''EXIT SIGNALS AND EXIT
        '''
        for exitSignal, exitDirection in self.CL.exitSignalDict.items():
            position = self.algo.Portfolio[self.symbol].Quantity
            if position!=0 and re.search(exitSignal, self.signals):
                if exitDirection == +1 and position < 0 and not longTrigger:
                    #Exit Short
                    self.algo.myPositionManagerB.LiquidatePosition(self.symbol, "L_ExitSignal", "L_ExitSignal")
                    if False: self.algo.MyDebug(f' Exit from Short: {self.symbol}, signal:{exitSignal}')
                elif exitDirection == -1 and position > 0 and not shortTrigger:
                    #Exit Long
                    self.algo.myPositionManagerB.LiquidatePosition(self.symbol, "S_ExitSignal", "S_ExitSignal")
                    if False: self.algo.MyDebug(f' Exit from Long: {self.symbol}, signal:{exitSignal}')
                    
        '''SIMULATION CALL
        '''
        debugSim  = False
        if longTriggerSim or shortTriggerSim: 
            myFeatures = self.rawFeatures1
        self.longSimDisabledBars=max(self.longSimDisabledBars-1,0)
        self.shortSimDisabledBars=max(self.shortSimDisabledBars-1,0)
        simTradeTypes = (   {'s':3,  't':35, 'mpo':2.0, 'sc':True, 'st':None, 'msd':2.0}, \
                            {'s':5,  't':35, 'mpo':2.0, 'sc':True, 'st':None, 'msd':2.0}, \
                            {'s':8,  't':35, 'mpo':2.0, 'sc':True, 'st':None, 'msd':2.0}, \
                            {'s':15, 't':35, 'mpo':1.7, 'sc':True, 'st':None, 'msd':2.0}, \
                            {'s':25, 't':50, 'mpo':1.7, 'sc':True, 'st':None, 'msd':2.0}, \
                            {'s':35, 't':70, 'mpo':1.7, 'sc':True, 'st':None, 'msd':2.0})
        simMinMaxTypes = (  {'n':5},  \
                            {'n':10}, \
                            {'n':20}, \
                            {'n':35})
        lastEntryDate = datetime(year = 3019, month = 10, day = 7)
        if self.CL.simulate and self.posEnabled and longTriggerSim and (self.algo.Time<self.algo.simEndDate or self.algo.simEndDate==None):
            if debugSim: self.algo.MyDebug(f' {self.symbol} Sim call Long: {self.signals}')
            sim = MySIMPosition(self, direction=1, timestamp=self.algo.Time.strftime("%Y%m%d %H:%M"), signal=self.signals, features=myFeatures, simTradeTypes=simTradeTypes, simMinMaxTypes=simMinMaxTypes, MinMaxNormMethod="pct")
            #use 0 if signalspecific
            self.longSimDisabledBars=0
        if self.CL.simulate and self.posEnabled and shortTriggerSim and (self.algo.Time<self.algo.simEndDate or self.algo.simEndDate==None):
            if debugSim: self.algo.MyDebug(f' {self.symbol} Sim call Short: {self.signals}')
            sim = MySIMPosition(self, direction=-1, timestamp=self.algo.Time.strftime("%Y%m%d %H:%M"), signal=self.signals, features=myFeatures, simTradeTypes=simTradeTypes, simMinMaxTypes=simMinMaxTypes, MinMaxNormMethod="pct") 
            #use 0 if signalspecific
            self.shortSimDisabledBars=0
        
        '''FEATURE DEBUG
        '''
        debugTicker = 'AES'
        debugSignal = 'ALL'
        if True and self.symbol.Value == debugTicker and ((re.search(debugSignal, self.signals) or debugSignal=='ALL')):
            if False and datetime(2020, 8, 4, 10, 00) == self.algo.Time:
                self.algo.MyDebug(f' {self.symbol} Feature Debug. gasf1_2 Price:{self.gasf1_2.myBars[0].Close}, EndTime:{self.gasf1_2.myBars[0].EndTime}')
                df_debug = self.algo.myHelpers.UnpackFeatures(self.rawFeatures1,  featureType=1, featureRegex='Feat', reshapeTuple=None)
                hp.MyModelLoader.PrintFeatures(self, df_debug, negativeOnly=False)
        
        '''RESET strategyCode, Signals and Events FOR THE NEXT BAR
        '''
        self.CL.strategyCode = self.CL.strategyCodeOriginal 
        #if False and re.search('ABCD_|DCH_', self.signals): self.algo.MyDebug(f' {self.symbol} self.signals before reset: {self.signals}')
        #if False and self.zz.patterns.doubleTop: self.algo.MyDebug(f' {self.symbol} self.signals before reset: {self.signals}')
        self.signals = ''
        self.events = ''
        return
    
    '''
    On Consolidated HIGHER TIMEFRAME =======================
    '''
    def OnDataConsolidated_2(self, sender, bar):
        '''Update
        '''
        self.bars_rw_2.Add(bar)
        
    '''
    UPDATE STATUS =======================
    '''
    def IsReady(self):
        indiacatorsReady = self.readyIndicator.IsReady
        if False and indiacatorsReady:
            self.algo.MyDebug(" Consolidation Is Ready " + str(self.bars_rw.IsReady) + str(self.dch2.IsReady) + str(self.WasJustUpdated(self.algo.Time)))
        return indiacatorsReady
    
    def WasJustUpdated(self, currentTime):
        return self.bars_rw.Count > 0 and (currentTime - self.barTimeDifference - self.bars_rw[0].EndTime) < timedelta(seconds=10) \
                and (currentTime - self.barTimeDifference - self.bars_rw[0].EndTime).total_seconds() > timedelta(seconds=-10).total_seconds()
    
    '''
    OTHER =======================
    '''
    def TestPredict(self, model, featureNo):
        device = 'cpu'
        for i in range (10):
            np.random.seed(i)
            x_sample = np.random.rand(1, featureNo)
            x_sample = torch.from_numpy(x_sample.reshape(-1, featureNo)).float().to(device)
            y_hat = model.Predict2(x_sample)
            self.algo.MyDebug(f'{y_hat}')
 
'''
TORCH MODEL(S)
'''
class NN_2(nn.Module):
   def __init__(self, features, inputSize=None, hiddenSize1=None, hiddenSize2=None, hiddenSize3=None, outFeed='_h3', outputs=2, softmaxout=False, dropoutRate_in=0.0, dropoutRate_h=0.0, bn_momentum=0.1):
       super(NN_2, self).__init__()
       self.outputs = outputs
       self.features = features
       self.softmaxout = softmaxout
       self.inputSize=inputSize if inputSize!=None else features
       self.hiddenSize1 = hiddenSize1 if hiddenSize1!=None else features
       self.hiddenSize2 = hiddenSize2 if hiddenSize2!=None else features
       self.hiddenSize3 = hiddenSize3 if hiddenSize3!=None else features
       self.out_inFeatures = {
           '_in': self.inputSize,
           '_h1': self.hiddenSize1,
           '_h2': self.hiddenSize2,
           '_h3': self.hiddenSize3,}[outFeed]

       #https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/ , https://blog.paperspace.com/busting-the-myths-about-batch-normalization/
       #https://pytorch.org/docs/stable/nn.html?highlight=nn%20batchnorm1d#torch.nn.BatchNorm1d
       self.bn_in = nn.BatchNorm1d(num_features=features, momentum=bn_momentum)
       self.layer_In = torch.nn.Linear(features, self.inputSize)
       self.bn_h1 = nn.BatchNorm1d(num_features=self.inputSize, momentum=bn_momentum)
       self.layer_h1 = torch.nn.Linear(self.inputSize, self.hiddenSize1)
       self.bn_h2 = nn.BatchNorm1d(num_features=self.hiddenSize1, momentum=bn_momentum)
       self.layer_h2 = torch.nn.Linear(self.hiddenSize1, self.hiddenSize2)
       self.bn_h3 = nn.BatchNorm1d(num_features=self.hiddenSize2, momentum=bn_momentum)
       self.layer_h3 = torch.nn.Linear(self.hiddenSize2, self.hiddenSize3)
       self.bn_out = nn.BatchNorm1d(num_features=self.out_inFeatures, momentum=bn_momentum)
       self.layer_Out = torch.nn.Linear(self.out_inFeatures, self.outputs)
       #https://towardsdatascience.com/simplified-math-behind-dropout-in-deep-learning-6d50f3f47275
       self.L_out = {}
       self.outFeed = outFeed 
       #https://towardsdatascience.com/simplified-math-behind-dropout-in-deep-learning-6d50f3f47275
       self.dropoutRate_in = dropoutRate_in
       self.dropoutRate_h = dropoutRate_h
       #https://mlfromscratch.com/activation-functions-explained/ 
       self.MyActivation = [F.relu, F.leaky_relu_, F.selu][2]
                        
   def forward(self, x):
       #https://datascience.st  ackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks
       self.L_out['_in']  =  self.MyActivation(self.layer_In(self.bn_in(F.dropout(x, p=self.dropoutRate_in , training=self.training))))
       self.L_out['_h1']  =  self.MyActivation(self.layer_h1(self.bn_h1(F.dropout(self.L_out['_in'], p=self.dropoutRate_h, training=self.training))))
       self.L_out['_h2']  =  self.MyActivation(self.layer_h2(self.bn_h2(F.dropout(self.L_out['_h1'], p=self.dropoutRate_h, training=self.training))))
       self.L_out['_h3']  =  self.MyActivation(self.layer_h3(self.bn_h3(F.dropout(self.L_out['_h2'], p=self.dropoutRate_h, training=self.training))))
       self.L_out['_out'] =  self.MyActivation(self.layer_Out(self.bn_out(self.L_out[self.outFeed])))
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
       prediction =np.argmax(prediction.data.numpy(), axis=1) #armagx returns a tuple
       if len(prediction)==1:
           return prediction[0]
       return prediction
 
   def Predict2(self, x):
       self.eval()
       prediction = self.forward(x).data.numpy()
       return prediction

class CNN_AE_1(nn.Module):
    def __init__(self, cnn_Dim, inH, inW, inChannels=1, cnnChannels1=16, cnnChannels2=16, hiddenSize=None, outputs=20, useConvTranspose=False, hiddenSizeDecode=None, dropoutRate=0.25, bn_momentum=0.1):
        super(CNN_AE_1, self).__init__()
        self.cnn_Dim = cnn_Dim-1 #1:1D, 2:2D
        self.inH = inH
        self.inW = inW
        self.inChannels = inChannels
        self.cnnChannels1 = cnnChannels1
        self.cnnChannels2 = cnnChannels2
        self.cnnFlattenedSize = round(self.inW/4) * self.cnnChannels2 if self.cnn_Dim==0 else round(self.inH/4)*round(self.inW/4) * self.cnnChannels2 
        self.hiddenSize = hiddenSize
        self.hiddenSizeDecode = hiddenSizeDecode
        self.dropoutRate = dropoutRate
        self.outputs = outputs
        self.kernel_size = [3, 5, 7, 9][1]
        self.padding = round((self.kernel_size-1)/2)
        self.useConvTranspose = useConvTranspose
        self.kernel_size_T = [2, 4, 6, 8][1]
        self.padding_T = round((self.kernel_size_T-2)/2)

        self.MyActivation = [nn.ReLU(), nn.LeakyReLU, nn.SELU][0]

        self.cnn_BatchNorm = [nn.BatchNorm1d, nn.BatchNorm2d][self.cnn_Dim]
        self.cnn_Conv =  [nn.Conv1d,  nn.Conv2d][self.cnn_Dim]
        self.cnn_MaxPool = [nn.MaxPool1d, nn.MaxPool2d][self.cnn_Dim]

        if self.useConvTranspose:
            self.cnn_ConvT = [nn.ConvTranspose1d,  nn.ConvTranspose2d][self.cnn_Dim]
        
        #ENCODER
        #n/4^2*self.cnnChannels2
        self.CNN = nn.Sequential(
                        self.cnn_BatchNorm(inChannels),
                        self.cnn_Conv(inChannels, self.cnnChannels1, kernel_size=self.kernel_size, stride=1, padding=self.padding),
                        self.MyActivation,
                        self.cnn_MaxPool(kernel_size=2, stride=2),
                        self.cnn_BatchNorm(self.cnnChannels1),
                        self.cnn_Conv(self.cnnChannels1, self.cnnChannels2, kernel_size=self.kernel_size, stride=1, padding=self.padding),
                        self.MyActivation,
                        self.cnn_MaxPool(kernel_size=2, stride=2))
        
        if self.hiddenSize!=None:
            self.BN1 = nn.BatchNorm1d(num_features=self.cnnFlattenedSize, momentum=bn_momentum)
            self.FC1 = nn.Linear(self.cnnFlattenedSize, self.hiddenSize)
            self.BN2 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
            self.FC2 = nn.Linear(self.hiddenSize, self.outputs)
        else:
            self.BN1 = nn.BatchNorm1d(num_features=self.cnnFlattenedSize, momentum=bn_momentum)
            self.FC1 = nn.Linear(self.cnnFlattenedSize, self.outputs)

        #DECODER
        if self.useConvTranspose:
                self.D_BN_T = nn.BatchNorm1d(num_features=self.outputs, momentum=bn_momentum)
                self.D_NN_T = nn.Linear(self.outputs, self.cnnChannels2*round(self.inH/4)*round(self.inW/4)) if self.cnn_Dim==1 else nn.Linear(self.outputs, self.cnnChannels2*round(self.inW/4))
                
                self.CNN_T = nn.Sequential(
                                self.cnn_ConvT(self.cnnChannels2, self.cnnChannels1, kernel_size=self.kernel_size_T, stride=2, padding=self.padding_T),
                                self.MyActivation,
                                self.cnn_ConvT(self.cnnChannels1, self.inChannels, kernel_size=self.kernel_size_T, stride=2, padding=self.padding_T),
                                nn.Tanh())
        else:
            if self.hiddenSizeDecode!=None:
                self.D_BN1 = nn.BatchNorm1d(num_features=self.outputs, momentum=bn_momentum)
                self.D_NN1 = nn.Linear(self.outputs, self.hiddenSizeDecode)
                self.D_BN2 = nn.BatchNorm1d(num_features=self.hiddenSizeDecode, momentum=bn_momentum)
                self.D_NN2 = nn.Linear(self.hiddenSizeDecode, self.inH*self.inW*self.inChannels)
            else:
                self.D_BN1 = nn.BatchNorm1d(num_features=self.outputs, momentum=bn_momentum)
                self.D_NN1 = nn.Linear(self.outputs, self.inH*self.inW*self.inChannels)

    def forward(self, x):
        x = self.Encode(x)
        x = self.Decode(x)
        return x  

    def Encode(self, x):
        samples = x.size(0) 
        x = self.CNN(x)
        x = x.view(samples, -1)
        if self.hiddenSizeDecode!=None:
            x = self.MyActivation(self.FC1(self.BN1(F.dropout(x, p=self.dropoutRate, training=self.training))))
            x = torch.sigmoid(self.FC2(self.BN2(F.dropout(x, p=self.dropoutRate, training=self.training))))
        else:
            x = torch.sigmoid(self.FC1(self.BN1(F.dropout(x, p=self.dropoutRate, training=self.training))))
        return x

    def Decode(self, x):
        samples = x.size(0) 
        if self.useConvTranspose:
            x = self.MyActivation(self.D_NN_T(self.D_BN_T(F.dropout(x, p=self.dropoutRate, training=self.training))))
            x = x.view([samples, self.cnnChannels2, round(self.inH/4), round(self.inW/4)]) if self.cnn_Dim==1 else x.view([samples, self.cnnChannels2, round(self.inW/4)])
            x = self.CNN_T(x)
        else:
            if self.hiddenSizeDecode!=None:
                x = self.MyActivation(self.D_NN1(self.D_BN1(F.dropout(x, p=self.dropoutRate, training=self.training))))
                x = self.MyActivation(self.D_NN2(self.D_BN2(F.dropout(x, p=self.dropoutRate, training=self.training))))
                x = x.view([samples, self.inChannels, self.inH, self.inW]) if self.cnn_Dim==1 else x.view([samples, self.inChannels, self.inW])
            else:
                x = self.MyActivation(self.D_NN1(self.D_BN1(F.dropout(x, p=self.dropoutRate, training=self.training))))
                x = x.view([samples, self.inChannels, self.inH, self.inW]) if self.cnn_Dim==1 else x.view([samples, self.inChannels, self.inW])
        return x 
    
    def PreProcess(self, x, returnTorch=False):
        self.eval()
        torch.no_grad() #https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/8
        dimsNeeded = 4 if self.cnn_Dim==1 else 3
        while len(x.shape)<dimsNeeded:
            x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).float().to('cpu')
        x = self.Encode(x)
        if returnTorch:
            x = torch.squeeze(x)
            return x
        x = x.data.numpy()
        x = np.squeeze(x)
        return x

    def EncodeDecode(self, x, returnTorch=False):
        self.eval()
        torch.no_grad() #https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/8
        dimsNeeded = 4 if self.cnn_Dim==1 else 3
        while len(x.shape)<dimsNeeded:
            x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).float().to('cpu')
        x = self.forward(x)
        if returnTorch:
            x = torch.squeeze(x)
            return x
        x = x.data.numpy()
        x = np.squeeze(x)
        return x