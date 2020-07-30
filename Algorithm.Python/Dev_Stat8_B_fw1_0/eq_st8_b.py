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

class Eq_St8_B():
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
    strategyCodeOriginal = __name__
    strategyCode = strategyCodeOriginal #used by order tags and debug
    isEquity = True
    customFillModel = 1
    customSlippageModel = 1
    customFeeModel = 0
    customBuyingPowerModel = 0
    #Resolution
    resolutionMinutes   = 60
    resolutionMinutes_2 = 24*60
    maxWarmUpPeriod   = 110
    maxWarmUpPeriod_2 = 110
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
    riskperLongTrade  = 0.50/100 
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
    closeOnTriggerLong  = True #works if Flip is disabled
    closeOnTriggerShort = True 
    entryLimitOrderLong  = True  #If False: enter the position with Market Order
    entryLimitOrderShort = True
    limitEntryATRLong  = 0.01
    limitEntryATRShort = 0.01
    entryTimeInForceLong = timedelta(minutes=5*60)
    entryTimeInForceShort = timedelta(minutes=5*60)
    #Orders
    stopPlacerLong  =   [("minStop"), ("dch", 'dch01'), ("rw",7)][2]     # Tuple (Type, arg2, arg2...)
    stopPlacerShort =   [("minStop"), ("dch", 'dch01'), ("rw",7)][1] 
    targetPlacerLong  = [(None,0), ("minPayOff",0), ("dch", 'dch2'), ("rw",70)][3]  # Tuple (Type, arg2, arg2...) If targetPlacer[0]!=None then customPositionSettings['targetPlacer']==None must raise an exeption during installation otherwise _UT would become inconsistent
    targetPlacerShort = [(None,0), ("minPayOff",0), ("dch", 'dch2'), ("rw",70)][2]  # TWS Sync is not ready yet to handle useTragets for Foreign Symbols  
    minPayOffLong  = 2.00
    minPayOffShort = 1.50
    scratchTradeLong  = True 
    scratchTradeShort = True 
    stopTrailerLong  =  [(None,0), ("dch", 'dch01'), ("rw",15)][0]      # Tuple (Type, arg2, arg2...): (None,0), ("dch",'dch_attributeName'), ("rw",n)
    stopTrailerShort =  [(None,0), ("dch", 'dch01'), ("rw",5)][0] 
    targetTrailerLong  = [(None,0), ("dch", 'dch2'), ("rw",5)][0]       # Tuple (Type, arg2, arg2...): (None,0), ("dch",'dch_attributeName'), ("rw",n)
    targetTrailerShort = [(None,0), ("dch", 'dch2'), ("rw",5)][0] 
    stopATRLong  = 0.5
    stopATRShort = 0.5
    minEntryStopATRLong  = 2.0    
    minEntryStopATRShort = 2.0    
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

    #SP100 (100)
    #myTickers = ["AAPL", "ABBV", "ABT", "ACN", "ADBE", "AGN", "AIG", "ALL", "AMGN", "AMZN", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CELG", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CSCO", "CVS", "CVX", "DD", "DHR", "DIS", "DUK", "EMR", "EXC", "F", "FB", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTN", "SBUX", "SLB", "SO", "SPG", "T", "TGT", "TXN", "UNH", "UNP", "UPS", "USB", "UTX", "V", "VZ", "WBA", "WFC", "WMT", "XOM"]
    #ES&NQ (181)
    #myTickers = ["AAPL", "ABBV", "ABT", "ACN", "ADBE", "AGN", "AIG", "ALL", "AMGN", "AMZN", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CELG", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CSCO", "CVS", "CVX", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FB", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTN", "SBUX", "SLB", "SO", "SPG", "T", "TGT", "TXN", "UNH", "UNP", "UPS", "USB", "UTX", "V", "VZ", "WBA", "WFC", "WMT", "XOM", "ATVI", "AMD", "ALXN", "ALGN", "AAL", "ADI", "AMAT", "ASML", "ADSK", "ADP", "BIDU", "BMRN", "AVGO", "CDNS", "CERN", "CHKP", "CTAS", "CTXS", "CTSH", "CSX", "CTRP", "DLTR", "EBAY", "EA", "EXPE", "FAST", "FISV", "FOX", "FOXA", "HAS", "HSIC", "IDXX", "ILMN", "INCY", "INTU", "ISRG", "JBHT", "JD", "KLAC", "LRCX", "LBTYA", "LBTYK", "LULU", "MAR", "MXIM", "MELI", "MCHP", "MU", "MNST", "MYL", "NTAP", "NTES", "NXPI", "ORLY", "PCAR", "PAYX", "REGN", "ROST", "SIRI", "SWKS", "SYMC", "SNPS", "TMUS", "TTWO", "TSLA", "ULTA", "UAL", "VRSN", "VRSK", "VRTX", "WDAY", "WDC", "WLTW", "WYNN", "XEL", "XLNX", "STX", "TSLA", "VRSK", "WYNN", "XLNX"]
    #ES&NQ SLICE_1 (90)
    #myTickers = ["BK", "CL", "SPG", "PFE", "MSFT", "NEE", "AGN", "INTC", "NFLX", "HD", "PYPL", "USB", "MDLZ", "OXY", "MS", "C", "GD", "BRK.B", "ALL", "DUK", "ABT", "TGT", "DD", "QCOM", "BMY", "LLY", "CHTR", "WBA", "BA", "PM", "LOW", "GILD", "GOOG", "MMM", "COF", "SO", "JNJ", "DIS", "WMT", "CVX", "UNH", "GS", "CELG", "MCD", "BIIB", "CAT", "COP", "T", "AMZN", "ABBV", "HON", "RTN", "VZ", "PG", "SLB", "KMI", "IBM", "BLK", "ACN", "GE", "CVS", "UPS", "MDT", "MRK", "DHR", "BKNG", "NVDA", "V", "AXP", "XOM", "EXC", "JPM", "ORCL", "KHC", "KO", "AIG", "CSCO", "UNP", "TXN", "CMCSA", "AAPL", "LMT", "PEP", "UTX", "F", "SBUX", "COST", "WFC", "FDX", "AMGN"]
    #ES&NQ SLICE_2 (90)
    #myTickers = ["NKE", "MET", "ADBE", "GM", "MA", "MO", "BAC", "EMR", "FB", "TSLA", "EXPE", "AAL", "CTAS", "FISV", "UAL", "BIDU", "CHKP", "WDC", "SIRI", "ULTA", "EA", "FAST", "MAR", "ASML", "ADSK", "PAYX", "ATVI", "NTES", "MXIM", "VRSN", "AMD", "FOXA", "TTWO", "CSX", "VRSK", "VRTX", "BMRN", "WYNN", "KLAC", "ALXN", "JBHT", "WYNN", "XLNX", "INCY", "CERN", "HSIC", "AMAT", "VRSK", "SWKS", "ALGN", "XLNX", "INTU", "LBTYK", "STX", "MNST", "WDAY", "REGN", "ROST", "IDXX", "ILMN", "MELI", "ADP", "AVGO", "PCAR", "FOX", "JD", "GOOGL", "MYL", "ISRG", "ADI", "LBTYA", "DLTR", "CTXS", "EBAY", "MCHP", "XEL", "NXPI", "CDNS", "SNPS", "ORLY", "HAS", "LRCX", "LULU", "WLTW", "MU", "TMUS", "NTAP", "CTSH", "SYMC", "TSLA", "CTRP"]
    
    #SP500-ES&NQ
    #myTickers = ["CRM", "TMO", "LIN", "AMT", "FIS", "CME", "CB", "BDX", "SYK", "TJX", "ANTM", "SPGI", "NOC", "D", "CCI", "ZTS", "BSX", "PNC", "CI", "PLD", "ECL", "ICE", "MMC", "DE", "APD", "KMB", "LHX", "EQIX", "WM", "NSC", "AON", "EW", "SCHW", "EL", "AEP", "ITW", "PGR", "EOG", "SHW", "BAX", "PSX", "DG", "PSA", "SRE", "TRV", "ROP", "HUM", "AFL", "WELL", "BBT", "YUM", "MCO", "SYY", "DAL", "STZ", "JCI", "ETN", "NEM", "PRU", "MPC", "HCA", "GIS", "VLO", "EQR", "TEL", "TWTR", "PEG", "WEC", "MSI", "SBAC", "AVB", "OKE", "IR", "ED", "WMB", "ZBH", "AZO", "HPQ", "VTR", "VFC", "TSN", "STI", "HLT", "BLL", "APH", "MCK", "TROW", "PPG", "DFS", "GPN", "ES", "TDG", "FLT", "LUV", "DLR", "EIX", "IQV", "DTE", "INFO", "O", "FE", "AWK", "A", "CTVA", "HSY", "TSS", "GLW", "APTV", "CMI", "ETR", "PPL", "HIG", "PH", "ADM", "ESS", "FTV", "PXD", "LYB", "SYF", "CMG", "CLX", "SWK", "MTB", "MKC", "MSCI", "RMD", "BXP", "CHD", "AME", "WY", "RSG", "STT", "FITB", "KR", "CNC", "NTRS", "AEE", "VMC", "HPE", "KEYS", "ROK", "CMS", "RCL", "EFX", "ANSS", "CCL", "AMP", "CINF", "TFX", "ARE", "OMC", "HCP", "DHI", "LH", "KEY", "AJG", "MTD", "COO", "CBRE", "HAL", "EVRG", "AMCR", "MLM", "HES", "K", "EXR", "CFG", "IP", "CPRT", "FANG", "BR", "CBS", "NUE", "DRI", "FRC", "MKTX", "BBY", "LEN", "WAT", "RF", "AKAM", "CXO", "MAA", "MGM", "CE", "HBAN", "CAG", "CNP", "KMX", "PFG", "XYL", "DGX", "WCG", "UDR", "DOV", "CBOE", "FCX", "HOLX", "GPC", "L", "ATO", "ABC", "KSU", "CAH", "TSCO", "LDOS", "IEX", "LNT", "EXPD", "GWW", "XRAY", "MAS", "ANET", "UHS", "DRE", "HST", "IT", "SJM", "NDAQ", "HRL", "CHRW", "FTNT", "FMC", "BHGE", "JKHY", "IFF", "NBL", "WAB", "NI", "CTL", "NCLH", "REG", "LNC", "PNW", "CF", "TXT", "VNO", "FTI", "ARNC", "LW", "ETFC", "JEC", "SIVB", "MRO", "AES", "RJF", "AAP", "AVY", "GRMN", "BF.B", "VAR", "RE", "FRT", "TAP", "NRG", "WU", "CMA", "PKG", "DISCK", "DVN", "TIF", "PKI", "IRM", "GL", "ALLE", "EMN", "VIAB", "DXC", "URI", "WRK", "PHM", "ABMD", "WHR", "HII", "QRVO", "CPB", "SNA", "APA", "LKQ", "JNPR", "DISH", "BEN", "IPG", "NOV", "FFIV", "KIM", "KSS", "AIV", "AIZ", "ZION", "ALK", "NLSN", "COG", "MHK", "FBHS", "HFC", "DVA", "SLG", "BWA", "FLIR", "AOS", "ALB", "RHI", "MOS", "NWL", "IVZ", "SEE", "TPR", "PRGO", "PVH", "XRX", "PNR", "PBCT", "ADS", "UNM", "FLS", "NWSA", "HOG", "HBI", "HRB", "PWR", "LEG", "ROL", "JEF", "RL", "M", "DISCA", "IPGP", "XEC", "HP", "CPRI", "AMG", "TRIP", "LB", "GPS", "UAA", "UA", "JWN", "MAC", "NKTR", "COTY", "NWS"]
    #SP500-ES&NQ SLICE_1 (115)
    #myTickers = ["XRAY", "CPB", "L", "TDG", "O", "FBHS", "RHI", "MAC", "CMS", "ANTM", "EQR", "KSU", "BR", "TFX", "CHD", "BBY", "XYL", "SJM", "HRL", "KEYS", "FTNT", "DAL", "A", "GRMN", "BXP", "GPC", "ROP", "WAB", "LIN", "BDX", "AMT", "SHW", "TSN", "NBL", "HUM", "ETN", "WRK", "FCX", "ETFC", "ES", "AEE", "HFC", "AME", "EXR", "EMN", "NUE", "LW", "CB", "LNT", "PPG", "NTRS", "HCA", "ARE", "HES", "DHI", "LEN", "MCO", "MKC", "KMB", "ZION", "EW", "MPC", "ETR", "FRC", "MTD", "EFX", "CLX", "UNM", "RE", "CPRI", "TMO", "DISH", "DRE", "KEY", "DVN", "PSA", "WCG", "MOS", "UHS", "PKI", "FANG", "URI", "CF", "AWK", "EIX", "MGM", "ICE", "LEG", "KIM", "CTL", "QRVO", "ALLE", "DE", "ZBH", "ED", "APA", "CE", "GL", "CBS", "DVA", "GPN", "EL", "ALK", "MSCI", "TXT", "PPL", "RMD", "CMG", "JWN", "COTY", "PNR", "DG", "KSS", "BF.B", "CNC"]
    #SP500-ES&NQ SLICE_2 (115)
    #myTickers = ["PKG", "SLG", "PFG", "AVY", "DTE", "FITB", "SYY", "MCK", "AFL", "AAP", "SNA", "HCP", "EXPD", "LNC", "CHRW", "OKE", "STT", "LUV", "IPGP", "TSS", "HOG", "VTR", "REG", "LYB", "VAR", "NCLH", "IP", "SBAC", "DXC", "TROW", "HP", "FLT", "HBAN", "PNW", "MLM", "NOV", "WU", "JKHY", "AIV", "GWW", "MTB", "HII", "CNP", "PVH", "AEP", "JEC", "PNC", "TRV", "MMC", "FMC", "RJF", "RCL", "HPQ", "HOLX", "IPG", "DGX", "ITW", "NDAQ", "KMX", "WHR", "PLD", "NWL", "TRIP", "NWSA", "ADS", "ROL", "APD", "COO", "NSC", "HIG", "ADM", "HBI", "IR", "EVRG", "MHK", "SRE", "JEF", "MAA", "CMI", "STI", "PBCT", "AON", "AJG", "LKQ", "STZ", "BSX", "EQIX", "ANSS", "TIF", "FIS", "SCHW", "CAG", "UA", "YUM", "ESS", "CXO", "BLL", "NEM", "SYF", "NKTR", "BEN", "SEE", "HAL", "PEG", "MRO", "AKAM", "VIAB", "RSG", "PRU", "TEL", "DLR", "CINF", "NI", "FRT", "VFC"]
    #SP500-ES&NQ SLICE_3 (107)
    #myTickers = ["CRM", "MSI", "INFO", "HSY", "OMC", "VMC", "BAX", "TJX", "EOG", "AMP", "MAS", "LH", "BHGE", "FLS", "K", "IEX", "ATO", "CI", "PXD", "NLSN", "ALB", "COG", "DOW", "CAH", "WM", "WMB", "AIZ", "HRB", "SPGI", "ABMD", "JCI", "AOS", "TAP", "JNPR", "IVZ", "ZTS", "WY", "LB", "AES", "PGR", "PHM", "FE", "VLO", "FLIR", "SIVB", "CMA", "WELL", "DISCK", "RF", "CBOE", "ROK", "CPRT", "PH", "GIS", "PWR", "HPE", "GLW", "BBT", "IFF", "IRM", "CCI", "DOV", "HLT", "NRG", "NWS", "UAA", "HST", "TPR", "XEC", "GPS", "CFG", "D", "PRGO", "WAT", "DISCA", "SYK", "CCL", "M", "FTI", "AZO", "IT", "LHX", "ANET", "TWTR", "ABC", "XRX", "FFIV", "NOC", "MKTX", "AVB", "AMG", "ECL", "KR", "UDR", "CME", "DRI", "WEC", "PSX", "SWK", "TSCO", "ARNC", "DFS", "VNO", "FTV", "APH", "LDOS", "APTV", "BWA"]
    #SP500-ES&NQ SLICE_1 and SLICE_2 (230) 
    #myTickers = ["XRAY", "CPB", "L", "TDG", "O", "FBHS", "RHI", "MAC", "CMS", "ANTM", "EQR", "KSU", "BR", "TFX", "CHD", "BBY", "XYL", "SJM", "HRL", "KEYS", "FTNT", "DAL", "A", "GRMN", "BXP", "GPC", "ROP", "WAB", "LIN", "BDX", "AMT", "SHW", "TSN", "NBL", "HUM", "ETN", "WRK", "FCX", "ETFC", "ES", "AEE", "HFC", "AME", "EXR", "EMN", "NUE", "LW", "CB", "LNT", "PPG", "NTRS", "HCA", "ARE", "HES", "DHI", "LEN", "MCO", "MKC", "KMB", "ZION", "EW", "MPC", "ETR", "FRC", "MTD", "EFX", "CLX", "UNM", "RE", "CPRI", "TMO", "DISH", "DRE", "KEY", "DVN", "PSA", "WCG", "MOS", "UHS", "PKI", "FANG", "URI", "CF", "AWK", "EIX", "MGM", "ICE", "LEG", "KIM", "CTL", "QRVO", "ALLE", "DE", "ZBH", "ED", "APA", "CE", "GL", "CBS", "DVA", "GPN", "EL", "ALK", "MSCI", "TXT", "PPL", "RMD", "CMG", "JWN", "COTY", "PNR", "DG", "KSS", "BF.B", "CNC", "PKG", "SLG", "PFG", "AVY", "DTE", "FITB", "SYY", "MCK", "AFL", "AAP", "SNA", "HCP", "EXPD", "LNC", "CHRW", "OKE", "STT", "LUV", "IPGP", "TSS", "HOG", "VTR", "REG", "LYB", "VAR", "NCLH", "IP", "SBAC", "DXC", "TROW", "HP", "FLT", "HBAN", "PNW", "MLM", "NOV", "WU", "JKHY", "AIV", "GWW", "MTB", "HII", "CNP", "PVH", "AEP", "JEC", "PNC", "TRV", "MMC", "FMC", "RJF", "RCL", "HPQ", "HOLX", "IPG", "DGX", "ITW", "NDAQ", "KMX", "WHR", "PLD", "NWL", "TRIP", "NWSA", "ADS", "ROL", "APD", "COO", "NSC", "HIG", "ADM", "HBI", "IR", "EVRG", "MHK", "SRE", "JEF", "MAA", "CMI", "STI", "PBCT", "AON", "AJG", "LKQ", "STZ", "BSX", "EQIX", "ANSS", "TIF", "FIS", "SCHW", "CAG", "UA", "YUM", "ESS", "CXO", "BLL", "NEM", "SYF", "NKTR", "BEN", "SEE", "HAL", "PEG", "MRO", "AKAM", "VIAB", "RSG", "PRU", "TEL", "DLR", "CINF", "NI", "FRT", "VFC"]

 #Simulation Signals [direction, disableBars, Enabled, signalDisabledBars]
    simALL=False
    simDict = {
       "ALL_L": [1,len(myTickers),simALL,0], "ALL_S": [-1,len(myTickers),simALL,0],
       "L_Str_": [1,8,not simALL,0], "S_Str_": [-1,8,not simALL,0], 
       "L_Rej_": [1,8,not simALL,0], "S_Rej_": [-1,8,not simALL,0],
       "L_BPA_": [1,8,not simALL,0], "S_BPA_": [-1,8,not simALL,0], 
       "DB_": [1,8,not simALL,0], "DT_": [-1,8,not simALL,0], 
       "TB_": [1,8,not simALL,0], "TT_": [-1,8,not simALL,0],
       "IHS_": [1,8,not simALL,0], "HS_": [-1,8,not simALL,0]}
    
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

#MODELS   -------------------------------------------   
    #RBPA: Trained with Rejection and run with BPA Signals
    aiDict["L_RBPA-A"] = {
        "enabled": False,
        "signalRegex": "L_BPA",
        "direction": 1,
        "Type" : ["SK", "LGB", "PT", "PT_CNN"][1],
        "firstTradeHour": 0,
        "lastTradeHour": lastTradeHour,
        "riskMultiple": 0.50,
        'customPositionSettings': {},
        "modelURL": "https://www.dropbox.com/s/sds0bhmrt0c47vb/LGB_L_2007_FeatAll_PCA_Rej_2MM_Model_20200720-19_3711_c_booster.txt?dl=1",
        "model": None,
        "usePCA": True,
        "pcaURL": "https://www.dropbox.com/s/6x46abage7djkqf/PCA_PCA20_20200724-04_5702.txt?dl=1",
        "pca": None,
        "rawFeatures": "rawFeatures1",
        "threshold": 0.50,
        "features": None,
        "featureFilter": "Feat",
        "customColumnFilter": [],
        "dfFilterPassed": True,
        "signal": False}

    aiDict["L_RBPA-B"] = {
        "enabled": True,
        "signalRegex": "L_BPA",
        "direction": 1,
        "Type" : ["SK", "LGB", "PT", "PT_CNN"][1],
        "firstTradeHour": 0,
        "lastTradeHour": lastTradeHour,
        "riskMultiple": 0.50,
        'customPositionSettings': {},
        "modelURL": "https://www.dropbox.com/s/ihbl7nsn6w7mq1k/LGB_L_2007_FeatAll_PCA_Rej_2MM_Model_20200721-13_5957_c_booster.txt?dl=1",
        "model": None,
        "usePCA": True,
        "pcaURL": "https://www.dropbox.com/s/6x46abage7djkqf/PCA_PCA20_20200724-04_5702.txt?dl=1",
        "pca": None,
        "rawFeatures": "rawFeatures1",
        "threshold": 0.50,
        "features": None,
        "featureFilter": "Feat",
        "customColumnFilter": [],
        "dfFilterPassed": True,
        "signal": False}
    
    aiDict["L_RBPA-C"] = {
        "enabled": True,
        "signalRegex": "L_BPA",
        "direction": 1,
        "Type" : ["SK", "LGB", "PT", "PT_CNN"][1],
        "firstTradeHour": 0,
        "lastTradeHour": lastTradeHour,
        "riskMultiple": 0.50,
        'customPositionSettings': {},
        "modelURL": "https://www.dropbox.com/s/rc9edlbwkmk9gu5/LGB_L_2007_FeatAll_PCA_Rej_2MM_Model_20200721-14_2508_c_booster.txt?dl=1",
        "model": None,
        "usePCA": True,
        "pcaURL": "https://www.dropbox.com/s/6x46abage7djkqf/PCA_PCA20_20200724-04_5702.txt?dl=1",
        "pca": None,
        "rawFeatures": "rawFeatures1",
        "threshold": 0.50,
        "features": None,
        "featureFilter": "Feat",
        "customColumnFilter": [],
        "dfFilterPassed": True,
        "signal": False}
 
    aiDict["S_Rej-A"] = {
        "enabled": False,
        "signalRegex": "S_BPA",
        "direction": -1,
        "Type" : ["SK", "LGB", "PT", "PT_CNN"][1],
        "firstTradeHour": 0,
        "lastTradeHour": lastTradeHour,
        "riskMultiple": 0.50,
        'customPositionSettings': {},
        "modelURL": "https://www.dropbox.com/s/0xuxujtnrkf83rw/LGB_S_2007_FeatAll_PCA_Rej_2MM_Model_20200721-13_4807_c_booster.txt?dl=1",
        "model": None,
        "usePCA": True,
        "pcaURL": "https://www.dropbox.com/s/6x46abage7djkqf/PCA_PCA20_20200724-04_5702.txt?dl=1",
        "pca": None,
        "rawFeatures": "rawFeatures1",
        "threshold": 0.50,
        "features": None,
        "featureFilter": "Feat",
        "customColumnFilter": [],
        "dfFilterPassed": True,
        "signal": False}

    aiDict["S_Rej-B"] = {
        "enabled": False,
        "signalRegex": "S_BPA",
        "direction": -1,
        "Type" : ["SK", "LGB", "PT", "PT_CNN"][1],
        "firstTradeHour": 0,
        "lastTradeHour": lastTradeHour,
        "riskMultiple": 0.50,
        'customPositionSettings': {},
        "modelURL": "https://www.dropbox.com/s/jitvqwff2be2ur3/LGB_S_2007_FeatAll_PCA_Rej_2MM_Model_20200721-13_5045_c_booster.txt?dl=1",
        "model": None,
        "usePCA": True,
        "pcaURL": "https://www.dropbox.com/s/6x46abage7djkqf/PCA_PCA20_20200724-04_5702.txt?dl=1",
        "pca": None,
        "rawFeatures": "rawFeatures1",
        "threshold": 0.50,
        "features": None,
        "featureFilter": "Feat",
        "customColumnFilter": [],
        "dfFilterPassed": True,
        "signal": False}

    aiDict["S_Rej-C"] = {
        "enabled": False,
        "signalRegex": "S_BPA",
        "direction": -1,
        "Type" : ["SK", "LGB", "PT", "PT_CNN"][1],
        "firstTradeHour": 0,
        "lastTradeHour": lastTradeHour,
        "riskMultiple": 0.50,
        'customPositionSettings': {},
        "modelURL": "https://www.dropbox.com/s/0xuxujtnrkf83rw/LGB_S_2007_FeatAll_PCA_Rej_2MM_Model_20200721-13_4807_c_booster.txt?dl=1",
        "model": None,
        "usePCA": True,
        "pcaURL": "https://www.dropbox.com/s/6x46abage7djkqf/PCA_PCA20_20200724-04_5702.txt?dl=1",
        "pca": None,
        "rawFeatures": "rawFeatures1",
        "threshold": 0.50,
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
        self.bars_rw = RollingWindow[IBaseDataBar](100)
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
        self.atr2 = AverageTrueRange(70)
        self.algo.RegisterIndicator(self.symbol, self.atr2, self.consolidator)
        
        self.sma1 = SimpleMovingAverage (70)
        self.algo.RegisterIndicator(self.symbol, self.sma1, self.consolidator)
        self.sma2 = SimpleMovingAverage (140)
        self.algo.RegisterIndicator(self.symbol, self.sma2, self.consolidator)
        self.sma3 = SimpleMovingAverage (280)
        self.algo.RegisterIndicator(self.symbol, self.sma3, self.consolidator)

        self.dch_s = DonchianChannel(3)
        self.algo.RegisterIndicator(self.symbol, self.dch_s, self.consolidator)
        self.dch0 = DonchianChannel(4)
        self.algo.RegisterIndicator(self.symbol, self.dch0, self.consolidator)
        self.dch01 = DonchianChannel(7)
        self.algo.RegisterIndicator(self.symbol, self.dch01, self.consolidator)
        self.dch02 = DonchianChannel(20)
        self.algo.RegisterIndicator(self.symbol, self.dch02, self.consolidator)
        self.dch1 = DonchianChannel(35)
        self.algo.RegisterIndicator(self.symbol, self.dch1, self.consolidator)
        self.dch2 = DonchianChannel(70)
        self.algo.RegisterIndicator(self.symbol, self.dch2, self.consolidator)
        self.dch3 = DonchianChannel(280)
        self.algo.RegisterIndicator(self.symbol, self.dch3, self.consolidator)
        self.dch4 = DonchianChannel(420)
        self.algo.RegisterIndicator(self.symbol, self.dch4, self.consolidator)
        
        self.rsi1 = RelativeStrengthIndex(15)
        self.algo.RegisterIndicator(self.symbol, self.rsi1, self.consolidator)
        self.rsi2 = RelativeStrengthIndex(40)
        self.algo.RegisterIndicator(self.symbol, self.rsi2, self.consolidator)
        
        self.myRelativePrice = MyRelativePrice(self, self.algo, self.symbol, "RelPrice", 200, self.atr1)
        self.algo.RegisterIndicator(self.symbol, self.myRelativePrice, self.consolidator)
        
        self.zz = MyZigZag(self, self.algo, self.symbol, name='zz', period=200, atr=self.atr1, lookback=10, thresholdType=2, threshold=10)
        self.algo.RegisterIndicator(self.symbol, self.zz, self.consolidator)
        
        self.vol = MyVolatility(self, self.algo, self.symbol, name='vol', period=200, atr=self.atr1)
        self.algo.RegisterIndicator(self.symbol, self.vol, self.consolidator)

        '''Indicators Higher Timeframe'''
        self.atr1_2 = AverageTrueRange(5)
        self.algo.RegisterIndicator(self.symbol, self.atr1_2, self.consolidator_2)
        self.sma1_2 = SimpleMovingAverage (50)
        self.algo.RegisterIndicator(self.symbol, self.sma1_2, self.consolidator_2)
        self.sma2_2 = SimpleMovingAverage (100)
        self.algo.RegisterIndicator(self.symbol, self.sma2_2, self.consolidator_2)
        self.dch1_2 = DonchianChannel(80)
        self.algo.RegisterIndicator(self.symbol, self.dch1_2, self.consolidator_2)
        self.dch2_2 = DonchianChannel(100)
        self.algo.RegisterIndicator(self.symbol, self.dch2_2, self.consolidator_2)
        
        self.rsi1_2 = RelativeStrengthIndex(10)
        self.algo.RegisterIndicator(self.symbol, self.rsi1_2, self.consolidator_2)
        self.rsi2_2 = RelativeStrengthIndex(20)
        self.algo.RegisterIndicator(self.symbol, self.rsi2_2, self.consolidator_2)
        
        self.myRelativePrice_2 = MyRelativePrice(self, self.algo, self.symbol, "RelPrice_2", 100, self.atr1_2)
        self.algo.RegisterIndicator(self.symbol, self.myRelativePrice_2, self.consolidator_2)
        
        self.zz_2 = MyZigZag(self, self.algo, self.symbol, name='zz_2', period=100, atr=self.atr1_2, lookback=6, thresholdType=2, threshold=10)
        self.algo.RegisterIndicator(self.symbol, self.zz_2, self.consolidator_2)
        
        self.vol_2 = MyVolatility(self, self.algo, self.symbol, name='vol_2', period=101, atr=self.atr1_2, benchmarkVolAttr='vol_2')
        self.algo.RegisterIndicator(self.symbol, self.vol_2, self.consolidator_2)
        
        '''Symbol State and Features'''
        self.state_sma1 = MyMAState (self, self.sma1, self.sma2)
        self.state_sma2 = MyMAState (self, self.sma2, self.sma3)
        self.state_sma3 = MyMAState (self, self.sma3, self.sma1_2)
        self.state_sma1_2 = MyMAState (self, self.sma1_2, self.sma2_2)
        self.state_sma2_2 = MyMAState (self, self.sma2_2)

        self.state_dch_s = MyDCHState (self, self.dch_s, self.dch2, name='s')
        self.state_dch0 = MyDCHState (self, self.dch0, self.dch1, name='0')
        self.state_dch1 = MyDCHState (self, self.dch1, self.dch2, name='1')
        self.state_dch2 = MyDCHState (self, self.dch2, self.dch3, name='2')
        self.state_dch3 = MyDCHState (self, self.dch3, self.dch4, name='3')
        self.state_dch4 = MyDCHState (self, self.dch4, self.dch1_2, name='4')
        self.state_dch1_2 = MyDCHState (self, self.dch1_2, self.dch2_2, name='1_2')
        self.state_dch2_2 = MyDCHState (self, self.dch2_2, name='2_2')

        '''Signals'''
        self.barStrength1 = MyBarStrength(self, self.algo, self.symbol, name='str1', period=10, atr=self.atr1, lookbackLong=2, lookbackShort=2, \
                priceActionMinATRLong=1.5, priceActionMaxATRLong=2.5, priceActionMinATRShort=1.5, priceActionMaxATRShort=2.5, referenceTypeLong='Close', referenceTypeShort='Close')
        self.algo.RegisterIndicator(self.symbol, self.barStrength1, self.consolidator)
        
        self.barRejection1 = MyBarRejection(self, self.algo, self.symbol, name='rej1', period=10, atr=self.atr1, lookbackLong=3, lookbackShort=3, \
               rejectionPriceTravelLong=1.75, rejectionPriceTravelShort=1.75, rejectionPriceRangeLong=0.40, rejectionPriceRangeShort=0.40, referenceTypeLong='Close', referenceTypeShort='Close')
        self.algo.RegisterIndicator(self.symbol, self.barRejection1, self.consolidator)
        
        self.bpa1 = MyBPA(self, self.algo, self.symbol, name='bpa1', period=10, atr=self.atr1, lookbackLong=1, lookbackShort=1, \
                    minBarAtrLong=1.50, minBarAtrShort=1.50, referenceTypeLong='Close', referenceTypeShort='Close', gapATRLimitLong=1.0, gapATRLimitShort=1.0, needBPAinTrend=True, atrTrendChange=10.0)
        self.algo.RegisterIndicator(self.symbol, self.bpa1, self.consolidator)
        
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
                    self.algo.Debug(f' Torch MODEL ({preprKey}/{self.CL.strategyCode}) LOADED from url: {preprObj["modelURL"]}')

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
                    self.algo.Debug(f' PCA ({aiKey}) LOADED from url: {aiObj["pcaURL"]}')
                #Loading MODEL
                if aiObj["Type"] == "PT" or aiObj["Type"] == "PT_CNN":
                    aiObj["model"] = globals()[aiObj["modelClass"]](**aiObj["modelParams"]).to('cpu')
                    aiObj["model"] = hp.MyModelLoader.LoadModelTorch(self, aiObj["modelURL"], existingmodel=aiObj["model"])
                    self.algo.Debug(f' Torch MODEL ({aiKey}/{self.CL.strategyCode}) LOADED from url: {aiObj["modelURL"]}')
                elif aiObj["Type"] == "SK":
                    aiObj["model"] = hp.MyModelLoader.LoadModelPickled(self, aiObj["modelURL"])
                    self.algo.Debug(f' SKLearn MODEL ({aiKey}/{self.CL.strategyCode}) LOADED from url: {aiObj["modelURL"]}')
                elif aiObj["Type"] == "LGB":
                    #aiObj["model"] = hp.MyModelLoader.LoadModelLGBtxt(self, aiObj["modelURL"])
                    aiObj["model"] = hp.MyModelLoader.LoadModelLGB(self, aiObj["modelURL"])
                    self.algo.Debug(f' LGB MODEL ({aiKey}/{self.CL.strategyCode}) LOADED from url: {aiObj["modelURL"]}')
            self.signalDisabledBars[aiKey] = 0
        
        #Initialize signalDisabledBarsSim
        self.signalDisabledBarsSim = {}
        for simKey, simObj in self.CL.simDict.items():
            self.signalDisabledBarsSim[simKey] = random.randint(1, len(self.CL.myTickers))

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
                MySIMPosition.saveFiles.append([MyRelativePrice.file, '_ta1'])
                MySIMPosition.saveFiles.append([MyBarStrength.file, '_taB1'])
 
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
            aiObj["signal"] = aiObj["enabled"] and (signalRegex=="ALL" or re.search(signalRegex, self.signals)) and aiObj["firstTradeHour"] <= bar.EndTime.hour and bar.EndTime.hour <= aiObj["lastTradeHour"]
            if aiObj["signal"]:
                #self.algo.MyDebug(f' Signal:{aiKey} {self.symbol}')
                #self.algo.signalsTotal+=1
                if aiObj["rawFeatures"]=="rawFeatures1": loadFeatures1 = True
                if aiObj["rawFeatures"]=="rawFeatures2": loadFeatures2 = True
        
        '''SIMULATION SIGNALS
        '''
        longTriggerSim, shortTriggerSim = False, False
        #simObj : [direction, disableBars, enabled]
        for simKey, simObj in self.CL.simDict.items():
            self.signalDisabledBarsSim[simKey] = max(0, self.signalDisabledBarsSim[simKey]-1)
            if self.CL.simulate and simObj[2] and self.signalDisabledBarsSim[simKey]==0 and (simKey[0:3]=="ALL" or re.search(simKey, self.signals)):
                self.signalDisabledBarsSim[simKey] = simObj[1]
                if   simObj[0]== 1:  longTriggerSim = True
                elif simObj[0]==-1: shortTriggerSim = True

        '''FEATURES
        '''
        #Simulation should use this feature
        gasfSim = False
        intCode = [None, np.uint8, np.uint16][2] 
        if False:
            preProcessedFeature1 = self.gasf1.FeatureExtractor(featureType="Close", useGASF=True, picleFeatures=False, preProcessor=self.CL.preprocDict["CNNAE_1"]["model"])
            self.algo.MyDebug(f'preProcessedFeature1: {preProcessedFeature1.shape}')
        if loadFeatures1 or (longTriggerSim or shortTriggerSim):
            self.rawFeatures1 = [ self.state_sma1.FeatureExtractor(Type=11), self.state_sma2.FeatureExtractor(Type=11), self.state_sma3.FeatureExtractor(Type=11), self.state_sma1_2.FeatureExtractor(Type=11), self.state_sma2_2.FeatureExtractor(Type=11), \
                                self.state_dch0.FeatureExtractor(Type=6), self.state_dch1.FeatureExtractor(Type=6), self.state_dch2.FeatureExtractor(Type=6), self.state_dch3.FeatureExtractor(Type=6), self.state_dch4.FeatureExtractor(Type=6), self.state_dch1_2.FeatureExtractor(Type=6), \
                                self.myRelativePrice.FeatureExtractor(Type=1, normalizationType=1, lookbacklist=[8,28,42], featureMask=[0,0,1,1]), self.myRelativePrice_2.FeatureExtractor(Type=1, normalizationType=1, lookbacklist=[10,20,50], featureMask=[0,0,1,1]), \
                                self.vol.FeatureExtractor(Type=5, lookbacklist=[28,28,42]), \
                                self.vol_2.FeatureExtractor(Type=5, lookbacklist=[10,20,50], avgPeriod=70), \
                                self.zz.FeatureExtractor(listLen=20, Type=11), self.zz.FeatureExtractor(listLen=20, Type=21), self.zz_2.FeatureExtractor(listLen=6, Type=11), self.zz_2.FeatureExtractor(listLen=6, Type=21), \
                                [self.rsi1.Current.Value, self.rsi2.Current.Value, self.rsi1_2.Current.Value, self.rsi2_2.Current.Value] ]
        
        if loadFeatures2:
            self.rawFeatures2 = []
 
        for aiObj in self.CL.aiDict.values():
            if aiObj["enabled"] and aiObj["signal"]: 
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
        #simTradeTypes = [[0,2], [0,2,3,4,7,8,10,14], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]][1]
        simTradeTypes = ({'s':5, 't':35, 'mpo':1.5, 'sc':True, 's_t':None}, {'s':7, 't':70, 'mpo':1.5, 'sc':True, 's_t':None})
        #simMinMaxTypes = [[0,1], [0,1,2,3]][1]
        simMinMaxTypes = ({'n':5}, {'n':10})
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
        debugTicker = 'EURUSD'
        debugSignal = 'L_Str'
        if False and self.symbol == self.algo.Securities[debugTicker].Symbol and re.search(debugSignal, self.signals):
            if True or ( datetime(2019, 1, 1, 12, 00) < self.algo.Time and self.algo.Time <= datetime(2019, 10, 11, 13, 00) ):
                df_debug = self.algo.myHelpers.UnpackFeatures(self.rawFeatures1,  featureType=1, featureRegex='Feat', reshapeTuple=None)
                self.PrintFeatures(df_debug, negativeOnly=False)
                pass
        
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
        pass
        
    '''
    UPDATE STATUS =======================
    '''
    def IsReady(self):
        indiacatorsReady = self.dch2_2.IsReady
        if False and indiacatorsReady:
            self.algo.MyDebug(" Consolidation Is Ready " + str(self.bars_rw.IsReady) + str(self.dch2.IsReady) + str(self.WasJustUpdated(self.algo.Time)))
        return indiacatorsReady
    
    def WasJustUpdated(self, currentTime):
        return self.bars_rw.Count > 0 and (currentTime - self.barTimeDifference - self.bars_rw[0].EndTime) < timedelta(seconds=10) \
                and (currentTime - self.barTimeDifference - self.bars_rw[0].EndTime).total_seconds() > timedelta(seconds=-10).total_seconds()
    
    '''
    OTHER =======================
    '''
    def PrintFeatures(self, df, negativeOnly=False):
        df = df.filter(regex = 'Feat')
        for col in df:
            item = df.loc[df.index[0], col]
            if not negativeOnly or (negativeOnly and item<0):
                self.algo.MyDebug(f' {col}: {item}')
            
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