### <summary>
### 
### </summary>
from QuantConnect import SecurityType, Resolution
from QuantConnect.Indicators import RollingWindow, ExponentialMovingAverage, SimpleMovingAverage, IndicatorExtensions, AverageTrueRange, DonchianChannel, RelativeStrengthIndex, IndicatorDataPoint
from QuantConnect.Data.Consolidators import TradeBarConsolidator, QuoteBarConsolidator
from QuantConnect.Data.Market import IBaseDataBar, TradeBar

from datetime import datetime, timedelta, date
from ta1 import MyRelativePrice, MyPriceNormaliser, MyZigZag, MyDCHState, MyMAState, MySupportResistance, MyPatterns, MyVolatility
from taB1 import MyBarStrength, MyBarRejection, MyGASF
from sim1 import MySIMPosition
#from m_pt1 import NN_1

import hp3
from pandas import DataFrame
import numpy as np
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class Eq2_ai_4():
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
    maxWarmUpPeriod   = 430
    maxWarmUpPeriod_2 = 120
    barPeriod   =  timedelta(minutes=resolutionMinutes)
    barPeriod_2 =  timedelta(minutes=resolutionMinutes_2)
    if isEquity:
        warmupcalendardays = max( round(7/5*maxWarmUpPeriod/(7*(60/min(resolutionMinutes,60*7) ))), round(7/5*maxWarmUpPeriod_2/(7*(60/min(resolutionMinutes_2,60*7) ))) )
    else:
        warmupcalendardays = max(round(7/5*maxWarmUpPeriod/(24*(60/resolutionMinutes))), round(7/5*maxWarmUpPeriod_2/(24*(60/resolutionMinutes_2))))
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
    maxTotalVaR = 0.100
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
    useTragetsLong  = True #TWS Sync is not ready yet to handle useTragets for Foreign Symbols
    useTragetsShort = True
    stopPlacerLong  = 0    #0)dch0 1)dch1 2)dch2) 3)dch3 4)minStopATR 5)bars_rw[0] 6)bars_rw[0-1] 7)dc01 8)dch02 else: dch1
    stopPlacerShort = 0
    targetPlacerLong  = 1    #1)max(dch2,minPayoff) 2)max(dch3,minPayoff)  else: minPayoff 
    targetPlacerShort = 1    
    minPayOffLong  = 1.75
    minPayOffShort = 1.75
    scratchTradeLong  = True 
    scratchTradeShort = True 
    stopTrailerLong  = 0 #0) no Trail 1)dhc2 2)dch3 3)dch1 4)dch0
    stopTrailerShort = 0 
    targetTrailerLong  = 0 #0) no Trail 1)dhc2 
    targetTrailerShort = 0 
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
    
    #Pi NoETF ALL
    #myTickers = ["A", "AA", "AABA", "AAL", "AAXN", "ABBV", "ACIA", "ADM", "ADT", "AIG", "AKAM", "AKS", "ALLY", "ALTR", "AMAT", "AMC", "AMCX", "AMD", "AMGN", "AMZN", "AN", "ANF", "ANTM", "AOBC", "APO", "APRN", "ARLO", "ATUS", "ATVI", "AUY", "AVGO", "AVTR", "AWK", "BABA", "BAC", "BAH", "BB", "BBBY", "BBY", "BIDU", "BJ", "BKNG", "BLK", "BOX", "BP", "BRK-B", "BSX", "BTU", "BURL", "BX", "BYND", "C", "CAKE", "CARS", "CBOE", "CCJ", "CDLX", "CELG", "CHK", "CHWY", "CIEN", "CLDR", "CLF", "CLNE", "CMCSA", "CME", "CMG", "CMI", "CNDT", "COP", "COST", "COUP", "CPB", "CREE", "CRM", "CRSP", "CRUS", "CRWD", "CSX", "CTRP", "CTSH", "CVS", "DBI", "DBX", "DD", "DE", "DECK", "DELL", "DG", "DKS", "DLTR", "DNKN", "DNN", "DO", "DOCU", "DRYS", "DT", "DUK", "EA", "EBAY", "ELAN", "EOG", "EQT", "ESTC", "ET", "ETFC", "ETRN", "ETSY", "EXC", "F", "FANG", "FB", "FCX", "FDX", "FEYE", "FISV", "FIT", "FIVE", "FLR", "FLT", "FMCC", "FNMA", "FSCT", "FSLR", "FTCH", "GDDY", "GE", "GH", "GLBR", "GLW", "GM", "GME", "GNRC", "GOLD", "GOOGL", "GOOS", "GPRO", "GPS", "GRPN", "GRUB", "GSK", "GSKY", "HAL", "HCA", "HCAT", "HIG", "HLF", "HLT", "HOG", "HON", "HPE", "HPQ", "HRI", "HTZ", "IBKR", "ICE", "INFO", "INMD", "IQ", "IQV", "ISRG", "JBLU", "JCP", "JMIA", "JNPR", "KBR", "KLAC", "KMI", "KMX", "KNX", "KSS", "LC", "LEVI", "LHCG", "LLY", "LN", "LOW", "LULU", "LVS", "LYFT", "MA", "MDLZ", "MDR", "MGM", "MLCO", "MNK", "MO", "MOMO", "MRNA", "MRVL", "MS", "MSI", "MU", "MXIM", "NAVI", "NEM", "NET", "NFLX", "NIO", "NOK", "NOV", "NOW", "NTNX", "NTR", "NUAN", "NUE", "NVDA", "NVR", "NVS", "NWSA", "NXPI", "OAS", "OKTA", "OPRA", "ORCL", "OXY", "PANW", "PAYX", "PBR", "PCG", "PDD", "PE", "PEP", "PHM", "PINS", "PIR", "PM", "PRGO", "PS", "PSTG", "PTON", "PVTL", "PYPL", "QCOM", "QRTEA", "QRVO", "RACE", "RAD", "REEMF", "RGR", "RIG", "RIO", "RMBS", "ROKU", "RRC", "S", "SAVE", "SBUX", "SCCO", "SCHW", "SD", "SDC", "SHAK", "SHLDQ", "SHOP", "SINA", "SIRI", "SLB", "SNAP", "SOHU", "SONO", "SPLK", "SPOT", "SQ", "STNE", "STX", "SU", "SWAV", "SWCH", "SWI", "SWN", "SYMC", "T", "TAL", "TDC", "TEVA", "TGT", "TIF", "TLRY", "TM", "TME", "TOL", "TPR", "TPTX", "TRU", "TRUE", "TSLA", "TTD", "TW", "TWLO", "TWTR", "TXN", "UAA", "UBER", "UPS", "UPWK", "USFD", "UUUU", "VICI", "VLO", "VMW", "VRSN", "VVV", "W", "WB", "WDAY", "WDC", "WFC", "WFTIQ", "WHR", "WORK", "WYNN", "X", "YELP", "YETI", "YNDX", "YRD", "YUM", "YUMC", "ZAYO", "ZEUS", "ZG", "ZM", "ZNGA"]
    #Lean noETFs SLICE_n (41)
    myTickers = ["SAVE", "SBUX", "SCCO", "SCHW", "SD", "SDC", "SHAK", "SHLDQ", "SHOP", "SINA", "SIRI", "SLB", "SNAP", "SOHU", "SONO", "SPLK", "SPOT", "SQ", "STNE", "STX", "SU", "SWAV", "SWCH", "SWI", "SWN", "SYMC", "T", "TAL", "TDC", "TEVA", "TGT", "TIF", "TLRY", "TM", "TME", "TOL", "TPR", "TPTX", "TRU", "TRUE", "TSLA"]
    myTickers =["A", "AA"]

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
    #SP500-ES&NQ SLICE_1 and SLICE_2 (230) 
    #myTickers = ["XRAY", "CPB", "L", "TDG", "O", "FBHS", "RHI", "MAC", "CMS", "ANTM", "EQR", "KSU", "BR", "TFX", "CHD", "BBY", "XYL", "SJM", "HRL", "KEYS", "FTNT", "DAL", "A", "GRMN", "BXP", "GPC", "ROP", "WAB", "LIN", "BDX", "AMT", "SHW", "TSN", "NBL", "HUM", "ETN", "WRK", "FCX", "ETFC", "ES", "AEE", "HFC", "AME", "EXR", "EMN", "NUE", "LW", "CB", "LNT", "PPG", "NTRS", "HCA", "ARE", "HES", "DHI", "LEN", "MCO", "MKC", "KMB", "ZION", "EW", "MPC", "ETR", "FRC", "MTD", "EFX", "CLX", "UNM", "RE", "CPRI", "TMO", "DISH", "DRE", "KEY", "DVN", "PSA", "WCG", "MOS", "UHS", "PKI", "FANG", "URI", "CF", "AWK", "EIX", "MGM", "ICE", "LEG", "KIM", "CTL", "QRVO", "ALLE", "DE", "ZBH", "ED", "APA", "CE", "GL", "CBS", "DVA", "GPN", "EL", "ALK", "MSCI", "TXT", "PPL", "RMD", "CMG", "JWN", "COTY", "PNR", "DG", "KSS", "BF.B", "CNC", "PKG", "SLG", "PFG", "AVY", "DTE", "FITB", "SYY", "MCK", "AFL", "AAP", "SNA", "HCP", "EXPD", "LNC", "CHRW", "OKE", "STT", "LUV", "IPGP", "TSS", "HOG", "VTR", "REG", "LYB", "VAR", "NCLH", "IP", "SBAC", "DXC", "TROW", "HP", "FLT", "HBAN", "PNW", "MLM", "NOV", "WU", "JKHY", "AIV", "GWW", "MTB", "HII", "CNP", "PVH", "AEP", "JEC", "PNC", "TRV", "MMC", "FMC", "RJF", "RCL", "HPQ", "HOLX", "IPG", "DGX", "ITW", "NDAQ", "KMX", "WHR", "PLD", "NWL", "TRIP", "NWSA", "ADS", "ROL", "APD", "COO", "NSC", "HIG", "ADM", "HBI", "IR", "EVRG", "MHK", "SRE", "JEF", "MAA", "CMI", "STI", "PBCT", "AON", "AJG", "LKQ", "STZ", "BSX", "EQIX", "ANSS", "TIF", "FIS", "SCHW", "CAG", "UA", "YUM", "ESS", "CXO", "BLL", "NEM", "SYF", "NKTR", "BEN", "SEE", "HAL", "PEG", "MRO", "AKAM", "VIAB", "RSG", "PRU", "TEL", "DLR", "CINF", "NI", "FRT", "VFC"]
       
    #SP500-ES&NQ/100_1
    #myTickers = ["FE", "AWK", "A", "CTVA", "HSY", "TSS", "GLW", "APTV", "CMI", "ETR", "PPL", "HIG", "PH", "ADM", "ESS", "FTV", "PXD", "LYB", "SYF", "CMG", "CLX", "SWK", "MTB", "MKC", "MSCI", "RMD", "BXP", "CHD", "AME", "WY", "RSG", "STT", "FITB", "KR", "CNC", "NTRS", "AEE", "VMC", "HPE", "KEYS", "ROK", "CMS", "RCL", "EFX", "ANSS", "CCL", "AMP", "CINF", "TFX", "ARE", "OMC", "HCP", "DHI", "LH", "KEY", "AJG", "MTD", "COO", "CBRE", "HAL", "EVRG", "AMCR", "MLM", "HES", "K", "EXR", "CFG", "IP", "CPRT", "FANG", "BR", "NUE", "DRI", "FRC", "MKTX", "BBY", "LEN", "WAT", "RF", "AKAM", "CXO", "MAA", "MGM", "CE", "HBAN", "CAG", "CNP", "KMX", "PFG", "XYL", "DGX", "WCG", "UDR", "DOV", "CBOE", "FCX", "HOLX", "GPC", "L"]
    #SP100 (100)
    #myTickers = ["AAPL", "ABBV", "ABT", "ACN", "ADBE", "AGN", "AIG", "ALL", "AMGN", "AMZN", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CELG", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CSCO", "CVS", "CVX", "DD", "DHR", "DIS", "DUK", "EMR", "EXC", "F", "FB", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTN", "SBUX", "SLB", "SO", "SPG", "T", "TGT", "TXN", "UNH", "UNP", "UPS", "USB", "UTX", "V", "VZ", "WBA", "WFC", "WMT", "XOM"]
    #SP100 1/2_a
    #mySymbols = ["AAPL", "ABBV", "ABT", "ACN", "ADBE", "AGN", "AIG", "ALL", "AMGN", "AMZN", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CELG", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CSCO", "CVS", "CVX", "DD", "DHR"]
    #SP&NQ 1/2_a
    #mySymbols = ["AAPL", "ABBV", "ABT", "ACN", "ADBE", "AGN", "AIG", "ALL", "AMGN", "AMZN", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CELG", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CSCO", "CVS", "CVX", "DD", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FB", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTN", "SBUX", "SLB", "SO", "SPG", "T", "TGT", "TXN", "UNH", "UNP"]
    #DOW30 1/2_a
    #mySymbols = ["IBM", "MSFT", "XOM", "MMM", "CVX", "PG", "GS", "HD", "CSCO", "INTC", "PFE", "WBA", "V", "WMT", "UTX", "MCD", "JPM", "NKE", "VZ", "KO", "DIS", "JNJ", "AAPL", "UNH", "MRK", "TRV", "CAT", "AXP", "BA"]
    #mySymbols = ["IBM", "MSFT", "XOM", "MMM"] #, "CVX", "PG", "GS", "HD", "CSCO", "INTC", "PFE", "WBA", "V", "WMT", "UTX"]
  
    
 #Simulation Signals [direction, disableBars, Enabled, signalDisabledBars]
    simDict = {
       "ALL_L": [1,len(myTickers),True,0], "ALL_S": [-1,len(myTickers),True,0],
       "L_Str_": [1,8,False,0], "S_Str_": [-1,8,False,0], 
       "L_Rej_": [1,8,False,0], "S_Rej_": [-1,8,False,0],
       "DB_": [1,8,False,0], "DT_": [-1,8,False,0], 
       "TB_": [1,8,False,0], "TT_": [-1,8,False,0],
       "IHS_": [1,8,False,0], "HS_": [-1,8,False,0]}
    
    exitSignalDict = {
       "L_Str_": 0, 
       "S_Str_": 0, 
       "L_Rej_": 0,
       "S_Rej_": 0}
    
    #AI ----
    loadAI = True
    preproDict = {}
    aiDict = {}
    
    preproDict ["CNNAE_1"] = {
        "enabled": True,
        "Type": "PT",
        "modelClass": "CNN_AE_1",
        "modelParams" : {"inHW": 50, 
                        "inChannels": 1, 
                        "cnnChannels1": 32, 
                        "cnnChannels2": 64, 
                        "outputs": 200, 
                        "hiddenSize": 200, 
                        "hiddenSizeDecode": 200},
        "modelURL": "https://www.dropbox.com/s/eggeeegwtce4zfz/Test-CNN_AE_Model_20200701-09_03_c.txt?dl=1",
        "model": None }

    aiDict["L_CNN1"] = {
        "enabled": False,
        "signalRegex": "ALL",
        "direction": 1,
        "Type" : "PT_CNN",
        "firstTradeHour": 0,
        "lastTradeHour": 24,
        "riskMultiple": 1.00,
        "modelClass": "NN_1",
        "modelParams" : {"featureCount": 0, 
                        "hiddenCount": 0, 
                        "outFeed": "",
                        "softmaxout": False,
                        "outputs": 2},       
        "modelURL": "https://www.dropbox.com/s/mw9pg986o59e2c8/LGB_FX15m_L_Str%26DCH_s_3MM_3MM_FeatSel_2003_Model_20200307-18_14_booster.txt?dl=1",
        "model": None,
        "usePCA": False,
        "pcaURL": '-',
        "pca": None,
        "rawFeatures": "rawFeatures2",
        "threshold": 0.0,
        "features": None,
        "featureFilter": "Feat5_4$|Feat13_2$|Feat13_1$|Feat11_1$|Feat13_0$|Feat11_2$|Feat5_0$|Feat13_3$|Feat17_0$|Feat15_0$|Feat14_0$|Feat15_11$|Feat8_0$|Feat6_4$|Feat6_3$|Feat18_11$|Feat7_0$|Feat11_0$|Feat18_1$|Feat5_2$|Feat8_4$|Feat0_0$|Feat9_0$|Feat11_3$|Feat9_4$|Feat14_1$|Feat15_10$|Feat10_4$|Feat14_2$|Feat13_4$|Feat16_18$|Feat10_0$|Feat6_1$|Feat15_4$|Feat7_3$|Feat15_12$|Feat16_21$|Feat16_1$|Feat16_15$|Feat18_3$|Feat16_17$|Feat16_0$|Feat18_9$|Feat15_5$|Feat0_1$|Feat14_3$|Feat7_1$|Feat10_1$|Feat1_2$|Feat10_2$",
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

        self.vol = MyVolatility(self, self.algo, self.symbol, name='vol', period=200, atr=self.atr1)
        self.algo.RegisterIndicator(self.symbol, self.vol, self.consolidator)
        
        self.gasf1 = MyGASF(self, self.algo, self.symbol, name='gasf1', period=100, atr=self.atr1, benchmarkTicker=None)
        self.algo.RegisterIndicator(self.symbol, self.gasf1, self.consolidator)

        '''Indicators Higher Timeframe'''
        self.atr1_2 = AverageTrueRange(5)
        self.algo.RegisterIndicator(self.symbol, self.atr1_2, self.consolidator_2)
        self.dch2_2 = DonchianChannel(100)
        self.algo.RegisterIndicator(self.symbol, self.dch2_2, self.consolidator_2)

        self.gasf1_2 = MyGASF(self, self.algo, self.symbol, name='gasf1_2', period=50, atr=self.atr1_2, benchmarkTicker=None)
        self.algo.RegisterIndicator(self.symbol, self.gasf1_2, self.consolidator)

        '''Symbol State and Features'''
        #GASF indicator implements FeatureExtraction method

        '''Signals'''
        self.barStrength1 = MyBarStrength(self, self.algo, self.symbol, name='barStrength1', period=10, atr=self.atr1, lookbackLong=2, lookbackShort=2, \
                priceActionMinATRLong=1.5, priceActionMaxATRLong=2.5, priceActionMinATRShort=1.5, priceActionMaxATRShort=2.5, referenceTypeLong='Close', referenceTypeShort='Close')
        self.algo.RegisterIndicator(self.symbol, self.barStrength1, self.consolidator)
        
        self.barRejection1 = MyBarRejection(self, self.algo, self.symbol, name='barRejection1', period=10, atr=self.atr1, lookbackLong=3, lookbackShort=3, \
               rejectionPriceTravelLong=1.75, rejectionPriceTravelShort=1.75, rejectionPriceRangeLong=0.40, rejectionPriceRangeShort=0.40, referenceTypeLong='Close', referenceTypeShort='Close')
        self.algo.RegisterIndicator(self.symbol, self.barRejection1, self.consolidator)
        
        '''Signals and Events string Container'''
        self.signals = ''
        self.events = ''

        '''Disable Bars'''
        self.longDisabledBars = 0
        self.shortDisabledBars = 0
        self.longSimDisabledBars = 0
        self.shortSimDisabledBars = 0
        
        '''Set up Preprocessors'''
        for preprKey, preprObj in self.CL.preproDict.items():
            if self.CL.loadAI and preprObj["enabled"] and preprObj["model"]==None:
                #Loading MODEL
                if preprObj["Type"] == "PT":
                    preprObj["model"] = globals()[preprObj["modelClass"]](**preprObj["modelParams"]).to('cpu')
                    preprObj["model"] = hp3.MyModelLoader.LoadModelTorch(self, preprObj["modelURL"], existingmodel=preprObj["model"])
                    self.algo.Debug(f' Torch MODEL ({preprKey}/{self.CL.strategyCode}) LOADED from url: {preprObj["modelURL"]}')

        '''Set up AI models'''
        self.signalDisabledBars = {}
        for aiKey, aiObj in self.CL.aiDict.items():
            if self.CL.loadAI and aiObj["enabled"] and aiObj["model"]==None:
                #Loading PCA Preprocessor if set
                if aiObj["usePCA"]: 
                    aiObj["pca"] = hp3.MyModelLoader.LoadModelPickled(self, aiObj["pcaURL"])
                    self.algo.Debug(f' PCA ({aiKey}) LOADED from url: {aiObj["pcaURL"]}')
                #Loading MODEL
                if aiObj["Type"] == "PT":
                    aiObj["model"] = globals()[aiObj["modelClass"]](**aiObj["modelParams"]).to('cpu')
                    aiObj["model"] = hp3.MyModelLoader.LoadModelTorch(self, aiObj["modelURL"], existingmodel=aiObj["model"])
                    self.algo.Debug(f' Torch MODEL ({aiKey}/{self.CL.strategyCode}) LOADED from url: {aiObj["modelURL"]}')
                elif aiObj["Type"] == "SK":
                    aiObj["model"] = hp3.MyModelLoader.LoadModelPickled(self, aiObj["modelURL"])
                    self.algo.Debug(f' SKLearn MODEL ({aiKey}/{self.CL.strategyCode}) LOADED from url: {aiObj["modelURL"]}')
                elif aiObj["Type"] == "LGB":
                    #aiObj["model"] = hp3.MyModelLoader.LoadModelLGBtxt(self, aiObj["modelURL"])
                    aiObj["model"] = hp3.MyModelLoader.LoadModelLGB(self, aiObj["modelURL"])
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
            aiObj["signal"] = aiObj["enabled"] and (signalRegex=="ALL" or re.search(signalRegex, self.signals)) and aiObj["firstTradeHour"] <= self.algo.Time.hour and self.algo.Time.hour <= aiObj["lastTradeHour"]
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
            preProcessedFeature1 = self.gasf1.FeatureExtractor(featureType="Close", useGASF=True, picleFeatures=False, preProcessor=self.CL.preproDict["CNNAE_1"]["model"])
            self.algo.MyDebug(f'preProcessedFeature1: {preProcessedFeature1.shape}')
        if loadFeatures1 or (longTriggerSim or shortTriggerSim):
            #self.rawFeatures1 = [self.gasf1.FeatureExtractor(featureType="CULBG", useGAP=gasfSim, picleFeatures=True, intCode=intCode), \
            #                    self.gasf1.FeatureExtractor(featureType="RelativePrice", useGASF=gasfSim, picleFeatures=True, intCode=intCode), \
            #                    self.gasf1.FeatureExtractor(featureType="Volume", useGASF=gasfSim, picleFeatures=True, intCode=intCode), \
            #                    self.gasf1.FeatureExtractor(featureType="Volatility", useGASF=gasfSim, picleFeatures=True, intCode=intCode)]
            self.rawFeatures1 = [self.gasf1.FeatureExtractor(featureType="Close", useGASF=False, picleFeatures=True, intCode=intCode), \
                                 self.gasf1.FeatureExtractor(featureType="ULBG", useGASF=False, picleFeatures=True, intCode=intCode), \
                                 self.gasf1.FeatureExtractor(featureType="Volume", useGASF=False, picleFeatures=True, intCode=intCode), \
                                 self.gasf1.FeatureExtractor(featureType="_B", useGASF=False, picleFeatures=True, intCode=intCode)]

        
        if loadFeatures2:
            self.rawFeatures2 = [self.gasf1.FeatureExtractor(featureType="CULBG", useGASF=True, picleFeatures=False), \
                                self.gasf1.FeatureExtractor(featureType="RelativePrice", useGASF=True, picleFeatures=False), \
                                self.gasf1.FeatureExtractor(featureType="Volume", useGASF=True, picleFeatures=False), \
                                self.gasf1.FeatureExtractor(featureType="Volatility", useGASF=True, picleFeatures=False)]
 
        for aiObj in self.CL.aiDict.values():
            if aiObj["enabled"] and aiObj["signal"]: 
                featureFilter = aiObj["featureFilter"]
                myFeatures = self.algo.myHelpers.UnpackFeatures(getattr(self, aiObj["rawFeatures"]),  featureType=6, featureRegex=featureFilter, reshapeTuple=None)
                customColumnFilters = aiObj["customColumnFilter"]
                aiObj["dfFilterPassed"] = self.algo.myHelpers.FeatureCustomColumnFilter(myFeatures, customColumnFilters=customColumnFilters) if len(customColumnFilters)!=0 else True
                if aiObj["Type"] != "PT_CNN": myFeatures = myFeatures.values
                if aiObj["usePCA"]:
                    myFeatures = aiObj["pca"].transform(myFeatures)
                if aiObj["Type"] == "PT":
                    myFeatures = torch.from_numpy(myFeatures.reshape(-1, aiObj["featureCount"])).float().to('cpu')
                aiObj["features"] = myFeatures

        '''SIGNAL FILTERING
        '''
        longTrigger, shortTrigger = False, False
        longRiskMultiple, shortRiskMultiple = 1.00, 1.00
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
                    longTrigger = True
                    longRiskMultiple = min(1.00,max(aiObj["riskMultiple"],0.00))
                    self.CL.strategyCode = self.CL.strategyCode + "|" + aiKey
                elif trigger and aiObj["direction"]==-1:
                    shortTrigger = True
                    shortRiskMultiple = min(1.00,max(aiObj["riskMultiple"],0.00))
                    self.CL.strategyCode = self.CL.strategyCode + "|" + aiKey
            if aiObj["signal"] and self.signalDisabledBars[aiKey]==0: self.signalDisabledBars[aiKey] = 8

        '''POSITION ENTRY/FLIP (Flip Enabled is checked in EnterPosition_2)
        '''
        self.longDisabledBars = max(self.longDisabledBars-1,0)
        self.shortDisabledBars = max(self.shortDisabledBars-1,0)
        #---LONG POSITION
        if self.posEnabled and self.CL.enableLong and self.longDisabledBars==0 and longTrigger:
            self.algo.myPositionManager.EnterPosition_2(self.symbol, 1, myRiskMultiple=longRiskMultiple)
            if False: self.algo.MyDebug(f' Long Trade: {self.symbol}')
            self.longDisabledBars=3
        #---SHORT POSITION
        elif self.posEnabled and self.CL.enableShort and self.shortDisabledBars==0 and shortTrigger:
            self.algo.myPositionManager.EnterPosition_2(self.symbol, -1, myRiskMultiple=shortRiskMultiple)
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
        simTradeTypes = [0,2] #[0,2,3,4,7,8,10] #[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] 
        simMinMaxTypes = [0,1] #[0,1,2,3]
        lastEntryDate = datetime(year = 3019, month = 10, day = 7)
        if self.CL.simulate and self.posEnabled and longTriggerSim and (self.algo.Time<self.algo.simEndDate or self.algo.simEndDate==None):
            if debugSim: self.algo.MyDebug(f' {self.symbol} Sim call Long: {self.signals}')
            sim = MySIMPosition(self, direction=1, timestamp=self.algo.Time.strftime("%Y%m%d %H:%M"), signal=self.signals, features=myFeatures, simTradeTypes=simTradeTypes, simMinMaxTypes=simMinMaxTypes)
            #use 0 if signalspecific
            self.longSimDisabledBars=0
        if self.CL.simulate and self.posEnabled and shortTriggerSim and (self.algo.Time<self.algo.simEndDate or self.algo.simEndDate==None):
            if debugSim: self.algo.MyDebug(f' {self.symbol} Sim call Short: {self.signals}')
            sim = MySIMPosition(self, direction=-1, timestamp=self.algo.Time.strftime("%Y%m%d %H:%M"), signal=self.signals, features=myFeatures, simTradeTypes=simTradeTypes, simMinMaxTypes=simMinMaxTypes) 
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
class NN_1(nn.Module):
   def __init__(self, features, hiddenSize=None, outFeed='_h3', outputs=2, dropoutRateIn=0.0, dropoutRate=0.0, dropoutRate2=None, softmaxout=False, bn_momentum=0.1):
       super(NN_1, self).__init__()
       self.outputs = outputs
       self.features = features
       self.softmaxout = softmaxout
       if hiddenSize!=None:
           self.hiddenSize = hiddenSize
       else:
           self.hiddenSize = round(1.0*features)
       #https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/ , https://blog.paperspace.com/busting-the-myths-about-batch-normalization/
       #https://pytorch.org/docs/stable/nn.html?highlight=nn%20batchnorm1d#torch.nn.BatchNorm1d
       self.bn0 = nn.BatchNorm1d(num_features=features, momentum=bn_momentum)
       self.layer_In = torch.nn.Linear(features, self.hiddenSize)
       self.bn01 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
       self.layer_h1 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.bn1 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
       self.layer_h2 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.bn2 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
       self.layer_h3 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.bn3 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
       self.layer_h4 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.bn4 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
       self.layer_h5 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.bn5 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
       self.layer_h6 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.bn6 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
       self.layer_h7 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.bn7 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
      #  self.layer_h8 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
      #  self.bn8 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
       self.layer_Out = torch.nn.Linear(self.hiddenSize, self.outputs)
       #https://towardsdatascience.com/simplified-math-behind-dropout-in-deep-learning-6d50f3f47275
       self.L_out = {}
       self.outFeed = outFeed 
       #https://towardsdatascience.com/simplified-math-behind-dropout-in-deep-learning-6d50f3f47275
       self.dropoutRateIn = dropoutRateIn
       self.dropoutRate = dropoutRate
       self.dropoutRate2 = self.dropoutRate if dropoutRate2==None else dropoutRate2
       #https://mlfromscratch.com/activation-functions-explained/ 
       self.MyActivation = [F.relu, F.leaky_relu_, F.selu][2]
       self.useOutBn = True
       if self.useOutBn: self.bn_out = nn.BatchNorm1d(num_features=self.outputs, momentum=bn_momentum)
                     
   def forward(self, x):
       self.L_out['_x'] =  F.dropout(self.bn0(x), p=self.dropoutRateIn, training=self.training)
       #https://datascience.st  ackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks

       self.L_out['_in'] =  F.dropout(self.bn01(self.MyActivation(self.layer_In(self.L_out['_x']))),  p=self.dropoutRate, training=self.training)
       self.L_out['_h1'] =  F.dropout(self.bn1(self.MyActivation(self.layer_h1(self.L_out['_in']))),  p=self.dropoutRate2, training=self.training)
       self.L_out['_h2'] =  F.dropout(self.bn2(self.MyActivation(self.layer_h2(self.L_out['_h1']))),  p=self.dropoutRate2, training=self.training)
       self.L_out['_h3'] =  F.dropout(self.bn3(self.MyActivation(self.layer_h3(self.L_out['_h2']))),  p=self.dropoutRate2, training=self.training)
       self.L_out['_h4'] =  F.dropout(self.bn4(self.MyActivation(self.layer_h4(self.L_out['_h3']))),  p=self.dropoutRate2, training=self.training)
       #self.L_out['_h5'] =  F.dropout(self.bn5(self.MyActivation(self.layer_h5(self.L_out['_h4']))),  p=self.dropoutRate2, training=self.training)
       #self.L_out['_h6'] =  F.dropout(self.bn6(self.MyActivation(self.layer_h6(self.L_out['_h5']))),  p=self.dropoutRate2, training=self.training)
       #self.L_out['_h7'] =  F.dropout(self.bn7(self.MyActivation(self.layer_h7(self.L_out['_h6']))),  p=self.dropoutRate2, training=self.training)
       #self.L_out['_h8'] =  F.dropout(self.bn8(self.MyActivation(self.layer_h8(self.L_out['_h7']))),  p=self.dropoutRate2, training=self.training)
       if self.useOutBn:
         self.L_out['_out'] = self.bn_out(self.MyActivation(self.layer_Out(self.L_out[self.outFeed])))
       else:
         self.L_out['_out'] = self.MyActivation(self.layer_Out(self.L_out[self.outFeed]))
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
       prediction = np.argmax(prediction.data.numpy(), axis=1) #armax returns a tuple
       if len(prediction)==1:
           return prediction[0]
       return prediction
   
   def Predict2(self, x):
       self.eval()
       prediction = self.forward(x).data.numpy()
       return prediction

class CNN_AE_1(nn.Module):
    def __init__(self, inHW, inChannels=1, cnnChannels1=16, cnnChannels2=16, outputs=20, hiddenSize=None, hiddenSizeDecode=None, dropoutRate=0.25, bn_momentum=0.1):
        super(CNN_AE_1, self).__init__()
        self.inHW = inHW
        self.inChannels = inChannels
        self.cnnChannels1 = cnnChannels1
        self.cnnChannels2 = cnnChannels2
        self.cnnFlattenedSize = round(self.inHW/4)**2 * self.cnnChannels2
        self.hiddenSize = hiddenSize
        self.hiddenSizeDecode = hiddenSizeDecode
        self.dropoutRate = dropoutRate
        self.outputs = outputs
        self.kernel_size = [3, 5, 7, 9][1]

        self.MyActivation = [nn.ReLU(), nn.LeakyReLU, nn.SELU][0]

        #ENCODER
        #n/4^2*self.cnnChannels2
        self.CNN = nn.Sequential(
            nn.BatchNorm2d(inChannels),
            nn.Conv2d(inChannels, self.cnnChannels1, kernel_size=self.kernel_size, stride=1, padding=2),
            self.MyActivation,
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.BatchNorm2d(self.cnnChannels1),
            nn.Conv2d(self.cnnChannels1, self.cnnChannels2, kernel_size=self.kernel_size, stride=1, padding=2),
            self.MyActivation,
            nn.MaxPool2d(kernel_size=2, stride=2))
                       
        if self.hiddenSize!=None:
            self.BN1 = nn.BatchNorm1d(num_features=self.cnnFlattenedSize, momentum=bn_momentum)
            self.FC1 = nn.Linear(self.cnnFlattenedSize, self.hiddenSize)
            self.BN2 = nn.BatchNorm1d(num_features=self.hiddenSize, momentum=bn_momentum)
            self.FC2 = nn.Linear(self.hiddenSize, self.outputs)
        else:
            self.BN1 = nn.BatchNorm1d(num_features=self.cnnFlattenedSize, momentum=bn_momentum)
            self.FC1 = nn.Linear(self.cnnFlattenedSize, self.outputs)

        #DECODER
        if self.hiddenSizeDecode!=None:
            self.D_BN1 = nn.BatchNorm1d(num_features=self.outputs, momentum=bn_momentum)
            self.D_NN1 = nn.Linear(self.outputs, self.hiddenSizeDecode)
            self.D_BN2 = nn.BatchNorm1d(num_features=self.hiddenSizeDecode, momentum=bn_momentum)
            self.D_NN2 = nn.Linear(self.hiddenSizeDecode, self.inHW*self.inHW*self.inChannels)
        else:
            self.D_BN1 = nn.BatchNorm1d(num_features=self.outputs, momentum=bn_momentum)
            self.D_NN1 = nn.Linear(self.outputs, self.inHW*self.inHW*self.inChannels)

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
            x = self.MyActivation(self.FC2(self.BN2(F.dropout(x, p=self.dropoutRate, training=self.training))))
        else:
            x = self.MyActivation(self.FC1(self.BN1(F.dropout(x, p=self.dropoutRate, training=self.training))))
        return x

    def Decode(self, x):
        samples = x.size(0) 
        if self.hiddenSizeDecode!=None:
            x = self.MyActivation(self.D_NN1(self.D_BN1(F.dropout(x, p=self.dropoutRate, training=self.training))))
            x = self.MyActivation(self.D_NN2(self.D_BN2(F.dropout(x, p=self.dropoutRate, training=self.training))))
            x = x.view([samples, self.inChannels, self.inHW, self.inHW])
        else:
            x = self.MyActivation(self.D_NN1(self.D_BN1(F.dropout(x, p=self.dropoutRate, training=self.training))))
            x = x.view([samples, self.inChannels, self.inHW, self.inHW])
        return x 
    
    def PreProcess(self, x, returnTorch=False):
        self.eval()
        while len(x.shape)<4:
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
        while len(x.shape)<4:
            x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).float().to('cpu')
        x = self.forward(x)
        if returnTorch:
            x = torch.squeeze(x)
            return x
        x = x.data.numpy()
        x = np.squeeze(x)
        return x

