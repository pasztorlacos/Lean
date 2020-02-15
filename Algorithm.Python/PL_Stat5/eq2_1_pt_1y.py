### <summary>
### 
### </summary>
from QuantConnect import SecurityType, Resolution
from QuantConnect.Indicators import RollingWindow, ExponentialMovingAverage, SimpleMovingAverage, IndicatorExtensions, AverageTrueRange, DonchianChannel, RelativeStrengthIndex, IndicatorDataPoint
from QuantConnect.Data.Consolidators import TradeBarConsolidator, QuoteBarConsolidator
from QuantConnect.Data.Market import IBaseDataBar, TradeBar

from datetime import datetime, timedelta, date
from ta1 import MyRelativePrice, MyPriceNormaliser, MyZigZag, MyDCHState, MyMAState, MySupportResistance, MyPatterns, MyVolatility
from taB1 import MyBarStrength, MyBarRejection
from sim1 import MySIMPosition
import hp3
from pandas import DataFrame
import numpy as np

class Equities2_1():
    file = __file__
    '''
    Strategy Implementation
    '''
    '''STRATEGY SETTINGS'''
    enabled = True
    manageOrderConsistency = True
    simulate = True
    saveFilesFramework = True
    saveFilesSim = True
    strategyCode = "e2_1" #used by order tags and debug
    isEquity = True
    customFillModel = 1
    customSlippageModel = 1
    customFeeModel = 0
    customBuyingPowerModel = 0
    #Resolution
    resolutionMinutes = 60
    maxWarmUpPeriod = 710
    barPeriod =  timedelta(minutes=resolutionMinutes) #TimeSpan.FromMinutes(resolutionMinutes)
    warmupcalendardays = round(7/5*maxWarmUpPeriod/(7*(60/resolutionMinutes))) if isEquity else round(7/5*maxWarmUpPeriod/(24*(60/resolutionMinutes)))
    #Switches
    plotIndicators = False
    #Risk Management
    strategyAllocation = 0.5 #Install can OverWrite it!!!
    maxAbsExposure = 2.0
    maxLongExposure = 2.0 
    maxNetLongExposure = 2.0
    maxShortExposure = -2.0
    maxNetShortExposure = -2.0
    maxSymbolAbsExposure = 0.50
    minSymbolAbsExposure = 0.05
    riskperLongTrade = 0.01
    riskperShortTrade = 0.01
    maxLongVaR = 0.075
    maxShortVaR = 0.075
    maxTotalVaR = 0.1
    mainVaRr = None #it is set by the first instance
    #Entry
    enableLong = False
    enableShort = False
    liquidateLong = False
    liquidateShort = False
    enableFlipLong = True
    enableFlipShort = True
    closeOnTriggerLong = True #works if Flip is didabled
    closeOnTriggerShort = True 
    entryLimitOrderLong = True  #If False: enter the position with Market Order
    entryLimitOrderShort = True
    limitEntryATRLong = 0.01
    limitEntryATRShort = 0.1
    entryTimeInForceLong = timedelta(minutes=5*60)
    entryTimeInForceShort = timedelta(minutes=5*60)
    #Orders
    useTragetsLong = True #TWS Sync is not ready yet to handle useTragets for Foreign Symbols
    useTragetsShort = True
    stopPlacerLong = 7    #0)dch0 1)dch1 2)dch2) 3)dch3 4)minStopATR 5)bars_rw[0] 6)bars_rw[0-1] 7)dc01 8)dch02 else: dch1
    stopPlacerShort = 7
    targetPlacerLong = 3    #1)max(dch2,minPayoff) 2)max(dch3,minPayoff)  else: minPayoff 
    targetPlacerShort = 3    
    minPayOffLong = 3.0
    minPayOffShort = 3.0
    scratchTradeLong = True 
    scratchTradeShort = True 
    stopTrailerLong = 0 #0) no Trail 1)dhc2 2)dch3 3)dch1 4)dch0
    stopTrailerShort = 0 
    targetTrailerLong = 0 #0) no Trail 1)dhc2 
    targetTrailerShort = 0 
    stopATRLong = 0.5
    stopATRShort = 0.25
    minEntryStopATRLong = 2.0    
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
    
    # SP500-ES&NQ/100_2
    #myTickers = ["FE", "AWK", "A", "CTVA", "HSY", "TSS", "GLW", "APTV", "CMI", "ETR", "PPL", "HIG", "PH", "ADM", "ESS", "FTV", "PXD", "LYB", "SYF", "CMG", "CLX", "SWK", "MTB", "MKC", "MSCI", "RMD", "BXP", "CHD", "AME", "WY", "RSG", "STT", "FITB", "KR", "CNC", "NTRS", "AEE", "VMC", "HPE", "KEYS", "ROK", "CMS", "RCL", "EFX", "ANSS", "CCL", "AMP", "CINF", "TFX", "ARE", "OMC", "HCP", "DHI", "LH", "KEY", "AJG", "MTD", "COO", "CBRE", "HAL", "EVRG", "AMCR", "MLM", "HES", "K", "EXR", "CFG", "IP", "CPRT", "FANG", "BR", "NUE", "DRI", "FRC", "MKTX", "BBY", "LEN", "WAT", "RF", "AKAM", "CXO", "MAA", "MGM", "CE", "HBAN", "CAG", "CNP", "KMX", "PFG", "XYL", "DGX", "WCG", "UDR", "DOV", "CBOE", "FCX", "HOLX", "GPC", "L"]
    
    #SP100 (100)
    #myTickers = ["AAPL", "ABBV", "ABT", "ACN", "ADBE", "AGN", "AIG", "ALL", "AMGN", "AMZN", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CELG", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CSCO", "CVS", "CVX", "DD", "DHR", "DIS", "DUK", "EMR", "EXC", "F", "FB", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTN", "SBUX", "SLB", "SO", "SPG", "T", "TGT", "TXN", "UNH", "UNP", "UPS", "USB", "UTX", "V", "VZ", "WBA", "WFC", "WMT", "XOM"]
    #SP&NQ 1/2_b
    #myTickers = ["UPS", "USB", "UTX", "V", "VZ", "WBA", "WFC", "WMT", "XOM", "ATVI", "AMD", "ALXN", "ALGN", "AAL", "ADI", "AMAT", "ASML", "ADSK", "ADP", "BIDU", "BMRN", "AVGO", "CDNS", "CERN", "CHKP", "CTAS", "CTXS", "CTSH", "CSX", "CTRP", "DLTR", "EBAY", "EA", "EXPE", "FAST", "FISV", "FOX", "HAS", "HSIC", "IDXX", "ILMN", "INCY", "INTU", "ISRG", "JBHT", "JD", "KLAC", "LRCX", "LBTYA", "LBTYK", "LULU", "MAR", "MXIM", "MELI", "MCHP", "MU", "MNST", "MYL", "NTAP", "NTES", "NXPI", "ORLY", "PCAR", "PAYX", "REGN", "ROST", "SIRI", "SWKS", "SYMC", "SNPS", "TMUS", "TTWO", "TSLA", "ULTA", "UAL", "VRSN", "VRSK", "VRTX", "WDAY", "WDC", "WLTW", "WYNN", "XEL", "XLNX", "STX", "TSLA", "VRSK", "WYNN", "XLNX"]       
    #SP100 1/2_b
    #myTickers = ["DIS", "DUK", "EMR", "EXC", "F", "FB", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTN", "SBUX", "SLB", "SO", "SPG", "T", "TGT", "TXN", "UNH", "UNP", "UPS", "USB", "UTX", "V", "VZ", "WBA", "WFC", "WMT", "XOM"]
    #DOW30 1/2_b
    #myTickers = ["IBM", "MSFT", "XOM", "MMM", "CVX", "PG", "GS", "HD", "CSCO", "INTC", "PFE", "WBA", "V", "WMT", "UTX", "MCD", "JPM", "NKE", "VZ", "KO", "DIS", "JNJ", "AAPL", "UNH", "MRK", "TRV", "CAT", "AXP", "BA"]
    #myTickers = ["MCD", "JPM","NKE", "VZ", "KO", "DIS", "JNJ", "AAPL", "UNH", "MRK", "TRV", "CAT", "AXP", "BA"]
    
    #PiTrading_ALL
    #myTickers = ["A", "AA", "AABA", "AAL", "AAXN", "ABBV", "ACIA", "ADM", "ADT", "AIG", "AKAM", "AKS", "ALLY", "ALTR", "AMAT", "AMC", "AMCX", "AMD", "AMGN", "AMZN", "AN", "ANF", "ANTM", "AOBC", "APO", "APRN", "ARLO", "ATUS", "ATVI", "AUY", "AVGO", "AVTR", "AWK", "BABA", "BAC", "BAH", "BB", "BBBY", "BBH", "BBY", "BIDU", "BJ", "BKNG", "BLK", "BOX", "BP", "BRK-B", "BSX", "BTU", "BURL", "BX", "BYND", "C", "CAKE", "CARS", "CBOE", "CCJ", "CDLX", "CELG", "CHK", "CHWY", "CIEN", "CLDR", "CLF", "CLNE", "CMCSA", "CME", "CMG", "CMI", "CNDT", "COP", "COST", "COUP", "CPB", "CREE", "CRM", "CRSP", "CRUS", "CRWD", "CSX", "CTRP", "CTSH", "CVS", "DBI", "DBX", "DD", "DE", "DECK", "DELL", "DG", "DIA", "DKS", "DLTR", "DNKN", "DNN", "DO", "DOCU", "DRYS", "DT", "DUK", "EA", "EBAY", "EEM", "ELAN", "EOG", "EQT", "ESTC", "ET", "ETFC", "ETRN", "ETSY", "EWJ", "EXC", "F", "FANG", "FAS", "FAZ", "FB", "FCX", "FDX", "FEYE", "FISV", "FIT", "FIVE", "FLR", "FLT", "FMCC", "FNMA", "FSCT", "FSLR", "FTCH", "FXE", "FXI", "GDDY", "GDX", "GE", "GH", "GLBR", "GLD", "GLW", "GM", "GME", "GNRC", "GOLD", "GOOGL", "GOOS", "GPRO", "GPS", "GRPN", "GRUB", "GSK", "GSKY", "HAL", "HCA", "HCAT", "HIG", "HLF", "HLT", "HOG", "HON", "HPE", "HPQ", "HRI", "HTZ", "IBKR", "ICE", "INFO", "INMD", "IQ", "IQV", "ISRG", "IWM", "IYR", "JBLU", "JCP", "JMIA", "JNPR", "KBR", "KLAC", "KMI", "KMX", "KNX", "KSS", "LC", "LEVI", "LHCG", "LLY", "LN", "LOW", "LULU", "LVS", "LYFT", "MA", "MDLZ", "MDR", "MDY", "MGM", "MLCO", "MNK", "MO", "MOMO", "MRNA", "MRVL", "MS", "MSI", "MU", "MXIM", "NAVI", "NEM", "NET", "NFLX", "NIO", "NOK", "NOV", "NOW", "NTNX", "NTR", "NUAN", "NUE", "NVDA", "NVR", "NVS", "NWSA", "NXPI", "OAS", "OIH", "OKTA", "OPRA", "ORCL", "OXY", "PANW", "PAYX", "PBR", "PCG", "PDD", "PE", "PEP", "PHM", "PINS", "PIR", "PM", "PPH", "PRGO", "PS", "PSTG", "PTON", "PVTL", "PYPL", "QCOM", "QQQ", "QRTEA", "QRVO", "RACE", "RAD", "REEMF", "RGR", "RIG", "RIO", "RMBS", "ROKU", "RRC", "RSX", "RTH", "S", "SAVE", "SBUX", "SCCO", "SCHW", "SD", "SDC", "SDS", "SHAK", "SHLDQ", "SHOP", "SINA", "SIRI", "SLB", "SLV", "SMH", "SNAP", "SOHU", "SONO", "SPLK", "SPOT", "SPY", "SQ", "STNE", "STX", "SU", "SWAV", "SWCH", "SWI", "SWN", "SYMC", "T", "TAL", "TDC", "TEVA", "TGT", "TIF", "TLRY", "TLT", "TM", "TME", "TNA", "TOL", "TPR", "TPTX", "TRU", "TRUE", "TSLA", "TTD", "TW", "TWLO", "TWTR", "TXN", "TZA", "UAA", "UBER", "UNG", "UPS", "UPWK", "USFD", "USO", "UUUU", "VICI", "VLO", "VMW", "VRSN", "VVV", "VXX", "W", "WB", "WDAY", "WDC", "WFC", "WFTIQ", "WHR", "WORK", "WYNN", "X", "XLC", "XLE", "XLF", "XLU", "XLV", "YELP", "YETI", "YNDX", "YRD", "YUM", "YUMC", "ZAYO", "ZEUS", "ZG", "ZM", "ZNGA", "ZS", "ZUO"]
    #PiTrading_1_2
    #myTickers = ["A", "AA", "AABA", "AAL", "AAXN", "ABBV", "ACIA", "ADM", "ADT", "AIG", "AKAM", "AKS", "ALLY", "ALTR", "AMAT", "AMC", "AMCX", "AMD", "AMGN", "AMZN", "AN", "ANF", "ANTM", "AOBC", "APO", "APRN", "ARLO", "ATUS", "ATVI", "AUY", "AVGO", "AVTR", "AWK", "BABA", "BAC", "BAH", "BB", "BBBY", "BBH", "BBY", "BIDU", "BJ", "BKNG", "BLK", "BOX", "BP", "BRK-B", "BSX", "BTU", "BURL", "BX", "BYND", "C", "CAKE", "CARS", "CBOE", "CCJ", "CDLX", "CELG", "CHK", "CHWY", "CIEN", "CLDR", "CLF", "CLNE", "CMCSA", "CME", "CMG", "CMI", "CNDT", "COP", "COST", "COUP", "CPB", "CREE", "CRM", "CRSP", "CRUS", "CRWD", "CSX", "CTRP", "CTSH", "CVS", "DBI", "DBX", "DD", "DE", "DECK", "DELL", "DG", "DIA", "DKS", "DLTR", "DNKN", "DNN", "DO", "DOCU", "DRYS", "DT", "DUK", "EA", "EBAY", "EEM", "ELAN", "EOG", "EQT", "ESTC", "ET", "ETFC", "ETRN", "ETSY", "EWJ", "EXC", "F", "FANG", "FAS", "FAZ", "FB", "FCX", "FDX", "FEYE", "FISV", "FIT", "FIVE", "FLR", "FLT", "FMCC", "FNMA", "FSCT", "FSLR", "FTCH", "FXE", "FXI", "GDDY", "GDX", "GE", "GH", "GLBR", "GLD", "GLW", "GM", "GME", "GNRC", "GOLD", "GOOGL", "GOOS", "GPRO", "GPS", "GRPN", "GRUB", "GSK", "GSKY", "HAL", "HCA", "HCAT", "HIG", "HLF", "HLT", "HOG", "HON", "HPE", "HPQ", "HRI", "HTZ", "IBKR", "ICE", "INFO", "INMD", "IQ", "IQV", "ISRG", "IWM", "IYR", "JBLU", "JCP", "JMIA", "JNPR", "KBR", "KLAC", "KMI"]
    #PiTrading_2_2
    #myTickers = ["KMX", "KNX", "KSS", "LC", "LEVI", "LHCG", "LLY", "LN", "LOW", "LULU", "LVS", "LYFT", "MA", "MDLZ", "MDR", "MDY", "MGM", "MLCO", "MNK", "MO", "MOMO", "MRNA", "MRVL", "MS", "MSI", "MU", "MXIM", "NAVI", "NEM", "NET", "NFLX", "NIO", "NOK", "NOV", "NOW", "NTNX", "NTR", "NUAN", "NUE", "NVDA", "NVR", "NVS", "NWSA", "NXPI", "OAS", "OIH", "OKTA", "OPRA", "ORCL", "OXY", "PANW", "PAYX", "PBR", "PCG", "PDD", "PE", "PEP", "PHM", "PINS", "PIR", "PM", "PPH", "PRGO", "PS", "PSTG", "PTON", "PVTL", "PYPL", "QCOM", "QQQ", "QRTEA", "QRVO", "RACE", "RAD", "REEMF", "RGR", "RIG", "RIO", "RMBS", "ROKU", "RRC", "RSX", "RTH", "S", "SAVE", "SBUX", "SCCO", "SCHW", "SD", "SDC", "SDS", "SHAK", "SHLDQ", "SHOP", "SINA", "SIRI", "SLB", "SLV", "SMH", "SNAP", "SOHU", "SONO", "SPLK", "SPOT", "SPY", "SQ", "STNE", "STX", "SU", "SWAV", "SWCH", "SWI", "SWN", "SYMC", "T", "TAL", "TDC", "TEVA", "TGT", "TIF", "TLRY", "TLT", "TM", "TME", "TNA", "TOL", "TPR", "TPTX", "TRU", "TRUE", "TSLA", "TTD", "TW", "TWLO", "TWTR", "TXN", "TZA", "UAA", "UBER", "UNG", "UPS", "UPWK", "USFD", "USO", "UUUU", "VICI", "VLO", "VMW", "VRSN", "VVV", "VXX", "W", "WB", "WDAY", "WDC", "WFC", "WFTIQ", "WHR", "WORK", "WYNN", "X", "XLC", "XLE", "XLF", "XLU", "XLV", "YELP", "YETI", "YNDX", "YRD", "YUM", "YUMC", "ZAYO", "ZEUS", "ZG", "ZM", "ZNGA", "ZS", "ZUO"]
    #PiTrading_1_3
    #myTickers = ["A", "AA", "AABA", "AAL", "AAXN", "ABBV", "ACIA", "ADM", "ADT", "AIG", "AKAM", "AKS", "ALLY", "ALTR", "AMAT", "AMC", "AMCX", "AMD", "AMGN", "AMZN", "AN", "ANF", "ANTM", "AOBC", "APO", "APRN", "ARLO", "ATUS", "ATVI", "AUY", "AVGO", "AVTR", "AWK", "BABA", "BAC", "BAH", "BB", "BBBY", "BBH", "BBY", "BIDU", "BJ", "BKNG", "BLK", "BOX", "BP", "BRK-B", "BSX", "BTU", "BURL", "BX", "BYND", "C", "CAKE", "CARS", "CBOE", "CCJ", "CDLX", "CELG", "CHK", "CHWY", "CIEN", "CLDR", "CLF", "CLNE", "CMCSA", "CME", "CMG", "CMI", "CNDT", "COP", "COST", "COUP", "CPB", "CREE", "CRM", "CRSP", "CRUS", "CRWD", "CSX", "CTRP", "CTSH", "CVS", "DBI", "DBX", "DD", "DE", "DECK", "DELL", "DG", "DIA", "DKS", "DLTR", "DNKN", "DNN", "DO", "DOCU", "DRYS", "DT", "DUK", "EA", "EBAY", "EEM", "ELAN", "EOG", "EQT", "ESTC", "ET", "ETFC", "ETRN", "ETSY", "EWJ", "EXC", "F", "FANG", "FAS", "FAZ", "FB", "FCX", "FDX"]
    
    #PiTrading_1-75
    #myTickers = ["A", "AA", "AABA", "AAL", "AAXN", "ABBV", "ACIA", "ADM", "ADT", "AIG", "AKAM", "AKS", "ALLY", "ALTR", "AMAT", "AMC", "AMCX", "AMD", "AMGN", "AMZN", "AN", "ANF", "ANTM", "AOBC", "APO", "APRN", "ARLO", "ATUS", "ATVI", "AUY", "AVGO", "AVTR", "AWK", "BABA", "BAC", "BAH", "BB", "BBBY", "BBH", "BBY", "BIDU", "BJ", "BKNG", "BLK", "BOX", "BP", "BRK-B", "BSX", "BTU", "BURL", "BX", "BYND", "C", "CAKE", "CARS", "CBOE", "CCJ", "CDLX", "CELG", "CHK", "CHWY", "CIEN", "CLDR", "CLF", "CLNE", "CMCSA", "CME", "CMG", "CMI", "CNDT", "COP", "COST", "COUP", "CPB", "CREE"]    
    #PiTrading_76-150
    #myTickers = ["CRM", "CRSP", "CRUS", "CRWD", "CSX", "CTRP", "CTSH", "CVS", "DBI", "DBX", "DD", "DE", "DECK", "DELL", "DG", "DIA", "DKS", "DLTR", "DNKN", "DNN", "DO", "DOCU", "DRYS", "DT", "DUK", "EA", "EBAY", "EEM", "ELAN", "EOG", "EQT", "ESTC", "ET", "ETFC", "ETRN", "ETSY", "EWJ", "EXC", "F", "FANG", "FAS", "FAZ", "FB", "FCX", "FDX", "FEYE", "FISV", "FIT", "FIVE", "FLR", "FLT", "FMCC", "FNMA", "FSCT", "FSLR", "FTCH", "FXE", "FXI", "GDDY", "GDX", "GE", "GH", "GLBR", "GLD", "GLW", "GM", "GME", "GNRC", "GOLD", "GOOGL", "GOOS", "GPRO", "GPS", "GRPN", "GRUB"]
    #PiTrading_151-225
    myTickers = ["GSK", "GSKY", "HAL", "HCA", "HCAT", "HIG", "HLF", "HLT", "HOG", "HON", "HPE", "HPQ", "HRI", "HTZ", "IBKR", "ICE", "INFO", "INMD", "IQ", "IQV", "ISRG", "IWM", "IYR", "JBLU", "JCP", "JMIA", "JNPR", "KBR", "KLAC", "KMI", "KMX", "KNX", "KSS", "LC", "LEVI", "LHCG", "LLY", "LN", "LOW", "LULU", "LVS", "LYFT", "MA", "MDLZ", "MDR", "MDY", "MGM", "MLCO", "MNK", "MO", "MOMO", "MRNA", "MRVL", "MS", "MSI", "MU", "MXIM", "NAVI", "NEM", "NET", "NFLX", "NIO", "NOK", "NOV", "NOW", "NTNX", "NTR", "NUAN", "NUE", "NVDA", "NVR", "NVS", "NWSA", "NXPI", "OAS"]
    #PiTrading_226-300
    #myTickers = ["OIH", "OKTA", "OPRA", "ORCL", "OXY", "PANW", "PAYX", "PBR", "PCG", "PDD", "PE", "PEP", "PHM", "PINS", "PIR", "PM", "PPH", "PRGO", "PS", "PSTG", "PTON", "PVTL", "PYPL", "QCOM", "QQQ", "QRTEA", "QRVO", "RACE", "RAD", "REEMF", "RGR", "RIG", "RIO", "RMBS", "ROKU", "RRC", "RSX", "RTH", "S", "SAVE", "SBUX", "SCCO", "SCHW", "SD", "SDC", "SDS", "SHAK", "SHLDQ", "SHOP", "SINA", "SIRI", "SLB", "SLV", "SMH", "SNAP", "SOHU", "SONO", "SPLK", "SPOT", "SPY", "SQ", "STNE", "STX", "SU", "SWAV", "SWCH", "SWI", "SWN", "SYMC", "T", "TAL", "TDC", "TEVA", "TGT", "TIF"]   
    #PiTrading_301-
    #myTickers = ["TLRY", "TLT", "TM", "TME", "TNA", "TOL", "TPR", "TPTX", "TRU", "TRUE", "TSLA", "TTD", "TW", "TWLO", "TWTR", "TXN", "TZA", "UAA", "UBER", "UNG", "UPS", "UPWK", "USFD", "USO", "UUUU", "VICI", "VLO", "VMW", "VRSN", "VVV", "VXX", "W", "WB", "WDAY", "WDC", "WFC", "WFTIQ", "WHR", "WORK", "WYNN", "X", "XLC", "XLE", "XLF", "XLU", "XLV", "YELP", "YETI", "YNDX", "YRD", "YUM", "YUMC", "ZAYO", "ZEUS", "ZG", "ZM", "ZNGA", "ZS", "ZUO"]  
        
    #myTickers = ["AA", "AAL", "AMAT"]
    myTickers = ["A"]
    
    #AI
    loadAI = True
    modelUrlLong="https://www.dropbox.com/s/kivq9d1ekxug5hk/RF_1_Model_Stat4_L.txt?dl=1"
    modelLoaderLong=None #self.CL.modelLoaderLong.model would be the model object
    modelUrlShort="https://www.dropbox.com/s/spcfxup7dsb46je/RF_1_Model_Stat4_S.txt?dl=1"
    modelLoaderShort=None #self.CL.modelLoaderShort.model would be the model object

    def __init__(self, caller, symbol, var):
        self.CL = self.__class__
        self.algo = caller
        self.symbol = symbol
        self.security = self.algo.Securities[symbol]
        self.var = var
        '''Consolidator'''
        self.consolidator = TradeBarConsolidator(self.CL.barPeriod) if symbol.SecurityType == SecurityType.Equity else QuoteBarConsolidator(self.CL.barPeriod)
        #self.consolidator = TradeBarConsolidator(1) if symbol.SecurityType == SecurityType.Equity else QuoteBarConsolidator(1)
        self.consolidator.DataConsolidated += self.OnDataConsolidated
        self.algo.SubscriptionManager.AddConsolidator(symbol, self.consolidator)
        '''Symbol Data''' 
        self.bars_rw = RollingWindow[IBaseDataBar](100)
        self.barTimeDifference = timedelta(0)
        self.posEnabled = True
        self.entryEnabled = False
        self.blockOrderCheck = False
        self.fillReleaseTime = self.algo.Time - timedelta(hours=20)
        self.fromTWS = False
        self.stateUpdateList = []
        '''Indicators and Rolling Windows'''
        self.sma1 = SimpleMovingAverage (70)
        self.algo.RegisterIndicator(self.symbol, self.sma1, self.consolidator)
        self.sma2 = SimpleMovingAverage (140)
        self.algo.RegisterIndicator(self.symbol, self.sma2, self.consolidator)
        self.sma3 = SimpleMovingAverage (280)
        self.algo.RegisterIndicator(self.symbol, self.sma3, self.consolidator)
        self.sma4 = SimpleMovingAverage (420)
        self.algo.RegisterIndicator(self.symbol, self.sma4, self.consolidator)
        self.atr1 = AverageTrueRange(15)
        self.algo.RegisterIndicator(self.symbol, self.atr1, self.consolidator)
        self.atr2 = AverageTrueRange(70)
        self.algo.RegisterIndicator(self.symbol, self.atr2, self.consolidator)
        
        self.dch_s = DonchianChannel(7)
        self.algo.RegisterIndicator(self.symbol, self.dch_s, self.consolidator)
        self.dch0 = DonchianChannel(3)
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
        
        self.rsi1 = RelativeStrengthIndex(35)
        self.algo.RegisterIndicator(self.symbol, self.rsi1, self.consolidator)
        
        '''Symbol State and Features'''
        self.longDisabledBars = 0
        self.shortDisabledBars = 0
        self.longSimDisabledBars = 0
        self.shortSimDisabledBars = 0

        self.state_sma1 = MyMAState (self, self.sma1, self.sma2)
        self.state_sma2 = MyMAState (self, self.sma2, self.sma3)
        self.state_sma3 = MyMAState (self, self.sma3, self.sma4)
        self.state_sma4 = MyMAState (self, self.sma4)

        self.state_dch_s = MyDCHState (self, self.dch_s, self.dch2)
        self.state_dch0 = MyDCHState (self, self.dch0, self.dch1)
        self.state_dch1 = MyDCHState (self, self.dch1, self.dch2)
        self.state_dch2 = MyDCHState (self, self.dch2, self.dch3)
        self.state_dch3 = MyDCHState (self, self.dch3, self.dch4)
        self.state_dch4 = MyDCHState (self, self.dch4)

        self.myRelativePrice = MyRelativePrice(self, self.algo, self.symbol, "RelPrice", 100, self.atr1)
        self.algo.RegisterIndicator(self.symbol, self.myRelativePrice, self.consolidator)
        
        self.zz1 = MyZigZag(self, self.algo, self.symbol, name='zz1', period=100, atr=self.atr1)
        self.algo.RegisterIndicator(self.symbol, self.zz1, self.consolidator)
        
        self.vol = MyVolatility(self, self.algo, self.symbol, name='vol_', period=110, atr=self.atr1)
        self.algo.RegisterIndicator(self.symbol, self.vol, self.consolidator)

        #self.priceNormaliser = MyPriceNormaliser(self, self.algo, self.symbol, "Normaliser", 100)
        #self.algo.RegisterIndicator(self.symbol, self.priceNormaliser, self.consolidator)

        self.barStrength1 = MyBarStrength(self, self.algo, self.symbol, name='barStrength1', period=10, atr=self.atr1, lookbackLong=2, lookbackShort=2, \
                priceActionMinATRLong=1.0, priceActionMaxATRLong=2.0, priceActionMinATRShort=1.0, priceActionMaxATRShort=2.0, referenceTypeLong='Close', referenceTypeShort='Close')
        self.algo.RegisterIndicator(self.symbol, self.barStrength1, self.consolidator)
        
        self.barRejection1 = MyBarRejection(self, self.algo, self.symbol, name='barRejection1', period=10, atr=self.atr1, lookbackLong=3, lookbackShort=3, \
               rejectionPriceTravelLong=2.0, rejectionPriceTravelShort=2.0, rejectionPriceRangeLong=1.0, rejectionPriceRangeShort=1.0, referenceTypeLong='Close', referenceTypeShort='Close')
        self.algo.RegisterIndicator(self.symbol, self.barRejection1, self.consolidator)
        
        '''Set up AI model if any'''
        if self.CL.loadAI and self.CL.modelUrlLong!="" and self.CL.modelLoaderLong==None:
            self.CL.modelLoaderLong = hp3.MyModelLoader(self.algo, loadtype=1, url1=self.CL.modelUrlLong, printSummary=True)
        if self.CL.loadAI and self.CL.modelUrlShort!="" and self.CL.modelLoaderShort==None:
            self.CL.modelLoaderShort = hp3.MyModelLoader(self.algo, loadtype=1, url1=self.CL.modelUrlShort, printSummary=True)

        #SET FILES TO BE SAVED AT THE END OF SIMULATION
        if self.CL.simulate:
            if self.CL.saveFilesFramework:        
                #Framework specific filelist to be saved
                MySIMPosition.saveFiles.append([self.algo.myPositionManager.CL.file, 'pmFile'])
                MySIMPosition.saveFiles.append([self.algo.myPositionManagerB.CL.file, 'pmBFile'])
                MySIMPosition.saveFiles.append([self.algo.myVaR.CL.file, 'varFile'])
                MySIMPosition.saveFiles.append([self.algo.myHelpers.CL.file, 'hpFile'])
            if self.CL.saveFilesSim: 
                #Simulation Specific Files    
                MySIMPosition.saveFiles.append([self.algo.__class__.file, 'algoFile'])
                MySIMPosition.saveFiles.append([self.CL.file, 'strategyFile'])
                MySIMPosition.saveFiles.append([MySIMPosition.file, 'simFile'])
                MySIMPosition.saveFiles.append([MyRelativePrice.file, 'ta1File'])
                MySIMPosition.saveFiles.append([MyBarStrength.file, 'taB1File'])
 
    '''
    On Consolidated
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

        if not self.CL.enabled or self.algo.IsWarmingUp: return

        if not self.IsReady() or not self.WasJustUpdated(self.algo.Time) or not self.posEnabled: 
            return

        '''VOLATILITY CUTOFF
        '''
        volatility = self.vol.atrVolatility[0]
        volatilityNorm = self.vol.atrNormVolatility[0]
        volatilityChange = self.vol.atrVolatility[0]-self.vol.atrVolatility[10]
        #self.algo.MyDebug(' ' + str(volatility) + '  ' + str(volatilityChange)) 
        
        volatilityLimitLong = 1.0 
        volatilityLimitShort = 1.15
        
        volatilityCutOffLong = 1.2 
        volatilityCutOffShort = 1.0

        if False and volatility > volatilityCutOffLong and self.algo.Portfolio[self.symbol].Quantity > 0:
            self.algo.myPositionManagerB.LiquidatePosition(self.symbol, "VolCutOff", "VolCutOff")
        if False and volatility < volatilityCutOffShort and self.algo.Portfolio[self.symbol].Quantity < 0:
            self.algo.myPositionManagerB.LiquidatePosition(self.symbol, "VolCutOff", "VolCutOff")
       
        '''TRADE SIGNAL
        '''
        longSignal = False
        shortSignal = False
        if self.posEnabled:
            #LONG SIGNAL
            if  True and self.algo.Time.hour < 15 \
                    and (False or self.barStrength1.Value == 1 or self.barRejection1.Value == 1) \
                    and (True or self.barStrength1.Value == 1) \
                    and (True or self.barRejection1.Value == 1) \
                    and (True or self.zz1.patterns.doubleBottom or self.zz1.patterns.tripleBottom):
                longSignal = True
                mySignalLong = 'L'
                if self.barStrength1.Value: mySignalLong += '-Str_'+str(self.barStrength1.priceActionMinATRLong)+'_'+str(self.barStrength1.priceActionMaxATRLong)
                if self.barRejection1.Value: mySignalLong += '-Rej_'+str(self.barRejection1.rejectionPriceTravelLong)+'_'+str(self.barRejection1.rejectionPriceRangeLong)
                if self.zz1.patterns.doubleBottom: mySignalLong += '-DB'
                if self.zz1.patterns.tripleBottom: mySignalLong += '-TB'
            #SHORT SIGNAL
            if True and self.algo.Time.hour < 15 \
                    and (False or self.barStrength1.Value == -1 or self.barRejection1.Value == -1) \
                    and (True or self.barStrength1.Value == -1) \
                    and (True or self.barRejection1.Value == -1) \
                    and (True or self.zz1.patterns.doubleTop or self.zz1.patterns.tripleTop):
                shortSignal = True
                mySignalShort = 'S'
                if self.barStrength1.Value==-1: mySignalShort += '-Str_'+str(self.barStrength1.priceActionMinATRShort)+'_'+str(self.barStrength1.priceActionMaxATRShort)  
                if self.barRejection1.Value==-1: mySignalShort += '-Rej_'+str(self.barRejection1.rejectionPriceTravelShort)+'_'+str(self.barRejection1.rejectionPriceRangeShort) 
                if self.zz1.patterns.doubleTop: mySignalShort += '-DT'
                if self.zz1.patterns.tripleTop: mySignalShort += '-TT'
        
        '''FEATURES
        '''
        if longSignal or shortSignal:
            #myFeatures = self.priceNormaliser.FeatureExtractor(onlyClose=False, icludeVolume=False, includeRelative=True, periods=10000, multiList=True)
            #myFeatures = [self.state_sma1.FeatureExtractor(Type=1), self.state_sma1.FeatureExtractor(Type=2), self.state_sma1.FeatureExtractor(Type=3), self.state_sma1.FeatureExtractor(Type=4), self.state_sma1.FeatureExtractor(Type=5)]
            #myFeatures = [self.state_dch1.FeatureExtractor(Type=1), self.state_dch1.FeatureExtractor(Type=2), self.state_dch1.FeatureExtractor(Type=3), self.state_dch1.FeatureExtractor(Type=4), self.state_dch1.FeatureExtractor(Type=5)]
            #myFeatures = [self.zz1.FeatureExtractor(listLen=10, Type=1), self.zz1.FeatureExtractor(listLen=10, Type=2)]
            #myFeatures = [self.vol.FeatureExtractor(Type=1), self.vol.FeatureExtractor(Type=2)]
            #myFeatures = [self.algo.mySymbolDict[self.algo.benchmarkSymbol].vol.FeatureExtractor(Type=1), self.algo.mySymbolDict[self.algo.benchmarkSymbol].vol.FeatureExtractor(Type=2)]
            #myFeatures = self.barStrength1.FeatureExtractor(Type=2)
            #myFeatures = self.barRejection1.FeatureExtractor(Type=1)
            #myFeatures = self.barRejection1.FeatureExtractor(Type=2)
            #myFeatures = [self.barStrength1.Value, self.barRejection1.Value]
            #relativePerformance = self.myBars.RelativePerformance(20,  self.algo.mySymbolDict[self.algo.benchmarkSymbol].myBars)))
        
            #myFeatures_sk = [self.state_sma1.FeatureExtractor(Type=1), self.state_sma2.FeatureExtractor(Type=1), self.state_sma3.FeatureExtractor(Type=1), \
            #                self.state_dch0.FeatureExtractor(Type=6), self.state_dch1.FeatureExtractor(Type=6), self.state_dch2.FeatureExtractor(Type=6), self.state_dch3.FeatureExtractor(Type=6), \
            #                self.algo.mySymbolDict[self.algo.benchmarkSymbol].vol.FeatureExtractor(Type=4)]
        
            myFeatures = [self.state_sma1.FeatureExtractor(Type=1), self.state_sma2.FeatureExtractor(Type=1), self.state_sma3.FeatureExtractor(Type=1), \
                            self.state_dch0.FeatureExtractor(Type=6), self.state_dch1.FeatureExtractor(Type=6), self.state_dch2.FeatureExtractor(Type=6), self.state_dch3.FeatureExtractor(Type=6), \
                            self.myRelativePrice.FeatureExtractor(Type=1, normalizationType=1, lookbacklist=[3,7,14,35,70], featureMask=[0,0,1,1]), \
                            self.vol.FeatureExtractor(Type=5, lookbacklist=[15,35,70,100]), \
                            self.zz1.FeatureExtractor(listLen=10, Type=1), self.zz1.FeatureExtractor(listLen=10, Type=2)]
            #myFeatures = [self.vol.FeatureExtractor(Type=5, lookbacklist=[15,35,70,100])]
            #myFeatures = [self.zz1.FeatureExtractor(listLen=10, Type=1)]
            #myFeatures = [self.zz1.FeatureExtractor(listLen=10, Type=2)]
            #myFeatures = [self.myRelativePrice.FeatureExtractor(Type=1, normalizationType=1, lookbacklist=[3,7,14,35,70], featureMask=[0,0,1,1])]

            myFeaturesLong = self.algo.myHelpers.UnpackFeatures(myFeatures,  featureType=1, myRegex='Feat', reshapeTuple=None)
            myFeaturesShort = myFeaturesLong
        
        '''SIGNAL FILTERING
        '''
        longTrigger = False
        shortTrigger = False
        if longSignal and False:
            longTrigger = self.CL.modelLoaderLong.model.predict(myFeaturesLong)
            #self.algo.MyDebug('Long prediction:' + str(longTrigger))
        elif shortSignal and False:
            shortTrigger = self.CL.modelLoaderShort.model.predict(myFeaturesShort)
            #self.algo.MyDebug('Short prediction:' + str(shortTrigger))
      
        '''POSITION ENTRY/FLIP/CLOSE
        '''
        # pendingEntries = self.algo.myPositionManager.CalculatePendingEntry(self.symbol)
        # if pendingEntries['_SP']<=0: longTrigger = True
        # if pendingEntries['_SP']>0: shortTrigger = True

        self.longDisabledBars = max(self.longDisabledBars-1,0)
        self.shortDisabledBars = max(self.shortDisabledBars-1,0)
        #---LONG POSITION
        if self.posEnabled and self.CL.enableLong and self.longDisabledBars==0 and longTrigger:
            self.algo.myPositionManager.EnterPosition_2(self.symbol, 1)
            self.longDisabledBars=3
        #---SHORT POSITION
        elif self.posEnabled and self.CL.enableShort and self.shortDisabledBars==0 and shortTrigger:
            self.algo.myPositionManager.EnterPosition_2(self.symbol, -1)
            self.shortDisabledBars=3

        '''SIMULATION CALL
        '''
        self.longSimDisabledBars=max(self.longSimDisabledBars-1,0)
        self.shortSimDisabledBars=max(self.shortSimDisabledBars-1,0)
        simTradeTypes=[0,2,3,4,7,8] #[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] 
        simMinMaxTypes=[0,1] #[0,1,2,3]
        lastEntryDate = datetime(year = 3019, month = 10, day = 7)
        if self.CL.simulate and longSignal and self.longSimDisabledBars==0 and self.algo.Time<=lastEntryDate:
            sim = MySIMPosition(self, direction=1, timestamp=self.algo.Time.strftime("%Y%m%d %H:%M"), signal=mySignalLong, features=myFeatures, simTradeTypes=simTradeTypes, simMinMaxTypes=simMinMaxTypes)
            self.longSimDisabledBars=8
        if self.CL.simulate and shortSignal and self.shortSimDisabledBars==0 and self.algo.Time<=lastEntryDate:
            sim = MySIMPosition(self, direction=-1, timestamp=self.algo.Time.strftime("%Y%m%d %H:%M"), signal=mySignalShort, features=myFeatures, simTradeTypes=simTradeTypes, simMinMaxTypes=simMinMaxTypes) 
            self.shortSimDisabledBars=8
        return

    '''
    Update Status
    '''
    def IsReady(self):
        indiacatorsReady = self.bars_rw.IsReady and self.dch4.IsReady
        if False and indiacatorsReady:
            self.algo.MyDebug(" Consolidation Is Ready " + str(self.bars_rw.IsReady) + str(self.dch2.IsReady) + str(self.WasJustUpdated(self.algo.Time)))
        return indiacatorsReady
    
    def WasJustUpdated(self, currentTime):
        return self.bars_rw.Count > 0 and (currentTime - self.barTimeDifference - self.bars_rw[0].EndTime) < timedelta(seconds=10) \
                and (currentTime - self.barTimeDifference - self.bars_rw[0].EndTime).total_seconds() > timedelta(seconds=-10).total_seconds()