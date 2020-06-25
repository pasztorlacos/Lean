### <summary>
### Helpers
### 
### </summary>
from QuantConnect.Orders import *
from QuantConnect.Orders.Fills import *
from QuantConnect.Orders.Fees import *
import tensorflow as tf
from QuantConnect.Orders import OrderStatus
from QuantConnect import Resolution, SecurityType

#import math
from math import log
#import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow
import json
import pickle
import codecs
import tempfile
import os
import io
import torch
import operator

#from pm3 import MyPositionManager
#from pmB3 import MyPositionManagerB
from var31 import MyVaR
import lightgbm
import sklearn

class MyHelpers:
    '''
    Commonly used functionality
    '''
    file = __file__
    '''
    SYMBOL LISTS
    '''
    #a) For quick Equity Debug (AAPL R735QTJ8XC9X)
    # ["AAPL" ,"AES", "WMT"]
    #b) DOW30 (29 excl. DOW) and 1/2 and 1/2
    # ["IBM", "MSFT", "XOM", "MMM", "CVX", "PG", "GS", "HD", "CSCO", "INTC", "PFE", "WBA", "V", "WMT", "UTX", "MCD", "JPM", "NKE", "VZ", "KO", "DIS", "JNJ", "AAPL", "UNH", "MRK", "TRV", "CAT", "AXP", "BA"]
    # ["IBM", "MSFT", "XOM", "MMM", "CVX", "PG", "GS", "HD", "CSCO", "INTC", "PFE", "WBA", "V", "WMT", "UTX"]
    # ["MCD", "JPM","NKE", "VZ", "KO", "DIS", "JNJ", "AAPL", "UNH", "MRK", "TRV", "CAT", "AXP", "BA"]
    #c) SP100 (100) and 1/2 and 1/2
    # ["AAPL", "ABBV", "ABT", "ACN", "ADBE", "AGN", "AIG", "ALL", "AMGN", "AMZN", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CELG", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CSCO", "CVS", "CVX", "DD", "DHR", "DIS", "DUK", "EMR", "EXC", "F", "FB", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTN", "SBUX", "SLB", "SO", "SPG", "T", "TGT", "TXN", "UNH", "UNP", "UPS", "USB", "UTX", "V", "VZ", "WBA", "WFC", "WMT", "XOM"]
    # ["AAPL", "ABBV", "ABT", "ACN", "ADBE", "AGN", "AIG", "ALL", "AMGN", "AMZN", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CELG", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CSCO", "CVS", "CVX", "DD", "DHR"]
    # ["DIS", "DUK", "EMR", "EXC", "F", "FB", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTN", "SBUX", "SLB", "SO", "SPG", "T", "TGT", "TXN", "UNH", "UNP", "UPS", "USB", "UTX", "V", "VZ", "WBA", "WFC", "WMT", "XOM"]
    #d) NQ100 (107)
    # ["ATVI", "ADBE", "AMD", "ALXN", "ALGN", "GOOG", "AMZN", "AAL", "AMGN", "ADI", "AAPL", "AMAT", "ASML", "ADSK", "ADP", "BIDU", "BIIB", "BMRN", "BKNG", "AVGO", "CDNS", "CELG", "CERN", "CHTR", "CHKP", "CTAS", "CSCO", "CTXS", "CTSH", "CMCSA", "COST", "CSX", "CTRP", "DLTR", "EBAY", "EA", "EXPE", "FB", "FAST", "FISV", "FOX", "FOXA", "GILD", "HAS", "HSIC", "IDXX", "ILMN", "INCY", "INTC", "INTU", "ISRG", "JBHT", "JD", "KLAC", "LRCX", "LBTYA", "LBTYK", "LULU", "MAR", "MXIM", "MELI", "MCHP", "MU", "MSFT", "MDLZ", "MNST", "MYL", "NTAP", "NTES", "NFLX", "NVDA", "NXPI", "ORLY", "PCAR", "PAYX", "PYPL", "PEP", "QCOM", "REGN", "ROST", "SIRI", "SWKS", "SBUX", "SYMC", "SNPS", "TMUS", "TTWO", "TSLA", "TXN", "KHC", "ULTA", "UAL", "VRSN", "VRSK", "VRTX", "WBA", "WDAY", "WDC", "WLTW", "WYNN", "XEL", "XLNX", "STX", "TSLA", "VRSK", "WYNN", "XLNX"]
    #e) SP&NQ (180) and 1/2 and 1/2
    # ["AAPL", "ABBV", "ABT", "ACN", "ADBE", "AGN", "AIG", "ALL", "AMGN", "AMZN", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CELG", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CSCO", "CVS", "CVX", "DD", "DHR", "DIS", "DUK", "EMR", "EXC", "F", "FB", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTN", "SBUX", "SLB", "SO", "SPG", "T", "TGT", "TXN", "UNH", "UNP", "UPS", "USB", "UTX", "V", "VZ", "WBA", "WFC", "WMT", "XOM", "ATVI", "AMD", "ALXN", "ALGN", "AAL", "ADI", "AMAT", "ASML", "ADSK", "ADP", "BIDU", "BMRN", "AVGO", "CDNS", "CERN", "CHKP", "CTAS", "CTXS", "CTSH", "CSX", "CTRP", "DLTR", "EBAY", "EA", "EXPE", "FAST", "FISV", "FOX", "FOXA", "HAS", "HSIC", "IDXX", "ILMN", "INCY", "INTU", "ISRG", "JBHT", "JD", "KLAC", "LRCX", "LBTYA", "LBTYK", "LULU", "MAR", "MXIM", "MELI", "MCHP", "MU", "MNST", "MYL", "NTAP", "NTES", "NXPI", "ORLY", "PCAR", "PAYX", "REGN", "ROST", "SIRI", "SWKS", "SYMC", "SNPS", "TMUS", "TTWO", "TSLA", "ULTA", "UAL", "VRSN", "VRSK", "VRTX", "WDAY", "WDC", "WLTW", "WYNN", "XEL", "XLNX", "STX", "TSLA", "VRSK", "WYNN", "XLNX"]
    # ["AAPL", "ABBV", "ABT", "ACN", "ADBE", "AGN", "AIG", "ALL", "AMGN", "AMZN", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CELG", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CSCO", "CVS", "CVX", "DD", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FB", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTN", "SBUX", "SLB", "SO", "SPG", "T", "TGT", "TXN", "UNH", "UNP"]
    # ["UPS", "USB", "UTX", "V", "VZ", "WBA", "WFC", "WMT", "XOM", "ATVI", "AMD", "ALXN", "ALGN", "AAL", "ADI", "AMAT", "ASML", "ADSK", "ADP", "BIDU", "BMRN", "AVGO", "CDNS", "CERN", "CHKP", "CTAS", "CTXS", "CTSH", "CSX", "CTRP", "DLTR", "EBAY", "EA", "EXPE", "FAST", "FISV", "FOX", "FOXA", "HAS", "HSIC", "IDXX", "ILMN", "INCY", "INTU", "ISRG", "JBHT", "JD", "KLAC", "LRCX", "LBTYA", "LBTYK", "LULU", "MAR", "MXIM", "MELI", "MCHP", "MU", "MNST", "MYL", "NTAP", "NTES", "NXPI", "ORLY", "PCAR", "PAYX", "REGN", "ROST", "SIRI", "SWKS", "SYMC", "SNPS", "TMUS", "TTWO", "TSLA", "ULTA", "UAL", "VRSN", "VRSK", "VRTX", "WDAY", "WDC", "WLTW", "WYNN", "XEL", "XLNX", "STX", "TSLA", "VRSK", "WYNN", "XLNX"]       
    # SP500-ES&NQ/100_1
    # ["CRM", "TMO", "LIN", "AMT", "FIS", "CME", "CB", "BDX", "SYK", "TJX", "ANTM", "SPGI", "NOC", "D", "CCI", "ZTS", "BSX", "PNC", "CI", "PLD", "ECL", "ICE", "MMC", "DE", "APD", "KMB", "LHX", "EQIX", "WM", "NSC", "AON", "EW", "SCHW", "EL", "AEP", "ITW", "PGR", "EOG", "SHW", "BAX", "PSX", "DG", "PSA", "SRE", "TRV", "ROP", "HUM", "AFL", "WELL", "BBT", "YUM", "MCO", "SYY", "DAL", "STZ", "JCI", "ETN", "NEM", "PRU", "MPC", "HCA", "GIS", "VLO", "EQR", "TEL", "TWTR", "PEG", "WEC", "MSI", "SBAC", "AVB", "OKE", "IR", "ED", "WMB", "ZBH", "AZO", "HPQ", "VTR", "VFC", "TSN", "STI", "HLT", "BLL", "APH", "MCK", "TROW", "PPG", "DFS", "GPN", "ES", "TDG", "FLT", "LUV", "DLR", "EIX", "IQV", "DTE", "INFO", "O"]
    # SP500-ES&NQ/100_2
    # ["FE", "AWK", "A", "CTVA", "HSY", "TSS", "GLW", "APTV", "CMI", "ETR", "PPL", "HIG", "PH", "ADM", "ESS", "FTV", "PXD", "LYB", "SYF", "CMG", "CLX", "SWK", "MTB", "MKC", "MSCI", "RMD", "BXP", "CHD", "AME", "WY", "RSG", "STT", "FITB", "KR", "CNC", "NTRS", "AEE", "VMC", "HPE", "KEYS", "ROK", "CMS", "RCL", "EFX", "ANSS", "CCL", "AMP", "CINF", "TFX", "ARE", "OMC", "HCP", "DHI", "LH", "KEY", "AJG", "MTD", "COO", "CBRE", "HAL", "EVRG", "AMCR", "MLM", "HES", "K", "EXR", "CFG", "IP", "CPRT", "FANG", "BR", "CBS", "NUE", "DRI", "FRC", "MKTX", "BBY", "LEN", "WAT", "RF", "AKAM", "CXO", "MAA", "MGM", "CE", "HBAN", "CAG", "CNP", "KMX", "PFG", "XYL", "DGX", "WCG", "UDR", "DOV", "CBOE", "FCX", "HOLX", "GPC", "L"]
    
    # FX (16)
    # ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD", "USDCNH", "EURJPY", "EURSEK", "EURNOK","USDMXN", "USDZAR", "USDSEK", "USDNOK", "EURHUF"]
    
    #PiTrading_All
    #["A", "AA", "AABA", "AAL", "AAXN", "ABBV", "ACIA", "ADM", "ADT", "AIG", "AKAM", "AKS", "ALLY", "ALTR", "AMAT", "AMC", "AMCX", "AMD", "AMGN", "AMZN", "AN", "ANF", "ANTM", "AOBC", "APO", "APRN", "ARLO", "ATUS", "ATVI", "AUY", "AVGO", "AVTR", "AWK", "BABA", "BAC", "BAH", "BB", "BBBY", "BBH", "BBY", "BIDU", "BJ", "BKNG", "BLK", "BOX", "BP", "BRK-B", "BSX", "BTU", "BURL", "BX", "BYND", "C", "CAKE", "CARS", "CBOE", "CCJ", "CDLX", "CELG", "CHK", "CHWY", "CIEN", "CLDR", "CLF", "CLNE", "CMCSA", "CME", "CMG", "CMI", "CNDT", "COP", "COST", "COUP", "CPB", "CREE", "CRM", "CRSP", "CRUS", "CRWD", "CSX", "CTRP", "CTSH", "CVS", "DBI", "DBX", "DD", "DE", "DECK", "DELL", "DG", "DIA", "DKS", "DLTR", "DNKN", "DNN", "DO", "DOCU", "DRYS", "DT", "DUK", "EA", "EBAY", "EEM", "ELAN", "EOG", "EQT", "ESTC", "ET", "ETFC", "ETRN", "ETSY", "EWJ", "EXC", "F", "FANG", "FAS", "FAZ", "FB", "FCX", "FDX", "FEYE", "FISV", "FIT", "FIVE", "FLR", "FLT", "FMCC", "FNMA", "FSCT", "FSLR", "FTCH", "FXE", "FXI", "GDDY", "GDX", "GE", "GH", "GLBR", "GLD", "GLW", "GM", "GME", "GNRC", "GOLD", "GOOGL", "GOOS", "GPRO", "GPS", "GRPN", "GRUB", "GSK", "GSKY", "HAL", "HCA", "HCAT", "HIG", "HLF", "HLT", "HOG", "HON", "HPE", "HPQ", "HRI", "HTZ", "IBKR", "ICE", "INFO", "INMD", "IQ", "IQV", "ISRG", "IWM", "IYR", "JBLU", "JCP", "JMIA", "JNPR", "KBR", "KLAC", "KMI", "KMX", "KNX", "KSS", "LC", "LEVI", "LHCG", "LLY", "LN", "LOW", "LULU", "LVS", "LYFT", "MA", "MDLZ", "MDR", "MDY", "MGM", "MLCO", "MNK", "MO", "MOMO", "MRNA", "MRVL", "MS", "MSI", "MU", "MXIM", "NAVI", "NEM", "NET", "NFLX", "NIO", "NOK", "NOV", "NOW", "NTNX", "NTR", "NUAN", "NUE", "NVDA", "NVR", "NVS", "NWSA", "NXPI", "OAS", "OIH", "OKTA", "OPRA", "ORCL", "OXY", "PANW", "PAYX", "PBR", "PCG", "PDD", "PE", "PEP", "PHM", "PINS", "PIR", "PM", "PPH", "PRGO", "PS", "PSTG", "PTON", "PVTL", "PYPL", "QCOM", "QQQ", "QRTEA", "QRVO", "RACE", "RAD", "REEMF", "RGR", "RIG", "RIO", "RMBS", "ROKU", "RRC", "RSX", "RTH", "S", "SAVE", "SBUX", "SCCO", "SCHW", "SD", "SDC", "SDS", "SHAK", "SHLDQ", "SHOP", "SINA", "SIRI", "SLB", "SLV", "SMH", "SNAP", "SOHU", "SONO", "SPLK", "SPOT", "SPY", "SQ", "STNE", "STX", "SU", "SWAV", "SWCH", "SWI", "SWN", "SYMC", "T", "TAL", "TDC", "TEVA", "TGT", "TIF", "TLRY", "TLT", "TM", "TME", "TNA", "TOL", "TPR", "TPTX", "TRU", "TRUE", "TSLA", "TTD", "TW", "TWLO", "TWTR", "TXN", "TZA", "UAA", "UBER", "UNG", "UPS", "UPWK", "USFD", "USO", "UUUU", "VICI", "VLO", "VMW", "VRSN", "VVV", "VXX", "W", "WB", "WDAY", "WDC", "WFC", "WFTIQ", "WHR", "WORK", "WYNN", "X", "XLC", "XLE", "XLF", "XLU", "XLV", "YELP", "YETI", "YNDX", "YRD", "YUM", "YUMC", "ZAYO", "ZEUS", "ZG", "ZM", "ZNGA", "ZS", "ZUO"]
    #PiTrading_1
    #["A", "AA", "AABA", "AAL", "AAXN", "ABBV", "ACIA", "ADM", "ADT", "AIG", "AKAM", "AKS", "ALLY", "ALTR", "AMAT", "AMC", "AMCX", "AMD", "AMGN", "AMZN", "AN", "ANF", "ANTM", "AOBC", "APO", "APRN", "ARLO", "ATUS", "ATVI", "AUY", "AVGO", "AVTR", "AWK", "BABA", "BAC", "BAH", "BB", "BBBY", "BBH", "BBY", "BIDU", "BJ", "BKNG", "BLK", "BOX", "BP", "BRK-B", "BSX", "BTU", "BURL", "BX", "BYND", "C", "CAKE", "CARS", "CBOE", "CCJ", "CDLX", "CELG", "CHK", "CHWY", "CIEN", "CLDR", "CLF", "CLNE", "CMCSA", "CME", "CMG", "CMI", "CNDT", "COP", "COST", "COUP", "CPB", "CREE", "CRM", "CRSP", "CRUS", "CRWD", "CSX", "CTRP", "CTSH", "CVS", "DBI", "DBX", "DD", "DE", "DECK", "DELL", "DG", "DIA", "DKS", "DLTR", "DNKN", "DNN", "DO", "DOCU", "DRYS", "DT", "DUK", "EA", "EBAY", "EEM", "ELAN", "EOG", "EQT", "ESTC", "ET", "ETFC", "ETRN", "ETSY", "EWJ", "EXC", "F", "FANG", "FAS", "FAZ", "FB", "FCX", "FDX", "FEYE", "FISV", "FIT", "FIVE", "FLR", "FLT", "FMCC", "FNMA", "FSCT", "FSLR", "FTCH", "FXE", "FXI", "GDDY", "GDX", "GE", "GH", "GLBR", "GLD", "GLW", "GM", "GME", "GNRC", "GOLD", "GOOGL", "GOOS", "GPRO", "GPS", "GRPN", "GRUB", "GSK", "GSKY", "HAL", "HCA", "HCAT", "HIG", "HLF", "HLT", "HOG", "HON", "HPE", "HPQ", "HRI", "HTZ", "IBKR", "ICE", "INFO", "INMD", "IQ", "IQV", "ISRG", "IWM", "IYR", "JBLU", "JCP", "JMIA", "JNPR", "KBR", "KLAC", "KMI"]
    #PiTrading_2
    #["KMX", "KNX", "KSS", "LC", "LEVI", "LHCG", "LLY", "LN", "LOW", "LULU", "LVS", "LYFT", "MA", "MDLZ", "MDR", "MDY", "MGM", "MLCO", "MNK", "MO", "MOMO", "MRNA", "MRVL", "MS", "MSI", "MU", "MXIM", "NAVI", "NEM", "NET", "NFLX", "NIO", "NOK", "NOV", "NOW", "NTNX", "NTR", "NUAN", "NUE", "NVDA", "NVR", "NVS", "NWSA", "NXPI", "OAS", "OIH", "OKTA", "OPRA", "ORCL", "OXY", "PANW", "PAYX", "PBR", "PCG", "PDD", "PE", "PEP", "PHM", "PINS", "PIR", "PM", "PPH", "PRGO", "PS", "PSTG", "PTON", "PVTL", "PYPL", "QCOM", "QQQ", "QRTEA", "QRVO", "RACE", "RAD", "REEMF", "RGR", "RIG", "RIO", "RMBS", "ROKU", "RRC", "RSX", "RTH", "S", "SAVE", "SBUX", "SCCO", "SCHW", "SD", "SDC", "SDS", "SHAK", "SHLDQ", "SHOP", "SINA", "SIRI", "SLB", "SLV", "SMH", "SNAP", "SOHU", "SONO", "SPLK", "SPOT", "SPY", "SQ", "STNE", "STX", "SU", "SWAV", "SWCH", "SWI", "SWN", "SYMC", "T", "TAL", "TDC", "TEVA", "TGT", "TIF", "TLRY", "TLT", "TM", "TME", "TNA", "TOL", "TPR", "TPTX", "TRU", "TRUE", "TSLA", "TTD", "TW", "TWLO", "TWTR", "TXN", "TZA", "UAA", "UBER", "UNG", "UPS", "UPWK", "USFD", "USO", "UUUU", "VICI", "VLO", "VMW", "VRSN", "VVV", "VXX", "W", "WB", "WDAY", "WDC", "WFC", "WFTIQ", "WHR", "WORK", "WYNN", "X", "XLC", "XLE", "XLF", "XLU", "XLV", "YELP", "YETI", "YNDX", "YRD", "YUM", "YUMC", "ZAYO", "ZEUS", "ZG", "ZM", "ZNGA", "ZS", "ZUO"]

    #BenchMarks
    #IWV iShares Russell 3000 ETF
    #IWB Russell 1000: 1,000 large-cap American companies in the Russell 3000 Index
    #IWM Russell 2000: 2,000 smallest-cap American companies in the Russell 3000 Index
    
    '''Global Variables
    '''
    _totalSymbolsAdded = 0
    
    def __init__(self, caller):
        self.CL = self.__class__
        self.algo = caller
        self.debug = self.algo.debug
    '''
    AFTER WARMUP
    '''
    def MyOnWarmupFinished(self):
        #Check Warmup Status for each Symbol
        for sd, value in self.algo.mySymbolDict.items():
            if not value.IsReady(): 
                self.algo.MyDebug(" Symbol: {}({}) is NOT READY AFTER WARMUP!".format(str(value.symbol), str(value.CL.strategyCode)))

        '''
        IN LIVE MODE: Syncs Orders with Broker, Checks Order Consistency, Lists Order and Portfolio Items
        '''
        self.PortfolioCheckSymbolDict()
        if not self.algo.LiveMode: self.algo.twsSynced = True
        
        if self.algo.LiveMode or False:
            self.algo.MyDebug(" ---- WarmUp Finished Startup Sync Started:" )
            
            self.PortfolioCheckSymbolDict()
        
            #Sync TWS orders
            totalOrdersAdded = self.algo.myPositionManager.TWS_Sync()
            #List Active Orders
            if totalOrdersAdded != 0:
                self.algo.myVaR.OrderList()
                #Check consistency for all symbols
                self.algo.myPositionManagerB.AllOrdersConsistency()
            self.algo.MyDebug(" ---- Initial TWS Sync and Consistency Check Finished") 
            
            #List Portfolio Items
            self.algo.myVaR.PortfolioList(True) #True if position only
            #Freeze consistency as things could mess up at startup due to sync with IB 
            self.algo.consistencyStartUpReleaseTime = self.algo.Time + timedelta(seconds=120)
            
            #SET SCHEDULED TASKS
            #AllOrdersConsistency so it is run regularly not only in onData
            self.algo.Schedule.On(self.algo.DateRules.EveryDay(), self.algo.TimeRules.Every(self.algo.myVaR.CL.consistencyCheckSec), \
                    Action(self.algo.myPositionManagerB.AllOrdersConsistency))
            #VaR Calculations so it is updated regularly in LiveMode not only in onData
            self.algo.Schedule.On(self.algo.DateRules.EveryDay(), self.algo.TimeRules.Every(timedelta(seconds=68.123456789)), Action(self.algo.myVaR.Update))
            #Pending Entries
            #self.algo.myPositionManagerB.AllOrdersConsistency() cannot call it due to RECURSIVE LOOP as CheckPendingEntry->EnterPosition->VaR->AllOrdersConsistency->CheckPendingEntry
            self.algo.Schedule.On(self.algo.DateRules.EveryDay(), self.algo.TimeRules.Every(timedelta(seconds=196.80625)), Action(self.algo.myPositionManager.CheckPendingEntry))

            if self.algo.updateSettings:
                #Update Setting For the first time
                self.algo.strategySettings.UpdateSettings()
                self.algo.MyDebug(" ---- UPDATE SETTINGS IS ON! First update is completed.") 
                self.algo.Schedule.On(self.algo.DateRules.EveryDay(), self.algo.TimeRules.Every(timedelta(minutes=6.251968)), Action(self.algo.strategySettings.UpdateSettings))
            
            #Update VaR and Order Statistics on DashBoard
            self.algo.myVaR.Update()
            self.algo.MyDebug(" ---- OnWarmupFinished     Total mySymbolDict:" + str(len(self.algo.mySymbolDict)) \
                    + "     Portfolio Holdings Value:" + str(round(self.algo.Portfolio.TotalHoldingsValue)))
        return
    '''
    ON DATA
    '''
    def MyOnData(self, data):
        #EXIT HERE IF WarmingUp or initial consistency blocked at Startup
        # none of the consolidators have new data
        if self.algo.IsWarmingUp or self.algo.Time < self.algo.consistencyStartUpReleaseTime: return
        
        #Only if at least one symbol is ready to speed up backtest
        isReady = False
        for sd, value in self.algo.mySymbolDict.items():
            if value.IsReady() and value.WasJustUpdated(self.algo.Time): isReady = True
        if not isReady: return
        
        #ORDER CONSISTENCY Check for all Symbols not only Portfolio
        self.algo.myPositionManagerB.AllOrdersConsistency()
        #TRAIL STOPS
        self.algo.myPositionManager.TrailStops()
        #TRAIL TARGETS
        self.algo.myPositionManager.TrailTargets()
        #REMOVE OLD ORDERS
        self.algo.myPositionManagerB.ClearOrderList()
        #EXPOSURE and VaR Calculation 
        self.algo.myVaR.Update()
        #PENDING ENTRIES
        #Tself.algo.myPositionManagerB.AllOrdersConsistency() cannot call it due to RECURSIVE LOOP as CheckPendingEntry->EnterPosition->VaR->AllOrdersConsistency->CheckPendingEntry
        self.algo.myPositionManager.CheckPendingEntry()
        
        # #STRESS TEST
        # if self.algo.Time.minute == 10 or self.algo.Time.minute == 30 or self.algo.Time.minute == 50:
        #     for x in self.algo.Portfolio:
        #         if self.algo.Portfolio[x.Key].Quantity != 0:
        #             self.algo.myPositionManagerB.LiquidatePosition(x.Key, "STest", " --- STRESS TEST")
        return

    '''
    INSTALLING NEW strategy
    '''
    def InstallStrategy (self, strategy, myAllocation=-1):
        if not strategy.enabled or myAllocation==0 or (myAllocation==-1 and strategy.strategyAllocation==0):
            self.algo.MyDebug(" STARTEGY: {} IS NOT INSTALLED! Enabled:{}, Allocation:{}/{}".format(str(strategy.strategyCode),str(strategy.enabled),str(myAllocation),str(strategy.strategyAllocation)))
            return
        
        #OverWrite strategyAllocation if needed
        if myAllocation !=-1: strategy.strategyAllocation = myAllocation
        
        #If this is the first strategy 
        if not self.algo.myStrategyClassList:
            #Setup VaR for benchmark and Chartsymbol
            self.algo.myVaR = MyVaR(self.algo, strategy)
            self.algo.myVaRList.append(self.algo.myVaR)
            #Setup VaR for TWS and Chartsymbol
            self.algo.foreignVaR = MyVaR(self.algo, strategy)
            self.algo.myVaRList.append(self.algo.foreignVaR)
            self.algo.foreignVaR.icnludeinTotalVaR = self.algo.myVaR.CL.manageTWSSymbols
        
        #Add VaR module to startegy
        self.algo.myStrategyClassList.append(strategy)
        strategy.mainVaR = MyVaR(self.algo, strategy)
        self.algo.myVaRList.append(strategy.mainVaR)
 
        #Tickers
        tickerlist = strategy.myTickers if hasattr(strategy, 'myTickers') else strategy.mySymbols #keep mySymbols for compatibility reasons
        #Check for ticker duplication
        for ticker in tickerlist:
            for symbol in self.algo.mySymbolDict:
                #Symbol.Value == ticker??
                if ticker == symbol.Value:
                    self.algo.MyDebug(" SYMBOL DUPLICATION IN STRATEGIES: "+str(ticker)+" IS IN: "+str(strategy.strategyCode)+" AND IS ALREADY IN: "+str(self.algo.mySymbolDict[symbol].CL.strategyCode))
        
        #Resolution
        resolution = Resolution.Daily
        if strategy.resolutionMinutes < 60:
            resolution = Resolution.Minute
        elif strategy.resolutionMinutes < 60*24:
            resolution = Resolution.Hour
        
        #Add tickers/symbols/securities    
        for ticker in tickerlist: 
            if strategy.isEquity:
                self.algo.AddEquity(ticker,  resolution)
                self.algo.Securities[ticker].SetDataNormalizationMode(self.algo.myDataNormalizationMode)
            else: 
                self.algo.AddForex(ticker, resolution)
            
            symbol = self.algo.Securities[ticker].Symbol
            security = self.algo.Securities[ticker]
            self.AddSymbolDict(symbol, strategy, strategy.mainVaR)
            if strategy.customFillModel != 0:
                security.SetFillModel(MyFillModel(self.algo, symbol))
            if strategy.customSlippageModel != 0:
                security.SetSlippageModel(MySlippageModel(self.algo, symbol))
               
        #Checking allocation breach
        totalAllocation = 0
        for strategy in self.algo.myStrategyClassList:
            totalAllocation += strategy.strategyAllocation
        
        self.algo.MyDebug(" STRATEGY INSTALLED: {} Strategy Allocation:{} Total Allocation:{}, Total Symbols:{}, Resolution(min):{}".format(str(strategy.strategyCode),str(strategy.strategyAllocation),str(round(totalAllocation,2)),str(self.CL._totalSymbolsAdded),str(strategy.resolutionMinutes)))

        if totalAllocation > 1:
            self.algo.MyDebug(" TOTAL ALLOCATION IS GREATER THAN 1.00: {} ALGO IS DISABLED!".format(str(round(totalAllocation,2))))
            self.algo.enabled = False
            raise Exception(" TOTAL ALLOCATION IS GREATER THAN 1.00: {} ALGO IS DISABLED!".format(str(round(totalAllocation,2))))
        return       
    
    '''
    SETTING RESOLUTION
    '''
    def MyResolution (self):
        resolution = Resolution.Daily
        minResolutionMinites = 60*24
        for st in self.algo.myStrategyClassList:
            if st.resolutionMinutes < minResolutionMinites and st.enabled: minResolutionMinites = st.resolutionMinutes
        self.algo.minResolutionMinutes = minResolutionMinites
        if minResolutionMinites < 60:
            resolution = Resolution.Minute
        elif minResolutionMinites < 6*24:
            resolution = Resolution.Hour
        return resolution
    '''
    WARMUP IN DAYS
    '''
    def WarUpDays (self):
        warmupcalendardays = 1
        extraDays = 1
        for strategy in self.algo.myStrategyClassList:
            if strategy.enabled and strategy.warmupcalendardays > warmupcalendardays:
                warmupcalendardays = strategy.warmupcalendardays
        warmupdays = timedelta(days=warmupcalendardays+extraDays)
        self.algo.MyDebug(" WarmUp Calendar Days: {} ({} Extra Days Added) ".format(str(warmupdays.days), str(extraDays)))
        return warmupdays
    
    '''
    ADDING NEW SYMBOL
    '''
    def AddSymbolDict (self, symbol, strategy, var):
        if symbol not in self.algo.mySymbolDict:
            self.algo.mySymbolDict[symbol] = strategy(self.algo, symbol, var)
            self.CL._totalSymbolsAdded +=1
            #if self.algo.LiveMode: self.algo.MyDebug(" Added to mySymbolDict:" + str(symbol))
    '''
    CHECK PORTFOLIO SYMBOLS
    '''
    def PortfolioCheckSymbolDict (self):
        '''Need this check if conversion rate currency is added
        '''
        for x in self.algo.Portfolio:
            if x.Key not in self.algo.mySymbolDict:
                #Subscribe to Data
                if x.Key.SecurityType == SecurityType.Equity:
                    self.algo.AddEquity(x.Key.Value, self.algo.mainResolution)
                elif x.Key.SecurityType == SecurityType.Forex: 
                    self.algo.AddForex(self.algo.Securities[x.Key].Symbol.Value, self.algo.mainResolution)
                #Add to mySymbolDict
                self.AddSymbolDict(x.Key, self.algo.myStrategyClassList[0], self.algo.foreignVaR)
                self.algo.mySymbolDict[x.Key].posEnabled = False
                if self.algo.Portfolio[x.Key].Quantity != 0: self.algo.mySymbolDict[x.Key].fromTWS = True
                if self.algo.LiveMode or self.debug: self.algo.MyDebug(" PORTFOLIO SYMBOL ADDED Symbol:{}, Position Quantity:{}"
                            .format(str(x.Key),
                                    str(self.algo.Portfolio[x.Key].Quantity)))
    
    '''
    Check if History Download was succesful
    NOT USED
    '''
    def AssertHistoryCount(self, tradeBarHistory, expected):
        count = len(tradeBarHistory.index)
        if count == expected: 
            return True
        else:
            return False
            
    '''
    SCURITY CHANGE EVENT HANDLER
    NOT USED
    '''
    def OnSecuritiesChanged (self, changes):
        '''This is not called during Warmup even if self.AddEquity is used! History data download can be put here
        '''
        return
        for security in changes.AddedSecurities:
            if security.Symbol not in self.algo.mySymbolDict:
                self.AddSymbolDict(security.Symbol, self.algo.myVaR)
                if self.algo.LiveMode: self.algo.MyDebug(" " + str(security.Symbol) + "Added OnSecuritiesChanged")
        for security in changes.RemovedSecurities:
            if security.Symbol in self.algo.mySymbolDict:
                del self.algo.mySymbolDict[security.Symbol]
                if self.algo.LiveMode: self.algo.MyDebug(" " + str(security.Symbol) + " Removed OnSecuritiesChanged")
    '''
    FEATURES TO PANDAS
    '''
    #slicer must be a slice object: slice(start, stop, step) or slice(stop) (https://data-flair.training/blogs/python-slice/) example: slice(0, 400, None)
    def UnpackFeatures (self, features, featureType=1, featureRegex='Feat', reshapeTuple=None, mySlicer=None):
        useSingleFeatureList = False
        dataBase = []
        rawDataHeader = []
        rawData = []

        #PT CNN just reshaping the numpy element of the original features list
        if featureType==6:
            #Return List of torch tensors from reshaped numpys
            for i in range(len(features)):
                x = features[i]
                #if n*n GASF matrtix
                if len(x.shape) == 2:
                    x = x.reshape(1, 1, x.shape[0], x.shape[1])
                #if channel*n*n GASF matrtix
                if len(x.shape) == 3:
                    x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
                x = torch.from_numpy(x).float().to('cpu')
                rawData.append(x)
                return rawData

        if isinstance(features[0], list) and not useSingleFeatureList:
            #if features is a list of lists
            for i in range(0, len(features)):
                for j in range(0, len(features[i])):
                    rawDataHeader.append("Feat"+str(i)+'_'+str(j))
                    rawData.append(features[i][j])
        else:
            #if features is a single list or useSingleFeatureList
            for i in range(len(features)):
                rawDataHeader.append("Feat"+str(i))
                rawData.append(features[i])
        dataBase.append(rawDataHeader)
        dataBase.append(rawData)
        df = pd.DataFrame(dataBase[1:], columns=dataBase[0])
        
        #SELECTING FEATURES with featureRegex and SLICING with mySlicer
        if mySlicer==None:
            df_filtered = df.filter(regex = featureRegex)[:]
        else:
            df_filtered = df.filter(regex = featureRegex)[mySlicer]
        
        #Types
        if featureType==1:
            #keep original Pandas
            convertedFeatures = df_filtered
        if featureType==2:
            #original Pandas Transposed
            convertedFeatures = df_filtered.T
        elif featureType==3:
            #converted to list
            convertedFeatures = df_filtered.values.tolist()[0]
        elif featureType==4:
            #numpy Array
            convertedFeatures = np.asarray(df_filtered)
        elif featureType==5:
            #numpy Array Reshaped (for old CNN)
            convertedFeatures = np.asarray(df_filtered)
            convertedFeatures = np.reshape(convertedFeatures, reshapeTuple)
        return convertedFeatures
    
    #CUSTOM FILTERS with customColumnFilters
    #expect one row on df
    #returns true if at least one row meets the conditions
    def FeatureCustomColumnFilter(self, df, customColumnFilters):
        #CUSTOM FILTERS with customColumnFilterslist of tuples ('col', 'opearator', 'treshold value') like [('Feat8_15', '>', 0.55)]
        myOperators = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '=': operator.eq}    
        for filter in customColumnFilters:
            opFilteredCol = filter[0]
            opRelate = filter[1]
            opTreshold = filter[2]
            if opFilteredCol in df.columns:
                df = df.loc[myOperators[opRelate](df[opFilteredCol], opTreshold)]
        #df.reset_index(inplace=True,drop=True)
        if df.empty:
            return False
        else: 
            return True
        
'''
Custom Fill Model Class
'''
class MyFillModel(FillModel):
    def __init__(self, algo, symbol):
        self.CL = self.__class__
        self.algo = algo
        self.symbol = symbol
        self.debug = False
        #super().__init__(self, algo)
        
        if self.debug: self.algo.MyDebug(" MyFillModel __init__ Symbol: " + str(symbol))
        
    #It look as QC doesn't use slippage so all the fill prices to be recalculated 
    
    #QC is too conservative if price walks through the stop
    def StopMarketFill(self, asset, order):
        fill = super().StopMarketFill(asset, order)
        prices = super().GetPrices(asset, order.Direction)
        slippage = asset.SlippageModel.GetSlippageApproximation(asset, order)
        oldfillprice = fill.FillPrice
        if self.debug: self.algo.MyDebug(" {} Quantity:{} oldFillPrice:{} StopPrice:{} Open:{} High:{} Low:{}".format(str(asset.Symbol), str(order.Quantity), str(oldfillprice), str(order.StopPrice), str(prices.Open), str(prices.High), str(prices.Low)))
        
        if order.Direction == OrderDirection.Sell and prices.Low <= order.StopPrice:
            #fill.Status = OrderStatus.Filled
            #fill.FillQuantity = order.Quantity
            #if self.debug: self.algo.MyDebug(" {} StopMarket Fill".format(str(asset.Symbol)))
            pass
        elif order.Direction == OrderDirection.Buy and prices.High >= order.StopPrice:
            #fill.Status = OrderStatus.Filled
            #fill.FillQuantity = order.Quantity
            #if self.debug: self.algo.MyDebug(" {} StopMarket Fill".format(str(asset.Symbol)))
            pass
        
        if fill.Status == OrderStatus.Filled or fill.Status == OrderStatus.PartiallyFilled:
            if order.Direction == OrderDirection.Sell:
                #Price walks through the Stop
                if prices.Open > order.StopPrice and prices.Close < order.StopPrice:
                    fill.FillPrice = order.StopPrice - slippage
                #Stops and reverses
                elif prices.Open > order.StopPrice and prices.Low <= order.StopPrice and prices.Close > order.StopPrice:
                    fill.FillPrice = order.StopPrice - slippage
                #Gaps Down
                elif prices.Open <= order.StopPrice:
                    fill.FillPrice = prices.Open - slippage
                if self.debug: self.algo.MyDebug(" StopMarketFill({}): Fill Price Modidied from:{} to:{} StopPrice:{} bar.Open:{} bar.High:{} bar.Low:{} bar.Close:{}".format(str(asset.Symbol), str(oldfillprice), str(fill.FillPrice), str(order.StopPrice), str(prices.Open), str(prices.High), str(prices.Low), str(prices.Close)))
            elif order.Direction == OrderDirection.Buy:
                #Price walks through the Stop
                if prices.Open < order.StopPrice and prices.Close > order.StopPrice:
                    fill.FillPrice = order.StopPrice + slippage
                #Stops and reverses
                elif prices.Open < order.StopPrice and prices.High >= order.StopPrice and prices.Close < order.StopPrice:
                    fill.FillPrice = order.StopPrice + slippage
                #Gaps Up
                elif prices.Open >= order.StopPrice:
                    fill.FillPrice = prices.Open + slippage
                if self.debug: self.algo.MyDebug(" StopMarketFill({}): Fill Price Modidied from:{} to:{} StopPrice:{} bar.Open:{} bar.High:{} bar.Low:{} bar.Close:{}".format(str(asset.Symbol), str(oldfillprice), str(fill.FillPrice), str(order.StopPrice), str(prices.Open), str(prices.High), str(prices.Low), str(prices.Close)))
        return fill
    
    #For market orders the slippage is correct
    def MarketFill(self, asset, order):
        fill = super().MarketFill(asset, order)
        prices = super().GetPrices(asset, order.Direction)
        slippage = asset.SlippageModel.GetSlippageApproximation(asset, order)
        oldfillprice = fill.FillPrice
        if self.debug: self.algo.MyDebug(" {} oldFillPrice:{} OpenPrice:{}".format(str(asset.Symbol), str(oldfillprice), str(prices.Open)))
        
        return fill
    
'''
Custom Slippage Model Class
'''        
class MySlippageModel:
    applyMinVariation = True
    roundSlippage = False
    
    def __init__(self, algo, symbol):
        self.CL = self.__class__
        self.algo = algo
        self.symbol = symbol
        self.debug = False

    def GetSlippageApproximation(self, asset, order):
        slippage = 0
        #Percent Based Slippage Model
        if self.algo.mySymbolDict[self.symbol].CL.customSlippageModel == 1:
            slippage = self.PercentSlippage1 (asset, order)
        #ATR Based Slippage Model
        elif self.algo.mySymbolDict[self.symbol].CL.customSlippageModel == 2:
            slippage = self.ATRSlippage1 (asset, order)
        
        if self.debug: self.algo.MyDebug("  {} CustomSlippageModel:{} ".format(str(asset.Symbol), str(slippage)))
        return slippage
        
    def PercentSlippage1 (self, asset, order):
        slippageRatioEq = 0.001/4
        slippageRatioFX = 0.0001/2

        minPriceVariation = self.algo.Securities[self.symbol].SymbolProperties.MinimumPriceVariation
        priceRoundingDigits = round(-1*log(minPriceVariation,10))
        #slippage = asset.Price * 0.0001 * np.log10(2*float(order.AbsoluteQuantity))
        
        if self.symbol.SecurityType == SecurityType.Equity:
            slippageRatio = slippageRatioEq
        else:
            slippageRatio = slippageRatioFX
        
        baseSlippage = asset.Price * slippageRatio
        if self.CL.applyMinVariation: baseSlippage = max(baseSlippage, minPriceVariation)
        if self.CL.roundSlippage:
            slippage = round(baseSlippage, priceRoundingDigits)
        else:
            slippage = baseSlippage
        return slippage
        
    def ATRSlippage1 (self, asset, order):
        slippageRatioEq = 0.1 
        slippageRatioFX = 0.1
        
        slippage = 0
        atr = self.algo.mySymbolDict[self.symbol].atr1.Current.Value
        
        minPriceVariation = self.algo.Securities[self.symbol].SymbolProperties.MinimumPriceVariation
        priceRoundingDigits = round(-1*log(minPriceVariation,10))
        #slippage = asset.Price * 0.0001 * np.log10(2*float(order.AbsoluteQuantity))
        
        if self.symbol.SecurityType == SecurityType.Equity:
            slippageRatio = slippageRatioEq
        else:
            slippageRatio = slippageRatioFX
        
        baseSlippage = atr * slippageRatio
        if self.CL.applyMinVariation: baseSlippage = max(baseSlippage, minPriceVariation)
        if self.CL.roundSlippage:
            slippage = round(baseSlippage, priceRoundingDigits)
        else:
            slippage = baseSlippage
        return slippage

'''
AI Model Loader
'''        

class MyModelLoader:
    session = 0
    
    @classmethod
    def LoadModelTorch(cls, caller, url, existingmodel=None):
        algo = caller.algo
        response = algo.Download(url) 
        decoded = codecs.decode(response.encode(), "base64")
        stream = io.BytesIO(decoded)
        if existingmodel==None:
            model = torch.load(stream, map_location='cpu')
        else:
            model = existingmodel
            model.load_state_dict(torch.load(stream, map_location='cpu'))
        if False:
            algo.Debug(str(model))
            algo.Debug(str(model.state_dict()))
        model.eval()
        # algo.Debug(' MODEL LOADED: '+str(url1))
        return model
    
    @classmethod
    def LoadModelPickled(cls, caller, url):
        response = caller.algo.Download(url)
        model = pickle.loads(codecs.decode(response.encode(), "base64"))
        return model
 
    @classmethod
    def LoadModelLGBtxt(cls, caller, url, features=50):
        response = caller.algo.Download(url)
        #https://github.com/microsoft/LightGBM/issues/2097#issuecomment-482332232
        booster = lightgbm.Booster({'model_str': response})
        #booster = lightgbm.Booster(model_str=response)
        #booster = lightgbm.Booster(train_set=lightgbm.Dataset(data=np.random.rand(10,features)))
        #model = booster.model_from_string(response)
        return model
 
    @classmethod
    def LoadModelLGB(cls, caller, url):
        response = caller.algo.Download(url)
        decoded = codecs.decode(response.encode(), "base64")
        with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as fd:
            fd.write(decoded)
            fd.flush()
            tmpFile = fd.name
        booster = lightgbm.Booster(model_file=tmpFile)
        if os.path.isfile(tmpFile): os.remove(tmpFile)
        return booster

    def __init__(self, algo, loadtype, url1, url2=None, printSummary=False):
        self.algo=algo
        self.loadtype=loadtype
        self.url1=url1
        self.url2=url2
        self.printSummary= printSummary
        self.model=None
        self.stream=None
        if self.loadtype in [2,3,4,5]:
            self.tfGraph = tensorflow.Graph() #tensorflow.Graph() #tensorflow.get_default_graph()
            #self.tfSession = tensorflow.keras.backend.get_session() #tensorflow.Session(graph=self.tfGraph)
            self.tfConfig = tensorflow.ConfigProto()
            self.tfConfig.operation_timeout_in_ms = 10000
            self.tfConfig.allow_soft_placement = True
        self.LoadModel()
        return
    
    def LoadModel(self):
        model = None
        #Pickle the whole model. Works for sklearn
        if self.loadtype==1:
            response = self.algo.Download(self.url1)
            self.model = pickle.loads(codecs.decode(response.encode(), "base64"))
        #keras only: load model from json and pickle weights
        #model.set_weights(weights) sets the values of the weights of the model, from a list of Numpy arrays. The arrays in the list should have the same shape as those returned by get_weights()
        #https://keras.io/models/about-keras-models/
        elif self.loadtype==2:
            #get the model first
            response = self.Download(self.url1)
            model_json = json.loads(response)
            self.model = tensorflow.keras.models.model_from_json(model_json)
            #get the pickled weights
            response = self.Download(self.url2)
            weights = pickle.loads(codecs.decode(response.encode(), "base64"))
            self.model.set_weights(weights)
            self.model._make_predict_function()
        #keras only: load model from json and h5 weights. Works if keras.get_file whitelisted on QC proxy
        elif self.loadtype==3:
            #get the model first
            response = self.Download(self.url1)
            self.model_json = json.loads(response)
            self.model = tensorflow.keras.models.model_from_json(model_json)
            #get the weights in h5 format
            weights_path = tensorflow.keras.utils.get_file('model.h5',self.url2)
            self.model.load_weights(weights_path)
            self.model._make_predict_function()
        #keras only: load model from h5 using tempfile
        elif self.loadtype==4:
            response = self.algo.Download(self.url1)   
            h5file_fromtxt = codecs.decode(response.encode(), "base64")
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
                fd.write(h5file_fromtxt)
                fd.flush()
                self.model = tensorflow.keras.models.load_model(fd.name)
                self.model._make_predict_function()
                try:
                    fd.close()
                    os.unlink(fd.name)
                except:
                    pass
            if self.printSummary:
                self.algo.MyDebug("Summary of the loaded model: " + self.url1)
                model.summary(print_fn=lambda x: self.algo.MyDebug(x))
        #keras only: load model from h5txt using BytesIO
        elif self.loadtype==5:
            dummyImput = [np.random.rand(1,400), np.random.rand(1,100)]
            response = self.algo.Download(self.url1) 
            decoded = codecs.decode(response.encode(), "base64")
            stream = io.BytesIO(decoded)
            self.stream = stream
            #self.tfGraph =  tensorflow.Graph()
            with self.tfGraph.as_default():
                self.tfSession = tensorflow.Session(config=self.tfConfig, graph=self.tfGraph)
                tensorflow.keras.backend.set_session(self.tfSession)
                with self.tfSession.as_default():
                    self.model = tensorflow.keras.models.load_model(stream)
                    #self.model.predict(dummyImput)
                    self.model._make_predict_function()
                    #self.tfSession.run(tensorflow.global_variables_initializer())
                    #self.tfSession.run(tensorflow.local_variables_initializer())
                    #self.tfGraph.finalize()
            if self.printSummary:
                self.algo.MyDebug("Summary of the loaded model: " + self.url1)
                self.model.summary(print_fn=lambda x: self.algo.MyDebug(x))
        self.algo.MyDebug(' MODEL LOADED: '+str(self.url1))
        return
    
    def tfPredict(self, features):
        #with self.tfGraph.as_default(), self.tfSession.as_default():
        # with self.tfGraph.as_default(): 
        #     with self.tfSession.as_default():
        #Brute force solution to the multithread problem to load the actual model again
        #Properly managing Graphs and Sessions it to be investigated
        if self.algo.LiveMode or True:
            #self.tfGraph = tensorflow.get_default_graph() #tensorflow.Graph()
            with self.tfGraph.as_default():
                #self.model = tensorflow.keras.models.load_model(self.stream)
                #self.model._make_predict_function()
                #tfSession = tensorflow.Session(graph=self.tfGraph, config=self.tfConfig)
                tensorflow.keras.backend.set_session(self.tfSession)
                with self.tfSession.as_default():
                    prediction = self.model.predict(features)
            #tensorflow.keras.backend.clear_session()
            #tensorflow.reset_default_graph()
        else:
            with self.tfGraph.as_default():
            #     self.tfSession = tensorflow.Session(graph=self.tfGraph)
            #     with self.tfSession.as_default():
                prediction = self.model.predict(features)
        return (np.argmax(prediction), prediction)

'''
Strategy Settings from Cloud
'''    
class MyStrategySettings():
    debugChanges = True
    debug = False
    
    dataValidation = {
        'enabled': (bool, True, False),
        'debug': (bool, True, False),
        'strategyAllocation': (float, 0.00, 1.00),
        'enableLong': (bool, True, False),
        'enableShort': (bool, True, False),
        'liquidateLong': (bool, True, False),
        'liquidateShort': (bool, True, False),
        'riskperLongTrade': (float, 0.00, 0.02),
        'riskperShortTrade': (float, 0.00, 0.02),
        'maxAbsExposure' : (float, 0.00, 4.00),
        'maxLongExposure' : (float, 0.00, 4.00),
        'maxNetLongExposure' : (float, 0.00, 4.00),
        'maxShortExposure' : (float, -4.00, 0.00),
        'maxNetShortExposure' : (float, -4.00, 0.00),
        'maxSymbolAbsExposure' : (float, 0.00, 2.00),
        'maxLongVaR' : (float, 0.00, 0.20),
        'maxShortVaR' : (float, 0.00, 0.20),
        'maxTotalVaR' : (float, 0.00, 0.20)
    }
    
    def __init__(self, algo):
        self.CL = self.__class__
        self.algo = algo
    
    def ReadSettings(self):
        try: 
            file_str = self.algo.Download(self.algo.settingsURL)
            csv_stream = io.StringIO(file_str)
            df = pd.read_csv(csv_stream, sep=',', index_col=0, header=0)
            df = self.ConvertDataType_pd(df)
            return df
        except:
            self.algo.MyDebug('--- SETTINGS READING ERROR!')
            return None
    
    def UpdateSettings(self):
        df = self.ReadSettings()
        if df is None:
            return
        
        if self.CL.debug: self.algo.MyDebug('Settings Up')
        #Update algo Settings
        if 'algo' in df:
            for row in range(df.shape[0]):
                prop = df.index[row]
                value = df.loc[df.index[row], 'algo']
                if hasattr(self.algo, prop) and not pd.isna(value):
                    oldvalue = getattr(self.algo, prop)
                    if value!=oldvalue and ((isinstance(value, float) and isinstance(oldvalue, float)) or (isinstance(value, bool) and isinstance(oldvalue, bool))) and self.ValidateData(value, prop):
                        setattr(self.algo, prop, value)
                        if self.CL.debugChanges: self.algo.MyDebug(' ---- SETTINGS HAS CHANGED!  algo.{} = {}, oldvalue:{}, equal:{}'.format(prop, str(getattr(self.algo, prop)), str(oldvalue), getattr(self.algo, prop)==df.loc[df.index[row], 'algo']))
                    if self.CL.debug: self.algo.MyDebug('algo.{} value:{} csv_value:{} equal:{}'.format(prop, str(getattr(self.algo, prop)),df.loc[df.index[row], 'algo'], getattr(self.algo, prop)==df.loc[df.index[row], 'algo']))
                    
        #Update Strategies
        for strategy in self.algo.myStrategyClassList:
            if hasattr(strategy, "strategyCodeOriginal"):
                strCode = strategy.strategyCodeOriginal
            else:
                strCode = strategy.strategyCode
            if strCode in df:
                for row in range(df.shape[0]):
                    prop = df.index[row]
                    value = df.loc[df.index[row], strCode]
                    if hasattr(strategy, prop) and not pd.isna(value):
                        oldvalue = getattr(strategy, prop) 
                        if value!=oldvalue and ((isinstance(value, float) and isinstance(oldvalue, float)) or (isinstance(value, bool) and isinstance(oldvalue, bool))) and self.ValidateData(value, prop):
                            setattr(strategy, prop, value)
                            if  self.CL.debugChanges: self.algo.MyDebug(' ---- SETTINGS HAS CHANGED!  {}.CL.{} = {}, oldvalue:{}, equal:{}'.format(strCode, prop, str(getattr(strategy, prop)), str(oldvalue), getattr(strategy, prop)==df.loc[df.index[row], strCode]))
                        if self.CL.debug: self.algo.MyDebug('{}.CL.{} value:{} csv_value:{}, equal:{}'.format(strCode, prop, str(getattr(strategy, prop)), df.loc[df.index[row], strCode], getattr(strategy, prop)==df.loc[df.index[row], strCode]))
        return

    def ValidateData(self, value, prop):
        if prop in self.CL.dataValidation:
            if self.CL.dataValidation[prop][0] == bool:
                return value==self.CL.dataValidation[prop][1] or value==self.CL.dataValidation[prop][2]
            if self.CL.dataValidation[prop][0] == float:
                return value>=self.CL.dataValidation[prop][1] and value<=self.CL.dataValidation[prop][2]
            else:
                return False
        else:
            return False

    def ConvertDataType_pd (self, df):
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                #Check if string is Boolean
                cell = df.iloc[row, col]
                cellStr = str(cell).lower()
                if cellStr in ("yes", "true", "t"):
                    df.iloc[row, col] = True
                elif cellStr in ("no", "false", "f"):
                    df.iloc[row, col] = False
                #Check if sting is Float
                cell = df.iloc[row, col]
                if cell!=True and  cell!=False and not pd.isna(cell):
                    try:
                        float(cell)
                        df.iloc[row, col] = float(cell)
                    except ValueError:
                        pass
        return df