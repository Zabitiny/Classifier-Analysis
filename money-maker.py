# based on https://towardsdatascience.com/forecasting-s-p-500-stock-index-using-classification-models-eb41510a896d

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quandl

#Daily data from Quandl
quandl_dict = {
	#futures
	'gold' :"CHRIS/CME_GC1.4", #Gold Futures, Continuous Contract #1 (GC1) (Front Month)
	'eurodollar': "CHRIS/CME_ED1.4", #Eurodollar Futures, Continuous Contract #1 (ED1) (Front Month)
	'silver': "CHRIS/CME_SI1.4", #Silver Futures, Continuous Contract #1 (SI1) (Front Month)
	'DCOILWTICO': "FRED/DCOILWTICO.4", #Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma
	#major currency pair spot rates
	'USDCAD': "FED/RXI_N_B_CA", #CANADA -- SPOT EXCHANGE RATE, CANADIAN $/US$, Business day
	'EURUSD': "FED/RXI_US_N_B_EU", #SPOT EXCHANGE RATE - EURO AREA, Business day
	'GBPUSD': "FED/RXI_US_N_B_UK", #UNITED KINGDOM -- SPOT EXCHANGE RATE, US$/POUND (1/RXI_N.B.UK), Business day
	'USDJPY': "FED/RXI_N_B_JA", #JAPAN -- SPOT EXCHANGE RATE, YEN/US$, Business day
	'AUDUSD': "FED/RXI_US_N_B_AL", #AUSTRALIA -- SPOT EXCHANGE RATE US$/AU$ (RECIPROCAL OF RXI_N.B.AL), Business day
	'NZDUSD': "FED/RXI_US_N_B_NZ", #NEW ZEALAND -- SPOT EXCHANGE RATE, US$/NZ$ RECIPROCAL OF RXI_N.B.NZ, Business day
	'USDCHF': "FED/RXI_N_B_SZ", #SWITZERLAND -- SPOT EXCHANGE RATE, FRANCS/US$, Business day
	'USDNOK': "FED/RXI_N_B_NO", #NORWAY -- SPOT EXCHANGE RATE, KRONER/US$, Business day
	'USDCNY': "FED/RXI_N_B_CH", #CHINA -- SPOT EXCHANGE RATE, YUAN/US$ P.R., Business day
	'USDINR': "FED/RXI_N_B_IN", #INDIA -- SPOT EXCHANGE RATE, RUPEES/US$, Business day
	'DTWEXM': "FRED/DTWEXM", #Trade Weighted U.S. Dollar Index: Major Currencies
	'DTWEXB': "FRED/DTWEXB", #Trade Weighted U.S. Dollar Index: Broad
	#Interest rates
	'DFF': "FRED/DFF", #Effective Federal Funds Rate
	'DTB3': "FRED/DTB3", #3-Month Treasury Bill: Secondary Market Rate
	'DGS5': "FRED/DGS5", #5-Year Treasury Constant Maturity Rate
	'DGS10': "FRED/DGS10",#10-Year Treasury Constant Maturity Rate
	'DGS30': "FRED/DGS30", #30-Year Treasury Constant Maturity Rate
	'T5YIE': "FRED/T5YIE", #5-year Breakeven Inflation Rate
	'T10YIE': "FRED/T10YIE", #10-year Breakeven Inflation Rate
	'T5YIFR': "FRED/T5YIFR",#5-Year, 5-Year Forward Inflation Expectation Rate 
	'TEDRATE': "FRED/TEDRATE", #TED Spread
	'DPRIME': "FRED/DPRIME" #Bank Prime Loan Rate
}

#Local files (Source: yahoo finance)
local_files = {
	'VIX':"https://raw.githubusercontent.com/YizhouTang/SP500_Classfication_Models/master/data/VIX.csv", #Vix index
	#Sector ETFs:
	'XLE':"https://raw.githubusercontent.com/YizhouTang/SP500_Classfication_Models/master/data/XLE.csv", #Energy Select Sector SPDR Fund 
	'XLF':"https://raw.githubusercontent.com/YizhouTang/SP500_Classfication_Models/master/data/XLF.csv", #Financial Select Sector SPDR Fund
	'XLU':"https://raw.githubusercontent.com/YizhouTang/SP500_Classfication_Models/master/data/XLU.csv", #Utilities Select Sector SPDR Fund
	'XLI':"https://raw.githubusercontent.com/YizhouTang/SP500_Classfication_Models/master/data/XLI.csv", #Industrial Select Sector SPDR Fund
	'XLK':"https://raw.githubusercontent.com/YizhouTang/SP500_Classfication_Models/master/data/XLK.csv", #Technology Select Sector SPDR Fund
	'XLV':"https://raw.githubusercontent.com/YizhouTang/SP500_Classfication_Models/master/data/XLV.csv", #Health Care Select Sector SPDR Fund
	'XLY':"https://raw.githubusercontent.com/YizhouTang/SP500_Classfication_Models/master/data/XLY.csv", #Consumer Discretionary Select Sector SPDR Fund
	'XLP':"https://raw.githubusercontent.com/YizhouTang/SP500_Classfication_Models/master/data/XLP.csv", #Consumer Staples Select Sector SPDR Fund
	'XLB':"https://raw.githubusercontent.com/YizhouTang/SP500_Classfication_Models/master/data/XLB.csv", #Materials Select Sector SPDR Fund
}

#econ features that are not daily (weekly, monthly & quarterly data - need to ffill)
sparse_econ_data = {
	#Growth
	'GDP' :"FRED/GDPPOT", #Gross Domestic Product
	'GDPC1': "FRED/GDPC1", #Real Gross Domestic Product
	'GDPPOT': "FRED/GDPC1", #Real Potential Gross Domestic Product
	#Prices and Inflation
	'CPIAUCSL': "FRED/CPIAUCSL", #Consumer Price Index for All Urban Consumers: All Items
	'CPILFESL': "FRED/CPILFESL", #Consumer Price Index for All Urban Consumers: All Items Less Food & Energy
	'GDPDEF': "FRED/GDPDEF", #Gross Domestic Product: Implicit Price Deflator
	#Money Supply
	'BASE': "FRED/BASE", #St. Louis Adjusted Monetary Base
	'M1': "FRED/M1", #M1 Money Stock
	'M2': "FRED/M2", #M2 Money Stock
	'M1V': "FRED/M1V", #Velocity of M1 Money Stock
	'M2V': "FRED/M2V", #Velocity of M2 Money Stock
	#Employment
	'UNRATE': "FRED/UNRATE", #Civilian Unemployment Rate
	'NROU': "FRED/NROU", #Natural Rate of Unemployment (Long-Term)
	'NROUST': "FRED/NROUST", #Natural Rate of Unemployment (Short-Term)
	'CIVPART': "FRED/CIVPART", #Civilian Labor Force Participation Rate
	'EMRATIO': "FRED/EMRATIO", #Civilian Employment-Population Ratio
	'UNEMPLOY': "FRED/UNEMPLOY", #Unemployed level
	'PAYEMS': "FRED/PAYEMS", #All Employees: Total nonfarm
	'MANEMP': "FRED/MANEMP", #All Employees: Manufacturing
	'ICSA': "FRED/ICSA", #Initial Claims
	#Income and Expenditure
	'MEHOINUSA672N': "FRED/MEHOINUSA672N", #Real Median Household Income in the United States
	'DSPIC96': "FRED/DSPIC96", #Real Disposable Personal Income
	'PCE': "FRED/PCE", #Personal Consumption Expenditures
	'PCEDG': "FRED/PCEDG", #Personal Consumption Expenditures: Durable Goods
	'PSAVERT': "FRED/PSAVERT", #Personal Saving Rate
	'RRSFS': "FRED/RRSFS", #Real Retail and Food Services Sales
	'DSPI': "FRED/DSPI", #Disposable personal income
	#Debt
	'GFDEBTN': "FRED/GFDEBTN", #Federal Debt: Total Public Debt
	'GFDEGDQ188S': "FRED/GFDEGDQ188S", #Federal Debt: Total Public Debt as Percent of Gross Domestic Product
	'EXCSRESNW': "FRED/EXCSRESNW", #Excess Reserves of Depository Institutions
	'TOTCI': "FRED/TOTCI", #Commercial and Industrial Loans, All Commercial Banks
	#Other Economic Indicators
	'INDPRO': "FRED/INDPRO", #Industrial Production Index
	'TCU': "FRED/TCU", #Capacity Utilization: Total Industry
	'HOUST': "FRED/HOUST", #Housing Starts: Total: New Privately Owned Housing Units Started
	'GPDI': "FRED/GPDI", #Gross Private Domestic Investment
	'CP': "FRED/CP", #Corporate Profits After Tax (without IVA and CCAdj)
	'STLFSI': "FRED/STLFSI", #St. Louis Fed Financial Stress Index
	'USSLIND': "FRED/USSLIND", #Leading Index for the United States
	'S&P 500 Dividend Yield by Month': "MULTPL/SP500_DIV_MONTH",#12-month real dividend per share inflation adjusted February, 2020 dollars. Data courtesy Standard & Poor's and Robert Shiller.
	'S&P 500 Earnings by Month':"MULTPL/SP500_EARNINGS_MONTH",#S&P 500 Earnings Per Share. 12-month real earnings per share inflation adjusted, constant February, 2020 dollars. Sources: Standard & Poor's for current S&P 500 Earnings. Robert Shiller and his book Irrational Exuberance for historic S&P 500 Earnings.
	'S&P 500 Earnings Yield by Month':"MULTPL/SP500_EARNINGS_YIELD_MONTH",#S&P 500 Earnings Yield. Earnings Yield = trailing 12 month earnings divided by index price (or inverse PE) Yields following September, 2019 (including current yield) are estimated based on 12 month earnings through September, 2019 the latest reported by S&P. Source: Standard & Poor's
	'S&P 500 PE Ratio by Month':"MULTPL/SP500_PE_RATIO_MONTH",#Price to earnings ratio, based on trailing twelve month as reported earnings. Current PE is estimated from latest reported earnings and current market price. Source: Robert Shiller and his book Irrational Exuberance for historic S&P 500 PE Ratio.
}


