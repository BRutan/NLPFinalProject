#################################################
# Main.py
#################################################
# Perform main steps.

from Classifiers import DataSplit, LogisticRegression, NaiveBayes
from CorpDataPuller import CorpDataPuller
from CorporateFiling import CorporateFiling, DocumentType, PullingSteps 
from Data import CompanyData, FilingData
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, MONTHLY
import os
import vaex

def PullTickers():
    """
    * Pull in all tickers from local Tickers.txt
    file.
    """
    start = datetime(year = 2006, month = 1, day = 1)
    end = datetime(year = 2020, month = 12, day = 16)
    # Read in all tickers:
    with open('Tickers.txt', 'r') as f:
        tickers = set(f.readline().split(','))
    months = list(rrule(MONTHLY, dtstart = start, until = end))
    months = [months[num] for num in range(0, len(months)) if num % 4 == 0]
    return tickers, months

def PullReturnsAndAttributes(tickers, months):
    """
    * Pull single-day historical returns for day-of/day-after
    earnings have been released for ticker, as well as
    attributes for given company.
    """
    priceTypes = ['Adj Close']
    attributes = ['sector', 'industry', 'earningsTimestamp', 'earningsTimestampStart', 'earningsTimestampEnd']
    puller = CorpDataPuller(attributes, priceTypes)
    dataFolder = 'CorpData'
    corpData = {}
    if not os.path.exists(dataFolder):
        os.mkdir(dataFolder)
    for group in tickers:
        if not os.path.exists('%s/%s.xml' % (dataFolder, group[0])):
            # Pull single day returns after/before earnings have been released: 
            attributes = puller.GetAttributes(group[0])
            prices = {}
            for num in len(months):
                prices[months[num]] = puller.GetAssetPrices(group[0], months[num])
            corpData[group[0]] = CompanyData(prices, attributes)
            corpData[group[0]].ToFile(group[0], dataFolder)
        else:
            corpData[group[0]] = CompanyData.FromFile(group[0], dataFolder)
    return corpData

def PullFilings(tickers, months):
    """
    * Pull all quarterly/yearly filings for given
    ticker during given period.
    """
    filings = {}
    filingsFolder = 'Filings'
    if not os.path.exists(filingsFolder):
        os.mkdir(filingsFolder)
    steps = PullingSteps(True, False, False)
    end_date = max(months)
    for ticker in tickers:
        targetFolder = '%s/%s' % (filingsFolder, ticker)
        if not os.path.exists(targetFolder):
            os.mkdir('%s/%s' % (filingsFolder, ticker))
            # Start by finding most recent 10-K:
            num = 0
            filings[ticker] = {}
            ten_k = CorporateFiling(ticker, DocumentType.TENK, steps, date = months[0])
            start_date = ten_k.Date
            filings[ticker][start_date] = ten_k
            while start_date < end_date:
                start_date += relativedelta(months = 3)
                num += 1
                type_ = DocumentType.TENQ if num % 4 == 0 else DocumentType.TENK
                filings[ticker][start_date] = CorporateFiling(ticker, type_, steps, date = start_date)
        else:
            # Pull in all filings from folder:
            filings[group[0]] = FilingData.FromFolder(filingsFolder)
            
def TrainModels(corpData, filings):
    """
    * Train various sentiment classifiers using 
    reuters dataset.
    """
    raw_data = None
    split = DataSplit(raw_data, .7)
    return [LogisticRegression(split), NaiveBayes(split)]

def EvaluatePerformance(corpData, filings, models):
    """
    * Evaluate impact of positive/negative earnings releases on
    same/next-day returns.
    """
    results = {}
    for corp in corpData:
        for model in models:
            pass

def Main():
    """
    * Perform key steps in order.
    """
    tickers, months = PullTickers()
    filings = PullFilings(tickers, months)
    corpData = PullReturnsAndAttributes(tickers)
    # Perform modeling using filing sentiments and day-of returns:
    models = TrainModels(corpData, filings)
    EvaluatePerformance(corpData, filings, models)

if __name__ == '__main__':
    Main()

