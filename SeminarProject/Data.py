##########################################
# Data.py
##########################################
# Wrappers for data used to train
# NLP models.

from bs4 import BeautifulSoup as Soup
from CorporateFiling import CorporateFiling
import dateutil.parser as dtparser
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re

class CompanyData:
    """
    * Company returns and attributes used
    for modeling impact of earnings release sentiment
    on returns.
    """
    def __init__(self, returns, attributes):
        """
        * Convert to form usable by machine learning modes.
        """
        self.__Initialize(returns, attributes)
    ##########
    # Properties:
    ##########
    @property
    def EarningTimeStamp(self):
        return self.__earningtimestamp
    @property
    def EarningTimeStampEnd(self):
        return self.__earningtimestampend
    @property
    def EarningTimeStampStart(self):
        return self.__earningtimestampstart
    @property
    def Industry(self):
        return self.__industry
    @property
    def Name(self):
        return self.__name
    @property
    def NumShares(self):
        return self.__numshares
    @property
    def Returns(self):
        """
        * { TimeStamp -> Return }.
        """
        return self.__returns
    @property
    def Sector(self):
        return self.__sector
    @EarningTimeStamp.setter
    def EarningTimeStamp(self, val):
        self.__earningtimestamp = dtparser.parse(val)
    @EarningTimeStampEnd.setter
    def EarningTimeStampEnd(self, val):
        self.__earningtimestampend = dtparser.parse(val)
    @EarningTimeStampStart.setter
    def EarningTimeStampStart(self, val):
        self.__earningtimestampstart = dtparser.parse(val)
    @Industry.setter
    def Industry(self):
        return self.__industry
    @Name.setter
    def Name(self, val):
        self.__name = val
    @NumShares.setter
    def NumShares(self, val):
        self.__numshares = int(val)
    @Returns.setter
    def Returns(self, val):
        """
        * { TimeStamp -> Return }.
        """
        self.__returns = val
    @Sector.setter
    def Sector(self, val):
        self.__sector = val
    ##########
    # Interface Method:
    ##########
    @staticmethod
    def FromFile(ticker, folder):
        """
        * Pull from html file.
        """
        attrs = [attr for attr in dir(CompanyData) if not attr.startswith('__')]
        with open('%s/%s.xml' % (folder, ticker), 'r') as f:
            tags = Soup(f.readlines(), 'lxml')
            returns = tags.find('Returns')
            attributes = {}
            for attr in attrs:
                attributes[attr] = tags.find(attr)
            return CompanyData(returns, attributes)
    def Transform(self):
        """
        * Transform into vector usable by machine learning models.
        """
        pass
    def ToFile(self, ticker, folder):
        """
        * Convert to xml file.
        """
        attrs = [attr for attr in dir(self) if not attr.startswith('__')]
        with open('%s/%s.xml' % (folder, ticker), 'w') as f:
            soup = Soup(f.readlines(), 'lxml')
            for attr in attrs:
                pass
                #soup.insert(attr, )
    ##########
    # Private Helpers:
    ##########
    def __Initialize(self, returns, attributes):
        """
        * 
        """
        pass


class FilingData:
    """
    * All data used to get filing sentiment.
    """
    __filingExt = re.compile('.filing')
    def __init__(self, filings):
        """
        * Convert multiple CorporateFilings into single dataset 
        useable by machine learning models.
        """
        self.__Initialize(filings)

    ############
    # Properties:
    ############
    @property
    def ReleaseDate(self):
        return self.__releasedate
    @property
    def ReleaseText(self):
        return self.__release
    @property
    def Ticker(self):
        return self.__ticker
    @property
    def WordTokens(self):
        return self.__wordtokens
    ############
    # Interface Methods:
    ############
    def ToFile(self, folder):
        """
        * Output to .filing file.
        """
        signature = '%s/%s_%s.filing' % (folder, self.__ticker, self.__releasedate.strptime('%m_%d_%Y'))
        with open(signature, 'w') as f:
            pass


    @staticmethod
    def FromFolder(self, folder):
        """
        * Read multiple .filing for given company from folder.
        """
        pass
        

