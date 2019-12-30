#################################
# Main.py
#################################
# Description:
# * 

import csv
import DataBase
import os
import re
from BrandQuery import BrandQuery
from PullTwitterData import TwitterPuller
from CorporateFiling import CorporateFiling, DocumentType
from SeminarProject import SeminarProject
    
if __name__ == '__main__':
    type = DocumentType.TENK
    doc = CorporateFiling('amzn', type, date = '20190201')
    query = BrandQuery([])

    #tickerPath = "C:\\Users\\rutan\\OneDrive\\Desktop\\Fordham MSQF Courses\\Fall 2019\\Research Seminar\\Project\\Project\\XLY_All_Holdings.csv"
    #db = DataBase.MYSQLDatabase("root", "Correlation$", "127.0.0.1", "Research_Seminar_Project")
    #seminar = SeminarProject(tickerPath, db)
    #seminar.ExecuteAll()

