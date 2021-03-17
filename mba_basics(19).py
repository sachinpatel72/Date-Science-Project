# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:50:53 2020

@author: Asus
"""

#Import Libraries----------
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from apyori import apriori

#Importing Dataset
dataset = pd.read_csv('F:/pyWork/pyData/MBA_Basic.csv', header = None)

#Now, Convert Pandas DataFrame into a list of lists.
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
    
#Build the Apriori model.
rules = apriori(transactions, min_support = 0.03, min_confidence = 0.2, min_lift = 2, min_length = 2)
r= list(rules)

#Total no. of rules
len(r)

#Print rules:
print(r)

from pandas import DataFrame
df = DataFrame(r,columns=['Itemsets','Support','Others-Confi & Lift'])
