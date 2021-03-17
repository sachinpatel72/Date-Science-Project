# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:22:45 2020

@author: Asus
"""

#-------------------------Reading & Writing data in Files----------------------

import pandas

# Reading CSV Files with Pandas:
df = pandas.read_csv('F:/pyWork/pyData/User_Data.csv')
print(df)

# Writing CSV Files with Pandas:
df.to_csv('F:/pyWork/pyData/IIT-B.csv')

# Reading Excel Files with Pandas
df1 = pandas.read_excel('F:/pyWork/pyData/User_Data.xlsx')

df1 = pandas.read_excel('User_Data.xlsx')
print(df1)

# Writing Excel Files with Pandas 
df1.to_excel('IIT-B.xlsx')
df2 = pandas.DataFrame(df1)
print (df2)