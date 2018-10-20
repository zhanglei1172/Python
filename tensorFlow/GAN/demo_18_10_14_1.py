#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Necessary librarys
import os # it's a operational system library, to set some informations
import random # random is to generate random values

import pandas as pd # to manipulate data frames
import numpy as np # to work with matrix
from scipy.stats import kurtosis, skew # it's to explore some statistics of numerical values

import matplotlib.pyplot as plt # to graphics plot
import seaborn as sns # a good library to graphic plots
# import squarify # to better understand proportion of categorys - it's a treemap layout algorithm

import json # to convert json in df
from pandas.io.json import json_normalize # to normalize the json file

# to set a style to all graphs
# plt.style.use('fivethirtyeight')
#
# columns = ['device', 'geoNetwork', 'totals', 'trafficSource']  # Columns that have json format
#
# dir_path = r"E:/webDownload/data/"
# # you can change to your local
#
# # p is a fractional number to skiprows and read just a random sample of the our dataset.
# p = 0.5  # *** In this case we will use 50% of data set *** #
#
#
# # Code to transform the json format columns in table
# def json_read(df):
#     # joining the [ path + df received]
#     data_frame = dir_path + df
#
#     # Importing the dataset
#     df = pd.read_csv(data_frame,
#                      converters={column: json.loads for column in columns},  # loading the json columns properly
#                      dtype={'fullVisitorId': 'str'},  # transforming this column to string
#                      skiprows=lambda i: i > 0 and random.random() > p)  # Number of rows that will be imported randomly
#
#     for column in columns:  # loop to finally transform the columns in data frame
#         # It will normalize and set the json to a table
#         column_as_df = json_normalize(df[column])
#         # here will be set the name using the category and subcategory of json columns
#         column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
#         # after extracting the values, let drop the original columns
#         df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
#
#     # Printing the shape of dataframes that was imported
#     print(f"Loaded {os.path.basename(data_frame)}. Shape: {df.shape}")
#     return df  # returning the df after importing and transforming
#
# # %%time
# # %%time is used to calculate the timing of this code chunk execution
#
# # We will import the data using the name and extension that will be concatenated with dir_path
# df_train = json_read("train.csv")
# # The same to test dataset
# #df_test = json_read("test.csv")