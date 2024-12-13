import pandas as pd
import sys

print(sys.version)
print("Pandas Version ",pd.__version__)

df = pd.read_csv('iris.arff.csv')
pd.set_option('display.max_columns',500)

# print(df)
print(df.head())
print(df.tail())
print(df.columns)
print(df.sample(5)) # Random Sample of the File
