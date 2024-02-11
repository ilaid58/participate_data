import pandas as pd

pd.options.display.max_columns = 10
train = pd.read_excel('Data_Train.xlsx')
test = pd.read_excel('Data_Test.xlsx')
subm = pd.read_excel('Sample_Submission.xlsx')
print(train.head())
print(train.info())
print('\n\n TEST')
print(test.head())
print(test.info())
print('\n\n SUBMISSION')
print(subm.head())
print(subm.info())
cpt = 0
title = train.groupby('Title')['Title'].value_counts()

