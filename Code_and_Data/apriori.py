# Apriori
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
# Training model
from apyori import apriori
rules = apriori(transactions,
                min_support = 0.003,
                min_confidence = 0.2,
                min_lift = 3,
                min_length = 2)

# Visualising the results

results = list(rules)
results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]) + '\nInfo:\t' + str(results[i][2]))