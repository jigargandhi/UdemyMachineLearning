# Support, Confidence, list
# Support (M) = Who watched M/ total Number 
# Confidence = Users who have seen M1 and M2/ Who has seen M1
# Lift = Confidence/Support
#1. set min support and confidence
#2. Take support > min support 
#3. Take all rules having confidence > min
#4. Sort the rules by decreasing lift
# Companies like amazon use not just apriori but more sophisticated. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Market_Basket_Optimisation.csv",header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

from apyori import apriori
# min_support = 3*7/7500
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

results = list(rules)
    