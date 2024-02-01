import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpmax, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv('content/Dataset.csv',sep=',')
df

items = set()
for col in df:
  items.update(df[col].unique())
print(items)

itemset = set(items)
encond_vals = []
for index, row in df.iterrows():
  rowset = set(row)
  labels = {}
  uncommons = list(itemset-rowset)
  commons = list(itemset.intersection(rowset))
  for uc in uncommons:
    labels[uc] = 0
  for com in commons:
    labels[com] = 1
  encond_vals.append(labels)
ohe_df = pd.DataFrame(encond_vals)
ohe_df

freq_items = apriori(ohe_df,min_support=0.1,
                     use_colnames=True,verbose=1)
freq_items

rules = association_rules(freq_items,metric='confidence',
                          min_threshold=0.1)
rules.head()
#confidence: Conf(X->Y) = Supp(X union Y)/Supp(Y)
#lift: Supp(X unio Y) / (Supp(X)*Supp(Y))
#Conviction: (1-Supp(Y)) / (1-Conf(X->Y))

dataset = df.values.tolist()
values = []
for ll in dataset:
  ll = [item for item in ll if not(pd.isna(item))==True]
  values.append(ll)

te = TransactionEncoder()
te_aux = te.fit(values).transform(values)
new_df = pd.DataFrame(te_aux,columns=te.columns_)


frequent_itemsets = fpgrowth(new_df,min_support=0.001,use_colnames=True)
frequent_itemsets

rules = association_rules(frequent_itemsets,
                          metric='confidence',min_threshold=0.001)
rules.head(100)

rules['antecedent_len'] = rules['antecedents'].apply(lambda x: len(x))

rules[(rules['lift'] >= 3)]