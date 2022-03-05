import numpy as np
import pandas as pd

# series

mydict = {'Talha': 50, 'Nebi': 40, 'Kumru': 30}

serie = pd.Series(mydict)

print(serie)

print('*' * 100)

# dataframe
s = serie.index.values
s1 = ['Name', 'Name', 'Name']
s2 = list(zip(s1, s))

index = pd.MultiIndex.from_tuples(s2)

data = np.random.randn(3, 3)

df = pd.DataFrame(data, index=index, columns=['Price', 'Age', 'Hours'])

df[['Price', 'Age']]  # select columns
df.loc['Name'] # select row

print(df)

print('*' * 100)

branch_dict = {
    "Department": ["Software", "Software", "Trade", "Trade", "Law", "Law"],
    "Name": ["Talha", "Nebi", "Kumru", "Ahmet", "Ali", "Veli"],
    "Price": [100, 150, 200, 300, 400, 500]
}

br_df = pd.DataFrame(branch_dict)
print(br_df)

print('-' * 100)

group_department = br_df.groupby("Department")
print(group_department.count())
print('-' * 100)
print(group_department.mean())
print('-' * 100)
print(group_department.max())
print('-' * 100)
print(group_department.describe())
print('-' * 100)
